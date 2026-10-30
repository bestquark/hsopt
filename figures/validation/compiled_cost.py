#!/usr/bin/env python3
"""Count or project a frozen protocol's T gates across HamLib, without simulation.

All counts are for one physical evolution U(t), including PF repetitions. They
are not QPE T95. Large-system parameters and analytical synthesis costs are
explicit resource extrapolations; no energy-accuracy claim follows from them.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from _hamlib_scaling import inspect_text, iter_terms, load_text, resolve_archives
from _tgate_scaling import allocate_parameters, count_policy, fit_beta_model
from hsopt.evaluation import compiled_unitary as ev


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def dump(path, data):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def instance_key(row):
    return "|".join(str(row[k]) for k in ("collection", "file", "dataset"))


def synthesis_model(rows, *, profile="pauli"):
    """Fit mean T/rotation versus log2(1/eps), using baseline counts only."""
    x = np.array([math.log2(1 / float(r["uniform_epsilon"])) for r in rows])
    rotations = []
    for row in rows:
        value = row.get("scheduled_rotations_per_step")
        if value is None:
            if profile != "pauli":
                raise ValueError("a grouped cost model requires scheduled_rotations_per_step")
            value = 2 * int(row["nonidentity_terms"]) - 1
        rotations.append(int(value))
    if any(value < 1 for value in rotations):
        raise ValueError("scheduled rotation counts must be positive")
    y = np.array(
        [
            float(row["baseline_one_step_t_count"]) / count
            for row, count in zip(rows, rotations, strict=True)
        ]
    )
    design = np.column_stack((x, np.ones(len(x))))
    gamma, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    if gamma <= 0:
        raise ValueError("baseline data do not support a positive synthesis-cost slope")
    predicted = np.maximum(0, design @ [gamma, offset])
    return {
        "method": "baseline-only-log-tolerance-least-squares",
        "formula": "max(0, gamma * log2(1/epsilon) + offset)",
        "gamma": float(gamma),
        "offset": float(offset),
        "training_count": len(rows),
        "per_rotation_training_rmse": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "compiler_exact": False,
        "energy_accuracy_certified": False,
        "scope": "cost surrogate, not an integer CUDA-Q circuit count or upper bound",
    }


def _method_ast(path, name):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProductFormula")
    return ast.dump(
        next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name),
        include_attributes=False,
    )


def streamable_pf2(baseline_path, policy_path):
    """The bounded-memory fast path is limited to the canonical PF2 schedule."""
    canonical = Path(__file__).resolve().parents[2] / "experiments/abs_protocol/initial_pf.py"
    expected = _method_ast(canonical, "build_step")
    for path in (baseline_path, policy_path):
        module = ev.load_policy(Path(path))
        if getattr(module.ProductFormula, "hamiltonian_representation", "pauli") != "pauli":
            raise ValueError("a fermionic representation requires --full-schedule")
        if _method_ast(path, "build_step") != expected:
            raise ValueError(
                "streaming requires canonical lexicographic PF2; use --method cudaq "
                "or --full-schedule for a grouped/custom protocol"
            )
        tree = ast.parse(Path(path).read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            if any(isinstance(n, ast.FunctionDef) and n.name == "basis_change" for n in cls.body):
                raise ValueError("a Clifford-frame policy requires --full-schedule")
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "synthesis_epsilon"
            )
            for node in ast.walk(method):
                target = node
                while isinstance(target, (ast.Attribute, ast.Subscript)):
                    target = target.value
                rooted_self = isinstance(target, ast.Name) and target.id == "self"
                if rooted_self and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
                    raise ValueError("stateful synthesis tolerance requires --full-schedule")
                if isinstance(node, ast.Call):
                    target = node.func
                    while isinstance(target, (ast.Attribute, ast.Subscript)):
                        target = target.value
                    if isinstance(target, ast.Name) and target.id == "self":
                        raise ValueError("synthesis methods on self require --full-schedule")


def analysis_sources():
    """Pin count semantics while allowing independent figure-style changes."""
    root = Path(__file__).resolve().parent
    sources = {
        name: sha(root / name)
        for name in ("_hamlib_scaling.py", "_tgate_scaling.py", "_streamed_tgate_count.py")
    }
    methods = {
        "sha",
        "read_csv",
        "instance_key",
        "synthesis_model",
        "_method_ast",
        "streamable_pf2",
        "analysis_sources",
        "bound_instance",
        "streamed_analytic_pair",
        "count",
        "ReplayableTerms",
    }
    nodes = [
        node
        for node in ast.parse(Path(__file__).read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in methods
    ]
    sources["count_cli_ast"] = hashlib.sha256(
        ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()
    ).hexdigest()
    package = Path(ev.__file__).resolve().parents[1]
    for name in (
        "evaluation/compiled_unitary.py",
        "evaluation/elote.py",
        "evaluation/clifford.py",
        "campaign/calibration.py",
    ):
        sources[name] = sha(package / name)
    return sources


class ReplayableTerms:
    """Expose actual terms to the frozen constructor without retaining a list."""

    def __init__(self, text, n_qubits, count):
        self.text, self.n_qubits, self.count = text, n_qubits, count

    def __iter__(self):
        return iter_terms(self.text, self.n_qubits)

    def __len__(self):
        return self.count


def bound_instance(problem, policy_path, allocation):
    module = ev.load_policy(Path(policy_path))
    name = problem["name"]
    r, epsilon, beta = (
        allocation[k] for k in ("n_steps", "uniform_epsilon", "reference_trotter_number_beta")
    )
    bound = type(
        "ProductFormula",
        (module.ProductFormula,),
        {
            "baseline_parameters": {name: (r, epsilon, beta)},
            "parameters": {name: (r, epsilon)},
        },
    )
    return bound(problem)


def streamed_analytic_pair(text, problem, statistics, allocation, baseline, policy, model):
    """Apply actual tolerance methods; accumulate the cost model for every gate."""
    terms = ReplayableTerms(text, problem["n_qubits"], statistics["nonidentity_terms"])
    instances = [
        bound_instance({**problem, "terms": terms}, path, allocation) for path in (baseline, policy)
    ]
    repetitions = [p.repetitions() for p in instances]
    if any(type(r) is not int or r <= 0 for r in repetitions):
        raise ValueError("invalid repetitions")
    totals = [0.0, 0.0]
    lo, hi = [1.0, 1.0], [0.0, 0.0]
    for term in terms:
        term = {**term, "original_label": term["label"], "frame_sign": 1}
        multiplicity = 1 if term["label"] == statistics["center_label"] else 2
        for i, (p, r) in enumerate(zip(instances, repetitions, strict=True)):
            step_time = problem["target_time"] / r
            angle = (
                term["coefficient"] * step_time
                if multiplicity == 1
                else 0.5 * term["coefficient"] * step_time
            )
            epsilon = float(p.synthesis_epsilon(dict(term), angle))
            if not math.isfinite(epsilon) or not 0 < epsilon < 1:
                raise ValueError("policy returned an invalid synthesis tolerance")
            lo[i], hi[i] = min(lo[i], epsilon), max(hi[i], epsilon)
            totals[i] += multiplicity * max(
                0.0, model["gamma"] * math.log2(1 / epsilon) + model["offset"]
            )
    return {
        "baseline_t_count": totals[0] * repetitions[0],
        "policy_t_count": totals[1] * repetitions[1],
        "baseline_repetitions": repetitions[0],
        "policy_repetitions": repetitions[1],
        "baseline_epsilon_min": lo[0],
        "baseline_epsilon_max": hi[0],
        "policy_epsilon_min": lo[1],
        "policy_epsilon_max": hi[1],
        "rotations_per_step": 2 * statistics["nonidentity_terms"] - 1,
        "count_method": "analytic_ross_selinger",
        "exact_compiled_count": False,
    }


def count(args):
    output = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo):
        raise ValueError("generated resource tables must live outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    training = read_csv(args.training_table)
    training = [r for r in training if int(r["n_qubits"]) <= args.training_max_qubits]
    if not training:
        raise ValueError("no training calibration rows")
    sealed = json.loads(args.calibrations.read_text())
    model = fit_beta_model(training, profile=args.profile)
    cost = (
        synthesis_model(training, profile=args.profile)
        if args.method == "analytic_ross_selinger"
        else {"method": "unused_literal_cudaq_counts", "gamma": 3.0, "offset": 0.0}
    )
    provenance = json.loads(args.policy_provenance.read_text()) if args.policy_provenance else None
    if not args.allow_different_policy:
        if provenance is None or provenance.get("policy_sha256") != sha(args.policy):
            raise ValueError(
                "archived comparisons require matching --policy-provenance; "
                "use --allow-different-policy for a new frozen policy"
            )
    rows = resolve_archives(args.catalog, args.archive_root)
    rows = sorted(
        (r for r in rows if int(r["nqubits"]) <= args.max_qubits),
        key=lambda r: (int(r["nqubits"]), instance_key(r)),
    )
    if args.case_id:
        rows = [r for r in rows if instance_key(r) in args.case_id]
        if len(rows) != len(set(args.case_id)):
            raise ValueError("unknown --case-id")
    config = {
        "schema": "hamlib-count-only-scaling-v1",
        "catalog_sha256": sha(args.catalog),
        "training_sha256": sha(args.training_table),
        "calibrations_sha256": sha(args.calibrations),
        "policy_sha256": sha(args.policy),
        "baseline_policy_sha256": sha(args.baseline_policy),
        "training_max_qubits": args.training_max_qubits,
        "max_qubits": args.max_qubits,
        "profile": args.profile,
        "method": args.method,
        "full_schedule": args.full_schedule,
        "archived_policy_comparison": not args.allow_different_policy,
        "policy_label": args.policy_label,
        "policy_provenance": provenance,
        "selected_case_ids": [instance_key(r) for r in rows],
        "shard_count": args.shard_count,
        "beta_model": model,
        "synthesis_model": cost,
        "count_scope": "T gates in one complete U(t), including r, excluding QPE and state preparation",
        "energy_accuracy_certified_for_extrapolation": False,
        "synthesis": ev._synthesis_metadata() if args.method == "cudaq" else None,
        "analysis_sources": analysis_sources(),
    }
    manifest = output / "scaling_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("output belongs to another configuration; choose a new output directory")
    if not manifest.exists():
        dump(manifest, config)
    training_by_key = {instance_key(r): r for r in training}
    stream = not args.full_schedule
    if stream:
        if args.profile != "pauli":
            raise ValueError("grouped profiles require --full-schedule")
        streamable_pf2(args.baseline_policy, args.policy)
    global_selected = len(rows)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require 0 <= shard-index < shard-count")
    rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]
    suffix = f".shard-{args.shard_index:03d}" if args.shard_count > 1 else ""
    table = output / f"scaling_counts{suffix}.jsonl"
    done = {}
    if table.exists():
        for line in table.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok":
                done[row["case_id"]] = row
    errors = []
    with table.open("a", buffering=1) as writer:
        for index, row in enumerate(rows, 1):
            key = instance_key(row)
            if key in done:
                continue
            started = time.monotonic()
            result = {
                "case_id": key,
                "n_qubits": int(row["nqubits"]),
                "molecule": row["molecule"],
                "status": "error",
            }
            try:
                if (
                    sha(args.policy) != config["policy_sha256"]
                    or sha(args.baseline_policy) != config["baseline_policy_sha256"]
                ):
                    raise ValueError("a policy changed during counting; use frozen snapshots")
                text, archive_sha = load_text(row, cache_dir=args.archive_cache)
                stats = inspect_text(text, int(row["nqubits"]), expected_terms=int(row["terms"]))
                small = training_by_key.get(key)
                problem = {
                    "name": small["problem_name"] if small else "HamLib " + key,
                    "n_qubits": int(row["nqubits"]),
                    "electrons": int(row["effective_electrons_for_algorithm"]),
                    "identity_coefficient": stats["identity_coefficient"],
                    "target_time": (
                        float(small["target_time"])
                        if small
                        else 1.0 / stats["one_norm_nonidentity_hartree"]
                    ),
                    "nonidentity_terms": stats["nonidentity_terms"],
                    "terms": ReplayableTerms(text, int(row["nqubits"]), stats["nonidentity_terms"]),
                }
                # Extrapolation and Eq34 need a term count, not a Hilbert-space array.
                record = sealed.get(small["case_id"]) if small else None
                if record is not None:
                    record = {**record}
                    record.setdefault("policy_profile", record.get("profile", "pauli"))
                extra = {}
                if args.profile == "fermionic_nested" and record is None:
                    problem["terms"] = list(iter_terms(text, problem["n_qubits"]))
                    # Reproduce the selected grouped initializer's forward decomposition,
                    # without solving an ideal PF or importing any saved exact energy.
                    module = ev.load_policy(args.baseline_policy)
                    name = problem["name"]
                    module.ProductFormula = type(
                        "ProductFormula",
                        (module.ProductFormula,),
                        {
                            "baseline_parameters": {name: (1, 1e-10, 0.0)},
                            "parameters": {name: (1, 1e-10)},
                        },
                    )
                    _, rotations, metrics = ev._product_formula_schedule(problem, module)
                    groups = metrics["elote"]["groups"]
                    ordered = [
                        {**term, "coefficient": term["angle"] / problem["target_time"]}
                        for group in groups
                        for term in group
                    ]
                    extra = {"allocation_terms": ordered, "rotations_per_step": len(rotations)}
                allocation = allocate_parameters(
                    problem, sealed=record, beta_model=model, profile=args.profile, **extra
                )
                if stream:
                    if args.method == "analytic_ross_selinger":
                        values = streamed_analytic_pair(
                            text,
                            problem,
                            stats,
                            allocation,
                            args.baseline_policy,
                            args.policy,
                            cost,
                        )
                    else:
                        from _streamed_tgate_count import streamed_cudaq_pair

                        formulas = [
                            bound_instance(problem, p, allocation)
                            for p in (args.baseline_policy, args.policy)
                        ]
                        values = streamed_cudaq_pair(
                            text,
                            problem["n_qubits"],
                            stats["center_label"],
                            *formulas,
                            problem["target_time"],
                            workers=args.synthesis_workers,
                        )
                else:
                    problem["terms"] = list(iter_terms(text, problem["n_qubits"]))
                    counts = [
                        count_policy(
                            problem,
                            p,
                            allocation,
                            method=args.method,
                            gamma=cost["gamma"],
                            offset=cost["offset"],
                        )
                        for p in (args.baseline_policy, args.policy)
                    ]
                    field = (
                        "compiled_evolution_t_count"
                        if args.method == "cudaq"
                        else "estimated_evolution_t_count"
                    )
                    values = {
                        "baseline_t_count": counts[0][field],
                        "policy_t_count": counts[1][field],
                        "baseline_repetitions": counts[0]["n_steps"],
                        "policy_repetitions": counts[1]["n_steps"],
                        "count_method": args.method,
                        "exact_compiled_count": args.method == "cudaq",
                        "count_details": counts,
                    }
                result.update(
                    values,
                    status="ok",
                    archive_sha256=archive_sha,
                    nonidentity_terms=stats["nonidentity_terms"],
                    allocation_source=allocation["allocation_source"],
                    allocation_extrapolated=allocation["allocation_extrapolated"],
                    beta=allocation["reference_trotter_number_beta"],
                    baseline_epsilon=allocation["uniform_epsilon"],
                    target_time=problem["target_time"],
                    energy_accuracy_certified=False,
                )
                if small and not args.allow_different_policy:
                    result["archived_baseline_t_count"] = int(
                        small["baseline_compiled_evolution_t_count"]
                    )
                    result["archived_policy_t_count"] = int(
                        small["learned_compiled_evolution_t_count"]
                    )
                    result["archived_energy_accuracy_passed"] = (
                        small["baseline_passed"] == small["learned_passed"] == "True"
                    )
                    if args.method == "cudaq" and not args.allow_different_policy:
                        if (
                            result["baseline_t_count"] != result["archived_baseline_t_count"]
                            or result["policy_t_count"] != result["archived_policy_t_count"]
                        ):
                            raise RuntimeError("literal counts disagree with frozen trajectory")
                if (
                    sha(args.policy) != config["policy_sha256"]
                    or sha(args.baseline_policy) != config["baseline_policy_sha256"]
                ):
                    raise ValueError("a policy changed during counting; results were not recorded")
                del text, problem
            except Exception as error:
                result.update(status="error", error=f"{type(error).__name__}: {error}")
                errors.append(result)
            if result["status"] == "ok":
                done[key] = result
            result["wall_seconds"] = time.monotonic() - started
            writer.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                json.dumps({"completed": index, "total": len(rows), **result}, allow_nan=False),
                flush=True,
            )
    summary = {
        "selected": len(rows),
        "global_selected": global_selected,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "completed": len(done),
        "failed": len(errors),
        "complete": len(done) == len(rows) and not errors,
        "errors": [{"case_id": r["case_id"], "error": r["error"]} for r in errors],
    }
    dump(output / f"scaling_summary{suffix}.json", summary)
    if not summary["complete"]:
        raise RuntimeError("count-only census is incomplete; inspect scaling_summary.json")


def merge(args):
    root = args.input_dir.resolve()
    summaries = [
        json.loads(p.read_text()) for p in sorted(root.glob("scaling_summary.shard-*.json"))
    ]
    if not summaries or any(not s["complete"] for s in summaries):
        raise ValueError("all shards must finish successfully before merging")
    expected_shards = summaries[0]["shard_count"]
    if len(summaries) != expected_shards or {s["shard_index"] for s in summaries} != set(
        range(expected_shards)
    ):
        raise ValueError("missing or duplicate shard summaries")
    rows = {}
    for path in sorted(root.glob("scaling_counts.shard-*.jsonl")):
        shard_rows = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "ok":
                shard_rows[row["case_id"]] = row
        if rows.keys() & shard_rows.keys():
            raise ValueError("shards contain duplicate instance IDs")
        rows.update(shard_rows)
    selected = summaries[0]["global_selected"]
    manifest = json.loads((root / "scaling_manifest.json").read_text())
    if (
        len(rows) != selected
        or set(rows) != set(manifest["selected_case_ids"])
        or any(s["global_selected"] != selected for s in summaries)
    ):
        raise ValueError("merged census does not match the selected catalog")
    with (root / "scaling_counts.jsonl").open("w") as stream:
        for row in sorted(rows.values(), key=lambda r: (r["n_qubits"], r["case_id"])):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    dump(
        root / "scaling_summary.json",
        {"selected": selected, "completed": len(rows), "failed": 0, "complete": True, "errors": []},
    )


def _geomean(values):
    return math.exp(float(np.mean(np.log(values))))


def calibrated_count(row):
    return "archived_baseline_t_count" in row or (
        row.get("exact_compiled_count") and not row.get("allocation_extrapolated")
    )


def plot(args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    from _paper_plot import GRAPHITE, NVIDIA_DARK_GREEN, apply_paper_style

    root = args.input_dir.resolve()
    manifest = json.loads((root / "scaling_manifest.json").read_text())
    summary = json.loads((root / "scaling_summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("refusing a final figure from an incomplete catalog census")
    data = {}
    for line in (root / "scaling_counts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["status"] == "ok":
            if row["case_id"] in data:
                raise ValueError("duplicate completed case in count table")
            data[row["case_id"]] = row
    expected = set(manifest.get("selected_case_ids", []))
    if (
        not expected
        or set(data) != expected
        or summary["selected"] != len(expected)
        or summary["completed"] != len(expected)
    ):
        raise ValueError("count table does not cover the complete selected catalog")
    rows = list(data.values())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 21,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 17,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["n_qubits"]].append(row)
        actual = calibrated_count(row)
        b = row.get("archived_baseline_t_count", row["baseline_t_count"])
        p = row.get("archived_policy_t_count", row["policy_t_count"])
        if b <= 0 or p <= 0:
            raise ValueError("log-scaled comparison requires positive counts")
        marker = "o" if actual else "."
        for count_, color in ((b, GRAPHITE), (p, NVIDIA_DARK_GREEN)):
            axes[0].scatter(
                row["n_qubits"],
                count_,
                color=color,
                marker=marker,
                s=18 if actual else 9,
                alpha=0.38 if actual else 0.15,
                edgecolors="none",
                rasterized=True,
            )
        axes[1].scatter(
            row["n_qubits"],
            p / b,
            color=NVIDIA_DARK_GREEN,
            marker=marker,
            s=18 if actual else 9,
            alpha=0.5 if actual else 0.2,
            edgecolors="none",
            rasterized=True,
        )
    aggregate = []
    for n, members in sorted(grouped.items()):
        actual = all(calibrated_count(row) for row in members)
        b = [
            r["archived_baseline_t_count"]
            if "archived_baseline_t_count" in r
            else r["baseline_t_count"]
            for r in members
        ]
        p = [
            r["archived_policy_t_count"] if "archived_policy_t_count" in r else r["policy_t_count"]
            for r in members
        ]
        aggregate.append(
            {
                "n_qubits": n,
                "instances": len(members),
                "actual_archived": actual,
                "baseline_geomean_t_count": _geomean(b),
                "policy_geomean_t_count": _geomean(p),
                "relative_geomean_t_count": _geomean([x / y for x, y in zip(p, b, strict=True)]),
            }
        )
    for actual, style in ((True, "-"), (False, "--")):
        selected = [r for r in aggregate if r["actual_archived"] == actual]
        n = [r["n_qubits"] for r in selected]
        for field, color in (
            ("baseline_geomean_t_count", GRAPHITE),
            ("policy_geomean_t_count", NVIDIA_DARK_GREEN),
        ):
            axes[0].plot(n, [r[field] for r in selected], style, color=color, linewidth=2.5)
        axes[1].plot(
            n,
            [r["relative_geomean_t_count"] for r in selected],
            style,
            color=NVIDIA_DARK_GREEN,
            linewidth=2.5,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$T$ gates per evolution $U(t)$")
    axes[1].set_ylabel(r"Relative $T$ count, $T_{\rm policy}/T_{\rm baseline}$")
    axes[1].axhline(1, color=GRAPHITE, linestyle=":", linewidth=1.5)
    cutoff = manifest["training_max_qubits"]
    maximum = max(r["n_qubits"] for r in rows)
    for i, ax in enumerate(axes):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({n for n in (4, 8, 16, 32, 64, 128) if n <= maximum} | {maximum}))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"Qubits, $n$")
        ax.axvline(cutoff + 0.5, color="0.6", linewidth=1.1, linestyle=":")
        ax.axvspan(cutoff + 0.5, maximum * 1.05, color="0.94", zorder=0)
        ax.set_xlim(3.7, maximum * 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.14)
        ax.text(0, 1.03, rf"\textbf{{({chr(97 + i)})}}", transform=ax.transAxes)
    axes[0].legend(
        handles=[
            Line2D([], [], color=GRAPHITE, label="Uniform PF2"),
            Line2D([], [], color=NVIDIA_DARK_GREEN, label=manifest["policy_label"]),
            Line2D([], [], color="0.5", linestyle="--", label="Extrapolated parameters"),
        ],
        loc="upper left",
    )
    axes[1].text(
        0.98,
        0.04,
        "Lines: geometric means",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
    )
    for suffix in ("pdf", "png"):
        path = output / f"hamlib_tgate_scaling.{suffix}"
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)
    dump(output / "hamlib_tgate_scaling_aggregate.json", aggregate)
    with (output / "hamlib_tgate_scaling.csv").open("w", newline="") as stream:
        columns = [
            "case_id",
            "molecule",
            "n_qubits",
            "nonidentity_terms",
            "baseline_t_count",
            "policy_t_count",
            "baseline_repetitions",
            "policy_repetitions",
            "allocation_source",
            "count_method",
            "exact_compiled_count",
            "energy_accuracy_certified",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["n_qubits"], r["case_id"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("count", help="checkpointed count-only or resource-projection census")
    for flag in (
        "catalog",
        "archive-root",
        "training-table",
        "calibrations",
        "policy",
        "baseline-policy",
        "output-dir",
    ):
        c.add_argument("--" + flag, type=Path, required=True)
    c.add_argument("--training-max-qubits", type=int, default=14)
    c.add_argument("--max-qubits", type=int, default=120)
    c.add_argument("--profile", choices=("pauli", "fermionic_nested"), default="pauli")
    c.add_argument("--method", choices=("cudaq", "analytic_ross_selinger"), default="cudaq")
    c.add_argument("--full-schedule", action="store_true")
    c.add_argument(
        "--allow-different-policy",
        action="store_true",
        help="allow a new frozen policy instead of verifying old archived counts",
    )
    c.add_argument(
        "--policy-provenance",
        type=Path,
        help="frozen source-policy provenance for archived small-case comparisons",
    )
    c.add_argument("--policy-label", default="Precision-only policy")
    c.add_argument(
        "--archive-cache", type=Path, help="external scratch for verified HDF5 archive extraction"
    )
    c.add_argument("--shard-index", type=int, default=0)
    c.add_argument("--shard-count", type=int, default=1)
    c.add_argument("--synthesis-workers", type=int, default=2)
    c.add_argument("--case-id", action="append", help="collection|file|dataset; repeatable")
    c.set_defaults(func=count)
    p = commands.add_parser("plot", help="render LaTeX figures from a completed count table")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=plot)
    m = commands.add_parser("merge", help="verify and merge completed count-only shards")
    m.add_argument("--input-dir", type=Path, required=True)
    m.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

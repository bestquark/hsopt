#!/usr/bin/env python3
"""Freeze one best shared ELOTE and reproduce Figure-2-style diagnostics.

Export on the data host with the campaign's pinned HSOpt on PYTHONPATH; plot the
small exported tables locally. Neither command synthesizes gates or solves an
eigenproblem. One global feasible policy is used, never per-case cherry picking.
See elote_precision_summary.md for reproduction commands and metric definitions.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def gm(values):
    values = list(values)
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def write_csv(path, rows):
    with (gzip.open(path, "wt", newline="") if path.suffix == ".gz" else path.open("w", newline="")) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(campaign, output):
    from hsopt.evaluation import compiled_unitary as judge

    accepted = []
    for seed in sorted(campaign.glob("seed-*")):
        for path in sorted((seed / "iterations").glob("[0-9][0-9][0-9][0-9]/decision.json")):
            decision = read(path)
            if int(path.parent.name) == 0 or decision.get("accepted") is not True:
                continue
            result = read(path.parent / "result.json")
            if result.get("status") == "ok" and result.get("passes_chemical_accuracy") is True:
                assert sha(path.parent / "candidate.py") == decision["shared_candidate_sha256"]
                accepted.append((result["score"], seed, path.parent, result, decision))
    score, seed, folder, result, decision = min(accepted, key=lambda row: row[0])
    suite = read(seed / "suite.json")
    assert sha(seed / "suite.json") == decision["suite_sha256"]
    assert result["case_count"] == len(suite["cases"])
    metrics = {row["case_id"]: row for row in result["case_metrics"]}
    assert all(row["passes_chemical_accuracy"] and row.get("evaluation_error") is None for row in metrics.values())
    assert math.isclose(gm(row["relative_t95"] for row in metrics.values()), score, rel_tol=1e-12)
    output.mkdir(parents=True, exist_ok=True)
    (output / "best_policy.py").write_bytes((folder / "candidate.py").read_bytes())
    # Full archived receipts are read on the data host, then reduced to their
    # scientific scalars. No Hamiltonian or full trajectory is copied locally.
    fields = ("t_count", "one_step_t_count", "n_steps", "rotations_per_step",
              "confidence_adjusted_t_count", "rotation_schedule_sha256", "passes_chemical_accuracy")
    scalar = []
    for iteration in (seed / "iterations/0000", folder):
        raw = read(iteration / "case_results.json")
        scalar.append({case: {key: row[key] for key in fields} for case, row in raw.items()})
        del raw
    baseline, scored = scalar
    expected = {case["id"] for case in suite["cases"]}
    assert set(baseline) == set(scored) == set(metrics) == expected
    molecules, rotations = [], []
    for case in suite["cases"]:
        key = case["id"]
        problem_path = seed / case["source"]
        problem = judge.load_problem(problem_path)
        parameters = case["parameters"][problem["name"]]
        epsilon0 = float(parameters["uniform_epsilon"])
        assert parameters["baseline"] == "tight"
        for stage, iteration, recorded in (("baseline", seed / "iterations/0000", baseline[key]),
                                            ("best", folder, scored[key])):
            policy_path = iteration / "bound_policies" / (key + ".py")
            policy = judge.load_policy(policy_path)
            pf, schedule, meta = judge._product_formula_schedule(problem, policy)
            assert meta["rotation_schedule_sha256"] == recorded["rotation_schedule_sha256"], key
            assert meta["rotations_per_step"] == recorded["rotations_per_step"]
            assert meta["n_steps"] == recorded["n_steps"]
            assert pf.epsilon == epsilon0
            counts = Counter((term["label"], angle) for term, angle in schedule)
            seen = set()
            cmax = max(abs(float(term["coefficient"])) for term in problem["terms"])
            for term, angle in schedule:
                identity = term["label"], angle
                if identity in seen:
                    continue
                seen.add(identity)
                eps = float(pf.synthesis_epsilon(dict(term), angle))
                assert math.isfinite(eps) and 0 < eps < 1
                if stage == "baseline":
                    assert eps == epsilon0
                label = term["label"]
                rotations.append({
                    "case_id": key, "stage": stage, "label": label,
                    "n_qubits": problem["n_qubits"], "coefficient": term["coefficient"],
                    "coefficient_ratio_cmax": abs(term["coefficient"]) / cmax,
                    "coefficient_ratio_policy_reference": abs(term["coefficient"]) / pf.max_abs_coeff if hasattr(pf, "max_abs_coeff") else None,
                    "support": sum(ch != "I" for ch in label), "diagonal": int(not any(ch in "XY" for ch in label)),
                    "pauli_angle": angle, "rz_angle": -2 * angle,
                    "epsilon": eps, "epsilon0": epsilon0, "epsilon_ratio": eps / epsilon0,
                    "multiplicity_per_step": counts[identity], "n_steps": meta["n_steps"],
                })
        a, b = baseline[key], scored[key]
        assert a["passes_chemical_accuracy"] and b["passes_chemical_accuracy"]
        assert b["confidence_adjusted_t_count"] == metrics[key]["t95"]
        relative = b["confidence_adjusted_t_count"] / a["confidence_adjusted_t_count"]
        assert math.isclose(relative, metrics[key]["relative_t95"], rel_tol=1e-12)
        molecules.append({
            "case_id": key, "n_qubits": problem["n_qubits"], "problem_sha256": sha(problem_path),
            "baseline_t_count": a["t_count"], "best_t_count": b["t_count"],
            "compiled_t_reduction_percent": 100 * (1 - b["t_count"] / a["t_count"]),
            "baseline_t95": a["confidence_adjusted_t_count"], "best_t95": b["confidence_adjusted_t_count"],
            "t95_reduction_percent": 100 * (1 - relative),
            "baseline_n_steps": a["n_steps"], "best_n_steps": b["n_steps"],
            "baseline_rotations_per_step": a["rotations_per_step"], "best_rotations_per_step": b["rotations_per_step"],
            "error_hartree": metrics[key]["error_hartree"], "passes_chemical_accuracy": True,
            "baseline_schedule_sha256": a["rotation_schedule_sha256"], "best_schedule_sha256": b["rotation_schedule_sha256"],
        })
    write_csv(output / "molecules.csv", molecules)
    write_csv(output / "rotations.csv.gz", rotations)
    summary = {"schema": "elote-current-figure2-v1", "captured_at": time.time(),
               "campaign": str(campaign), "seed": seed.name, "iteration": int(folder.name),
               "policy_sha256": sha(folder / "candidate.py"), "suite_sha256": sha(seed / "suite.json"),
               "baseline": "tight", "case_count": len(molecules), "all_pass_chemical_accuracy": True,
               "score_relative_t95_geometric_mean": score, "t95_geometric_reduction_percent": 100 * (1 - score),
               "compiled_geometric_reduction_percent": 100 * (1 - gm(row["best_t_count"] / row["baseline_t_count"] for row in molecules)),
               "schedule_resolver": str(judge.__file__),
               "recorded_result_file": str(folder / "result.json"),
               "recorded_result_sha256": sha(folder / "result.json"),
               "recorded_decision_sha256": sha(folder / "decision.json"),
               "unique_rotation_rows": len(rotations),
               "schedule_hashes_verified": 2 * len(molecules),
               "rotation_scatter": "One point per distinct (case, stage, Pauli label, signed angle); multiplicities and repetitions exported separately.",
               "angle_convention": "pauli_angle=alpha in exp(+i alpha P); compiler Rz angle theta=-2 alpha.",
               "compiled_metric": "t_count: literal Clifford+T circuit for one full evolution, including all Trotter steps; excludes QPE query/repetition factors.",
               "t95_metric": "confidence_adjusted_t_count: full QPE queries and repetitions for 95% success.",
               "case_counts_by_qubits": dict(sorted(Counter(row["n_qubits"] for row in molecules).items())),
               "epsilon_ratio_range": [min(row["epsilon_ratio"] for row in rotations if row["stage"] == "best"),
                                       max(row["epsilon_ratio"] for row in rotations if row["stage"] == "best")]}
    for key in ("compiled_t_reduction_percent", "t95_reduction_percent"):
        values = [row[key] for row in molecules]
        summary[key] = {"mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
    summary["files"] = {name: sha(output / name) for name in ("best_policy.py", "molecules.csv", "rotations.csv.gz")}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def plot(snapshot, output, heldout_summary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator, LogLocator, FuncFormatter
    from _paper_plot import apply_paper_style, NVIDIA_GREEN, NVIDIA_DARK_GREEN, GRAPHITE

    manifest = read(snapshot / "manifest.json")
    for name, digest in manifest["files"].items():
        assert sha(snapshot / name) == digest, name
    with (snapshot / "molecules.csv").open() as stream:
        molecules = list(csv.DictReader(stream))
    with gzip.open(snapshot / "rotations.csv.gz", "rt") as stream:
        rotations = list(csv.DictReader(stream))
    assert len(molecules) == manifest["case_count"]
    assert len(rotations) == manifest["unique_rotation_rows"]
    heldout = []
    heldout_statuses = None
    if heldout_summary is not None:
        test = read(heldout_summary)
        if test.get("schema") != "elote-heldout-audit-v1":
            raise ValueError("unsupported held-out count summary")
        if test.get("policy_sha256") != manifest["policy_sha256"]:
            raise ValueError("held-out counts use a different frozen ELOTE policy")
        heldout_statuses = test["count_statuses"]
        seen = set()
        for row in test["results"]:
            key = row["case_id"]
            if key in seen or int(row["n_qubits"]) <= 16:
                raise ValueError("held-out cases must be unique and larger than training cases")
            seen.add(key)
            if row["count_status"] != "ok":
                continue
            if row["allocation_extrapolated"] is not True:
                raise ValueError("held-out allocation provenance is missing")
            baseline, learned = int(row["baseline_t_count"]), int(row["elote_t_count"])
            if baseline <= 0 or learned < 0:
                raise ValueError("invalid held-out compiled count")
            heldout.append(100 * (1 - learned / baseline))
        if not heldout:
            raise ValueError("no countable held-out Hamiltonians")
    output.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 24, "axes.titlesize": 23,
                         "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18})

    def scatter(ax, parameter):
        xkey = "rz_angle" if parameter == "angle" else "coefficient_ratio_cmax"
        # Uniform precision is a constant reference for every angle. Drawing
        # its exact line avoids stretching the axis for negligible near-zero
        # rotations in the baseline's different splitting. All raw baseline
        # rotations remain in the exported table.
        ax.axhline(1, color="0.55", linestyle="--", linewidth=1.7, label="Tight baseline")
        points = [row for row in rotations if row["stage"] == "best" and abs(float(row[xkey])) > 0]
        xx = [abs(float(row[xkey])) for row in points]
        yy = [float(row["epsilon_ratio"]) for row in points]
        ax.scatter(xx, yy, s=5, color=NVIDIA_GREEN, alpha=0.30, linewidths=0,
                   rasterized=True, label="Best ELOTE")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:g}$"))
        ax.set_xlabel(r"Rotation angle $|\theta|$ (rad)" if parameter == "angle" else r"Relative coefficient $|c_j|/c_{\max}$")
        ax.set_ylabel(r"Relative tolerance $\epsilon_j/\epsilon_{0}$")
        ax.grid(False, which="both")
        legend = ax.legend(loc="upper left", markerscale=2, frameon=False)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        ax.set_ylim(bottom=0.83)

    def histogram(ax, metric):
        key = "compiled_t_reduction_percent" if metric == "compiled" else "t95_reduction_percent"
        values = np.asarray([float(row[key]) for row in molecules])
        mean, median = float(values.mean()), float(np.median(values))
        # Show the actual distribution with data-driven bins; retain every
        # Hamiltonian, including any negative reductions, without clipping.
        bins = np.histogram_bin_edges(
            np.concatenate((values, heldout)) if heldout and metric == "compiled" else values,
            bins="fd",
        )
        padding = max(0.5, 0.08 * float(np.ptp(values)))
        low, high = float(values.min()) - padding, float(values.max()) + padding
        if heldout and metric == "compiled":
            hold = np.asarray(heldout)
            low = min(low, float(hold.min()) - padding)
            high = max(high, float(hold.max()) + padding)
            ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                    color=NVIDIA_GREEN, edgecolor="black", linewidth=0.7, alpha=0.65,
                    label=rf"Training ($n={len(values)}$)")
            ax.hist(hold, bins=bins, weights=np.full(len(hold), 100 / len(hold)),
                    histtype="step", color="#396FA8", linewidth=2.4,
                    label=rf"Held-out ($n={len(hold)}$)")
            ax.axvline(median, color=NVIDIA_DARK_GREEN, linestyle=":", linewidth=1.9)
            ax.axvline(float(np.median(hold)), color="#396FA8", linestyle=":", linewidth=1.9)
        else:
            ax.hist(values, bins=bins, color=NVIDIA_GREEN, edgecolor="black", linewidth=0.7, alpha=0.82)
            ax.axvline(mean, color=NVIDIA_DARK_GREEN, linestyle="--", linewidth=2,
                       label=rf"Mean ${mean:.2f}\%$")
            ax.axvline(median, color=GRAPHITE, linestyle=":", linewidth=2,
                       label=rf"Median ${median:.2f}\%$")
        ax.set_xlabel(r"Compiled $T$-gate reduction (\%)" if metric == "compiled" else r"$T_{95}$ reduction (\%)")
        ax.set_ylabel(r"Hamiltonians (\%)" if heldout and metric == "compiled" else "Hamiltonians")
        ax.set_xlim(low, high); ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(integer=not (heldout and metric == "compiled")))
        ax.legend(loc="upper left", fontsize=17)

    def finish(fig, axes, name):
        for i, ax in enumerate(axes):
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(-0.12, 1.02, rf"({chr(97+i)})", transform=ax.transAxes, va="bottom", fontsize=25)
        for ext in ("pdf", "png"):
            fig.savefig(output / (name + "." + ext), dpi=300)
        plt.close(fig)

    for parameter, metric, name in (("angle", "compiled", "elote_precision_and_compiled_savings"),
                                     ("coefficient", "compiled", "elote_coefficient_precision_and_compiled_savings"),
                                     ("angle", "t95", "elote_precision_and_t95_savings")):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout="constrained")
        scatter(axes[0], parameter); histogram(axes[1], metric)
        finish(fig, axes, name)
    # Standalone panels for direct two-subfigure use in the manuscript.
    for kind, draw in (("tolerance_angle", lambda ax: scatter(ax, "angle")),
                       ("compiled_reduction_histogram", lambda ax: histogram(ax, "compiled")),
                       ("t95_reduction_histogram", lambda ax: histogram(ax, "t95"))):
        fig, ax = plt.subplots(figsize=(6.1, 4.9), layout="constrained")
        draw(ax); finish(fig, [ax], "elote_" + kind)
    (output / "caption.tex").write_text(
        "Mixed-precision allocations and compiled $T$-gate savings for the current "
        f"best shared ELOTE (seed {int(manifest['seed'][-2:])}, mutation {manifest['iteration']}) "
        f"over {manifest['case_count']} HamLib Hamiltonians. "
        "(a) Relative synthesis tolerance $\\epsilon_j/\\epsilon_0$ versus absolute "
        "rotation angle $|\\theta_j|$, where the compiler synthesizes "
        "$R_z(\\theta_j)$ and $\\theta_j=-2\\alpha_j$ for the Pauli rotation "
        "$e^{i\\alpha_jP_j}$. The gray dashed line denotes the tight uniform-precision "
        "nested PF2 baseline; green points denote the single learned shared policy. "
        "Larger tolerances permit less accurate rotation synthesis. "
        "(b) Reduction in the literal compiled $T$ count for one full evolution "
        "relative to the same tight baseline, with one observation per Hamiltonian. "
        "This count includes all Trotter repetitions and excludes QPE query and "
        "success-amplification factors; a separate $T_{95}$ version includes them. "
        "Both policies pass the recorded chemical-accuracy criterion on every optimization case. "
        + (f"Panel (b) additionally outlines {len(heldout)} held-out 18--22-qubit literal circuit counts "
           "under training-extrapolated starting parameters; those counts alone do not certify energy accuracy. "
           if heldout else "") +
        "The learned policy also changes the splitting to flat PF1, so the cost "
        "reduction is not a precision-only ablation.\n"
    )
    (output / "figure_manifest.json").write_text(json.dumps({
        "rendered_at": time.time(), "plot_script_sha256": sha(Path(__file__)),
        "snapshot_manifest_sha256": sha(snapshot / "manifest.json"),
        "latex_rendering": True, "policy_sha256": manifest["policy_sha256"],
        "case_count": manifest["case_count"],
        "heldout_countable_cases": len(heldout),
        "heldout_statuses": heldout_statuses,
        "heldout_summary_sha256": sha(heldout_summary) if heldout_summary else None,
        "baseline_display": "Constant uniform-tolerance reference at epsilon/epsilon0=1; raw baseline rotations retained in snapshot.",
        "histogram": "Freedman-Diaconis bins, all Hamiltonians retained, one observation per Hamiltonian.",
        "files": {path.name: sha(path) for path in sorted(output.iterdir())
                  if path.is_file() and path.suffix in ("png", ".png", ".pdf", ".tex")},
    }, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("seed", "iteration", "case_count", "policy_sha256", "epsilon_ratio_range",
                                                    "compiled_t_reduction_percent", "t95_reduction_percent")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze = modes.add_parser("export")
    freeze.add_argument("--campaign", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    render = modes.add_parser("plot")
    render.add_argument("--snapshot", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--heldout-summary", type=Path,
                        help="optional frozen >16-qubit literal-count audit for panel (b)")
    args = parser.parse_args()
    if args.mode == "export":
        export(args.campaign.resolve(), args.output.resolve())
    else:
        plot(args.snapshot.resolve(), args.output.resolve(),
             args.heldout_summary.resolve() if args.heldout_summary else None)


if __name__ == "__main__":
    main()

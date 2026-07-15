#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from campaign_common import (
    CHEMICAL_ACCURACY_HARTREE,
    config_to_dict,
    ensure_dirs,
    load_json,
    load_policy_module,
    locate_zip,
    order_terms,
    parse_hamlib_terms,
    write_json,
)


def read_lane_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    lane_ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return lane_ids


def import_pygridsynth():
    import pygridsynth
    import pygridsynth.mymath
    import pygridsynth.ring

    def ntz_compat(n: int) -> int:
        n = int(n)
        if n == 0:
            return 0
        lowbit = n & -n
        return lowbit.bit_length() - 1

    pygridsynth.mymath.ntz = ntz_compat
    pygridsynth.ring.ntz = ntz_compat

    return pygridsynth


def import_array_backend():
    requested = os.environ.get("CAMPAIGN_ARRAY_BACKEND", "cupynumeric").strip().lower()
    if requested in {"numpy", "cpu"}:
        return "numpy", np
    if requested not in {"cupynumeric", "cunumeric", "gpu"}:
        raise RuntimeError(
            "HamLib campaign evaluations must run with CAMPAIGN_ARRAY_BACKEND=cupynumeric or numpy; "
            f"got {requested!r}. CPU/NumPy fallback is disabled."
        )
    try:
        import cupynumeric as xp
    except Exception as exc:
        raise RuntimeError(
            "CAMPAIGN_ARRAY_BACKEND=cupynumeric was requested, but cuPyNumeric could not be imported. "
            "Run the worker through Legate inside a GPU allocation."
        ) from exc
    return "cupynumeric", xp


def gpu_backend_preflight(*, xp, worker_id: int) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; refusing to run without an assigned GPU")

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if smi.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed in worker {worker_id}: {smi.stderr.strip()}")

    started = time.perf_counter()
    probe = xp.eye(512, dtype=xp.float64)
    values = xp.linalg.eigh(probe)[0]
    first = float(np.asarray(values)[0])
    if abs(first - 1.0) > 1e-12:
        raise RuntimeError(
            f"cuPyNumeric eigensolver preflight returned unexpected eigenvalue {first}"
        )

    return {
        "worker_id": worker_id,
        "array_backend": "cupynumeric",
        "cuda_visible_devices": visible,
        "nvidia_smi": smi.stdout.strip().splitlines(),
        "preflight_eigh_seconds": time.perf_counter() - started,
        "preflight_array_type": f"{type(probe).__module__}.{type(probe).__name__}",
    }


def protocol_parameters(one_norm: float) -> dict[str, Any]:
    qpe_budget = CHEMICAL_ACCURACY_HARTREE / 3.0
    tau = math.pi / one_norm
    qpe_bits = math.ceil(math.log2((2.0 * one_norm) / qpe_budget))
    qpe_unitary_applications = (1 << qpe_bits) - 1
    qpe_grid_spacing = (2.0 * one_norm) / (1 << qpe_bits)
    return {
        "tau": tau,
        "qpe_bits": qpe_bits,
        "qpe_unitary_applications": qpe_unitary_applications,
        "qpe_grid_spacing_hartree": qpe_grid_spacing,
        "qpe_error_budget_hartree": qpe_budget,
        "pf_error_budget_hartree": qpe_budget,
        "synthesis_error_budget_hartree": qpe_budget,
    }


def aggregate_rotations(
    *,
    terms: list[dict[str, Any]],
    config: Any,
    choose_synthesis_epsilon,
    tau: float,
) -> tuple[Counter[tuple[str, str]], int, int, float, float]:
    formula_order = int(getattr(config, "formula_order"))
    n_steps = int(getattr(config, "n_steps"))
    if formula_order not in {1, 2}:
        raise ValueError(f"formula_order={formula_order} is unsupported; use 1 or 2")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    zero_threshold = float(getattr(config, "zero_angle_threshold", 0.0))
    if not math.isfinite(zero_threshold) or zero_threshold < 0.0:
        raise ValueError(
            f"zero_angle_threshold must be finite and nonnegative, got {zero_threshold}"
        )
    step_time = tau / float(n_steps)
    rotations: Counter[tuple[str, str]] = Counter()
    scheduled = 0
    nonzero = 0
    eps_sum_per_base = 0.0
    omitted_identity_error_per_base = 0.0

    def add(term: dict[str, Any], alpha: float, multiplicity: int) -> None:
        nonlocal scheduled, nonzero, eps_sum_per_base, omitted_identity_error_per_base
        scheduled += multiplicity
        if abs(alpha) <= zero_threshold:
            # Replacing exp(i * alpha * P) with identity incurs exact operator-norm
            # error 2|sin(alpha/2)| for a Pauli P. Charge that error through the
            # same triangle-inequality budget used for synthesized rotations.
            omitted_error = 2.0 * abs(math.sin(float(alpha) / 2.0)) * multiplicity
            omitted_identity_error_per_base += omitted_error
            eps_sum_per_base += omitted_error
            return
        epsilon = float(choose_synthesis_epsilon(term, alpha, 0, config))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"invalid synthesis epsilon {epsilon} for term {term['label']}")
        theta = abs(-2.0 * float(alpha))
        theta_key = f"{theta:.17e}"
        eps_key = f"{epsilon:.17e}"
        rotations[(theta_key, eps_key)] += multiplicity
        nonzero += multiplicity
        eps_sum_per_base += epsilon * multiplicity

    if formula_order == 1:
        for term in terms:
            add(term, float(term["coefficient"]) * step_time, n_steps)
    else:
        if terms:
            for term in terms[:-1]:
                add(term, 0.5 * float(term["coefficient"]) * step_time, 2 * n_steps)
            add(terms[-1], float(terms[-1]["coefficient"]) * step_time, n_steps)

    return (
        rotations,
        scheduled,
        nonzero,
        eps_sum_per_base,
        omitted_identity_error_per_base,
    )


def synthesize_rotation_item(item: tuple[tuple[str, str], int]) -> tuple[int, int, int, int]:
    (theta_text, eps_text), multiplicity = item
    pygridsynth = import_pygridsynth()
    theta = pygridsynth.mpmath.mpf(theta_text)
    epsilon = pygridsynth.mpmath.mpf(eps_text)
    if theta <= epsilon:
        return 0, 0, 1, 0
    gates = pygridsynth.gridsynth_gates(theta, epsilon)
    t_count = gates.count("T")
    gate_chars = len(gates)
    return int(t_count) * int(multiplicity), int(gate_chars) * int(multiplicity), 0, 1


def synthesize_t_count(rotations: Counter[tuple[str, str]]) -> dict[str, Any]:
    started = time.perf_counter()
    total_t = 0
    total_gate_chars = 0
    zero_t_unique = 0
    synthesized_unique = 0
    items = list(rotations.items())
    requested_workers = int(os.environ.get("CAMPAIGN_SYNTH_WORKERS", "1"))
    synth_workers = max(1, min(requested_workers, len(items) or 1))
    if synth_workers > 1 and len(items) >= 512:
        chunksize = max(1, len(items) // (synth_workers * 8))
        with ProcessPoolExecutor(max_workers=synth_workers) as pool:
            iterator = pool.map(synthesize_rotation_item, items, chunksize=chunksize)
            for t_part, chars_part, zero_part, synthesized_part in iterator:
                total_t += t_part
                total_gate_chars += chars_part
                zero_t_unique += zero_part
                synthesized_unique += synthesized_part
    else:
        for item in items:
            t_part, chars_part, zero_part, synthesized_part = synthesize_rotation_item(item)
            total_t += t_part
            total_gate_chars += chars_part
            zero_t_unique += zero_part
            synthesized_unique += synthesized_part
    return {
        "unique_rotation_angles": len(rotations),
        "synthesized_unique_angles": synthesized_unique,
        "zero_t_unique_angles": zero_t_unique,
        "base_unitary_t_gates": total_t,
        "base_unitary_gate_chars": total_gate_chars,
        "pygridsynth_workers": synth_workers,
        "pygridsynth_wall_seconds": time.perf_counter() - started,
    }


def pf_error_bound_hartree(
    *, formula_order: int, n_steps: int, qpe_unitary_applications: int, tau: float
) -> float:
    # Conservative l1-style product-formula constants. With tau = pi/lambda,
    # lambda*tau = pi, matching the earlier HamLib protocol script.
    if formula_order == 1:
        unitary_error = qpe_unitary_applications * (math.pi**2) / float(n_steps)
    elif formula_order == 2:
        unitary_error = qpe_unitary_applications * (math.pi**3) / float(n_steps * n_steps)
    else:
        raise ValueError(f"unsupported formula_order={formula_order}")
    return unitary_error / tau


def score(total_t_gates: int, total_error: float) -> float:
    if total_error <= CHEMICAL_ACCURACY_HARTREE:
        return float(total_t_gates)
    return float(total_t_gates) + 1.0e24 * (total_error / CHEMICAL_ACCURACY_HARTREE) ** 2


def apply_pauli_to_indices(
    indices: np.ndarray, label: str, nqubits: int
) -> tuple[np.ndarray, np.ndarray]:
    out = indices.copy()
    phase = np.ones(indices.shape, dtype=np.complex128)
    for qubit, pauli in enumerate(label):
        if pauli == "I":
            continue
        bit = 1 << (nqubits - 1 - qubit)
        occupied = (indices & bit) != 0
        if pauli == "X":
            out ^= bit
        elif pauli == "Y":
            phase *= np.where(occupied, -1j, 1j)
            out ^= bit
        elif pauli == "Z":
            phase *= np.where(occupied, -1.0, 1.0)
        else:
            raise ValueError(f"invalid Pauli char {pauli!r}")
    return out, phase


def compute_dense_sector_ground_energy(
    *,
    terms: list[dict[str, Any]],
    identity: float,
    nqubits: int,
    electrons: int,
    backend_name: str,
    xp,
) -> dict[str, Any]:
    if backend_name != "cupynumeric":
        raise RuntimeError(
            f"CPU dense-sector diagonalization is disabled; got backend={backend_name!r}"
        )

    started = time.perf_counter()
    sector_indices = np.asarray(
        [idx for idx in range(1 << nqubits) if bin(idx).count("1") == electrons],
        dtype=np.int64,
    )
    dim = int(sector_indices.size)
    lookup = np.full(1 << nqubits, -1, dtype=np.int64)
    lookup[sector_indices] = np.arange(dim, dtype=np.int64)
    columns = np.arange(dim, dtype=np.int64)
    h_host = np.eye(dim, dtype=np.complex128) * float(identity)
    for term in terms:
        rows_full, phase = apply_pauli_to_indices(sector_indices, str(term["label"]), nqubits)
        rows = lookup[rows_full]
        valid = rows >= 0
        if np.any(valid):
            h_host[rows[valid], columns[valid]] += float(term["coefficient"]) * phase[valid]
    build_seconds = time.perf_counter() - started
    h_sector = xp.asarray(h_host, dtype=xp.complex128)
    diag_started = time.perf_counter()
    values = xp.linalg.eigh(h_sector)[0]
    ground = float(np.asarray(values)[0])
    return {
        "baseline_status": "computed",
        "baseline_backend": backend_name,
        "baseline_build_backend": "numpy_host_dense_then_cupynumeric_transfer",
        "baseline_array_type": f"{type(h_sector).__module__}.{type(h_sector).__name__}",
        "ground_state_energy_hartree": ground,
        "sector_dim": dim,
        "baseline_build_seconds": build_seconds,
        "baseline_diag_seconds": time.perf_counter() - diag_started,
    }


def evaluate_lane(
    *,
    campaign_root: Path,
    iteration: int,
    lane: dict[str, Any],
    data_root: Path,
    compute_baseline: bool,
    policy_snapshot: str,
    backend_name: str,
    xp,
) -> dict[str, Any]:
    started = time.perf_counter()
    if policy_snapshot == "current":
        policy_path = campaign_root / "lanes" / lane["lane_id"] / "simple_pf.py"
    elif policy_snapshot == "initial":
        policy_path = campaign_root / "lanes" / lane["lane_id"] / "initial_simple_pf.py"
        if not policy_path.is_file():
            policy_path = (
                campaign_root
                / "iterations"
                / "iter_0001"
                / "mutations"
                / lane["lane_id"]
                / "before_simple_pf.py"
            )
    else:
        raise ValueError(f"unsupported policy snapshot {policy_snapshot!r}")
    module = load_policy_module(policy_path)
    config = module.get_method_config()
    config_dict = config_to_dict(config)

    zip_path = locate_zip(data_root, lane)
    identity, terms, parse_info = parse_hamlib_terms(
        zip_path, lane["dataset"], int(lane["nqubits"])
    )
    ordered_terms = order_terms(
        terms, str(config_dict.get("term_order_strategy", "input")), module, config
    )
    protocol = protocol_parameters(float(lane["one_norm"]))

    baseline: dict[str, Any] = {"baseline_status": "skipped", "baseline_backend": backend_name}
    baseline_path = campaign_root / "baselines" / f"{lane['lane_id']}.json"
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline_path.unlink(missing_ok=True)
    if baseline.get("baseline_status") == "skipped" and compute_baseline:
        baseline = compute_dense_sector_ground_energy(
            terms=ordered_terms,
            identity=identity,
            nqubits=int(lane["nqubits"]),
            electrons=int(lane["electrons"]),
            backend_name=backend_name,
            xp=xp,
        )
        write_json(baseline_path, baseline)

    (
        rotations,
        scheduled,
        nonzero,
        eps_sum_per_base,
        omitted_identity_error_per_base,
    ) = aggregate_rotations(
        terms=ordered_terms,
        config=config,
        choose_synthesis_epsilon=module.choose_synthesis_epsilon,
        tau=float(protocol["tau"]),
    )
    synth = synthesize_t_count(rotations)
    qpe_apps = int(protocol["qpe_unitary_applications"])
    formula_order = int(config_dict["formula_order"])
    n_steps = int(config_dict["n_steps"])
    pf_error = pf_error_bound_hartree(
        formula_order=formula_order,
        n_steps=n_steps,
        qpe_unitary_applications=qpe_apps,
        tau=float(protocol["tau"]),
    )
    rotation_synthesis_error_per_base = eps_sum_per_base - omitted_identity_error_per_base
    rotation_synthesis_unitary_error = qpe_apps * rotation_synthesis_error_per_base
    zero_angle_unitary_error = qpe_apps * omitted_identity_error_per_base
    zero_angle_error = zero_angle_unitary_error / float(protocol["tau"])
    synthesis_unitary_error = rotation_synthesis_unitary_error + zero_angle_unitary_error
    synthesis_error = synthesis_unitary_error / float(protocol["tau"])
    total_error = float(protocol["qpe_grid_spacing_hartree"]) + pf_error + synthesis_error
    total_t = int(synth["base_unitary_t_gates"]) * qpe_apps

    return {
        "lane_id": lane["lane_id"],
        "iteration": iteration,
        "status": "ok",
        "molecule": lane["molecule"],
        "file": lane["file"],
        "dataset": lane["dataset"],
        "nqubits": int(lane["nqubits"]),
        "electrons": int(lane["electrons"]),
        "terms": len(terms),
        "identity_coefficient": identity,
        "policy_path": str(policy_path),
        "policy_snapshot": policy_snapshot,
        "config": config_dict,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        **parse_info,
        **protocol,
        **synth,
        **baseline,
        "scheduled_rotation_count_per_base_unitary": scheduled,
        "nonzero_rotation_count_per_base_unitary": nonzero,
        "omitted_rotation_count_per_base_unitary": scheduled - nonzero,
        "omitted_rotation_identity_error_bound_per_base_unitary": (omitted_identity_error_per_base),
        "rotation_synthesis_unitary_error_bound": rotation_synthesis_unitary_error,
        "zero_angle_unitary_error_bound": zero_angle_unitary_error,
        "zero_angle_error_bound_hartree": zero_angle_error,
        "synthesis_unitary_error_bound": synthesis_unitary_error,
        "synthesis_error_bound_hartree": synthesis_error,
        "pf_error_bound_hartree": pf_error,
        "total_error_bound_hartree": total_error,
        "total_t_gates_qpe": total_t,
        "score": score(total_t, total_error),
        "passes_chemical_accuracy_bound": total_error <= CHEMICAL_ACCURACY_HARTREE,
        "array_backend": backend_name,
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one shard of a HamLib PF campaign iteration."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lane-ids-file", type=Path)
    parser.add_argument("--compute-baseline", action="store_true")
    parser.add_argument("--policy-snapshot", choices=("current", "initial"), default="current")
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    ensure_dirs(root)
    backend_name, xp = import_array_backend()
    if backend_name == "cupynumeric":
        backend_preflight = gpu_backend_preflight(xp=xp, worker_id=args.worker_id)
        print(json.dumps({"status": "gpu_preflight_ok", **backend_preflight}), flush=True)
    else:
        print(
            json.dumps(
                {
                    "status": "cpu_policy_eval_ok",
                    "worker_id": args.worker_id,
                    "array_backend": backend_name,
                    "compute_baseline": bool(args.compute_baseline),
                }
            ),
            flush=True,
        )

    manifest = load_json(root / "manifests" / "campaign_manifest.json")
    selected_lane_ids = read_lane_ids(args.lane_ids_file)
    lanes = sorted(
        [lane for lane in manifest["lanes"] if lane["exact_fit"]],
        key=lambda item: (
            -float(item.get("estimated_sector_workspace_gib", 0.0)),
            -float(item.get("estimated_state_workspace_gib", 0.0)),
            -int(item.get("terms", 0)),
            item["lane_id"],
        ),
    )
    if selected_lane_ids is not None:
        lane_ids_in_manifest = {lane["lane_id"] for lane in lanes}
        missing = selected_lane_ids - lane_ids_in_manifest
        if missing:
            raise ValueError(f"lane ids are not active in manifest: {sorted(missing)}")
        lanes = [lane for lane in lanes if lane["lane_id"] in selected_lane_ids]
    shard = [
        lane for offset, lane in enumerate(lanes) if offset % args.num_workers == args.worker_id
    ]

    iter_dir = root / "iterations" / f"iter_{args.iteration:04d}"
    out_dir = iter_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_log: list[dict[str, Any]] = []
    for lane in shard:
        out_path = out_dir / f"{lane['lane_id']}.json"
        try:
            result = evaluate_lane(
                campaign_root=root,
                iteration=args.iteration,
                lane=lane,
                data_root=args.data_root,
                compute_baseline=args.compute_baseline,
                policy_snapshot=args.policy_snapshot,
                backend_name=backend_name,
                xp=xp,
            )
        except Exception as exc:
            result = {
                "lane_id": lane["lane_id"],
                "iteration": args.iteration,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json(out_path, result)
        worker_log.append({"lane_id": lane["lane_id"], "status": result["status"]})
        print(json.dumps(worker_log[-1]), flush=True)
    write_json(iter_dir / f"worker_{args.worker_id:02d}.json", worker_log)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def validate_locked_python_runtime() -> None:
    expected_text = os.environ.get("CAMPAIGN_LOCKED_PYTHON", "")
    if not expected_text:
        raise RuntimeError(
            "CAMPAIGN_LOCKED_PYTHON is required for a Slurm evaluation; "
            "refusing an unpinned Python runtime"
        )
    expected = Path(expected_text)
    virtual_env_text = os.environ.get("VIRTUAL_ENV", "")
    if not expected.is_absolute() or not virtual_env_text:
        raise RuntimeError("locked Python identity must use an absolute executable and VIRTUAL_ENV")
    virtual_env = Path(virtual_env_text)
    if expected != virtual_env / "bin" / "python":
        raise RuntimeError("CAMPAIGN_LOCKED_PYTHON does not belong to VIRTUAL_ENV")
    try:
        expected_real = expected.resolve(strict=True)
        actual_real = Path(sys.executable).resolve(strict=True)
        expected_prefix = virtual_env.resolve(strict=True)
        actual_prefix = Path(sys.prefix).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("locked Python identity could not be resolved") from exc
    if actual_real != expected_real:
        raise RuntimeError(
            f"Python runtime drift: expected {expected_real}, launched {actual_real}"
        )
    if actual_prefix != expected_prefix:
        raise RuntimeError(
            f"Python environment drift: expected prefix {expected_prefix}, got {actual_prefix}"
        )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "HamLib campaign evaluations require the locked Python 3.12 runtime; "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )


if os.environ.get("SLURM_JOB_ID") or os.environ.get("CAMPAIGN_LOCKED_PYTHON"):
    validate_locked_python_runtime()

import numpy as np  # noqa: E402

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-hamlib121-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import (  # noqa: E402
    CHEMICAL_ACCURACY_HARTREE,
    config_to_dict,
    ensure_dirs,
    load_json,
    load_policy_module,
    order_terms,
    parse_hamlib_terms,
    sha256_file,
    validate_campaign_identity,
    validate_hamlib_archive,
    validate_runner_provenance,
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
        raise RuntimeError(
            "HamLib proposal evaluations require cuPyNumeric/Legate on an allocated A100; "
            "NumPy/CPU proposal scoring is disabled"
        )
    if requested not in {"cupynumeric", "cunumeric", "gpu"}:
        raise RuntimeError(
            "HamLib campaign evaluations must run with CAMPAIGN_ARRAY_BACKEND=cupynumeric; "
            f"got {requested!r}"
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
    backend_module = str(getattr(xp, "__name__", "")).lower()
    if xp is np or backend_module not in {"cupynumeric", "cunumeric"}:
        raise RuntimeError(
            f"paper baseline requires the real cuPyNumeric module, got {backend_module!r}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; refusing to run without an assigned GPU")
    visible_devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(visible_devices) != 1:
        raise RuntimeError(
            f"paper baseline requires exactly one allocated GPU, got CUDA_VISIBLE_DEVICES={visible!r}"
        )

    smi_path_text = os.environ.get("CAMPAIGN_NVIDIA_SMI", "")
    if not smi_path_text or not os.path.isabs(smi_path_text):
        raise RuntimeError("CAMPAIGN_NVIDIA_SMI must name a trusted absolute system tool")
    smi_path = Path(smi_path_text)
    try:
        smi_metadata = smi_path.stat()
    except OSError as exc:
        raise RuntimeError(f"trusted NVIDIA system tool is unavailable: {smi_path}") from exc
    if (
        smi_path.is_symlink()
        or not stat.S_ISREG(smi_metadata.st_mode)
        or smi_metadata.st_uid != 0
        or stat.S_IMODE(smi_metadata.st_mode) & 0o022
        or not os.access(smi_path, os.X_OK)
    ):
        raise RuntimeError(
            f"CAMPAIGN_NVIDIA_SMI is not a root-owned immutable executable: {smi_path}"
        )

    smi = subprocess.run(
        [
            str(smi_path),
            "-i",
            visible_devices[0],
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if smi.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed in worker {worker_id}: {smi.stderr.strip()}")
    gpu_lines = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    if len(gpu_lines) != 1:
        raise RuntimeError(
            "paper baseline requires exactly one queryable allocated GPU; "
            f"nvidia-smi reported {gpu_lines!r}"
        )
    gpu_fields = [field.strip() for field in gpu_lines[0].split(",", 3)]
    if len(gpu_fields) != 4:
        raise RuntimeError(f"could not parse allocated GPU evidence: {gpu_lines[0]!r}")
    gpu_index, gpu_name, gpu_uuid, gpu_memory_text = gpu_fields
    memory_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s+MiB", gpu_memory_text)
    gpu_memory_mib = float(memory_match.group(1)) if memory_match else -1.0
    normalized_name = re.sub(r"\s+", "", gpu_name.upper())
    if "A100" not in normalized_name or "80GB" not in normalized_name or gpu_memory_mib < 80_000:
        raise RuntimeError(
            "paper baseline requires the allocated NVIDIA A100 80GB GPU; "
            f"nvidia-smi reported {gpu_lines[0]!r}"
        )

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
        "array_backend_module": backend_module,
        "cuda_visible_devices": visible,
        "allocated_gpu_count": len(visible_devices),
        "nvidia_smi": gpu_lines,
        "nvidia_smi_path": str(smi_path),
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_memory_mib": gpu_memory_mib,
        "accelerator_family": "NVIDIA A100 80GB",
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


def score(total_t_gates: int, total_error: float | None) -> float:
    if total_error is not None and math.isfinite(total_error) and total_error <= CHEMICAL_ACCURACY_HARTREE:
        return float(total_t_gates)
    if total_error is None or not math.isfinite(total_error):
        return float(total_t_gates) + 1.0e48
    return float(total_t_gates) + 1.0e24 * (total_error / CHEMICAL_ACCURACY_HARTREE) ** 2


def pygridsynth_matrix_from_gates(gate_string: str) -> np.ndarray:
    pygridsynth = import_pygridsynth()
    matrix = pygridsynth.DOmegaUnitary.from_gates(gate_string).to_complex_matrix
    return np.asarray(
        [[complex(matrix[row, column]) for column in range(2)] for row in range(2)],
        dtype=np.complex128,
    )


def rz_matrix(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def unitary_error_up_to_global_phase(target: np.ndarray, approximate: np.ndarray) -> float:
    phase = np.vdot(target[:, 0], approximate[:, 0])
    if abs(phase) <= 1e-30:
        phase = 1.0 + 0.0j
    else:
        phase /= abs(phase)
    return float(np.linalg.norm(target - approximate / phase, ord=2))


def apply_single_qubit_gate_to_state(*, state, gate, qubit: int, nqubits: int, xp):
    if not 0 <= qubit < nqubits:
        raise ValueError(f"qubit {qubit} is outside a {nqubits}-qubit register")
    prefix = 1 << qubit
    suffix = 1 << (nqubits - qubit - 1)
    tensor = xp.asarray(state, dtype=xp.complex128).reshape(prefix, 2, suffix)
    updated = tensor.copy()
    updated[:, 0, :] = gate[0, 0] * tensor[:, 0, :] + gate[0, 1] * tensor[:, 1, :]
    updated[:, 1, :] = gate[1, 0] * tensor[:, 0, :] + gate[1, 1] * tensor[:, 1, :]
    return updated.reshape(-1)


def apply_cnot_to_state(*, state, control: int, target: int, nqubits: int, xp):
    if control == target:
        raise ValueError("CNOT control and target must be different")
    if not 0 <= control < nqubits or not 0 <= target < nqubits:
        raise ValueError(
            f"CNOT ({control}, {target}) is outside a {nqubits}-qubit register"
        )

    first = min(control, target)
    second = max(control, target)
    prefix = 1 << first
    middle = 1 << (second - first - 1)
    suffix = 1 << (nqubits - second - 1)
    tensor = xp.asarray(state, dtype=xp.complex128).reshape(prefix, 2, middle, 2, suffix)
    updated = tensor.copy()
    if control == first:
        updated[:, 1, :, 0, :] = tensor[:, 1, :, 1, :]
        updated[:, 1, :, 1, :] = tensor[:, 1, :, 0, :]
    else:
        updated[:, 0, :, 1, :] = tensor[:, 1, :, 1, :]
        updated[:, 1, :, 1, :] = tensor[:, 0, :, 1, :]
    return updated.reshape(-1)


def build_pauli_rotation_compiler(*, nqubits: int, xp):
    hadamard = (1.0 / math.sqrt(2.0)) * xp.asarray(
        [[1.0, 1.0], [1.0, -1.0]], dtype=xp.complex128
    )
    s_gate = xp.asarray([[1.0, 0.0], [0.0, 1j]], dtype=xp.complex128)
    s_dagger = xp.asarray([[1.0, 0.0], [0.0, -1j]], dtype=xp.complex128)
    named_gates = {"H": hadamard, "S": s_gate, "Sdg": s_dagger}

    @functools.lru_cache(maxsize=None)
    def term_scaffold(label: str):
        support = [index for index, pauli in enumerate(label) if pauli != "I"]
        if not support:
            raise ValueError("cannot compile an identity Pauli rotation")
        pivot = support[-1]
        basis_forward: list[tuple[int, str]] = []
        basis_inverse: list[tuple[int, str]] = []
        for qubit, pauli in enumerate(label):
            if pauli == "X":
                basis_forward.append((qubit, "H"))
                basis_inverse.append((qubit, "H"))
            elif pauli == "Y":
                basis_forward.extend(((qubit, "Sdg"), (qubit, "H")))
                basis_inverse.extend(((qubit, "H"), (qubit, "S")))
        cnot_pairs = tuple((qubit, pivot) for qubit in support[:-1])
        return tuple(basis_forward), tuple(basis_inverse), cnot_pairs, pivot

    @functools.lru_cache(maxsize=None)
    def compile_rz(theta: float, epsilon: float) -> dict[str, Any]:
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"synthesis epsilon must be positive and finite, got {epsilon}")
        if abs(theta) <= epsilon:
            gate_string = ""
            approximate = np.eye(2, dtype=np.complex128)
        else:
            pygridsynth = import_pygridsynth()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"pygridsynth is .*", category=UserWarning)
                gate_string = pygridsynth.gridsynth_gates(theta, epsilon)
            approximate = pygridsynth_matrix_from_gates(gate_string)
        return {
            "theta": theta,
            "epsilon": epsilon,
            "gate_string": gate_string,
            "gate_count": len(gate_string),
            "t_gate_count": gate_string.count("T"),
            "operator_error": unitary_error_up_to_global_phase(rz_matrix(theta), approximate),
            "matrix": xp.asarray(approximate, dtype=xp.complex128),
        }

    @functools.lru_cache(maxsize=None)
    def compile_pauli_rotation(label: str, alpha: float, epsilon: float) -> dict[str, Any]:
        normalized_alpha = float(alpha)
        normalized_epsilon = float(epsilon)
        theta = -2.0 * normalized_alpha
        rz = compile_rz(theta, normalized_epsilon)
        basis_forward, basis_inverse, cnot_pairs, pivot = term_scaffold(label)

        def apply_state(state):
            updated = xp.asarray(state, dtype=xp.complex128)
            if rz["gate_count"] == 0:
                return updated
            for qubit, gate_name in basis_forward:
                updated = apply_single_qubit_gate_to_state(
                    state=updated,
                    gate=named_gates[gate_name],
                    qubit=qubit,
                    nqubits=nqubits,
                    xp=xp,
                )
            for control, target in cnot_pairs:
                updated = apply_cnot_to_state(
                    state=updated,
                    control=control,
                    target=target,
                    nqubits=nqubits,
                    xp=xp,
                )
            updated = apply_single_qubit_gate_to_state(
                state=updated,
                gate=rz["matrix"],
                qubit=pivot,
                nqubits=nqubits,
                xp=xp,
            )
            for control, target in cnot_pairs:
                updated = apply_cnot_to_state(
                    state=updated,
                    control=control,
                    target=target,
                    nqubits=nqubits,
                    xp=xp,
                )
            for qubit, gate_name in basis_inverse:
                updated = apply_single_qubit_gate_to_state(
                    state=updated,
                    gate=named_gates[gate_name],
                    qubit=qubit,
                    nqubits=nqubits,
                    xp=xp,
                )
            return updated

        return {
            "cache_key": f"{label}|{normalized_alpha:.17e}|{normalized_epsilon:.17e}",
            "label": label,
            "alpha": normalized_alpha,
            "theta": theta,
            "epsilon": normalized_epsilon,
            "gate_count": int(rz["gate_count"]),
            "t_gate_count": int(rz["t_gate_count"]),
            "operator_error": float(rz["operator_error"]),
            "apply_state": apply_state,
        }

    return compile_pauli_rotation


def hartree_fock_basis_index(*, nqubits: int, electrons: int) -> int:
    if electrons < 0 or electrons > nqubits:
        raise ValueError(f"invalid electron count {electrons} for {nqubits} qubits")
    return sum(1 << (nqubits - 1 - qubit) for qubit in range(electrons))


def product_formula_step_schedule(
    *, terms: list[dict[str, Any]], formula_order: int, step_time: float
) -> list[tuple[dict[str, Any], float]]:
    if formula_order == 1:
        return [(term, float(term["coefficient"]) * step_time) for term in terms]
    if formula_order == 2:
        if not terms:
            return []
        leading = [
            (term, 0.5 * float(term["coefficient"]) * step_time) for term in terms[:-1]
        ]
        center = [(terms[-1], float(terms[-1]["coefficient"]) * step_time)]
        return leading + center + list(reversed(leading))
    raise ValueError(f"unsupported formula_order={formula_order}")


def run_compiled_product_formula(
    *,
    terms: list[dict[str, Any]],
    config: Any,
    choose_synthesis_epsilon,
    target_time: float,
    nqubits: int,
    electrons: int,
    xp,
) -> dict[str, Any]:
    n_steps = int(getattr(config, "n_steps"))
    formula_order = int(getattr(config, "formula_order"))
    zero_threshold = float(getattr(config, "zero_angle_threshold", 0.0))
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if formula_order not in {1, 2}:
        raise ValueError(f"unsupported formula_order={formula_order}")
    if not math.isfinite(zero_threshold) or zero_threshold < 0.0:
        raise ValueError("zero_angle_threshold must be finite and nonnegative")

    state = xp.zeros(1 << nqubits, dtype=xp.complex128)
    state[hartree_fock_basis_index(nqubits=nqubits, electrons=electrons)] = 1.0 + 0.0j
    compile_rotation = build_pauli_rotation_compiler(nqubits=nqubits, xp=xp)
    schedule = product_formula_step_schedule(
        terms=terms,
        formula_order=formula_order,
        step_time=float(target_time) / float(n_steps),
    )
    epsilon_hist: Counter[str] = Counter()
    unique_compilations: dict[str, dict[str, Any]] = {}
    t_gate_count = 0
    nonzero_rotation_count = 0
    skipped_rotation_count = 0
    propagation_started = time.perf_counter()

    # Policies are required to be stationary: each of the n_steps repetitions
    # applies the same compiled slice.  Evaluate that slice once on the A100 and
    # extract its phase over target_time / n_steps.  Replaying an identical
    # slice n_steps times is unnecessary for the spectral estimator and made
    # conservative high-step seeds prohibitively expensive.
    for term, alpha in schedule:
        if abs(alpha) <= zero_threshold:
            skipped_rotation_count += n_steps
            continue
        epsilon = float(choose_synthesis_epsilon(term, alpha, 0, config))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"invalid synthesis epsilon {epsilon} for term {term['label']}")
        compiled = compile_rotation(str(term["label"]), float(alpha), epsilon)
        state = compiled["apply_state"](state)
        nonzero_rotation_count += n_steps
        t_gate_count += n_steps * int(compiled["t_gate_count"])
        epsilon_hist[f"{epsilon:.3e}"] += n_steps
        unique_compilations[compiled["cache_key"]] = {
            key: compiled[key]
            for key in (
                "label",
                "alpha",
                "theta",
                "epsilon",
                "t_gate_count",
                "gate_count",
                "operator_error",
            )
        }

    norm = float(np.asarray(xp.linalg.norm(state)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"invalid propagated-state norm {norm}")
    state = state / norm
    return {
        "state": state,
        "state_array_type": f"{type(state).__module__}.{type(state).__name__}",
        "state_dimension": 1 << nqubits,
        "propagation_mode": "one_stationary_compiled_trotter_slice",
        "simulated_trotter_steps": 1,
        "physical_trotter_steps": n_steps,
        "spectral_phase_time": float(target_time) / float(n_steps),
        "base_unitary_t_gates": t_gate_count,
        "nonzero_rotation_count_per_base_unitary": nonzero_rotation_count,
        "skipped_rotation_count_per_base_unitary": skipped_rotation_count,
        "scheduled_rotation_count_per_base_unitary": len(schedule) * n_steps,
        "synthesis_epsilon_histogram": dict(sorted(epsilon_hist.items())),
        "unique_compilation_count": len(unique_compilations),
        "unique_compilations": sorted(
            unique_compilations.values(), key=lambda item: (item["label"], item["epsilon"])
        ),
        "propagation_wall_seconds": time.perf_counter() - propagation_started,
    }


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


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def compute_dense_sector_reference(
    *,
    terms: list[dict[str, Any]],
    identity: float,
    nqubits: int,
    electrons: int,
    backend_name: str,
    xp,
    peak_state_path: Path,
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
    values, vectors = xp.linalg.eigh(h_sector)
    values_host = np.asarray(values, dtype=np.float64)
    hf_index = hartree_fock_basis_index(nqubits=nqubits, electrons=electrons)
    hf_sector_rows = np.flatnonzero(sector_indices == hf_index)
    if hf_sector_rows.size != 1:
        raise RuntimeError("Hartree-Fock basis state is absent from the fixed-electron sector")
    hf_sector_row = int(hf_sector_rows[0])
    hf_row = np.asarray(vectors[hf_sector_row, :], dtype=np.complex128)
    weights = np.abs(hf_row) ** 2
    resolved = np.flatnonzero(weights >= 1.0e-12)
    if resolved.size == 0:
        raise RuntimeError("no fixed-sector eigenstate has resolvable Hartree-Fock overlap")
    selected = int(resolved[np.argmin(values_host[resolved])])
    selected_vector = np.asarray(vectors[:, selected], dtype=np.complex128)
    write_npy_atomic(peak_state_path, selected_vector)
    reference_amplitude = complex(np.conjugate(selected_vector[hf_sector_row]))
    return {
        "baseline_status": "computed",
        "baseline_backend": backend_name,
        "baseline_build_backend": "numpy_host_dense_then_cupynumeric_transfer",
        "baseline_array_type": f"{type(h_sector).__module__}.{type(h_sector).__name__}",
        "ground_state_energy_hartree": float(values_host[0]),
        "sector_dim": dim,
        "hf_basis_index": hf_index,
        "hf_sector_row": hf_sector_row,
        "spectral_peak_weight_threshold": 1.0e-12,
        "resolved_spectral_peak_count": int(resolved.size),
        "selected_spectral_peak_index": selected,
        "selected_spectral_peak_energy_hartree": float(values_host[selected]),
        "selected_spectral_peak_weight": float(weights[selected]),
        "selected_reference_amplitude_real": float(reference_amplitude.real),
        "selected_reference_amplitude_imag": float(reference_amplitude.imag),
        "selected_peak_state_file": peak_state_path.name,
        "selected_peak_state_sha256": sha256_file(peak_state_path),
        "baseline_build_seconds": build_seconds,
        "baseline_diag_seconds": time.perf_counter() - diag_started,
    }


def unwrap_energy_near_reference(
    *, phase: float, target_time: float, reference_energy: float
) -> float:
    principal_energy = float(phase / target_time)
    phase_period = float((2.0 * math.pi) / target_time)
    wrap_index = round((reference_energy - principal_energy) / phase_period)
    return principal_energy + wrap_index * phase_period


def estimate_reference_peak_energy(
    *,
    propagated_state,
    baseline: dict[str, Any],
    baseline_dir: Path,
    identity_shift: float,
    phase_time: float,
    nqubits: int,
    electrons: int,
    xp,
) -> dict[str, Any]:
    peak_file_name = str(baseline.get("selected_peak_state_file", ""))
    if not peak_file_name or Path(peak_file_name).name != peak_file_name:
        raise RuntimeError("baseline selected peak state filename is invalid")
    peak_path = baseline_dir / peak_file_name
    if peak_path.is_symlink() or not peak_path.is_file():
        raise RuntimeError("baseline selected peak state is missing or unsafe")
    if sha256_file(peak_path) != baseline.get("selected_peak_state_sha256"):
        raise RuntimeError("baseline selected peak state hash changed")

    peak_sector = xp.asarray(np.load(peak_path, allow_pickle=False), dtype=xp.complex128)
    sector_indices_host = np.asarray(
        [index for index in range(1 << nqubits) if index.bit_count() == electrons],
        dtype=np.int64,
    )
    if peak_sector.size != sector_indices_host.size:
        raise RuntimeError("baseline peak vector and sector-index dimensions differ")
    sector_indices = xp.asarray(sector_indices_host, dtype=xp.int64)
    propagated_amplitude = xp.vdot(peak_sector, propagated_state[sector_indices])
    reference_amplitude = complex(
        float(baseline["selected_reference_amplitude_real"]),
        float(baseline["selected_reference_amplitude_imag"]),
    )
    peak_weight = float(baseline["selected_spectral_peak_weight"])
    phase_component = np.conjugate(reference_amplitude) * propagated_amplitude / peak_weight
    phase_magnitude = float(np.asarray(xp.abs(phase_component)))
    if phase_magnitude <= 1.0e-15:
        return {
            "spectral_peak_phase_overlap_magnitude": phase_magnitude,
            "ground_state_energy_estimate_hartree": None,
            "selected_spectral_peak_phase_error_hartree": None,
        }

    selected_energy = float(baseline["selected_spectral_peak_energy_hartree"])
    centered_reference = selected_energy - float(identity_shift)
    if not math.isfinite(phase_time) or phase_time <= 0.0:
        raise ValueError(f"phase_time must be finite and positive, got {phase_time}")
    centered_estimate = unwrap_energy_near_reference(
        phase=float(np.asarray(xp.angle(phase_component))),
        target_time=phase_time,
        reference_energy=centered_reference,
    )
    energy_estimate = centered_estimate + float(identity_shift)
    return {
        "spectral_peak_phase_overlap_magnitude": phase_magnitude,
        "ground_state_energy_estimate_hartree": float(energy_estimate),
        "selected_spectral_peak_phase_error_hartree": float(
            abs(energy_estimate - selected_energy)
        ),
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
    campaign_manifest: dict[str, Any],
    campaign_identity: dict[str, Any],
    runner_provenance: dict[str, Any],
    execution_configuration: dict[str, Any],
    backend_preflight: dict[str, Any] | None,
    batch_manifest_sha256: str | None,
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
    if policy_path.is_symlink() or not policy_path.is_file():
        raise RuntimeError(f"policy snapshot is missing, a symlink, or not regular: {policy_path}")
    policy_sha256 = sha256_file(policy_path)
    module = load_policy_module(policy_path)
    config = module.get_method_config()
    config_dict = config_to_dict(config)

    zip_path, archive_provenance = validate_hamlib_archive(
        data_root=data_root,
        lane=lane,
        archive_records=list(campaign_manifest.get("hamlib_archives", [])),
    )
    identity, terms, parse_info = parse_hamlib_terms(
        zip_path, lane["dataset"], int(lane["nqubits"])
    )
    ordered_terms = order_terms(
        terms, str(config_dict.get("term_order_strategy", "input")), module, config
    )
    protocol = protocol_parameters(float(lane["one_norm"]))

    initial_policy_path = campaign_root / "lanes" / lane["lane_id"] / "initial_simple_pf.py"
    if initial_policy_path.is_symlink() or not initial_policy_path.is_file():
        raise RuntimeError(f"{lane['lane_id']}: initial policy snapshot is missing or unsafe")
    initial_policy_sha256 = sha256_file(initial_policy_path)
    baseline: dict[str, Any] = {"baseline_status": "missing"}
    baseline_path = campaign_root / "baselines" / f"{lane['lane_id']}.json"
    peak_state_path = campaign_root / "baselines" / f"{lane['lane_id']}.peak_state.npy"
    if baseline_path.exists():
        if baseline_path.is_symlink() or not baseline_path.is_file():
            raise RuntimeError(f"{lane['lane_id']}: baseline cache is unsafe")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("baseline_status") == "missing" and compute_baseline:
        if backend_name != "cupynumeric" or backend_preflight is None:
            raise RuntimeError(
                "baseline computation requires the verified cuPyNumeric/A100 backend"
            )
        baseline = compute_dense_sector_reference(
            terms=ordered_terms,
            identity=identity,
            nqubits=int(lane["nqubits"]),
            electrons=int(lane["electrons"]),
            backend_name=backend_name,
            xp=xp,
            peak_state_path=peak_state_path,
        )
        baseline.update(
            {
                "baseline_schema_version": 3,
                "baseline_accelerator": "NVIDIA A100",
                "baseline_computed_campaign_id": campaign_manifest["campaign_id"],
                "baseline_computed_manifest_sha256": campaign_identity["campaign_manifest_sha256"],
                "hamlib_archive": archive_provenance,
                "initial_policy_sha256": initial_policy_sha256,
                "gpu_preflight": backend_preflight,
                "baseline_execution_configuration": execution_configuration,
                "baseline_runner_provenance": runner_provenance,
            }
        )
        write_json(baseline_path, baseline)
    expected_baseline = {
        "baseline_schema_version": 3,
        "baseline_status": "computed",
        "baseline_backend": "cupynumeric",
        "baseline_accelerator": "NVIDIA A100",
        "hamlib_archive": archive_provenance,
        "initial_policy_sha256": initial_policy_sha256,
        "baseline_runner_provenance": runner_provenance,
    }
    baseline_drift = {
        key: {"actual": baseline.get(key), "expected": value}
        for key, value in expected_baseline.items()
        if baseline.get(key) != value
    }
    if baseline_drift:
        raise RuntimeError(f"{lane['lane_id']}: missing or invalid A100 baseline: {baseline_drift}")
    baseline_execution = baseline.get("baseline_execution_configuration")
    baseline_gpu = baseline.get("gpu_preflight")
    if (
        not isinstance(baseline_execution, dict)
        or baseline_execution.get("execution_mode") != "gpu_baseline"
        or baseline_execution.get("array_backend") != "cupynumeric"
        or baseline_execution.get("compute_baseline") is not True
        or baseline_execution.get("policy_snapshot") != "initial"
        or not baseline_execution.get("slurm_job_id")
        or not baseline_execution.get("slurm_account")
        or not baseline_execution.get("slurm_partition")
        or not isinstance(baseline_gpu, dict)
        or baseline_gpu.get("accelerator_family") != "NVIDIA A100 80GB"
        or float(baseline_gpu.get("gpu_memory_mib", 0.0)) < 80_000
        or not str(baseline_gpu.get("gpu_uuid", "")).startswith("GPU-")
        or baseline_gpu.get("array_backend_module") not in {"cupynumeric", "cunumeric"}
    ):
        raise RuntimeError(f"{lane['lane_id']}: incomplete A100/Slurm baseline provenance")
    baseline_sha256 = sha256_file(baseline_path)

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
    analytic_total_error = (
        float(protocol["qpe_grid_spacing_hartree"]) + pf_error + synthesis_error
    )

    propagation = run_compiled_product_formula(
        terms=ordered_terms,
        config=config,
        choose_synthesis_epsilon=module.choose_synthesis_epsilon,
        target_time=float(protocol["tau"]),
        nqubits=int(lane["nqubits"]),
        electrons=int(lane["electrons"]),
        xp=xp,
    )
    propagated_state = propagation.pop("state")
    phase_metrics = estimate_reference_peak_energy(
        propagated_state=propagated_state,
        baseline=baseline,
        baseline_dir=baseline_path.parent,
        identity_shift=identity,
        phase_time=float(propagation["spectral_phase_time"]),
        nqubits=int(lane["nqubits"]),
        electrons=int(lane["electrons"]),
        xp=xp,
    )
    energy_estimate = phase_metrics["ground_state_energy_estimate_hartree"]
    ground_energy = float(baseline["ground_state_energy_hartree"])
    energy_error = (
        None if energy_estimate is None else float(abs(float(energy_estimate) - ground_energy))
    )
    total_t = int(propagation["base_unitary_t_gates"]) * qpe_apps
    passes_energy = (
        energy_error is not None and energy_error <= CHEMICAL_ACCURACY_HARTREE
    )

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
        "policy_sha256": policy_sha256,
        "initial_policy_sha256": initial_policy_sha256,
        "campaign_id": campaign_manifest["campaign_id"],
        "campaign_manifest_sha256": campaign_identity["campaign_manifest_sha256"],
        "runner_provenance": runner_provenance,
        "hamlib_archive": archive_provenance,
        "baseline_cache_sha256": baseline_sha256,
        "execution_configuration": execution_configuration,
        "batch_manifest_sha256": batch_manifest_sha256,
        "config": config_dict,
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        **parse_info,
        **protocol,
        **propagation,
        **baseline,
        **phase_metrics,
        "objective": "hf_reference_spectral_energy",
        "spectral_estimator": "lowest_resolved_peak_one_stationary_slice_projection",
        "reference_state": "hartree_fock",
        "accuracy_metric_name": "ground_state_energy_error_hartree",
        "accuracy_metric_value": energy_error,
        "accuracy_threshold_name": "chemical_accuracy_hartree",
        "accuracy_threshold_value": CHEMICAL_ACCURACY_HARTREE,
        "ground_state_energy_error_hartree": energy_error,
        "analytic_scheduled_rotation_count_per_base_unitary": scheduled,
        "analytic_nonzero_rotation_count_per_base_unitary": nonzero,
        "analytic_omitted_rotation_count_per_base_unitary": scheduled - nonzero,
        "omitted_rotation_identity_error_bound_per_base_unitary": (omitted_identity_error_per_base),
        "rotation_synthesis_unitary_error_bound": rotation_synthesis_unitary_error,
        "zero_angle_unitary_error_bound": zero_angle_unitary_error,
        "zero_angle_error_bound_hartree": zero_angle_error,
        "synthesis_unitary_error_bound": synthesis_unitary_error,
        "synthesis_error_bound_hartree": synthesis_error,
        "pf_error_bound_hartree": pf_error,
        "analytic_total_error_bound_hartree": analytic_total_error,
        "total_t_gates_qpe": total_t,
        "score": score(total_t, energy_error),
        "passes_accuracy_constraint": passes_energy,
        "passes_energy_constraint": passes_energy,
        "passes_chemical_accuracy_bound": passes_energy,
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
    manifest, campaign_identity = validate_campaign_identity(root)
    runner_provenance = validate_runner_provenance(root)
    archive_preflight_path = root / "manifests" / "hamlib_archive_preflight.json"
    expected_preflight_sha256 = os.environ.get("CAMPAIGN_HAMLIB_PREFLIGHT_SHA256", "")
    if (
        archive_preflight_path.is_symlink()
        or not archive_preflight_path.is_file()
        or not expected_preflight_sha256
        or sha256_file(archive_preflight_path) != expected_preflight_sha256
    ):
        raise RuntimeError("worker HamLib archive preflight receipt is missing or changed")
    backend_name, xp = import_array_backend()
    backend_preflight = gpu_backend_preflight(xp=xp, worker_id=args.worker_id)
    print(json.dumps({"status": "gpu_preflight_ok", **backend_preflight}), flush=True)

    expected_execution_mode = "gpu_baseline" if args.compute_baseline else "gpu_policy"
    if os.environ.get("CAMPAIGN_EXECUTION_MODE") != expected_execution_mode:
        raise RuntimeError(
            "HamLib worker execution mode drift: "
            f"expected {expected_execution_mode!r}, got "
            f"{os.environ.get('CAMPAIGN_EXECUTION_MODE')!r}"
        )

    if args.compute_baseline and args.policy_snapshot != "initial":
        raise RuntimeError(
            "iteration-0 baseline must use cuPyNumeric/A100 and policy_snapshot=initial"
        )
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
    if args.compute_baseline and args.iteration != 0:
        raise RuntimeError(
            "baseline chunks may only evaluate iteration 0"
        )
    campaign_id_env = os.environ.get("CAMPAIGN_ID")
    manifest_sha_env = os.environ.get("CAMPAIGN_MANIFEST_SHA256")
    if campaign_id_env is not None and campaign_id_env != manifest["campaign_id"]:
        raise RuntimeError("worker CAMPAIGN_ID differs from campaign manifest")
    if (
        manifest_sha_env is not None
        and manifest_sha_env != campaign_identity["campaign_manifest_sha256"]
    ):
        raise RuntimeError("worker CAMPAIGN_MANIFEST_SHA256 differs from campaign identity")
    batch_id = os.environ.get("BATCH_ID")
    batch_manifest_sha256: str | None = None
    if batch_id:
        batch_manifest_path = (
            root
            / "iterations"
            / f"iter_{args.iteration:04d}"
            / "batches"
            / batch_id
            / "batch_manifest.json"
        )
        if batch_manifest_path.is_symlink() or not batch_manifest_path.is_file():
            raise RuntimeError("batch manifest is missing or unsafe")
        batch_manifest = load_json(batch_manifest_path)
        batch_manifest_sha256 = sha256_file(batch_manifest_path)
        expected_batch = {
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_sha256": campaign_identity["campaign_manifest_sha256"],
            "batch_id": batch_id,
            "iteration": args.iteration,
            "policy_snapshot": args.policy_snapshot,
            "submission_key": os.environ.get("CAMPAIGN_SUBMISSION_KEY"),
            "job_name": os.environ.get("CAMPAIGN_JOB_NAME"),
            "runner_provenance": runner_provenance,
        }
        drift = {
            key: {"actual": batch_manifest.get(key), "expected": value}
            for key, value in expected_batch.items()
            if batch_manifest.get(key) != value
        }
        batch_lane_ids = batch_manifest.get("lane_ids")
        if (
            drift
            or not isinstance(batch_lane_ids, list)
            or len(batch_lane_ids) != len(set(batch_lane_ids))
            or set(batch_lane_ids) != {lane["lane_id"] for lane in lanes}
        ):
            raise RuntimeError(f"batch manifest identity/lane drift: {drift}")
        policy_hashes = batch_manifest.get("policy_sha256s")
        if not isinstance(policy_hashes, dict):
            raise RuntimeError("batch manifest policy hash map is missing")
        for lane in lanes:
            policy_name = (
                "initial_simple_pf.py" if args.policy_snapshot == "initial" else "simple_pf.py"
            )
            policy = root / "lanes" / lane["lane_id"] / policy_name
            if policy_hashes.get(lane["lane_id"]) != sha256_file(policy):
                raise RuntimeError(f"{lane['lane_id']}: policy bytes changed after batch intent")
    execution_configuration = {
        "schema_version": 1,
        "execution_mode": os.environ.get("CAMPAIGN_EXECUTION_MODE", "direct"),
        "array_backend": backend_name,
        "compute_baseline": bool(args.compute_baseline),
        "policy_snapshot": args.policy_snapshot,
        "batch_id": batch_id,
        "submission_key": os.environ.get("CAMPAIGN_SUBMISSION_KEY"),
        "job_name": os.environ.get("CAMPAIGN_JOB_NAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_account": os.environ.get("CAMPAIGN_SLURM_ACCOUNT"),
        "slurm_partition": os.environ.get("CAMPAIGN_SLURM_PARTITION"),
        "legate_fbmem": os.environ.get("CAMPAIGN_LEGATE_FBMEM"),
        "uv_base_dir": os.environ.get("CAMPAIGN_UV_BASE_DIR"),
        "uv_cache_dir": os.environ.get("CAMPAIGN_UV_CACHE_DIR"),
        "uv_python_install_dir": os.environ.get("CAMPAIGN_UV_PYTHON_INSTALL_DIR"),
        "hamlib_archive_preflight_sha256": expected_preflight_sha256,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
    }
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
                campaign_manifest=manifest,
                campaign_identity=campaign_identity,
                runner_provenance=runner_provenance,
                execution_configuration=execution_configuration,
                backend_preflight=backend_preflight,
                batch_manifest_sha256=batch_manifest_sha256,
            )
        except Exception as exc:
            result = {
                "lane_id": lane["lane_id"],
                "iteration": args.iteration,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "campaign_id": manifest["campaign_id"],
                "campaign_manifest_sha256": campaign_identity["campaign_manifest_sha256"],
                "runner_provenance": runner_provenance,
                "execution_configuration": execution_configuration,
                "batch_manifest_sha256": batch_manifest_sha256,
            }
        write_json(out_path, result)
        worker_log.append({"lane_id": lane["lane_id"], "status": result["status"]})
        print(json.dumps(worker_log[-1]), flush=True)
    write_json(iter_dir / f"worker_{args.worker_id:02d}.json", worker_log)


if __name__ == "__main__":
    main()

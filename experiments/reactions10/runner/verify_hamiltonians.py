#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-runner-pycache-")
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
    campaign_lock,
    ensure_campaign_runner_runtime,
    now_iso,
    require_campaign,
    sha256_file,
    write_json,
    write_state,
)


PAULI_DENSE_RESIDUAL_TOLERANCE = 1.0e-10


def pauli_dense_residuals(
    *,
    hamiltonian: np.ndarray,
    identity_coefficient: float,
    labels: np.ndarray,
    coefficients: np.ndarray,
    n_qubits: int,
) -> tuple[float, float]:
    """Compare the evaluator's cached Pauli sum with its cached dense Hamiltonian.

    Each Pauli is applied as a signed permutation, matching the evaluator's
    label convention. This costs O(number_of_terms * 2**n_qubits) rather than
    constructing one dense Kronecker product per term.
    """

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or not 1 <= n_qubits <= 10:
        raise ValueError(f"n_qubits must be an integer in [1, 10], got {n_qubits!r}")
    dimension = 1 << n_qubits
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError(
            f"dense Hamiltonian shape {hamiltonian.shape} does not match n_qubits={n_qubits}"
        )
    labels = np.asarray(labels)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if labels.ndim != 1 or coefficients.ndim != 1 or labels.size != coefficients.size:
        raise ValueError("cached Pauli labels/coefficients have inconsistent shapes")
    if not math.isfinite(identity_coefficient) or not np.all(np.isfinite(coefficients)):
        raise ValueError("cached Pauli decomposition contains non-finite coefficients")

    normalized_labels = [str(raw_label) for raw_label in labels.tolist()]
    if len(normalized_labels) != len(set(normalized_labels)):
        raise ValueError("cached Pauli decomposition contains duplicate labels")
    identity_label = "I" * n_qubits
    for label in normalized_labels:
        if len(label) != n_qubits or set(label) - set("IXYZ"):
            raise ValueError(f"invalid cached Pauli label {label!r}")
        if label == identity_label:
            raise ValueError("cached Pauli terms must not repeat the separate identity coefficient")

    # Accumulate (identity + Pauli terms - dense Hamiltonian) directly to
    # avoid holding both a reconstructed matrix and a second difference matrix.
    residual = -np.asarray(hamiltonian, dtype=np.complex128).copy()
    diagonal = np.diag_indices(dimension)
    residual[diagonal] += identity_coefficient
    source = np.arange(dimension, dtype=np.int64)
    for label, coefficient in zip(normalized_labels, coefficients.tolist(), strict=True):
        destination = source.copy()
        phase = np.ones(dimension, dtype=np.complex128)
        for qubit, pauli in enumerate(label):
            if pauli == "I":
                continue
            mask = 1 << (n_qubits - qubit - 1)
            bit_is_one = (source & mask) != 0
            if pauli == "X":
                destination ^= mask
            elif pauli == "Y":
                destination ^= mask
                phase *= np.where(bit_is_one, -1.0j, 1.0j)
            else:  # Z
                phase *= np.where(bit_is_one, -1.0, 1.0)
        residual[destination, source] += coefficient * phase

    max_abs = float(np.max(np.abs(residual)))
    dense_frobenius = float(np.linalg.norm(hamiltonian))
    residual_frobenius = float(np.linalg.norm(residual))
    relative_frobenius = residual_frobenius / max(dense_frobenius, np.finfo(float).tiny)
    return max_abs, relative_frobenius


def verify_one(npz_path: Path, expected_hash: str) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = npz_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(npz_path) != expected_hash:
        raise RuntimeError(f"cache hash drift before verification: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        hamiltonian = np.asarray(
            arrays["hamiltonian_real"] + 1j * arrays["hamiltonian_imag"],
            dtype=np.complex128,
        )
        sector_indices = np.asarray(arrays["sector_indices"], dtype=np.int64)
        cached_values = np.asarray(arrays["sector_eigenvalues"], dtype=np.float64)
        cached_vectors = np.asarray(
            arrays["sector_eigenvectors_real"] + 1j * arrays["sector_eigenvectors_imag"],
            dtype=np.complex128,
        )
        evaluator_ground_state = np.asarray(
            arrays["ground_state_real"] + 1j * arrays["ground_state_imag"],
            dtype=np.complex128,
        )
        pauli_labels = np.asarray(arrays["pauli_labels"])
        pauli_coefficients = np.asarray(arrays["pauli_coefficients"], dtype=np.float64)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"Hamiltonian is not square: {npz_path}")
    n_qubits = int(metadata["n_qubits"])
    if n_qubits <= 0 or n_qubits > 10:
        raise ValueError(f"unsupported n_qubits={n_qubits}: {npz_path}")
    if int(metadata.get("total_pauli_rotation_terms", -1)) != pauli_labels.size:
        raise ValueError(f"cached Pauli term count mismatch: {npz_path}")
    pauli_max_abs_residual, pauli_relative_frobenius_residual = pauli_dense_residuals(
        hamiltonian=hamiltonian,
        identity_coefficient=float(metadata["identity_coefficient"]),
        labels=pauli_labels,
        coefficients=pauli_coefficients,
        n_qubits=n_qubits,
    )
    if (
        sector_indices.ndim != 1
        or sector_indices.size == 0
        or len(set(int(item) for item in sector_indices)) != sector_indices.size
        or int(sector_indices.min()) < 0
        or int(sector_indices.max()) >= hamiltonian.shape[0]
    ):
        raise ValueError(f"invalid fixed-electron sector indices: {npz_path}")
    hermiticity_error = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
    sector_hamiltonian = hamiltonian[np.ix_(sector_indices, sector_indices)]
    values, vectors = np.linalg.eigh(sector_hamiltonian)
    if cached_values.shape != values.shape:
        raise ValueError(f"cached sector spectrum shape mismatch: {npz_path}")
    cached_ground_vector = np.asarray(cached_vectors[sector_indices, 0], dtype=np.complex128)
    norm = float(np.linalg.norm(cached_ground_vector))
    if norm <= 0.0:
        raise ValueError(f"cached ground vector has zero norm: {npz_path}")
    cached_ground_vector /= norm
    residual = float(
        np.linalg.norm(
            sector_hamiltonian @ cached_ground_vector - cached_values[0] * cached_ground_vector
        )
    )
    ground_delta = float(values[0] - cached_values[0])
    max_spectrum_delta = float(np.max(np.abs(values - cached_values)))
    evaluator_norm = float(np.linalg.norm(evaluator_ground_state))
    if evaluator_norm <= 0.0:
        raise ValueError(f"evaluator ground_state has zero norm: {npz_path}")
    evaluator_ground_state /= evaluator_norm
    sector_mask = np.zeros(hamiltonian.shape[0], dtype=bool)
    sector_mask[sector_indices] = True
    outside_sector_norm = float(np.linalg.norm(evaluator_ground_state[~sector_mask]))
    evaluator_sector_vector = evaluator_ground_state[sector_indices]
    evaluator_residual = float(
        np.linalg.norm(
            sector_hamiltonian @ evaluator_sector_vector - values[0] * evaluator_sector_vector
        )
    )
    degeneracy_tolerance = 1.0e-9
    recomputed_ground_indices = np.flatnonzero(values <= values[0] + degeneracy_tolerance)
    cached_ground_indices = np.flatnonzero(cached_values <= cached_values[0] + degeneracy_tolerance)
    recomputed_ground_overlap = float(
        np.sum(
            np.abs(vectors[:, recomputed_ground_indices].conj().T @ evaluator_sector_vector) ** 2
        )
    )
    cached_ground_subspace = cached_vectors[sector_indices][:, cached_ground_indices]
    cached_ground_overlap = float(
        np.sum(np.abs(cached_ground_subspace.conj().T @ evaluator_sector_vector) ** 2)
    )
    verified = (
        hermiticity_error <= 1.0e-12
        and pauli_max_abs_residual <= PAULI_DENSE_RESIDUAL_TOLERANCE
        and pauli_relative_frobenius_residual <= PAULI_DENSE_RESIDUAL_TOLERANCE
        and abs(ground_delta) <= 1.0e-10
        and max_spectrum_delta <= 1.0e-10
        and residual <= 1.0e-8
        and abs(evaluator_norm - 1.0) <= 1.0e-10
        and outside_sector_norm <= 1.0e-10
        and evaluator_residual <= 1.0e-8
        and recomputed_ground_overlap >= 1.0 - 1.0e-8
        and cached_ground_overlap >= 1.0 - 1.0e-8
    )
    return {
        "species": str(metadata["molecule"]),
        "cache_file": npz_path.name,
        "cache_sha256": expected_hash,
        "metadata_sha256": sha256_file(metadata_path),
        "n_qubits": n_qubits,
        "active_electrons": int(metadata["active_electrons"]),
        "sector_dimension": int(sector_indices.size),
        "hermiticity_error": hermiticity_error,
        "pauli_dense_max_abs_residual_hartree": pauli_max_abs_residual,
        "pauli_dense_relative_frobenius_residual": pauli_relative_frobenius_residual,
        "cached_ground_energy_hartree": float(cached_values[0]),
        "recomputed_ground_energy_hartree": float(values[0]),
        "ground_energy_delta_hartree": ground_delta,
        "max_spectrum_delta_hartree": max_spectrum_delta,
        "cached_ground_vector_residual": residual,
        "evaluator_ground_state_input_norm": evaluator_norm,
        "evaluator_ground_state_outside_sector_norm": outside_sector_norm,
        "evaluator_ground_state_residual": evaluator_residual,
        "ground_eigenspace_dimension": int(recomputed_ground_indices.size),
        "evaluator_recomputed_ground_eigenspace_overlap": recomputed_ground_overlap,
        "evaluator_cached_ground_eigenspace_overlap": cached_ground_overlap,
        "verified": bool(verified),
        "wall_seconds": time.perf_counter() - started,
    }


def run_verification(root: Path, workers: int) -> None:
    manifest, state = require_campaign(root)
    ensure_campaign_runner_runtime(root, Path(__file__), manifest)
    checksum_map = dict(manifest["inputs"]["problem_cache_files"])
    npz_paths = sorted((root / "problem_cache").glob("*.npz"))
    if len(npz_paths) != 21:
        raise RuntimeError(f"expected 21 Hamiltonian NPZ files, found {len(npz_paths)}")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(npz_paths))) as pool:
        futures = {
            pool.submit(verify_one, path, checksum_map[path.name]): path for path in npz_paths
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['species']} sector={row['sector_dimension']} "
                f"delta={row['ground_energy_delta_hartree']:+.3e} "
                f"eigen_residual={row['cached_ground_vector_residual']:.3e} "
                f"pauli_residual={row['pauli_dense_max_abs_residual_hartree']:.3e}",
                flush=True,
            )
    rows.sort(key=lambda item: str(item["species"]))
    payload = {
        "schema_version": 1,
        "experiment": "reactions10_hamiltonian_cache_verification",
        "diagonalization": "numpy.linalg.eigh on fixed-electron Hamiltonian sectors",
        "pauli_dense_check": (
            "identity_coefficient plus cached evaluator Pauli labels/coefficients "
            "against the cached dense Hamiltonian"
        ),
        "hamiltonian_count": len(rows),
        "all_verified": all(bool(row["verified"]) for row in rows),
        "manifest_sha256": state["manifest_sha256"],
        "verified_at": now_iso(),
        "rows": rows,
    }
    destination = root / "baselines" / "hamiltonian_verification.json"
    write_json(destination, payload)
    if not payload["all_verified"]:
        raise SystemExit("one or more cached Hamiltonians failed verification")
    state["hamiltonian_verification_sha256"] = sha256_file(destination)
    write_state(root, state)
    print(f"verified {len(rows)} cached Hamiltonians: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-diagonalize and verify all 21 fixed reaction Hamiltonian caches."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    root = args.campaign_root.resolve()
    with campaign_lock(root, "orchestration"):
        run_verification(root, args.workers)


if __name__ == "__main__":
    main()

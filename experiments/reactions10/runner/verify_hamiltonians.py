#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from campaign_common import (
    campaign_lock,
    ensure_campaign_runner_runtime,
    now_iso,
    require_campaign,
    sha256_file,
    write_json,
    write_state,
)


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
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"Hamiltonian is not square: {npz_path}")
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
        "n_qubits": int(metadata["n_qubits"]),
        "active_electrons": int(metadata["active_electrons"]),
        "sector_dimension": int(sector_indices.size),
        "hermiticity_error": hermiticity_error,
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
                f"residual={row['cached_ground_vector_residual']:.3e}",
                flush=True,
            )
    rows.sort(key=lambda item: str(item["species"]))
    payload = {
        "schema_version": 1,
        "experiment": "reactions10_hamiltonian_cache_verification",
        "diagonalization": "numpy.linalg.eigh on fixed-electron Hamiltonian sectors",
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

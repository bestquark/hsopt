from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np

from chemistry import load_problem_data
from model_registry import ACTIVE_MOLECULES, get_spec


DEFAULT_CANDIDATES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def apply_pauli_rotation(state: np.ndarray, pauli_matrix: np.ndarray, alpha: float, identity: np.ndarray) -> np.ndarray:
    return (math.cos(alpha) * identity + 1j * math.sin(alpha) * pauli_matrix) @ state


def unwrap_energy_near_reference(phase: float, target_time: float, reference_energy: float) -> float:
    principal_energy = float(phase / target_time)
    phase_period = float((2.0 * math.pi) / target_time)
    wrap_index = round((reference_energy - principal_energy) / phase_period)
    return principal_energy + wrap_index * phase_period


def exact_pf1_ground_state_energy_error(molecule: str, n_steps: int) -> float | None:
    spec = get_spec(molecule)
    payload = load_problem_data(molecule)
    state = np.array(payload["ground_state"], dtype=np.complex128, copy=True)
    identity = np.eye(payload["hamiltonian"].shape[0], dtype=np.complex128)
    exact_apply = payload.get("apply_exact_pauli_rotation")
    step_time = spec.target_time / float(n_steps)
    for _ in range(n_steps):
        for term in payload["pauli_terms"]:
            alpha = float(term["coefficient"]) * step_time
            if term.get("matrix") is not None:
                state = apply_pauli_rotation(state, term["matrix"], alpha, identity)
            elif exact_apply is not None:
                state = exact_apply(state, term["label"], alpha)
            else:
                raise RuntimeError(f"missing exact rotation path for {term['label']}")
    state = state / np.linalg.norm(state)
    overlap = np.vdot(payload["ground_state"], state)
    overlap_magnitude = float(abs(overlap))
    if overlap_magnitude <= 1e-15:
        return None
    ground_energy = float(payload["metadata"]["ground_state_energy_hartree"])
    identity_coefficient = float(payload["metadata"]["identity_coefficient"])
    centered_ground_energy = ground_energy - identity_coefficient
    estimate = unwrap_energy_near_reference(float(np.angle(overlap)), spec.target_time, centered_ground_energy)
    estimate += identity_coefficient
    return float(abs(estimate - ground_energy))


def patch_n_steps(script_path, n_steps: int) -> None:
    text = script_path.read_text()
    updated = re.sub(r"n_steps:\s*int\s*=\s*\d+", f"n_steps: int = {n_steps}", text, count=1)
    script_path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate baseline n_steps for small-molecule PF lanes.")
    parser.add_argument("--molecule", choices=ACTIVE_MOLECULES)
    parser.add_argument("--patch", action="store_true", help="Write the chosen n_steps back into the lane simple_pf.py.")
    args = parser.parse_args()

    molecules = [args.molecule] if args.molecule else list(ACTIVE_MOLECULES)
    summaries = []
    for molecule in molecules:
        spec = get_spec(molecule)
        payload = load_problem_data(molecule)
        accuracy_target = float(payload["metadata"]["chemical_accuracy_hartree"])
        best = None
        for candidate in DEFAULT_CANDIDATES:
            energy_error = exact_pf1_ground_state_energy_error(molecule, candidate)
            record = {
                "molecule": molecule,
                "candidate_n_steps": candidate,
                "ground_state_energy_error_hartree": energy_error,
                "chemical_accuracy_hartree": accuracy_target,
            }
            summaries.append(record)
            if energy_error is not None and energy_error <= accuracy_target and best is None:
                best = candidate
        if best is None:
            raise SystemExit(f"no passing candidate found for {molecule}")
        chosen = min(DEFAULT_CANDIDATES[-1], next((value for value in DEFAULT_CANDIDATES if value >= 2 * best), DEFAULT_CANDIDATES[-1]))
        if args.patch:
            patch_n_steps(spec.editable_script, chosen)
        print(
            json.dumps(
                {
                    "molecule": molecule,
                    "display_name": spec.display_name,
                    "chemical_accuracy_hartree": accuracy_target,
                    "first_passing_n_steps": best,
                    "chosen_n_steps": chosen,
                }
            )
        )


if __name__ == "__main__":
    main()

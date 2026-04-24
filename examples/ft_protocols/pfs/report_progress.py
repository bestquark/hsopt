from __future__ import annotations

import csv
import json
from pathlib import Path

from model_registry import ACTIVE_MOLECULES, get_spec


OBJECTIVE_NAME = "ground_state_phase_estimation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def baseline_result(molecule: str) -> dict | None:
    path = get_spec(molecule).snapshots_dir / "iter_0000" / "result.json"
    if not path.exists():
        return None
    return load_json(path)


def row_passes_constraint(row: dict[str, str]) -> bool:
    value = (row.get("passes_accuracy_constraint") or "").strip().lower()
    return value in {"true", "1", "yes"}


def row_matches_objective(row: dict[str, str]) -> bool:
    objective = (row.get("objective") or "").strip()
    return objective in {"", OBJECTIVE_NAME} and "passes_accuracy_constraint" in row


def best_valid_row(molecule: str) -> dict | None:
    results_path = get_spec(molecule).results_path
    if not results_path.exists():
        return None

    best_row = None
    best_score = None
    with results_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            if not row_matches_objective(row):
                continue
            if not row_passes_constraint(row):
                continue
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_row = row
    return best_row


def next_iteration(molecule: str) -> int:
    snapshots_dir = get_spec(molecule).snapshots_dir
    if not snapshots_dir.exists():
        return 0
    values = []
    for child in snapshots_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            values.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(values, default=-1) + 1


def pct_reduction(baseline_t: int, best_t: int) -> float:
    if baseline_t <= 0:
        return 0.0
    return 100.0 * (baseline_t - best_t) / baseline_t


def main() -> None:
    header = (
        f"{'molecule':<8} {'iter':>5} {'baseline_T':>12} {'best_T':>12} "
        f"{'reduction_%':>12} {'fidelity':>14}"
    )
    print(header)
    print("-" * len(header))

    for molecule in ACTIVE_MOLECULES:
        baseline = baseline_result(molecule)
        best = best_valid_row(molecule)
        if baseline is None:
            print(f"{molecule:<8} {'-':>5} {'-':>12} {'-':>12} {'-':>12} {'-':>14}")
            continue

        baseline_t = int(baseline["t_gate_count"])
        if best is None:
            best_iteration = 0
            best_t = baseline_t
            fidelity = float(baseline["fidelity"])
        else:
            best_iteration = int(best["iteration"])
            best_t = int(float(best["t_gate_count"]))
            fidelity = float(best["fidelity"])

        reduction = pct_reduction(baseline_t, best_t)
        print(
            f"{molecule:<8} {best_iteration:>5} {baseline_t:>12,d} {best_t:>12,d} "
            f"{reduction:>11.3f}% {fidelity:>14.12f}"
        )

    print()
    for molecule in ACTIVE_MOLECULES:
        print(f"{molecule}: last archived iteration {next_iteration(molecule) - 1}")


if __name__ == "__main__":
    main()

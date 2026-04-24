from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from model_registry import ACTIVE_MOLECULES, get_spec


OBJECTIVE_NAME = "ground_state_phase_estimation"


def row_passes_constraint(row: dict[str, str]) -> bool:
    value = (row.get("passes_accuracy_constraint") or "").strip().lower()
    return value in {"true", "1", "yes"}


def row_matches_objective(row: dict[str, str]) -> bool:
    objective = (row.get("objective") or "").strip()
    return objective in {"", OBJECTIVE_NAME} and "passes_accuracy_constraint" in row


def best_kept_iteration(molecule: str) -> int:
    results_path = get_spec(molecule).results_path
    if not results_path.exists():
        raise SystemExit(f"missing results file: {results_path}")
    best_iteration = None
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
                iteration = int(row["iteration"])
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            if best_score is None or score < best_score:
                best_iteration = iteration
                best_score = score
    if best_iteration is None:
        raise SystemExit(f"no kept iterations found for {molecule}")
    return best_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the best kept small-molecule PF iteration.")
    parser.add_argument("--molecule", required=True, choices=ACTIVE_MOLECULES)
    parser.add_argument("--script", required=True, help="Path to the editable simple_pf.py file.")
    args = parser.parse_args()

    spec = get_spec(args.molecule)
    iteration = best_kept_iteration(args.molecule)
    source = spec.snapshots_dir / f"iter_{iteration:04d}" / "simple_pf.py"
    destination = Path(args.script).resolve()
    if not source.exists():
        raise SystemExit(f"missing archived script: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"restored {args.molecule} best kept iteration {iteration} -> {destination}")


if __name__ == "__main__":
    main()

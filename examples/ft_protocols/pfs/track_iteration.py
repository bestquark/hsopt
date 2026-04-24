from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from evaluate import evaluate
from model_registry import SCRIPT_DIR, get_spec
from render_circuit_evolution import render_snapshot_artifacts


OBJECTIVE_NAME = "ground_state_phase_estimation"
RESULT_COLUMNS = [
    "iteration",
    "objective",
    "score",
    "status",
    "passes_accuracy_constraint",
    "accuracy_metric_name",
    "accuracy_metric_value",
    "accuracy_threshold_name",
    "accuracy_threshold_value",
    "passes_state_fidelity_diagnostic",
    "fidelity",
    "fidelity_floor",
    "t_gate_count",
    "description",
    "snapshot_dir",
]
AUTORESEARCH_LOG_COLUMNS = [
    "timestamp_utc",
    "iteration",
    "objective",
    "status",
    "passes_accuracy_constraint",
    "accuracy_metric_name",
    "accuracy_metric_value",
    "accuracy_threshold_name",
    "accuracy_threshold_value",
    "passes_state_fidelity_diagnostic",
    "score",
    "t_gate_count",
    "fidelity",
    "fidelity_floor",
    "description",
    "previous_best_iteration",
    "previous_best_score",
    "source_hash_before",
    "source_hash_after",
    "snapshot_dir",
]


def snapshot_dir(molecule: str, iteration: int) -> Path:
    return get_spec(molecule).snapshots_dir / f"iter_{iteration:04d}"


def next_iteration(molecule: str) -> int:
    snapshots_root = get_spec(molecule).snapshots_dir
    if not snapshots_root.exists():
        return 0
    values = []
    for child in snapshots_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            values.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(values, default=-1) + 1


def resolve_script(path_like: str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_source_file(source_file: Path) -> Tuple[Path, tempfile.TemporaryDirectory, bytes, str]:
    source_bytes = source_file.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    staging_dir = tempfile.TemporaryDirectory(prefix=".track_", dir=str(SCRIPT_DIR))
    frozen_file = Path(staging_dir.name) / source_file.name
    frozen_file.write_bytes(source_bytes)
    return frozen_file, staging_dir, source_bytes, source_hash


def previous_snapshot_file(molecule: str, iteration: int) -> Optional[Path]:
    for idx in range(iteration - 1, -1, -1):
        candidate = snapshot_dir(molecule, idx) / "simple_pf.py"
        if candidate.exists():
            return candidate
    return None


def write_diff(previous_file: Optional[Path], current_file: Path, output_file: Path) -> None:
    if previous_file is None:
        output_file.write_text("")
        return
    previous_lines = previous_file.read_text().splitlines(keepends=True)
    current_lines = current_file.read_text().splitlines(keepends=True)
    diff = difflib.unified_diff(previous_lines, current_lines, fromfile=str(previous_file), tofile=str(current_file))
    output_file.write_text("".join(diff))


def row_passes_constraint(row: Dict[str, str]) -> bool:
    value = (row.get("passes_accuracy_constraint") or "").strip().lower()
    return value in {"true", "1", "yes"}


def row_matches_objective(row: Dict[str, str]) -> bool:
    objective = (row.get("objective") or "").strip()
    return objective in {"", OBJECTIVE_NAME} and "passes_accuracy_constraint" in row


def best_kept_score(molecule: str) -> Tuple[Optional[int], Optional[float]]:
    results_path = get_spec(molecule).results_path
    if not results_path.exists():
        return None, None
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
            iteration_text = row.get("iteration", "")
            score_text = row.get("score", "")
            if not iteration_text or not score_text:
                continue
            try:
                iteration = int(iteration_text)
                score = float(score_text)
            except ValueError:
                continue
            if best_score is None or score < best_score:
                best_iteration = iteration
                best_score = score
    return best_iteration, best_score


def tsv_has_column(path: Path, column_name: str) -> bool:
    if not path.exists():
        return False
    with path.open() as handle:
        header = handle.readline().strip().split("\t")
    return column_name in header


def legacy_tsv_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.stem}_legacy_fidelity{path.suffix}")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        numbered = path.with_name(f"{path.stem}_legacy_fidelity_{index}{path.suffix}")
        if not numbered.exists():
            return numbered
        index += 1


def ensure_tsv_schema(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(columns) + "\n"
    if not path.exists():
        path.write_text(header)
        return
    existing_header = path.read_text().splitlines()[0].split("\t") if path.stat().st_size else []
    if existing_header == columns:
        return
    path.rename(legacy_tsv_path(path))
    path.write_text(header)


def append_results(molecule: str, row: Dict[str, object]) -> None:
    results_path = get_spec(molecule).results_path
    ensure_tsv_schema(results_path, RESULT_COLUMNS)
    with results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def append_autoresearch_log(molecule: str, row: Dict[str, object]) -> None:
    log_path = get_spec(molecule).lane_dir / "autoresearch_iteration_log.tsv"
    ensure_tsv_schema(log_path, AUTORESEARCH_LOG_COLUMNS)
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=AUTORESEARCH_LOG_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writerow({column: row.get(column, "") for column in AUTORESEARCH_LOG_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and archive one small-molecule PF iteration.")
    parser.add_argument("--molecule", required=True, choices=tuple(sorted(name for name in get_spec.__globals__["MOLECULE_SPECS"])))
    parser.add_argument("--script", required=True, help="Path to the editable simple_pf.py file.")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--description", default="", help="Short note describing the code mutation.")
    args = parser.parse_args()

    source_file = resolve_script(args.script)
    if not source_file.exists():
        raise SystemExit(f"missing source file: {source_file}")

    iteration = args.iteration if args.iteration is not None else next_iteration(args.molecule)
    out_dir = snapshot_dir(args.molecule, iteration)
    if out_dir.exists():
        raise SystemExit(f"snapshot directory already exists: {out_dir}")

    previous_best_iteration, previous_best_score = best_kept_score(args.molecule)
    frozen_file, staging_dir, source_bytes, source_hash_before = freeze_source_file(source_file)
    try:
        result = evaluate(args.molecule, frozen_file)
    except Exception as exc:  # noqa: BLE001
        result = None
        error = str(exc)
    else:
        error = None
    finally:
        staging_dir.cleanup()

    out_dir.mkdir(parents=True, exist_ok=False)
    frozen_output_file = out_dir / "simple_pf.py"
    frozen_output_file.write_bytes(source_bytes)
    write_diff(previous_snapshot_file(args.molecule, iteration), frozen_output_file, out_dir / "changes.diff")

    source_hash_after = sha256_bytes(source_file.read_bytes()) if source_file.exists() else None

    if result is None:
        status = "crash"
        result_payload: Dict[str, object] = {
            "objective": OBJECTIVE_NAME,
            "script": str(source_file),
            "error": error,
        }
        score = ""
        fidelity = ""
        fidelity_floor = ""
        t_gate_count = ""
        passes_constraint = ""
        accuracy_metric_name = ""
        accuracy_metric_value = ""
        accuracy_threshold_name = ""
        accuracy_threshold_value = ""
        passes_state_fidelity_diagnostic = ""
    else:
        candidate_score = float(result["score"])
        passes_constraint = bool(result.get("passes_accuracy_constraint", result["passes_fidelity_constraint"]))
        status = "keep" if passes_constraint and (previous_best_score is None or candidate_score <= previous_best_score) else "discard"
        result_payload = result
        score = candidate_score
        fidelity = float(result["fidelity"])
        fidelity_floor = float(result["chemical_accuracy_fidelity_floor"])
        t_gate_count = int(result["t_gate_count"])
        accuracy_metric_name = result.get("accuracy_metric_name", "")
        accuracy_metric_value = result.get("accuracy_metric_value", "")
        accuracy_threshold_name = result.get("accuracy_threshold_name", "")
        accuracy_threshold_value = result.get("accuracy_threshold_value", "")
        passes_state_fidelity_diagnostic = result.get("passes_state_fidelity_diagnostic", "")

    metadata = {
        "iteration": iteration,
        "status": status,
        "description": args.description,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "molecule": args.molecule,
        "objective": OBJECTIVE_NAME,
        "script": str(source_file),
        "previous_best_iteration": previous_best_iteration,
        "previous_best_score": previous_best_score,
        "source_hash_before": source_hash_before,
        "source_hash_after": source_hash_after,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "result.json").write_text(json.dumps(result_payload, indent=2))

    append_results(
        args.molecule,
        {
            "iteration": iteration,
            "objective": OBJECTIVE_NAME,
            "score": score,
            "status": status,
            "passes_accuracy_constraint": passes_constraint,
            "accuracy_metric_name": accuracy_metric_name,
            "accuracy_metric_value": accuracy_metric_value,
            "accuracy_threshold_name": accuracy_threshold_name,
            "accuracy_threshold_value": accuracy_threshold_value,
            "passes_state_fidelity_diagnostic": passes_state_fidelity_diagnostic,
            "fidelity": fidelity,
            "fidelity_floor": fidelity_floor,
            "t_gate_count": t_gate_count,
            "description": args.description,
            "snapshot_dir": str(out_dir),
        },
    )

    append_autoresearch_log(
        args.molecule,
        {
            "timestamp_utc": metadata["timestamp_utc"],
            "iteration": iteration,
            "objective": OBJECTIVE_NAME,
            "status": status,
            "passes_accuracy_constraint": passes_constraint,
            "accuracy_metric_name": accuracy_metric_name,
            "accuracy_metric_value": accuracy_metric_value,
            "accuracy_threshold_name": accuracy_threshold_name,
            "accuracy_threshold_value": accuracy_threshold_value,
            "passes_state_fidelity_diagnostic": passes_state_fidelity_diagnostic,
            "score": score,
            "t_gate_count": t_gate_count,
            "fidelity": fidelity,
            "fidelity_floor": fidelity_floor,
            "description": args.description,
            "previous_best_iteration": previous_best_iteration,
            "previous_best_score": previous_best_score,
            "source_hash_before": source_hash_before,
            "source_hash_after": source_hash_after,
            "snapshot_dir": str(out_dir),
        },
    )

    if result is not None:
        try:
            render_snapshot_artifacts(args.molecule, iteration)
        except Exception as exc:  # noqa: BLE001
            (out_dir / "precision_circuit_error.txt").write_text(str(exc))

    summary = {
        "iteration": iteration,
        "status": status,
        "snapshot_dir": str(out_dir),
        "previous_best_iteration": previous_best_iteration,
        "previous_best_score": previous_best_score,
        "result": result_payload,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

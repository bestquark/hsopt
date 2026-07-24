"""Evaluate this directory's sealed absolute-energy experiment."""

from pathlib import Path

from hsopt.experiment_evaluator import evaluate_absolute_directory


if __name__ == "__main__":
    raise SystemExit(evaluate_absolute_directory(Path(__file__).resolve().parent))

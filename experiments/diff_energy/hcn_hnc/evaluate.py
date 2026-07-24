"""Evaluate this directory's sealed signed-energy experiment."""

from pathlib import Path

from hsopt.experiment_evaluator import evaluate_difference_directory


if __name__ == "__main__":
    raise SystemExit(evaluate_difference_directory(Path(__file__).resolve().parent))

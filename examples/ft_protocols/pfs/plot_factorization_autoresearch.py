from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MOLECULES = ("n2", "co", "nh3", "hcn")
CONSTRAINTS = ("energy", "fidelity")
FACTORIZATIONS = ("unfactorized", "sf", "df", "thc")
LABELS = {
    "n2": "N2",
    "co": "CO",
    "nh3": "NH3",
    "hcn": "HCN",
    "energy": "Energy constraint",
    "fidelity": "Fidelity constraint",
    "unfactorized": "Unfactorized",
    "sf": "SF",
    "df": "DF",
    "thc": "THC",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _running_best_valid_t_counts(run_dir: Path) -> tuple[list[int], list[float]]:
    rows = []
    for event in _read_jsonl(run_dir / "logs" / "evaluations.jsonl"):
        if event.get("type") != "evaluation":
            continue
        result = event.get("result") or {}
        iteration = event.get("iteration")
        if not isinstance(iteration, int):
            continue
        t_count = result.get("t_gate_count")
        if t_count is None:
            continue
        passes = bool(result.get("passes_accuracy_constraint", False))
        rows.append((iteration, float(t_count), passes))

    rows.sort(key=lambda item: item[0])
    xs: list[int] = []
    ys: list[float] = []
    best = np.inf
    for iteration, t_count, passes in rows:
        if passes:
            best = min(best, t_count)
        if np.isfinite(best):
            xs.append(iteration)
            ys.append(float(best))
    return xs, ys


def _run_dir(run_root: Path, molecule: str, constraint: str, factorization: str) -> Path:
    return run_root / f"{molecule}_{constraint}_{factorization}"


def plot_matrix(run_root: Path, output: Path) -> None:
    colors = {
        "unfactorized": "#4C78A8",
        "sf": "#F58518",
        "df": "#54A24B",
        "thc": "#B279A2",
    }
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 7.8), sharex=True, constrained_layout=True)

    for col, molecule in enumerate(MOLECULES):
        for row, constraint in enumerate(CONSTRAINTS):
            ax = axes[row, col]
            for factorization in FACTORIZATIONS:
                xs, ys = _running_best_valid_t_counts(_run_dir(run_root, molecule, constraint, factorization))
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    label=LABELS[factorization],
                    color=colors[factorization],
                    linewidth=2.0,
                )
            ax.set_title(LABELS[molecule] if row == 0 else "")
            if col == 0:
                ax.set_ylabel(f"{LABELS[constraint]}\nBest valid T count")
            if row == 1:
                ax.set_xlabel("Iteration")
            ax.grid(alpha=0.25)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 4x2 PF factorization autoresearch T-count curves.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.run_root / "pfs_factorization_autoresearch.pdf")
    plot_matrix(args.run_root.resolve(), output.resolve())
    print(json.dumps({"output": str(output.resolve()), "png": str(output.resolve().with_suffix(".png"))}, indent=2))


if __name__ == "__main__":
    main()

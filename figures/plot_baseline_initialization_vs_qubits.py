#!/usr/bin/env python3
"""Plot the Reiher-derived baseline initialization versus qubit count."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.mlab import GaussianKDE
from matplotlib.patches import Patch


from _paths import data_path, output_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = data_path("data/baseline_initialization_vs_qubits.csv")
DEFAULT_OUTPUT = output_path("baseline_initialization_vs_qubits.pdf")
EXPECTED_QUBIT_COUNTS = {
    4: 20,
    6: 2,
    8: 24,
    10: 11,
    12: 25,
    14: 9,
    16: 24,
    18: 1,
    20: 3,
    22: 1,
    30: 1,
}

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK_GREEN = "#3E7E00"


@dataclass(frozen=True)
class BaselineRecord:
    lane: str
    n_qubits: int
    n_steps: int
    rotation_tolerance: float


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_records(path: Path) -> list[BaselineRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[BaselineRecord] = []
    seen: set[str] = set()
    for row in rows:
        lane = str(row["lane_id"])
        if lane in seen:
            raise ValueError(f"duplicate lane in {path}: {lane}")
        seen.add(lane)
        n_qubits = int(row["n_qubits"])
        n_steps = int(row["n_steps"])
        rotation_tolerance = float(row["uniform_epsilon"])
        if (
            n_qubits <= 0
            or n_steps <= 0
            or not math.isfinite(rotation_tolerance)
            or rotation_tolerance <= 0.0
        ):
            raise ValueError(f"invalid baseline data for {lane}")
        records.append(
            BaselineRecord(
                lane=lane,
                n_qubits=n_qubits,
                n_steps=n_steps,
                rotation_tolerance=rotation_tolerance,
            )
        )

    counts = Counter(record.n_qubits for record in records)
    if len(records) != 121 or dict(sorted(counts.items())) != EXPECTED_QUBIT_COUNTS:
        raise ValueError(
            f"expected the 121-system HamLib census {EXPECTED_QUBIT_COUNTS}, "
            f"found {dict(sorted(counts.items()))}"
        )
    return records


def quartiles_by_qubit(
    records: list[BaselineRecord],
    *,
    metric: str,
    qubit_counts: list[int],
) -> tuple[list[float], list[float], list[float]]:
    first_quartiles: list[float] = []
    medians: list[float] = []
    third_quartiles: list[float] = []
    for n_qubits in qubit_counts:
        values = np.asarray(
            [getattr(record, metric) for record in records if record.n_qubits == n_qubits],
            dtype=float,
        )
        first_quartile, median, third_quartile = np.quantile(values, [0.25, 0.5, 0.75])
        first_quartiles.append(float(first_quartile))
        medians.append(float(median))
        third_quartiles.append(float(third_quartile))
    return first_quartiles, medians, third_quartiles


def draw_log_violin(
    ax: mpl.axes.Axes,
    *,
    position: int,
    values: np.ndarray,
    maximum_half_width: float = 0.62,
) -> None:
    """Draw a violin whose density estimate is computed in log10 space."""

    log_values = np.log10(values)
    if values.size < 2 or np.ptp(log_values) <= np.finfo(float).eps:
        return

    density_estimate = GaussianKDE(log_values, bw_method=0.45)
    log_grid = np.linspace(float(log_values.min()), float(log_values.max()), 256)
    density = density_estimate(log_grid)
    half_width = maximum_half_width * density / float(density.max())
    ax.fill_betweenx(
        np.power(10.0, log_grid),
        position - half_width,
        position + half_width,
        facecolor=NVIDIA_GREEN,
        edgecolor=NVIDIA_DARK_GREEN,
        linewidth=1.25,
        alpha=0.34,
        zorder=2,
    )


def plot(records: list[BaselineRecord], output_path: Path) -> None:
    apply_style()
    qubit_counts = sorted({record.n_qubits for record in records})
    panels = (
        (
            "n_steps",
            r"Trotter repetitions, $r$",
        ),
        (
            "rotation_tolerance",
            r"Rotation tolerance, $\eta_{\mathrm{init}}$",
        ),
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10.8, 4.4),
        sharex=True,
        layout="constrained",
    )
    for panel_label, ax, (metric, y_label) in zip(("a", "b"), axes, panels):
        first_quartiles, medians, third_quartiles = quartiles_by_qubit(
            records,
            metric=metric,
            qubit_counts=qubit_counts,
        )
        for n_qubits in qubit_counts:
            values = np.asarray(
                [getattr(record, metric) for record in records if record.n_qubits == n_qubits],
                dtype=float,
            )
            draw_log_violin(ax, position=n_qubits, values=values)
        ax.vlines(
            qubit_counts,
            first_quartiles,
            third_quartiles,
            color=NVIDIA_DARK_GREEN,
            linewidth=2.5,
            zorder=3,
        )
        ax.scatter(
            qubit_counts,
            medians,
            marker="D",
            s=48,
            facecolor=NVIDIA_GREEN,
            edgecolor=NVIDIA_DARK_GREEN,
            linewidth=1.0,
            zorder=4,
        )
        ax.set_yscale("log")
        ax.set_xticks(qubit_counts)
        ax.set_xlabel(r"Qubits, $n_q$")
        ax.set_ylabel(y_label)
        ax.set_xlim(min(qubit_counts) - 0.8, max(qubit_counts) + 0.8)
        ax.grid(True, axis="y", which="major", color="0.87", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(rf"\textbf{{({panel_label})}}", loc="left", pad=4, fontsize=18)

    axes[0].legend(
        handles=[
            Patch(
                facecolor=NVIDIA_GREEN,
                edgecolor=NVIDIA_DARK_GREEN,
                alpha=0.34,
                label=r"Distribution",
            ),
            Line2D(
                [0],
                [0],
                color=NVIDIA_DARK_GREEN,
                linewidth=2.5,
                marker="D",
                markersize=6.8,
                markerfacecolor=NVIDIA_GREEN,
                markeredgecolor=NVIDIA_DARK_GREEN,
                label=r"Median and IQR",
            ),
        ],
        loc="lower right",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = load_records(args.data)
    plot(records, args.output)
    print(f"wrote {args.output} from {len(records)} Hamiltonians")


if __name__ == "__main__":
    main()

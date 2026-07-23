#!/usr/bin/env python3
"""Recreate the aggregate HamLib T95-reduction and trajectory figure."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    GRAPHITE,
    LIGHT_GREEN,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    apply_paper_style,
    as_bool,
    as_float,
    as_int,
    external_output_dir,
    group_rows,
    output_file,
    read_csv,
    save_figure,
    strict_export,
)


HISTOGRAM_OUTPUT = "hamlib_pf_tgate_reduction_histogram.pdf"
TRAJECTORY_OUTPUT = "hamlib_pf_tgate_trajectories.pdf"


def normalized_tgate_trajectories(
    summaries: list[dict[str, str]],
    candidates: list[dict[str, str]],
    *,
    iterations: int,
) -> tuple[list[str], np.ndarray]:
    candidates_by_lane = group_rows(candidates, "lane_id", "run")
    traces_by_instance: dict[str, list[np.ndarray]] = defaultdict(list)

    for summary in summaries:
        lane = summary["lane_id"]
        run = summary["run"]
        rows = sorted(
            candidates_by_lane.get((lane, run), []),
            key=lambda row: as_int(row["iteration"]),
        )
        grid = [as_int(row["iteration"]) for row in rows]
        if grid != list(range(1, iterations + 1)):
            raise ValueError(f"{lane}/{run} does not span iterations 1--{iterations}")
        baseline = as_float(
            summary["baseline_confidence_adjusted_t_gates_qpe"],
            label=f"{lane}/{run} baseline T95",
        )
        if baseline <= 0.0:
            raise ValueError(f"{lane}/{run} has a nonpositive baseline T95")
        incumbent = baseline
        trace = [1.0]
        for row in rows:
            if as_bool(row["accepted"]):
                incumbent = as_float(
                    row["confidence_adjusted_t_gates_qpe"],
                    label=f"{lane}/{run} candidate T95",
                )
            trace.append(incumbent / baseline)
        traces_by_instance[lane].append(np.asarray(trace, dtype=float))

    lanes = sorted(traces_by_instance)
    instance_medians = [np.median(np.vstack(traces_by_instance[lane]), axis=0) for lane in lanes]
    return lanes, np.vstack(instance_medians)


def plot_reduction_histogram(matrix: np.ndarray, output_path: Path) -> None:
    apply_paper_style()
    reductions = 100.0 * (1.0 - matrix[:, -1])
    mean = float(np.mean(reductions))
    median = float(np.median(reductions))
    upper = max(5.0, float(np.max(reductions)) * 1.08)
    bins = np.linspace(min(0.0, float(np.min(reductions))), upper, 13)

    fig, ax = plt.subplots(figsize=(5.4, 4.2), layout="constrained")
    ax.hist(
        reductions,
        bins=bins,
        color=NVIDIA_GREEN,
        edgecolor=NVIDIA_DARK_GREEN,
        linewidth=1.0,
        alpha=0.85,
    )
    ax.axvline(
        mean,
        color=NVIDIA_DARK_GREEN,
        linestyle="--",
        linewidth=2.0,
        label=rf"mean ${mean:.1f}\%$",
    )
    ax.axvline(
        median,
        color=GRAPHITE,
        linestyle=":",
        linewidth=2.0,
        label=rf"median ${median:.1f}\%$",
    )
    ax.set_xlabel(r"$T_{95}$ reduction (\%)")
    ax.set_ylabel(r"Hamiltonians")
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(True, axis="y", color="0.90", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    save_figure(fig, output_path)
    print(f"mean reduction: {mean:.3f}%")
    print(f"median reduction: {median:.3f}%")


def plot_trajectories(matrix: np.ndarray, output_path: Path) -> None:
    apply_paper_style()
    iterations = np.arange(matrix.shape[1])
    median = np.median(matrix, axis=0)
    lower = np.quantile(matrix, 0.25, axis=0)
    upper = np.quantile(matrix, 0.75, axis=0)

    fig, ax = plt.subplots(figsize=(5.4, 4.2), layout="constrained")
    for row in matrix:
        ax.plot(iterations, row, color=GRAPHITE, linewidth=0.8, alpha=0.14)
    ax.fill_between(
        iterations,
        lower,
        upper,
        color=LIGHT_GREEN,
        alpha=0.45,
        linewidth=0,
        label=r"IQR",
        zorder=2,
    )
    ax.plot(
        iterations,
        median,
        color=NVIDIA_DARK_GREEN,
        linewidth=2.7,
        label=r"median",
        zorder=3,
    )
    ax.set_xlim(0, iterations[-1])
    ax.set_ylim(
        min(0.98, 0.98 * float(np.min(matrix))),
        1.005,
    )
    ax.set_xlabel(r"Autoresearch iteration")
    ax.set_ylabel(r"$T_{95}^{\star}/T_{95}^{\mathrm{initial}}$")
    ax.grid(True, axis="y", color="0.90", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_dir, manifest = strict_export(args.export_dir, "abs_energy")
    output_dir = external_output_dir(args.output_dir, export_dir)
    summaries = read_csv(export_dir / "lane_run_summary.csv")
    candidates = read_csv(export_dir / "candidate_log.csv")
    lanes, matrix = normalized_tgate_trajectories(
        summaries,
        candidates,
        iterations=as_int(manifest["iterations"]),
    )
    if len(lanes) != as_int(manifest["instance_count"]):
        raise ValueError("T95-trajectory instance count disagrees with export manifest")
    plot_reduction_histogram(
        matrix,
        output_file(output_dir, HISTOGRAM_OUTPUT),
    )
    plot_trajectories(
        matrix,
        output_file(output_dir, TRAJECTORY_OUTPUT),
    )


if __name__ == "__main__":
    main()

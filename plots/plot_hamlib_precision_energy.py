#!/usr/bin/env python3
"""Recreate the HamLib precision-allocation and energy-trajectory figure."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    CHEMICAL_ACCURACY_MHA,
    GRAPHITE,
    LIGHT_GREEN,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    PALE_GRAPHITE,
    RED,
    apply_paper_style,
    as_bool,
    as_float,
    as_int,
    deterministic_sample,
    external_output_dir,
    group_rows,
    output_file,
    read_csv,
    read_json,
    save_figure,
    strict_export,
)


TOLERANCE_OUTPUT = "hamlib_pf_tolerance_angle_scatter.pdf"
ENERGY_OUTPUT = "hamlib_pf_energy_error_trajectories.pdf"


def compilation_cloud(
    result: Mapping[str, Any],
    *,
    reference_epsilon: float,
) -> tuple[list[float], list[float]]:
    angles: list[float] = []
    relative_epsilons: list[float] = []
    for row in result.get("unique_compilations", []):
        theta = abs(as_float(row["theta"], label="compiled theta"))
        epsilon = as_float(row["epsilon"], label="compiled epsilon")
        if theta <= 0.0 or epsilon <= 0.0:
            continue
        angles.append(theta)
        relative_epsilons.append(epsilon / reference_epsilon)
    return angles, relative_epsilons


def rotation_clouds(
    export_dir: Path,
    summaries: list[dict[str, str]],
    *,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial_angles: list[float] = []
    initial_relative: list[float] = []
    optimized_angles: list[float] = []
    optimized_relative: list[float] = []
    summaries_by_lane = group_rows(summaries, "lane_id")

    for (lane,), rows in sorted(summaries_by_lane.items()):
        baseline_row = sorted(rows, key=lambda row: row["run"])[0]
        baseline_epsilons = [
            as_float(
                row["baseline_uniform_epsilon"],
                label=f"{lane}/{row['run']} baseline epsilon",
            )
            for row in rows
        ]
        if not all(
            math.isclose(
                epsilon,
                baseline_epsilons[0],
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
            for epsilon in baseline_epsilons[1:]
        ):
            raise ValueError(f"{lane} has inconsistent baseline epsilon across replicates")
        baseline_record_dir = export_dir / "records" / lane / baseline_row["run"]
        baseline = read_json(baseline_record_dir / "baseline_result.json")
        reference_epsilon = as_float(
            baseline.get("policy", {}).get(
                "uniform_epsilon",
                baseline_row["baseline_uniform_epsilon"],
            ),
            label=f"{lane} baseline epsilon",
        )
        if reference_epsilon <= 0.0:
            raise ValueError(f"{lane} has a nonpositive baseline epsilon")
        angles, relative = compilation_cloud(
            baseline,
            reference_epsilon=reference_epsilon,
        )
        initial_angles.extend(angles)
        initial_relative.extend(relative)

        best_row = min(
            rows,
            key=lambda row: as_float(
                row["best_confidence_adjusted_t_gates_qpe"],
                label=f"{lane}/{row['run']} best T95",
            ),
        )
        best_record_dir = export_dir / "records" / lane / best_row["run"]
        best_iteration = as_int(best_row["best_iteration"], label="best iteration")
        best = (
            baseline
            if best_iteration == 0
            else read_json(best_record_dir / "incumbent_result.json")
        )
        angles, relative = compilation_cloud(best, reference_epsilon=reference_epsilon)
        optimized_angles.extend(angles)
        optimized_relative.extend(relative)

    initial_x = np.asarray(initial_angles, dtype=float)
    initial_y = np.asarray(initial_relative, dtype=float)
    optimized_x = np.asarray(optimized_angles, dtype=float)
    optimized_y = np.asarray(optimized_relative, dtype=float)
    if initial_x.size == 0 or optimized_x.size == 0:
        raise ValueError("strict export contains no compiled-rotation records")
    initial_x, initial_y = deterministic_sample(
        initial_x,
        initial_y,
        maximum=maximum_points,
        seed=17,
    )
    optimized_x, optimized_y = deterministic_sample(
        optimized_x,
        optimized_y,
        maximum=maximum_points,
        seed=23,
    )
    return initial_x, initial_y, optimized_x, optimized_y


def running_energy_trajectories(
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
        incumbent = 1000.0 * as_float(
            summary["baseline_ground_state_energy_error_hartree"],
            label=f"{lane}/{run} baseline energy error",
        )
        trace = [incumbent]
        for row in rows:
            if as_bool(row["accepted"]):
                incumbent = 1000.0 * as_float(
                    row["ground_state_energy_error_hartree"],
                    label=f"{lane}/{run} candidate energy error",
                )
            trace.append(incumbent)
        traces_by_instance[lane].append(np.asarray(trace, dtype=float))

    lanes = sorted(traces_by_instance)
    aggregated = []
    for lane in lanes:
        matrix = np.vstack(traces_by_instance[lane])
        aggregated.append(np.median(matrix, axis=0))
    return lanes, np.vstack(aggregated)


def plot_tolerance_cloud(
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    optimized_x: np.ndarray,
    optimized_y: np.ndarray,
    output_path: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    ax.scatter(
        initial_x,
        initial_y,
        s=10,
        color=PALE_GRAPHITE,
        alpha=0.30,
        linewidth=0,
        label=r"initial",
        rasterized=True,
    )
    ax.scatter(
        optimized_x,
        optimized_y,
        s=10,
        color=NVIDIA_GREEN,
        alpha=0.48,
        linewidth=0,
        label=r"optimized",
        rasterized=True,
    )
    ax.axhline(1.0, color=GRAPHITE, linestyle="--", linewidth=1.4, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Rotation-angle magnitude, $|\theta|$")
    ax.set_ylabel(r"Relative synthesis tolerance, $\varepsilon/\varepsilon_0$")
    ax.grid(True, which="major", color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    save_figure(fig, output_path)


def plot_energy_trajectories(lanes: list[str], matrix: np.ndarray, output_path: Path) -> None:
    apply_paper_style()
    iterations = np.arange(matrix.shape[1])
    median = np.median(matrix, axis=0)
    lower = np.quantile(matrix, 0.25, axis=0)
    upper = np.quantile(matrix, 0.75, axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    for row in matrix:
        ax.plot(iterations, row, color=GRAPHITE, alpha=0.13, linewidth=0.8)
    ax.fill_between(
        iterations,
        lower,
        upper,
        color=LIGHT_GREEN,
        alpha=0.40,
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
    ax.axhspan(0.0, CHEMICAL_ACCURACY_MHA, color=LIGHT_GREEN, alpha=0.16, zorder=0)
    ax.axhline(
        CHEMICAL_ACCURACY_MHA,
        color=RED,
        linewidth=2.0,
        label=r"$1~\mathrm{kcal}\,\mathrm{mol}^{-1}$",
        zorder=4,
    )
    robust_max = float(np.quantile(matrix, 0.995))
    ax.set_ylim(0.0, max(2.0 * CHEMICAL_ACCURACY_MHA, 1.08 * robust_max))
    ax.set_xlim(0, iterations[-1])
    ax.set_xlabel(r"Autoresearch iteration")
    ax.set_ylabel(r"Absolute energy error, $|\Delta E|$ (mHa)")
    ax.grid(True, axis="y", color="0.90", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    save_figure(fig, output_path)
    print(f"aggregated {matrix.shape[0]} Hamiltonians across {len(lanes)} instance keys")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=75_000,
        help="Deterministic per-cloud cap used to keep the vector PDF responsive.",
    )
    args = parser.parse_args()
    if args.max_scatter_points < 100:
        parser.error("--max-scatter-points must be at least 100")

    export_dir, manifest = strict_export(args.export_dir, "abs_energy")
    output_dir = external_output_dir(args.output_dir, export_dir)
    summaries = read_csv(export_dir / "lane_run_summary.csv")
    candidates = read_csv(export_dir / "candidate_log.csv")
    initial_x, initial_y, optimized_x, optimized_y = rotation_clouds(
        export_dir,
        summaries,
        maximum_points=args.max_scatter_points,
    )
    lanes, energy_matrix = running_energy_trajectories(
        summaries,
        candidates,
        iterations=as_int(manifest["iterations"]),
    )
    if len(lanes) != as_int(manifest["instance_count"]):
        raise ValueError("energy-trajectory instance count disagrees with export manifest")

    plot_tolerance_cloud(
        initial_x,
        initial_y,
        optimized_x,
        optimized_y,
        output_file(output_dir, TOLERANCE_OUTPUT),
    )
    plot_energy_trajectories(
        lanes,
        energy_matrix,
        output_file(output_dir, ENERGY_OUTPUT),
    )


if __name__ == "__main__":
    main()

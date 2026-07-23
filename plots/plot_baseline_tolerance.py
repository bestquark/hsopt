#!/usr/bin/env python3
"""Recreate the baseline rotation-tolerance versus one-norm figure."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    apply_paper_style,
    as_float,
    as_int,
    external_output_dir,
    group_rows,
    output_file,
    read_csv,
    save_figure,
    strict_export,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "experiments/abs_energy/inputs/instances.csv"
OUTPUT_NAME = "baseline_epsilon_vs_lambda.pdf"


@dataclass(frozen=True)
class BaselineRecord:
    lane: str
    one_norm: float
    epsilon: float
    nqubits: int


def load_records(
    summaries: list[dict[str, str]],
    selection_manifest: Path,
) -> list[BaselineRecord]:
    selected = {row["lane_id"]: row for row in read_csv(selection_manifest)}
    by_lane = group_rows(summaries, "lane_id")
    records: list[BaselineRecord] = []
    for (lane,), rows in sorted(by_lane.items()):
        if lane not in selected:
            raise ValueError(f"{lane} is absent from {selection_manifest}")
        epsilons = [
            as_float(row["baseline_uniform_epsilon"], label=f"{lane} baseline epsilon")
            for row in rows
        ]
        if any(
            not math.isclose(epsilons[0], value, rel_tol=1.0e-12, abs_tol=0.0)
            for value in epsilons[1:]
        ):
            raise ValueError(f"{lane} baseline epsilon differs across replicates")
        manifest_row = selected[lane]
        records.append(
            BaselineRecord(
                lane=lane,
                one_norm=as_float(manifest_row["one_norm"], label=f"{lane} one-norm"),
                epsilon=epsilons[0],
                nqubits=as_int(manifest_row["nqubits"], label=f"{lane} qubits"),
            )
        )
    return records


def plot(records: list[BaselineRecord], output_path: Path) -> None:
    apply_paper_style()
    one_norms = np.asarray([record.one_norm for record in records], dtype=float)
    epsilons = np.asarray([record.epsilon for record in records], dtype=float)
    qubits = np.asarray([record.nqubits for record in records], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), layout="constrained")
    scatter = ax.scatter(
        one_norms,
        epsilons,
        c=qubits,
        cmap="viridis",
        s=58,
        alpha=0.88,
        linewidth=0.4,
        edgecolor="white",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Hamiltonian coefficient one-norm, $\lambda$ (Ha)")
    ax.set_ylabel(r"Initial rotation tolerance, $\varepsilon_0$")
    ax.grid(True, which="major", color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"Qubits, $n_q$")
    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_dir, manifest = strict_export(args.export_dir, "abs_energy")
    output_dir = external_output_dir(
        args.output_dir,
        export_dir,
        args.selection_manifest,
    )
    summaries = read_csv(export_dir / "lane_run_summary.csv")
    records = load_records(summaries, args.selection_manifest)
    if len(records) != as_int(manifest["instance_count"]):
        raise ValueError("baseline-tolerance instance count disagrees with export manifest")
    plot(records, output_file(output_dir, OUTPUT_NAME))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recreate the paper's four-panel HamLib coefficient-distribution figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    LIGHT_GREEN,
    NVIDIA_DARK_GREEN,
    apply_paper_style,
    external_output_dir,
    molecule_label,
    output_file,
    read_csv,
    read_json,
    save_figure,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "baseline/molecules.json"
OUTPUT_NAME = "hamlib_hamiltonian_coefficient_histograms.pdf"


@dataclass(frozen=True)
class Instance:
    key: str
    molecule: str
    nqubits: int
    terms: int
    problem_path: Path


def load_instances(input_dir: Path, selection_manifest: Path) -> list[Instance]:
    input_dir = input_dir.expanduser().resolve()
    prepared = read_json(input_dir / "manifest.json")
    selection = {row["lane_id"]: row for row in read_csv(selection_manifest)}
    instances: list[Instance] = []
    for row in prepared.get("instances", []):
        key = str(row["instance"])
        if key not in selection:
            raise ValueError(f"{key} is absent from {selection_manifest}")
        selected = selection[key]
        problem_path = input_dir / str(row["problem"])
        instances.append(
            Instance(
                key=key,
                molecule=str(selected["molecule"]),
                nqubits=int(row["n_qubits"]),
                terms=int(row["terms"]),
                problem_path=problem_path,
            )
        )
    if not instances:
        raise ValueError(f"{input_dir} contains no prepared HamLib instances")
    return instances


def choose_instances(
    instances: list[Instance],
    requested: list[str] | None,
) -> list[Instance]:
    by_key = {instance.key: instance for instance in instances}
    if requested:
        missing = [key for key in requested if key not in by_key]
        if missing:
            raise ValueError(f"unknown instance keys: {missing}")
        if len(requested) != 4 or len(set(requested)) != 4:
            raise ValueError("--instances must name four distinct instances")
        return [by_key[key] for key in requested]

    chosen: list[Instance] = []
    molecules: set[str] = set()
    for instance in sorted(instances, key=lambda item: (-item.terms, item.key)):
        if instance.molecule in molecules:
            continue
        chosen.append(instance)
        molecules.add(instance.molecule)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise ValueError("the prepared set contains fewer than four distinct molecules")
    return chosen


def coefficient_magnitudes(path: Path) -> np.ndarray:
    problem: dict[str, Any] = read_json(path)
    values = np.asarray(
        [
            abs(float(term["coefficient"]))
            for term in problem["terms"]
            if abs(float(term["coefficient"])) > 1.0e-15
        ],
        dtype=float,
    )
    if values.size == 0:
        raise ValueError(f"{path} has no nonzero nonidentity coefficients")
    return values


def plot(instances: list[Instance], output_path: Path, *, bins: int) -> None:
    apply_paper_style(compact=True)
    values_by_instance = [coefficient_magnitudes(item.problem_path) for item in instances]
    pooled = np.concatenate(values_by_instance)
    bin_edges = np.logspace(
        np.floor(np.log10(pooled.min())),
        np.ceil(np.log10(pooled.max())),
        bins,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(11.8, 3.55),
        sharex=True,
        layout="constrained",
    )
    for ax, instance, values in zip(axes, instances, values_by_instance):
        ax.hist(
            values,
            bins=bin_edges,
            color=LIGHT_GREEN,
            edgecolor=NVIDIA_DARK_GREEN,
            linewidth=1.35,
            alpha=0.82,
            label=rf"${values.size:,}$ terms",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"{molecule_label(instance.molecule)} ({instance.nqubits} qubits)")
        ax.grid(True, which="major", color="0.86", linewidth=0.65)
        ax.grid(True, which="minor", color="0.93", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94)
    axes[0].set_ylabel(r"Count")
    fig.supxlabel(r"Pauli-coefficient magnitude, $|c_g|$ (Ha)")
    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hamlib-input-dir",
        type=Path,
        required=True,
        help="Prepared abs_energy input root containing manifest.json.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Current checked HamLib selection CSV used for molecule labels.",
    )
    parser.add_argument(
        "--instances",
        nargs=4,
        metavar=("INSTANCE_1", "INSTANCE_2", "INSTANCE_3", "INSTANCE_4"),
        help="Four explicit instance keys; defaults to the largest distinct molecules.",
    )
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bins < 8:
        parser.error("--bins must be at least 8")

    available = load_instances(args.hamlib_input_dir, args.selection_manifest)
    chosen = choose_instances(available, args.instances)
    output_dir = external_output_dir(
        args.output_dir,
        args.hamlib_input_dir,
        args.selection_manifest,
    )
    output = output_file(output_dir, OUTPUT_NAME)
    plot(chosen, output, bins=args.bins)
    print("instances:", ", ".join(item.key for item in chosen))


if __name__ == "__main__":
    main()

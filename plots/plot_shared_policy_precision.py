#!/usr/bin/env python3
"""Inspect a shared policy's per-term tolerances from a frozen external snapshot.

One row represents one nonidentity Pauli term, rather than every repeated
rotation. No synthesis, eigensolver, or campaign mutation is run here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from _paper_plot import GRAPHITE, NVIDIA_GREEN, apply_paper_style


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def plot(data: Path, manifest_path: Path, molecule: str, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    rows = read_rows(data)
    cases = {row["case_id"] for row in rows}
    if len(rows) != manifest["term_count"] or len(cases) != manifest["molecule_count"]:
        raise ValueError("table does not match the frozen snapshot manifest")
    matches = [case for case in cases if molecule in case]
    if len(matches) != 1:
        raise ValueError("--molecule must select exactly one case ID")
    selected = matches[0]
    representative = [row for row in rows if row["case_id"] == selected]
    case = next(row for row in manifest["cases"] if row["case_id"] == selected)

    def array(points, key):
        value = np.array([float(row[key]) for row in points])
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite {key}")
        return value

    support = array(rows, "support_size")
    epsilon = array(rows, "epsilon")
    ratio = array(rows, "epsilon_ratio")
    coefficient = array(rows, "abs_coefficient_hartree")
    relative_coefficient = array(rows, "coefficient_ratio")
    # Zero coefficients cannot appear on logarithmic axes; retain their count
    # in the snapshot and only omit them from the graphical representation.
    retained = (coefficient > 0) & (epsilon > 0) & (ratio > 0)
    if not retained.any():
        raise ValueError("no positive terms to plot")
    apply_paper_style()
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 23, "axes.titlesize": 22,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 17})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "pauli_weight", ["#154F36", "#329360", NVIDIA_GREEN, "#DFE6A5"]
    )
    norm = Normalize(vmin=1, vmax=int(support.max()))
    a = array(representative, "abs_coefficient_hartree")
    e = array(representative, "epsilon")
    keep = (a > 0) & (e > 0)
    axes[0].scatter(a[keep], e[keep], c=array(representative, "support_size")[keep],
                    s=20, alpha=0.7, linewidths=0, cmap=cmap, norm=norm, rasterized=True)
    axes[0].axhline(case["baseline_epsilon"], color=GRAPHITE, linestyle="--", linewidth=1.7,
                    label=r"Uniform baseline $\epsilon_0$")
    axes[0].set(xlabel=r"$|c_j|$ (Ha)", ylabel=r"Rotation tolerance $\epsilon_j$")
    short_name = selected.split("_standard_")[-1].split("_")[0]
    if short_name == "OH":
        short_name = r"$\mathrm{OH}$"
    else:
        short_name = short_name.replace("_", r"\_")
    axes[0].set_title(f"{short_name}, {case['qubits']} qubits")
    axes[0].legend(loc="lower left")

    dots = axes[1].scatter(relative_coefficient[retained], ratio[retained],
                           c=support[retained], s=7, alpha=0.4, linewidths=0,
                           cmap=cmap, norm=norm, rasterized=True)
    axes[1].axhline(1, color=GRAPHITE, linestyle="--", linewidth=1.7)
    axes[1].set(xlabel=r"Relative coefficient $|c_j|/c_{\max}$",
                ylabel=r"Tolerance multiplier $\epsilon_j/\epsilon_0$",
                title=f"All {len(cases)} molecules")
    for index, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", alpha=0.12, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0, 1.04, rf"\textbf{{({chr(97 + index)})}}", transform=ax.transAxes,
                fontsize=21, va="bottom", ha="left")
    colorbar = fig.colorbar(dots, ax=axes, fraction=0.028, pad=0.025)
    colorbar.set_label(r"Pauli weight $w_j$")
    colorbar.set_ticks(sorted({1, 4, 8, 12, int(support.max())} & set(map(int, support))))
    colorbar.solids.set_alpha(1)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output / f"mixed_precision.{extension}"
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--molecule", required=True, help="Unique case ID or substring")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data.resolve(), args.manifest.resolve(), args.molecule, args.output_dir.resolve())


if __name__ == "__main__":
    main()

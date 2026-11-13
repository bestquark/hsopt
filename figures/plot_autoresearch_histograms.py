#!/usr/bin/env python3
"""Rebuild Figure 1's histogram inset as native PGF with readable typography.

Native inclusion lets the figure's LaTeX run embed all fonts together, avoiding
missing glyphs when a nested Matplotlib PDF is included in the manuscript.
The frozen data and 55 logarithmic bin edges are unchanged.
"""

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt
import numpy as np

from _coefficient_data import (
    INSTANCES,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    coefficient_magnitudes,
)
from _paths import output_path

OUTPUT = output_path(".build/coefficient_histograms_2x2.pgf")


def main() -> None:
    preamble = r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": preamble,
            "text.usetex": True,
            "text.latex.preamble": preamble,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 27,
            "axes.titlesize": 26,
            "axes.labelsize": 29,
            "xtick.labelsize": 24,
            "ytick.labelsize": 26,
            "axes.linewidth": 1,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "savefig.bbox": None,
        }
    )
    data = [coefficient_magnitudes(instance) for instance in INSTANCES]
    all_values = np.concatenate(data)
    bins = np.logspace(
        np.floor(np.log10(all_values.min())),
        np.ceil(np.log10(all_values.max())),
        55,
    )
    fig, axes = plt.subplots(2, 2, figsize=(5.25, 4.25), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.23, right=0.985, bottom=0.20, top=0.88, wspace=0.20, hspace=0.50
    )
    for ax, instance, values in zip(axes.flat, INSTANCES, data):
        ax.hist(values, bins=bins, histtype="stepfilled", color=NVIDIA_GREEN,
                alpha=0.35, linewidth=0)
        ax.hist(values, bins=bins, histtype="step", color=NVIDIA_DARK_GREEN,
                linewidth=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 1e4)
        ax.set_title(f"{instance.label} ({instance.nqubits} Q)", pad=7)
        ax.set_xticks([1e-6, 1e-1])
        ax.set_yticks([1, 1e2, 1e4])
        ax.minorticks_off()
        ax.label_outer()
        ax.tick_params(pad=3)
    fig.supylabel("Count", x=0.018, y=0.56, fontsize=30)
    fig.supxlabel(r"$|c_g|$ (Ha)", x=0.59, y=0.005, fontsize=30)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the exact coefficient-histogram inset used in Figure 1."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from _coefficient_data import (
    INSTANCES,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    apply_mlp_style,
    coefficient_magnitudes,
)


from _paths import output_path

OUTPUT = output_path(".build/coefficient_histograms_2x2.pdf")


def main() -> None:
    apply_mlp_style()
    mpl.rcParams.update(
        {
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}\usepackage{lmodern}"
                r"\usepackage{amsmath}\usepackage{amssymb}"
            ),
            "font.size": 40,
            "axes.labelsize": 48,
            "axes.titlesize": 42,
            "xtick.labelsize": 40,
            "ytick.labelsize": 40,
        }
    )
    data = [coefficient_magnitudes(instance) for instance in INSTANCES]
    all_values = np.concatenate(data)
    bins = np.logspace(
        np.floor(np.log10(all_values.min())),
        np.ceil(np.log10(all_values.max())),
        55,
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.20,
        right=0.99,
        bottom=0.19,
        top=0.89,
        wspace=0.12,
        hspace=0.38,
    )
    for ax, instance, values in zip(axes.flat, INSTANCES, data):
        ax.hist(
            values, bins=bins, histtype="stepfilled", color=NVIDIA_GREEN, alpha=0.35, linewidth=0
        )
        ax.hist(values, bins=bins, histtype="step", color=NVIDIA_DARK_GREEN, linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{instance.label} ({instance.nqubits} Q)", pad=8)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.08, linewidth=0.5)
        ax.set_xticks([1e-6, 1e-2])
        ax.set_yticks([1, 1e2, 1e4])
        ax.label_outer()
    axes[0, 0].set_ylim(0.8, 1e4)
    fig.supylabel("Count", x=0.01, fontsize=48)
    fig.supxlabel(r"$|c_g|$ (Ha)", y=0.015, fontsize=48)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

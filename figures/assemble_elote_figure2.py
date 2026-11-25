#!/usr/bin/env python3
"""Reproduce Figure 2 from frozen rotation data and QPE-inclusive T95 costs.

The NPZ supplies CUDA-Q per-rotation data for panel a. The JSON supplies matched
baseline/ELOTE T95 costs for panel b, including QPE queries and repeated attempts.
HF-18 is included in the testing histogram. Its baseline accuracy failure is
disclosed in the manuscript and frozen data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np

from _paths import data_path, output_path

ROOT = Path(__file__).resolve().parents[1]
DATA = data_path("data/elote115_figure2.npz")
T95_DATA = data_path("data/elote115_figure2_t95.json")
NVIDIA_GREEN = "#76B900"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=output_path("elote115_precision_and_compiled_savings.pdf")
    )
    args = parser.parse_args()
    with np.load(DATA, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {name: archive[name] for name in metadata["arrays"]}
    for name, value in arrays.items():
        assert (
            hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            == metadata["arrays"][name]
        ), name
        assert np.isfinite(value).all(), name
    x, y, t = [arrays[name] for name in ("angle", "relative_tolerance", "rotation_t_count")]
    snapshot = json.loads(T95_DATA.read_text())
    assert snapshot["schema"] == "elote-figure2-t95-v1"
    assert snapshot["policy_sha256"] == metadata["policy_sha256"]
    for path, expected in snapshot["source_sha256"].items():
        assert hashlib.sha256(data_path(path, expected).read_bytes()).hexdigest() == expected, path
    rows = snapshot["rows"]
    assert len({row["case_id"] for row in rows}) == len(rows) == 191
    for row in rows:
        assert row["baseline_t95"] > 0 and row["elote_t95"] > 0
        assert np.isclose(
            row["reduction_percent"],
            100 * (1 - row["elote_t95"] / row["baseline_t95"]),
            rtol=1e-13,
            atol=0,
        )
    training, testing = [
        np.array([row["reduction_percent"] for row in rows if row["dataset"] == dataset])
        for dataset in ("training", "testing")
    ]
    assert len(training) == 115 and len(testing) == 76
    assert all(row["elote_passes_chemical_accuracy"] for row in rows)
    assert sum(not row["baseline_passes_chemical_accuracy"] for row in rows) == 1
    assert x.shape == y.shape == t.shape and np.all(x > 0) and np.all(y > 0)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 21,
            "axes.labelsize": 23,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 17,
            "axes.linewidth": 1.25,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "hatch.linewidth": 4.0,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "green_inferno",
        [
            (0, "#000000"),
            (0.20, "#0A3324"),
            (0.42, "#126C43"),
            (0.65, "#24A65D"),
            (0.82, "#80D971"),
            (1, "#EAF9A3"),
        ],
        N=512,
    )
    norm = Normalize(0, int(t.max()))
    order = np.random.default_rng(20261023).permutation(len(x))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), layout="constrained")
    left, right = axes
    left.axhline(1, color=".55", ls="--", lw=1.7, zorder=0)
    points = left.scatter(
        x[order],
        y[order],
        c=t[order],
        cmap=cmap,
        norm=norm,
        s=5,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    left.set(
        xscale="log",
        yscale="log",
        xlabel=r"Rotation angle $|\theta|$ (rad)",
        ylabel=r"Relative tolerance $\epsilon_j/\epsilon_0$",
    )
    left.set_ylim(bottom=0.83)
    left.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:g}$"))
    left.yaxis.set_minor_locator(NullLocator())
    left.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                markersize=8,
                markerfacecolor="#24A65D",
                markeredgecolor="none",
                label="Best ELOTE",
            ),
            Line2D([], [], color=".55", ls="--", lw=1.7, label="Tight baseline"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.08, 1.0),
        borderaxespad=0,
        handlelength=1.7,
        labelspacing=0.3,
    )
    cax = left.inset_axes((0.10, 0.63, 0.57, 0.055))
    ticks = sorted({v for v in (0, 48, 80, int(t.max())) if v <= t.max()})
    cb = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=15, length=4, pad=2)
    cb.ax.set_title(r"$T$ gates per rotation", fontsize=17, pad=7)
    bins = np.histogram_bin_edges(np.r_[training, testing], bins="fd")
    total = len(training) + len(testing)
    a = np.histogram(training, bins=bins)[0] * 100 / total
    b = np.histogram(testing, bins=bins)[0] * 100 / total
    assert np.isclose(a.sum() + b.sum(), 100)
    right.bar(
        bins[:-1],
        a,
        width=np.diff(bins),
        align="edge",
        color=NVIDIA_GREEN,
        edgecolor="#3E7E00",
        lw=0.65,
        label=r"Training ($n=115$)",
    )
    right.bar(
        bins[:-1],
        b,
        width=np.diff(bins),
        bottom=a,
        align="edge",
        # Opaque pastel colors avoid disappearing translucent hatches in PDF viewers.
        color="#E6F2D1",
        edgecolor="#BCDC82",
        hatch="/",
        lw=0.8,
        label=r"Testing ($n=76$)",
    )
    # Draw outlines separately so the testing hatches keep their lighter green.
    visible = b > 0
    right.bar(
        bins[:-1][visible],
        b[visible],
        width=np.diff(bins)[visible],
        bottom=a[visible],
        align="edge",
        facecolor="none",
        edgecolor="#3E7E00",
        lw=0.65,
        label="_nolegend_",
    )
    for values, color in [(training, "#174D2B"), (testing, "#4FB88A")]:
        right.axvline(np.median(values), color=color, ls=":", lw=1.9)
    right.set(xlabel=r"$T_{95}$ reduction (\%)", ylabel=r"All Hamiltonians (\%)", ylim=(0, None))
    right.xaxis.set_major_locator(MaxNLocator(nbins=6))
    right.legend(loc="upper left", fontsize=17)
    for index, ax in enumerate(axes):
        ax.grid(False, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.12, 1.02, rf"({chr(97 + index)})", transform=ax.transAxes, va="bottom", fontsize=25
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
    preview = output_path(".build/figure2-final-preview.png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=300)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                policy_sha256=metadata["policy_sha256"],
                training_pairs=len(training),
                testing_pairs=len(testing),
                baseline_accuracy_failures=len(snapshot["baseline_accuracy_failures"]),
                rotation_points=len(x),
            )
        )
    )


if __name__ == "__main__":
    main()

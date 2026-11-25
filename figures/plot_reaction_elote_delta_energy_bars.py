"""Rebuild Figure 4b and its legend from frozen reaction-energy measurements."""
from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.transforms import Bbox
import numpy as np

from _paths import data_path, output_path
from plot_reaction_elote_delta_radial import LABELS

SOURCE = data_path("data/reaction_elote_delta_energy_summary.json")
FIGURE4_SOURCE = data_path("data/reaction_elote_delta_a100_summary.json")
OUTPUT = output_path("reaction_elote_delta_a100_energy_bars_log.pdf")
LEGEND_OUTPUT = output_path("reaction_elote_delta_legend.pdf")
GREEN, PURPLE, RED = "#76B900", "#6541A5", "#BC4A4A"
FLOOR, CEILING = 0.0007, 2.6
ORDER = (
    "acetylene_vinylidene", "hcn_hnc", "hco_hoc_cation", "co_h2o_co2_h2",
    "co_h2_ch4_h2o", "n2_h2_nh3", "h2_f2_hf", "h2_cl2_hcl", "h2co_hcoh", "hno_noh",
)


def tint(color, strength):
    return tuple(1.0 - strength + strength * channel for channel in to_rgb(color))


def pale(color):
    return tint(color, 0.32)


def main():
    energy = json.loads(SOURCE.read_text())
    cost = json.loads(FIGURE4_SOURCE.read_text())
    rows = {r["key"]: r for r in energy["rows"]}
    ranked = sorted(cost["order"], key=lambda c: (cost["reaction_max_qubits"][c], cost["baseline_t95"][c]))
    order = ranked[::2] + ranked[::-2]
    if tuple(order) != ORDER or set(rows) != set(ORDER) or set(ORDER) != set(LABELS):
        raise ValueError("Figure 4 panels must share the frozen ten-reaction order")
    if (energy["schema"] != "reaction-elote-delta-cudaq-a100-energy-v1"
            or energy["backend"] != "cupy-lanczos"
            or energy["evaluation_contract_version"] != cost["evaluation_contract_version"]
            or energy["evaluation_contract_version"] != "normalized_hamiltonian_ground_input_signed_difference_cudaq_elote_v1"):
        raise ValueError("Figure 4 energy provenance differs from the frozen campaign")

    # Both panels have the same printed width. Compensate for their different
    # PDF widths so reaction numbers, value ticks, and axis labels match (a).
    font_scale = 544.414 / 537.214
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 24, "axes.labelsize": 24 * font_scale,
        "xtick.labelsize": 30 * font_scale, "ytick.labelsize": 26 * font_scale,
        "legend.fontsize": 26, "legend.frameon": False,
        "axes.linewidth": 1.25, "xtick.major.width": 1.15,
        "ytick.major.width": 1.15, "xtick.major.size": 5.5,
        "ytick.major.size": 5.5, "savefig.dpi": 300, "hatch.linewidth": 0.65,
    })

    # Match the current panel (b)'s PDF MediaBox exactly, including label space.
    size = (544.414 / 72, 525.8 / 72)
    fig = plt.figure(figsize=size)
    # Leave enough room for the enlarged value ticks and the complete y label
    # inside the fixed PDF crop; preserve the right edge of the plotting area.
    ax = fig.add_axes([0.215, 0.09, 0.755, 0.825])
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, CEILING)
    ax.set_xlim(0.35, 10.65)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D8D8D8", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ticks = [0.001, 0.01, 0.1, 1.0]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([r"$0.001$", r"$0.01$", r"$0.1$", r"$1$"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(np.arange(1, 11), [rf"${i}$" for i in range(1, 11)])
    # Numbered reactions share the key beside both panels, as in the original.
    ax.set_ylabel("Energy error [mHa]", labelpad=10)
    fig.text(0.015, 0.955, "(b)", ha="left", va="top", fontsize=31)

    bar_width = 0.35
    for policy, color, offset in (("elote", GREEN, -bar_width / 2), ("delta_elote", PURPLE, bar_width / 2)):
        x = np.arange(1, 11) + offset
        signed = np.array([rows[c][policy]["signed_error_mha"] for c in order])
        bound = np.array([rows[c][policy]["sumabs_error_mha"] for c in order])
        assert np.all(signed <= bound + 1e-9)
        assert np.all(signed > FLOOR) and np.all(bound < CEILING)
        patterned = ax.bar(x, bound - FLOOR, bottom=FLOOR, width=bar_width,
                           facecolor=tint(color, 0.06), edgecolor=tint(color, 0.32), hatch="//",
                           linewidth=0.9, zorder=3)
        filled = ax.bar(x, signed - FLOOR, bottom=FLOOR, width=bar_width,
                        facecolor=pale(color), edgecolor=color, linewidth=1.1, zorder=4)
        for c, s, b, fs, hb in zip(order, signed, bound, filled, patterned):
            # Endpoints must be the recorded values, despite the logarithmic floor.
            assert np.isclose(fs.get_y() + fs.get_height(), s, rtol=1e-13)
            assert np.isclose(hb.get_y() + hb.get_height(), b, rtol=1e-13)

    ax.axhline(energy["chemical_accuracy_mha"], color=RED, lw=2.1, ls=(0, (5, 4)), zorder=5)
    ax.annotate("Maximum allowed error", xy=(0.985, energy["chemical_accuracy_mha"]),
                xycoords=ax.get_yaxis_transform(), xytext=(0, -6), textcoords="offset points",
                ha="right", va="top", color=RED, fontsize=22, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2})
    fig.savefig(OUTPUT, bbox_inches=Bbox.from_bounds(0, 0, *size),
                metadata={"CreationDate": None})
    plt.close(fig)

    legend_fig = plt.figure(figsize=(16, 1.2))
    handles = [Line2D([], [], color=GREEN, lw=3), Line2D([], [], color=PURPLE, lw=3),
               Patch(facecolor="#D6D6D6", edgecolor="#626262", lw=1.4),
               Patch(facecolor="#F8F8F8", edgecolor="#BDBDBD", hatch="//", lw=0.9),
               Line2D([], [], color=RED, lw=2, ls="--")]
    legend_fig.legend(handles, ["ELOTE", r"$\delta$-ELOTE", r"$\left|\sum_i\nu_i\delta E_i\right|$",
                               r"$\sum_i\left|\nu_i\delta E_i\right|$", r"$\epsilon_{\rm chem}$"],
                      ncol=5, loc="center", fontsize=26, columnspacing=1.0,
                      handlelength=1.5, handleheight=1.1, handletextpad=0.5)
    legend_fig.savefig(LEGEND_OUTPUT, bbox_inches="tight", pad_inches=0.08,
                       metadata={"CreationDate": None})
    plt.close(legend_fig)
    print(f"wrote {OUTPUT}")
    print(f"wrote {LEGEND_OUTPUT}")


if __name__ == "__main__":
    main()

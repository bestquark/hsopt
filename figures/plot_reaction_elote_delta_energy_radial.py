"""Rebuild Figure 4b from the checked-in CUDA-Q reaction-energy snapshot."""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import numpy as np

from plot_reaction_elote_delta_radial import LABELS, RADIAL_TICK_BBOX

from _paths import data_path, output_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = data_path("data/reaction_elote_delta_energy_summary.json")
FIGURE4_SOURCE = data_path("data/reaction_elote_delta_a100_summary.json")
OUTPUT = output_path("reaction_elote_delta_a100_energy_radial_log.pdf")
LEGEND_OUTPUT = output_path("reaction_elote_delta_legend.pdf")
ORDER = (
    "acetylene_vinylidene",
    "hcn_hnc",
    "hco_hoc_cation",
    "co_h2o_co2_h2",
    "co_h2_ch4_h2o",
    "n2_h2_nh3",
    "h2_f2_hf",
    "h2_cl2_hcl",
    "h2co_hcoh",
    "hno_noh",
)
GREEN = "#76B900"
PURPLE = "#6541A5"
RED = "#BC4A4A"
FLOOR = 0.0007
CEILING = 2.6


class StackedLinesHandler(HandlerBase):
    """Show a policy's solid signed-error and dashed bound in one legend key."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [
            Line2D(
                [-xdescent, width - xdescent],
                [height * fraction - ydescent] * 2,
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                transform=trans,
            )
            for line, fraction in zip(orig_handle, (0.75, 0.25))
        ]


mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 15,
        "axes.labelsize": 17,
        "legend.fontsize": 25,
        "axes.linewidth": 1.25,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def radial(value):
    return np.log10(np.asarray(value) / FLOOR)


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    t95_data = json.loads(FIGURE4_SOURCE.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in data["rows"]}
    ordered = sorted(
        t95_data["order"],
        key=lambda case: (t95_data["reaction_max_qubits"][case], t95_data["baseline_t95"][case]),
    )
    if (
        tuple(ordered[::2] + ordered[::-2]) != ORDER
        or set(rows) != set(ORDER)
        or set(ORDER) != set(LABELS)
    ):
        raise ValueError("Figure 4 panels do not share the same ten-reaction ordering")
    if (
        data["schema"] != "reaction-elote-delta-cudaq-a100-energy-v1"
        or data["backend"] != "cupy-lanczos"
        or data["evaluation_contract_version"]
        != "normalized_hamiltonian_ground_input_signed_difference_cudaq_elote_v1"
    ):
        raise ValueError("Figure 4b energy provenance does not match the CUDA-Q run")
    if any(
        row[policy]["signed_error_mha"] > row[policy]["sumabs_error_mha"] + 1e-9
        for row in rows.values()
        for policy in ("elote", "delta_elote")
    ):
        raise ValueError("A signed reaction error exceeds its no-cancellation bound")
    theta = np.linspace(0, 2 * np.pi, len(ORDER), endpoint=False)
    fig, ax = plt.subplots(figsize=(9.5, 8.4), subplot_kw={"projection": "polar"})
    ax.set_position([0.16, 0.14, 0.68, 0.72])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, float(radial(CEILING)))
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    ax.set_yticks([float(radial(value)) for value in (0.01, 0.1, 1.0)])
    ax.set_yticklabels([])
    ax.grid(color="#D8D8D8", linewidth=0.8)
    ax.spines["polar"].set_color("#D8D8D8")

    for index, angle in enumerate(theta, start=1):
        x, y = math.sin(angle), math.cos(angle)
        ax.text(
            0.5 + 0.56 * x,
            0.5 + 0.56 * y,
            rf"${index}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=30,
            clip_on=False,
        )

    angles = np.linspace(0, 2 * np.pi, 512)
    chem = float(radial(data["chemical_accuracy_mha"]))
    ax.plot(angles, np.full_like(angles, chem), color=RED, lw=2.2, ls=(0, (5, 4)), zorder=10)

    def pale(color):
        return tuple(0.68 + 0.32 * channel for channel in to_rgb(color))

    for policy, color in (("elote", GREEN), ("delta_elote", PURPLE)):
        x = theta
        signed = np.array([rows[key][policy]["signed_error_mha"] for key in ORDER])
        bound = np.array([rows[key][policy]["sumabs_error_mha"] for key in ORDER])
        sr = radial(signed)
        br = radial(bound)
        ax.fill_between(
            np.r_[x, x[0] + 2 * np.pi],
            0,
            np.r_[sr, sr[0]],
            color=pale(color),
            linewidth=0,
            zorder=2 if policy == "elote" else 3,
        )
        ax.fill_between(
            np.r_[x, x[0] + 2 * np.pi],
            np.r_[sr, sr[0]],
            np.r_[br, br[0]],
            color=color,
            alpha=0.09,
            linewidth=0,
            zorder=4,
        )
        ax.plot(
            np.r_[x, x[0] + 2 * np.pi],
            np.r_[sr, sr[0]],
            color=color,
            lw=2.8 if policy == "delta_elote" else 2.5,
            zorder=5,
        )
        ax.plot(
            np.r_[x, x[0] + 2 * np.pi],
            np.r_[br, br[0]],
            color=color,
            lw=2.1,
            ls=(0, (5, 3)),
            alpha=0.9,
            zorder=6,
        )
        ax.scatter(x, sr, s=36, color=color, edgecolor="white", lw=0.5, zorder=7)
        ax.scatter(x, br, s=75, facecolor="white", edgecolor=color, lw=2.0, zorder=8)

    ax.text(
        -0.10,
        0.12,
        "Energy error [mHa]\n(log scale)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=24,
        clip_on=False,
        zorder=30,
        bbox=RADIAL_TICK_BBOX,
    )

    # A short scale in the open sector between the first two reactions stays
    # legible even when the policy fills and radial bands cross other rings.
    scale_angle = np.pi / 10
    for value, label in ((0.01, r"$0.01$"), (0.1, r"$0.1$"), (1.0, r"$1$")):
        ax.text(
            scale_angle,
            float(radial(value)),
            label,
            ha="center",
            va="center",
            fontsize=26,
            zorder=30,
            bbox=RADIAL_TICK_BBOX,
        )
    fig.text(0.07, 0.89, "(b)", ha="left", va="top", fontsize=31)

    legend = [
        (
            Line2D([0], [0], color=GREEN, lw=2.6),
            Line2D([0], [0], color=GREEN, lw=2.1, ls=(0, (5, 3))),
        ),
        (
            Line2D([0], [0], color=PURPLE, lw=2.8),
            Line2D([0], [0], color=PURPLE, lw=2.1, ls=(0, (5, 3))),
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#474747",
            markeredgecolor="white",
            markersize=9,
            label=r"$\left|\sum_i\nu_i\delta_i\right|$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#474747",
            markeredgewidth=2.1,
            markersize=10,
            label=r"$\sum_i\left|\nu_i\delta_i\right|$",
        ),
        Line2D([0], [0], color=RED, lw=2, ls="--", label=r"$\epsilon_{\mathrm{chem}}$"),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"CreationDate": None})
    print(f"wrote {OUTPUT}")
    plt.close(fig)
    legend_fig = plt.figure(figsize=(16, 1.2))
    legend_fig.legend(
        handles=legend,
        labels=[
            "ELOTE",
            r"$\delta$-ELOTE",
            r"$\left|\sum_i\nu_i\delta E_i\right|$",
            r"$\sum_i\left|\nu_i\delta E_i\right|$",
            r"$\epsilon_{\mathrm{chem}}$",
        ],
        handler_map={tuple: StackedLinesHandler()},
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=5,
        frameon=False,
        fontsize=26,
        columnspacing=1.0,
        handlelength=1.5,
        handleheight=1.5,
        handletextpad=0.5,
    )
    legend_fig.savefig(LEGEND_OUTPUT, metadata={"CreationDate": None})
    print(f"wrote {LEGEND_OUTPUT}")
    plt.close(legend_fig)


if __name__ == "__main__":
    main()

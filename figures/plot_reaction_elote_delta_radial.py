"""Rebuild Figure 4 from the checked-in ten-reaction T95 summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
import numpy as np


from _paths import data_path, output_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = data_path("data/reaction_elote_delta_a100_summary.json")
OUTPUT = output_path("reaction_elote_delta_a100_radial_log.pdf")
KEY_OUTPUT = output_path(".build/figure4_reaction_key.tex")
GREEN = "#76B900"
PURPLE = "#6541A5"
RADIAL_TICK_BBOX = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.8,
    "pad": 2.5,
}


LABELS = {
    "hcn_hnc": r"$\mathrm{HCN}\!\to\!\mathrm{HNC}$",
    "hco_hoc_cation": r"$\mathrm{HCO}^{+}\!\to\!\mathrm{HOC}^{+}$",
    "hno_noh": r"$\mathrm{HNO}\!\to\!\mathrm{NOH}$",
    "h2co_hcoh": r"$\mathrm{H}_{2}\mathrm{CO}\!\to\!\mathrm{HCOH}$",
    "acetylene_vinylidene": r"$\mathrm{HCCH}\!\to\!\mathrm{H}_{2}\mathrm{CC}$",
    "h2_cl2_hcl": r"$\mathrm{H}_{2}+\mathrm{Cl}_{2}\!\to\!2\mathrm{HCl}$",
    "n2_h2_nh3": r"$\mathrm{N}_{2}+3\mathrm{H}_{2}\!\to\!2\mathrm{NH}_{3}$",
    "co_h2o_co2_h2": (
        r"$\mathrm{CO}+\mathrm{H}_{2}\mathrm{O}\!\to\!"
        r"\mathrm{CO}_{2}+\mathrm{H}_{2}$"
    ),
    "h2_f2_hf": r"$\mathrm{H}_{2}+\mathrm{F}_{2}\!\to\!2\mathrm{HF}$",
    "co_h2_ch4_h2o": (
        r"$\mathrm{CO}+3\mathrm{H}_{2}\!\to\!"
        r"\mathrm{CH}_{4}+\mathrm{H}_{2}\mathrm{O}$"
    ),
}


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = set(LABELS)
    if set(data["order"]) != cases or len(data["order"]) != len(cases):
        raise ValueError("Figure 4 reaction membership differs from the recorded set")
    if data["schema"] != "reaction-elote-delta-cudaq-a100-figure4-v1":
        raise ValueError("Figure 4 requires the completed CUDA-Q A100 snapshot")
    for field in ("baseline_t95", "elote_t95"):
        if set(data[field]) != cases or any(data[field][case] <= 0 for case in cases):
            raise ValueError(f"Missing or nonpositive T95 values in {field}")
    if (
        "CUDA-Q Version 0.16.0" not in data["compiler"]
        or data["eigensolver"] != "cupy-lanczos"
        or data["evaluation_contract_version"]
        != "normalized_hamiltonian_ground_input_signed_difference_cudaq_elote_v1"
    ):
        raise ValueError("Figure 4 has mismatched compiler or evaluation provenance")
    ranked = data["ranked_policies"]
    if len(ranked) != 5 or len({candidate["seed"] for candidate in ranked}) != 5:
        raise ValueError("Figure 4 requires all five independent A100 seeds")
    if [candidate["seed"] for candidate in ranked[:3]] != ["seed-05", "seed-01", "seed-04"]:
        raise ValueError("The top-three A100 seed ranking changed")
    for candidate in ranked:
        if (
            set(candidate["t95"]) != cases
            or any(candidate["t95"][case] <= 0 for case in cases)
            or candidate["max_reaction_error_mha"] >= candidate["chemical_accuracy_mha"]
        ):
            raise ValueError("A policy lacks complete, accurate reaction scores")
        score = math.exp(
            sum(math.log(candidate["t95"][case] / data["elote_t95"][case]) for case in cases)
            / len(cases)
        )
        if not math.isclose(score, candidate["score_relative_to_transferred_elote"], rel_tol=1e-12):
            raise ValueError("A policy score does not match its reaction costs")
    if any(
        a["score_relative_to_transferred_elote"] > b["score_relative_to_transferred_elote"]
        for a, b in zip(ranked, ranked[1:])
    ):
        raise ValueError("The five policies are not ranked by the recorded objective")

    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.titlesize": 19,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    widths = data["reaction_max_qubits"]
    ordered = sorted(data["order"], key=lambda case: (widths[case], data["baseline_t95"][case]))
    order = ordered[::2] + ordered[::-2]
    key_rows = [r"\begin{tabular}{@{}r@{\;}l@{}}"]
    for index, case in enumerate(order, start=1):
        key_rows.append(rf"{index}. & {LABELS[case]} \\")
    key_rows.append(r"\end{tabular}")
    KEY_OUTPUT.write_text("\n".join(key_rows) + "\n", encoding="utf-8")
    theta = np.linspace(0.0, 2.0 * np.pi, len(order), endpoint=False)
    closed = np.r_[theta, theta[0]]
    baseline = np.array([data["baseline_t95"][case] for case in order], dtype=float)
    elote = np.array([data["elote_t95"][case] for case in order]) / baseline
    delta = np.array([ranked[0]["t95"][case] for case in order]) / baseline
    if max(elote.max(), delta.max()) >= 0.3:
        raise ValueError("A policy lies beyond the displayed radial range")
    if min(elote.min(), delta.min()) <= 0.02:
        raise ValueError("A policy lies below the displayed radial range")

    fig, ax = plt.subplots(figsize=(9.5, 8.4), subplot_kw={"projection": "polar"})
    ax.set_position([0.16, 0.14, 0.68, 0.72])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 0.3)
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    ticks = [0.03, 0.05, 0.1, 0.2, 0.3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
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

    inner = np.full_like(closed, 0.02)

    def pale(color: str) -> tuple[float, float, float]:
        return tuple(0.68 + 0.32 * channel for channel in to_rgb(color))

    ax.fill_between(closed, inner, np.r_[elote, elote[0]], color=pale(GREEN), linewidth=0, zorder=1)
    ax.fill_between(
        closed, inner, np.r_[delta, delta[0]], color=pale(PURPLE), linewidth=0, zorder=2
    )
    ax.plot(closed, np.r_[elote, elote[0]], color=GREEN, linewidth=2.5, zorder=4, clip_on=False)
    ax.plot(closed, np.r_[delta, delta[0]], color=PURPLE, linewidth=3.0, zorder=5, clip_on=False)
    ax.text(
        -0.10,
        0.12,
        r"$T_{95}/T_{95}^{(0)}$" + "\n(log scale)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=24,
        clip_on=False,
        zorder=30,
        bbox=RADIAL_TICK_BBOX,
    )
    for tick in (0.05, 0.1, 0.2, 0.3):
        ax.text(
            np.pi / 10,
            tick,
            rf"${tick:g}$",
            ha="center",
            va="center",
            fontsize=26,
            zorder=30,
            bbox=RADIAL_TICK_BBOX,
        )
    fig.text(0.07, 0.89, "(a)", ha="left", va="top", fontsize=31)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot accepted shared T95 trajectories, averaging molecules and independent seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STYLE_VERSION = "relative-t95-five-seeds-latex-v1"


def aggregate(snapshot: dict) -> dict:
    seeds = snapshot["seeds"]
    if len(seeds) != 5 or len({s["seed"] for s in seeds}) != 5:
        raise ValueError("Five distinct independent seeds are required")
    if any(s.get("trajectory", {}).get("status") != "ready" for s in seeds):
        raise ValueError("Every seed needs a verified accepted-policy trajectory")
    count = int(snapshot["case_count"])
    case_ids = [r["case_id"] for r in seeds[0]["trajectory"]["molecules"]]
    if len(case_ids) != count or len(set(case_ids)) != count:
        raise ValueError("The molecule cohort is incomplete")
    common = min(s["trajectory"]["iterations"][-1] for s in seeds)
    grid = list(range(common + 1))
    values = []
    for seed in seeds:
        t = seed["trajectory"]
        if t["iterations"][:common + 1] != grid:
            raise ValueError("The seed iteration grids differ")
        rows = {r["case_id"]: r["drop_percent"] for r in t["molecules"]}
        if len(t["molecules"]) != count or set(rows) != set(case_ids):
            raise ValueError("All seeds must cover the same molecules")
        if any(len(rows[name]) != len(t["iterations"]) for name in case_ids):
            raise ValueError("A molecule trajectory is incomplete")
        values.append([[1 - drop / 100 for drop in rows[name][:common + 1]]
                       for name in case_ids])
    matrix = np.asarray(values, dtype=float)  # seed, molecule, iteration
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise ValueError("Invalid relative T95 values")
    if not np.allclose(matrix[:, :, 0], 1, rtol=0, atol=1e-12):
        raise ValueError("Every iteration-zero reference must equal one")
    seed_means = matrix.mean(axis=1)
    result = {
        "schema": "hsopt-shared-figure-statistics-v1",
        "style_version": STYLE_VERSION, "common_round": common,
        "iterations": grid, "seed_ids": [s["seed"] for s in seeds], "case_ids": case_ids,
        "seed_completed_rounds": [s["trajectory"]["iterations"][-1] for s in seeds],
        "molecule_mean_relative_t95": matrix.mean(axis=0).tolist(),
        "molecule_std_relative_t95": matrix.std(axis=0, ddof=1).tolist(),
        "seed_mean_relative_t95": seed_means.tolist(),
        "mean_relative_t95": seed_means.mean(axis=0).tolist(),
        "std_relative_t95": seed_means.std(axis=0, ddof=1).tolist(),
        "uncertainty": "Sample standard deviation across five independent seed means (ddof=1); not a confidence interval.",
        "semantics": "Accepted shared incumbents; matched rounds only; no extrapolation of incomplete seeds.",
    }
    identity = {k: v for k, v in result.items() if k != "seed_completed_rounds"}
    result["figure_key"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return result


def plot(summary: dict, output: Path) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 20, "axes.labelsize": 22, "axes.titlesize": 22,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 13,
        "lines.linewidth": 2.2, "axes.linewidth": 1.3,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.major.size": 5.5, "ytick.major.size": 5.5,
        "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": .03,
    })
    output.mkdir(parents=True, exist_ok=True)
    x = np.asarray(summary["iterations"])
    molecule = np.asarray(summary["molecule_mean_relative_t95"])
    molecule_std = np.asarray(summary["molecule_std_relative_t95"])
    mean, std = np.asarray(summary["mean_relative_t95"]), np.asarray(summary["std_relative_t95"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    for index, values in enumerate(molecule):
        ax.step(x, values, where="post", color="0.60", alpha=.36, linewidth=.7,
                label=r"molecule means" if index == 0 else None, zorder=2)
    ax.fill_between(x, mean-std, mean+std, step="post", color="#B7E36A", alpha=.38,
                    linewidth=0, label=r"$\pm 1\sigma$ (seeds)", zorder=1)
    ax.step(x, mean, where="post", color="#3E7E00", linewidth=2.8,
            label=r"mean of seed means", zorder=4)
    ax.axhline(1, color="0.35", linestyle="--", linewidth=.9, alpha=.7, zorder=0)
    ax.set_xlim(0, max(1, int(x[-1])))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel(r"Autoresearch iteration")
    ax.set_ylabel(r"Relative $T_{95}/T_{95}^{(0)}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", handlelength=1.7)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"relative_t95_trajectories.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.0, 4.6), layout="constrained")
    indices = np.arange(1, len(molecule) + 1)
    ax.bar(indices, molecule[:, -1]-1, bottom=1, width=.8, color="0.62", linewidth=0)
    ax.errorbar(indices, molecule[:, -1], yerr=molecule_std[:, -1], fmt="none",
                ecolor="0.25", elinewidth=.65, capsize=1.2,
                label=r"mean $\pm 1\sigma$ (seeds)")
    ax.axhline(1, color="0.35", linestyle="--", linewidth=1)
    lower, upper = molecule[:, -1]-molecule_std[:, -1], molecule[:, -1]+molecule_std[:, -1]
    span = max(.03, max(1, float(upper.max()))-min(1, float(lower.min())))
    ax.set_ylim(min(1, float(lower.min()))-.12*span, max(1, float(upper.max()))+.12*span)
    ax.set_xlim(.3, len(molecule)+.7)
    ax.set_xticks(sorted(set([1, len(molecule)//3, 2*len(molecule)//3, len(molecule)])))
    ax.set_xlabel(r"Hamiltonian index")
    ax.set_ylabel(r"Relative $T_{95}/T_{95}^{(0)}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"relative_t95_molecules.{ext}")
    plt.close(fig)
    (output / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    summary = data if data.get("schema") == "hsopt-shared-figure-statistics-v1" else aggregate(data)
    plot(summary, args.output)
    print(json.dumps({"common_round": summary["common_round"], "figure_key": summary["figure_key"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot accepted shared T95 trajectories, averaging molecules and independent seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STYLE_VERSION = "relative-t95-calibrated-energy-budget-five-seeds-latex-v3"


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
    values, energy_values, thresholds, baseline_budgets = [], [], [], []
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
        errors = {r["case_id"]: r["energy_error_hartree"] for r in t["molecules"]}
        if any(len(errors[name]) != len(t["iterations"]) for name in case_ids):
            raise ValueError("A molecule energy-error trajectory is incomplete")
        energy_values.append([[1000 * abs(error) for error in errors[name][:common + 1]]
                              for name in case_ids])
        budgets = {r["case_id"]: r["baseline_simulation_budget_hartree"] for r in t["molecules"]}
        baseline_budgets.append([1000 * float(budgets[name]) for name in case_ids])
        thresholds.append(float(t["chemical_accuracy_hartree"]) * 1000)
    matrix = np.asarray(values, dtype=float)  # seed, molecule, iteration
    if not np.isfinite(matrix).all() or np.any(matrix <= 0):
        raise ValueError("Invalid relative T95 values")
    if not np.allclose(matrix[:, :, 0], 1, rtol=0, atol=1e-12):
        raise ValueError("Every iteration-zero reference must equal one")
    seed_means = matrix.mean(axis=1)
    energy_matrix = np.asarray(energy_values, dtype=float)
    if not np.isfinite(energy_matrix).all():
        raise ValueError("Invalid energy-error values")
    if not np.isfinite(thresholds).all() or thresholds[0] <= 0 or not np.allclose(
            thresholds, thresholds[0], rtol=0, atol=1e-9):
        raise ValueError("All seeds need the same chemical-accuracy threshold")
    seed_energy_means = energy_matrix.mean(axis=1)
    budget_matrix = np.asarray(baseline_budgets, dtype=float)
    if not np.isfinite(budget_matrix).all() or np.any(budget_matrix <= 0) or np.any(
            budget_matrix >= thresholds[0]):
        raise ValueError("Invalid baseline simulation budget")
    if not np.allclose(budget_matrix, budget_matrix[0], rtol=0, atol=1e-9):
        raise ValueError("All seeds need the same per-molecule calibration budgets")
    budgets = budget_matrix[0]
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
        "molecule_mean_energy_error_mha": energy_matrix.mean(axis=0).tolist(),
        "molecule_std_energy_error_mha": energy_matrix.std(axis=0, ddof=1).tolist(),
        "seed_mean_energy_error_mha": seed_energy_means.tolist(),
        "mean_energy_error_mha": seed_energy_means.mean(axis=0).tolist(),
        "std_energy_error_mha": seed_energy_means.std(axis=0, ddof=1).tolist(),
        "chemical_accuracy_mha": thresholds[0],
        "molecule_baseline_simulation_budget_mha": budgets.tolist(),
        "mean_baseline_simulation_budget_mha": float(budgets.mean()),
        "min_baseline_simulation_budget_mha": float(budgets.min()),
        "max_baseline_simulation_budget_mha": float(budgets.max()),
        "energy_budget_semantics": "Fixed baseline calibration: chemical accuracy minus its QPE precision allocation. Reference only; current T95 adapts QPE precision to measured simulation error. The horizontal line is the molecule mean; the band is the molecule range, not seed uncertainty.",
        "energy_semantics": "Absolute molecular energy error before averaging; accepted shared policies including iteration zero.",
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
    if "molecule_mean_energy_error_mha" in summary:
        errors = np.asarray(summary["molecule_mean_energy_error_mha"])
        mean_error = np.asarray(summary["mean_energy_error_mha"])
        std_error = np.asarray(summary["std_energy_error_mha"])
        calibrated_budget = "mean_baseline_simulation_budget_mha" in summary
        budget = float(summary["mean_baseline_simulation_budget_mha"] if calibrated_budget
                       else summary["chemical_accuracy_mha"])
        budget_min = float(summary["min_baseline_simulation_budget_mha"]) if calibrated_budget else budget
        budget_max = float(summary["max_baseline_simulation_budget_mha"]) if calibrated_budget else budget
        fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
        for index, values in enumerate(errors):
            ax.step(x, values, where="post", color="0.60", alpha=.36, linewidth=.7,
                    label=r"molecule means" if index == 0 else None, zorder=2)
        ax.fill_between(x, mean_error-std_error, mean_error+std_error, step="post",
                        color="#B7E36A", alpha=.38, linewidth=0,
                        label=r"$\pm 1\sigma$ (seeds)", zorder=1)
        ax.step(x, mean_error, where="post", color="#3E7E00", linewidth=2.8,
                label=r"mean of seed means", zorder=4)
        if calibrated_budget:
            ax.axhspan(budget_min, budget_max, color="#C00000", alpha=.10, linewidth=0,
                       label=r"baseline budget range", zorder=0)
        ax.axhline(budget, color="#C00000", linewidth=1.8,
                   label=(r"$\overline{\epsilon_{\rm chem}-\epsilon_{\rm QPE}^{(0)}}$"
                          if calibrated_budget else r"chemical accuracy ($1$ kcal/mol)"), zorder=5)
        ax.set_xlim(0, max(1, int(x[-1])))
        ax.set_ylim(0, max(1.15*budget_max, 1.08*float(errors.max()),
                           1.08*float((mean_error+std_error).max())))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.set_xlabel(r"Autoresearch iteration")
        ax.set_ylabel(r"$|E_{\rm compiled}-E_{\rm exact}|$ (mHa)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", handlelength=1.7, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=1)
        for ext in ("png", "pdf"):
            fig.savefig(output / f"energy_error_trajectories.{ext}")
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

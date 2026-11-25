#!/usr/bin/env python3
"""Render a matched-round, selected-seed preview without changing a manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(snapshot, last):
    seeds = snapshot["seeds"]
    seed_ids = [seed["seed"] for seed in seeds]
    if len(seeds) < 2 or len(set(seed_ids)) != len(seeds):
        raise ValueError("At least two distinct seeds are required")
    case_ids = [m["case_id"] for m in seeds[0]["trajectory"]["molecules"]]
    if len(set(case_ids)) != len(case_ids) or len(case_ids) != snapshot["case_count"]:
        raise ValueError("Incomplete Hamiltonian cohort")
    ratios, errors, budgets, thresholds = [], [], [], []
    for seed in seeds:
        trajectory = seed["trajectory"]
        if trajectory["status"] != "ready" or trajectory["iterations"][: last + 1] != list(
            range(last + 1)
        ):
            raise ValueError("Every selected seed must have completed every requested round")
        molecules = {m["case_id"]: m for m in trajectory["molecules"]}
        if set(molecules) != set(case_ids) or len(trajectory["molecules"]) != len(case_ids):
            raise ValueError("Selected seeds must cover the same Hamiltonians")
        for molecule in molecules.values():
            if any(
                len(molecule[key]) != len(trajectory["iterations"])
                for key in ("drop_percent", "energy_error_hartree")
            ):
                raise ValueError("An incumbent trajectory is incomplete")
        ratios.append(
            [
                [1 - drop / 100 for drop in molecules[name]["drop_percent"][: last + 1]]
                for name in case_ids
            ]
        )
        errors.append(
            [
                [1000 * abs(error) for error in molecules[name]["energy_error_hartree"][: last + 1]]
                for name in case_ids
            ]
        )
        budgets.append(
            [1000 * molecules[name]["baseline_simulation_budget_hartree"] for name in case_ids]
        )
        thresholds.append(1000 * trajectory["chemical_accuracy_hartree"])
    ratios, errors, budgets = (
        np.asarray(values, dtype=float) for values in (ratios, errors, budgets)
    )
    if (
        not np.isfinite(ratios).all()
        or np.any(ratios <= 0)
        or not np.allclose(ratios[:, :, 0], 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid baseline-relative costs")
    if (
        not np.isfinite(thresholds).all()
        or thresholds[0] <= 0
        or not np.allclose(thresholds, thresholds[0], atol=1e-9, rtol=0)
    ):
        raise ValueError("Chemical-accuracy thresholds differ")
    if not np.isfinite(errors).all() or np.any(errors >= thresholds[0]):
        raise ValueError("An incumbent lacks a passing accuracy result")
    if (
        not np.isfinite(budgets).all()
        or np.any(budgets <= 0)
        or np.any(budgets >= thresholds[0])
        or not np.allclose(budgets, budgets[0], atol=1e-9, rtol=0)
    ):
        raise ValueError("Invalid or mismatched baseline calibration budgets")
    summary = {
        "schema": "hsopt-selected-seed-trajectory-preview-v1",
        "campaign_id": snapshot["campaign_id"],
        "energy_target": snapshot["energy_target"],
        "baseline": snapshot["baseline"],
        "seed_ids": seed_ids,
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "iterations": list(range(last + 1)),
        "seed_completed_rounds_at_snapshot": [s["completed_at_snapshot"] for s in seeds],
        "selection": snapshot["selection"],
        "chemical_accuracy_mha": thresholds[0],
        "mean_baseline_simulation_budget_mha": float(budgets.mean()),
        "min_baseline_simulation_budget_mha": float(budgets.min()),
        "max_baseline_simulation_budget_mha": float(budgets.max()),
        "semantics": "Retained shared incumbents, including the unchanged iteration-0 baseline; no rejected-trial scatter or extrapolation. Individual Hamiltonians can regress after an aggregate improvement.",
        "averaging": "Arithmetic mean of each Hamiltonian's own baseline-relative T95 across selected seeds, followed by an equal-weight Hamiltonian mean; this display is distinct from the geometric-mean acceptance objective.",
        "shading": "Observed minimum-to-maximum of the selected seed-level Hamiltonian means at each round; not a confidence interval or standard deviation.",
        "energy_semantics": "Absolute compiled-unitary energy error before averaging, in mHa. The baseline simulation budget is a fixed calibration reference; candidate QPE precision adapts to measured error.",
    }
    for metric, values in (("relative_t95", ratios), ("energy_error_mha", errors)):
        seed_means = values.mean(axis=1)
        summary["molecule_mean_" + metric] = values.mean(axis=0).tolist()
        summary["seed_mean_" + metric] = seed_means.tolist()
        summary["mean_" + metric] = seed_means.mean(axis=0).tolist()
        summary["min_" + metric] = seed_means.min(axis=0).tolist()
        summary["max_" + metric] = seed_means.max(axis=0).tolist()
        summary["std_" + metric] = seed_means.std(axis=0, ddof=1).tolist()
    return summary


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _paper_plot import apply_paper_style

    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), layout="constrained")
    # Reserve a separate header so the raised legend clears both panel labels.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))
    x = np.asarray(summary["iterations"])
    legend_handles = None
    for index, (ax, metric) in enumerate(zip(axes, ("relative_t95", "energy_error_mha"))):
        for i, values in enumerate(summary["molecule_mean_" + metric]):
            (gray,) = ax.step(
                x,
                values,
                where="post",
                color="0.60",
                alpha=0.40,
                linewidth=0.65,
                label="Hamiltonians" if i == 0 else None,
                zorder=2,
            )
        band = ax.fill_between(
            x,
            summary["min_" + metric],
            summary["max_" + metric],
            step="post",
            color="#B7E36A",
            alpha=0.40,
            linewidth=0,
            label="Seed range",
            zorder=1,
        )
        (mean,) = ax.step(
            x,
            summary["mean_" + metric],
            where="post",
            color="#3E7E00",
            linewidth=2.8,
            label="Mean",
            zorder=4,
        )
        if index == 0:
            legend_handles = [mean, band, gray]
        ax.set_xlim(0, x[-1])
        ax.set_xticks(np.arange(0, x[-1] + 1, 10))
        ax.set_xlabel("Autoresearch iteration")
        ax.grid(False, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.12, 1.02, f"({chr(97 + index)})", transform=ax.transAxes, fontsize=25, va="bottom"
        )
    axes[0].axhline(1, color="0.45", linestyle="--", linewidth=1.2, zorder=0)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel(r"Relative $T_{95}/T_{95}^{(0)}$")
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    error_ax = axes[1]
    chemical = summary["chemical_accuracy_mha"]
    error_ax.axhline(chemical, color="#8C2727", linestyle="--", linewidth=1.3, zorder=5)
    error_ax.annotate(
        "Chemical accuracy",
        xy=(0.98, chemical),
        xycoords=error_ax.get_yaxis_transform(),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=16,
        color="#8C2727",
    )
    error_ax.set_yscale("symlog", linthresh=0.01)
    error_ax.set_ylim(0, chemical * 1.45)
    error_ax.set_ylabel(r"$|\widehat E-E|$ (mHa)")
    fig.legend(
        legend_handles,
        ["Mean", "Seed range", "Hamiltonians"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=2.0,
        columnspacing=2.0,
    )
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output / ("elote_trajectories_four_seeds_60." + ext), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-iteration", type=int, default=60)
    args = parser.parse_args()
    if args.last_iteration < 1:
        parser.error("--last-iteration must be positive")
    raw = args.input.read_bytes()
    summary = summarize(json.loads(raw), args.last_iteration)
    summary["input_sha256"] = hashlib.sha256(raw).hexdigest()
    summary["plot_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plot(summary, args.output)
    summary["output_sha256"] = {
        ext: hashlib.sha256(
            (args.output / ("elote_trajectories_four_seeds_60." + ext)).read_bytes()
        ).hexdigest()
        for ext in ("png", "pdf")
    }
    (args.output / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "seeds": summary["seed_ids"],
                "last_iteration": args.last_iteration,
                "hamiltonians": summary["case_count"],
                "final_mean_relative_t95": summary["mean_relative_t95"][-1],
                "final_mean_energy_error_mha": summary["mean_energy_error_mha"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

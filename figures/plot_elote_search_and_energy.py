"""Reproduce Figure 3: shared search progress and selected-policy energy errors.

Run from any directory with NumPy, Matplotlib, and a LaTeX installation.
"""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import numpy as np

from _paths import MANIFEST, data_path, output_path
from audit import read_campaign

ROOT = Path(__file__).resolve().parents[1]
DATA = data_path("data/elote115_search_and_energy.json")
HELDOUT = data_path("data/elote_heldout_summary.json")
OUTPUT = output_path("elote115_search_and_energy.pdf")


def apply_paper_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "figure.titlesize": 22,
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
            "hatch.linewidth": 4.0,
        }
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_data(source):
    """Replay retained policies and complete proposal costs from frozen records."""
    published = source["trajectory"]
    campaign_path = ROOT / json.loads(MANIFEST.read_text())["scientific_trajectories"]["molecular"]
    _, header, grouped = read_campaign(campaign_path)
    seeds = published["seed_ids"]
    cases = published["case_ids"]
    iterations = np.asarray(published["iterations"])
    assert len(seeds) == 5 and len(cases) == header["case_count"] == 115
    assert np.array_equal(iterations, np.arange(101))

    trajectories, policies, proposals = [], [], []
    plot_ceiling = 1.3
    for seed in seeds:
        rows = grouped[seed]
        baseline = {m["case_id"]: float(m["t95"]) for m in rows[0]["result"]["case_metrics"]}
        assert set(baseline) == set(cases) and all(v > 0 for v in baseline.values())
        incumbent = None
        history, policy_history = [], []
        for row in rows:
            if row["iteration"] > 0:
                trial = {m["case_id"]: m for m in row["result"]["case_metrics"]}
                assert set(trial) == set(cases)
                measured = [float(trial[c]["t95"]) / baseline[c] for c in cases
                            if trial[c].get("t95") is not None and np.isfinite(float(trial[c]["t95"]))]
                complete = len(measured) == len(cases)
                mean_ratio = float(np.mean(measured)) if complete else None
                # Population SD over the complete finite benchmark, after
                # normalizing each Hamiltonian by its own baseline cost.
                sd_ratio = float(np.std(measured, ddof=0)) if complete else None
                if complete:
                    np.testing.assert_allclose(mean_ratio, row["result"]["mean_relative_t95"], rtol=1e-12, atol=1e-12)
                    assert bool(row["result"]["passes_chemical_accuracy"]) == all(
                        m["passes_chemical_accuracy"] for m in trial.values())
                proposals.append({"seed": seed, "iteration": row["iteration"],
                                  "candidate_sha256": row["candidate_sha256"],
                                  "accepted": row["decision"]["accepted"],
                                  "passes_chemical_accuracy": row["result"]["passes_chemical_accuracy"],
                                  "decision_reason": row["decision"]["acceptance_reason"],
                                  "recorded_costs": len(measured), "complete": complete,
                                  "mean_relative_t95": mean_ratio,
                                  "hamiltonian_population_sd": sd_ratio,
                                  "lower_sd": mean_ratio - sd_ratio if complete else None,
                                  "upper_sd": mean_ratio + sd_ratio if complete else None,
                                  "mean_reduction_percent": 100 * (1 - mean_ratio) if complete else None,
                                  "above_plot_range": bool(complete and mean_ratio > plot_ceiling)})
            if row["decision"]["accepted"]:
                incumbent = row
            assert incumbent is not None and incumbent["result"]["passes_chemical_accuracy"]
            metrics = {m["case_id"]: m for m in incumbent["result"]["case_metrics"]}
            assert set(metrics) == set(cases)
            assert all(m["passes_chemical_accuracy"] for m in metrics.values())
            history.append([float(metrics[c]["t95"]) / baseline[c] for c in cases])
            policy_history.append({"iteration": incumbent["iteration"], "sha256": incumbent["candidate_sha256"]})
        trajectories.append(history)
        policies.append(policy_history)
    ratios = np.asarray(trajectories)  # seed, iteration, Hamiltonian
    assert ratios.shape == (5, 101, 115) and np.all(np.isfinite(ratios))
    np.testing.assert_array_equal(ratios[:, 0], np.ones((5, 115)))
    seed_means = ratios.mean(axis=2)
    np.testing.assert_allclose(seed_means, published["seed_mean_relative_t95"], rtol=0, atol=5e-9)

    # Pick one whole shared protocol at each iteration, rather than a separate
    # minimum for each molecule. Keep the requested best-policy centre line.
    winners = np.argmin(seed_means, axis=0)
    selected = ratios[winners, iterations, :]
    best = selected.mean(axis=1)
    np.testing.assert_allclose(best, published["min_relative_t95"], rtol=0, atol=5e-9)
    assert policies[winners[-1]][-1]["sha256"] == source["provenance"]["policy_sha256"]
    assert len(proposals) == 500
    assert len({(p["seed"], p["iteration"]) for p in proposals}) == 500
    for proposal in proposals:
        proposal["from_leading_seed"] = proposal["seed"] == seeds[winners[proposal["iteration"]]]
    complete = [p for p in proposals if p["complete"]]
    visible = [p for p in complete if not p["above_plot_range"]]
    overflow = [p for p in complete if p["above_plot_range"]]
    assert len(complete) == 350 and len(overflow) == 6
    assert all(sum(p["iteration"] == k for p in complete) <= 5 for k in iterations)

    return iterations, seed_means, best, proposals, {
        "trajectory_archive_sha256": sha(campaign_path.with_name("trajectories.jsonl.gz")),
        "proposal_count": len(proposals),
        "complete_proposal_count": len(complete),
        "incomplete_proposals_omitted": len(proposals) - len(complete),
        "above_range_proposals_omitted": len(overflow),
        "displayed_proposals": len(visible),
        "displayed_accuracy_passing": sum(p["passes_chemical_accuracy"] for p in visible),
        "displayed_accuracy_failing": sum(not p["passes_chemical_accuracy"] for p in visible),
        "displayed_proposals_with_clipped_sd": sum(
            p["lower_sd"] < 0 or p["upper_sd"] > plot_ceiling for p in visible
        ),
        "best_seed_at_endpoint": seeds[winners[-1]],
        "best_arithmetic_mean_relative_t95_at_endpoint": float(best[-1]),
    }


def search(ax, iterations, seed_means, best, proposals):
    """Plot the approved best-policy view, with proposal accuracy markers."""
    plot_ceiling = 1.3
    visible = [p for p in proposals if p["complete"] and not p["above_plot_range"]]
    for leading, color in ((False, "#929292"), (True, "#76B900")):
        points = [p for p in visible if p["from_leading_seed"] == leading]
        ax.errorbar([p["iteration"] for p in points], [p["mean_relative_t95"] for p in points],
                    yerr=[p["hamiltonian_population_sd"] for p in points], fmt="none",
                    ecolor=color, alpha=0.20, elinewidth=0.7, capsize=1.1, capthick=0.7, zorder=1)
        for passes, marker in ((True, "^"), (False, "X")):
            subset = [p for p in points if p["passes_chemical_accuracy"] == passes]
            ax.scatter([p["iteration"] for p in subset], [p["mean_relative_t95"] for p in subset],
                       s=33 if passes else 38, marker=marker, color=color, alpha=0.40,
                       zorder=3, clip_on=False, edgecolors="none", linewidths=0)
    # Out-of-range and incomplete proposals remain in the frozen archive.
    for values in seed_means:
        ax.step(iterations, values, where="post", color="#929292", linewidth=1.65,
                alpha=0.90, zorder=2)
    ax.step(iterations, best, where="post", color="#3E7E00", linewidth=2.8, zorder=4)
    ax.axhline(1, color=".45", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, plot_ceiling)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 1.21, 0.2))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Relative $T_{95}/T_{95}^{(0)}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False, which="both")
    proposal_keys = {}
    for marker in ("^", "X"):
        proposal_keys[marker] = []
        for color in ("#929292", "#76B900"):
            # Invisible NaN points create capped legend handles without
            # plotting extra data; an empty input would omit the caps.
            key = ax.errorbar([np.nan], [np.nan], yerr=[0.1], fmt=marker, markersize=6,
                              color=color, alpha=0.65, elinewidth=0.9, capsize=1.1, capthick=0.7)
            key.lines[0].set_markeredgewidth(0)
            assert len(key.lines[1]) == 2
            proposal_keys[marker].append(key)
    handles = [Line2D([], [], color="#3E7E00", lw=2.8, label="Best policy"),
               Line2D([], [], color="#929292", lw=1.65, label="Independent seeds"),
               tuple(proposal_keys["^"]), tuple(proposal_keys["X"])]
    return handles


def main():
    source = json.loads(DATA.read_text())
    data = source["trajectory"]
    provenance = source["provenance"]
    rows = source["molecules"]
    assert data["case_count"] == 115
    heldout = []
    if HELDOUT.exists():
        audit = json.loads(HELDOUT.read_text())
        assert audit["schema"] == "elote-heldout-audit-v1"
        assert audit["policy_sha256"] == provenance["policy_sha256"]
        assert audit["total_count_cases"] == 78
        assert audit["count_statuses"] == {"ok": 76, "error": 2}
        heldout = [
            r for r in audit["results"] if r["energy_status"] in ("both_pass", "accuracy_failed")
        ]
        assert len(heldout) >= 4
        assert len({r["case_id"] for r in heldout}) == len(heldout)
        assert all(r["count_status"] == "ok" for r in heldout)
        if audit.get("complete_countable_energy_sweep"):
            assert len(heldout) == audit["count_statuses"]["ok"]
        assert all(int(r["n_qubits"]) > 16 for r in heldout)
    rows.sort(key=lambda r: (int(r["n_qubits"]), float(r["error_hartree"]), r["case_id"]))
    sizes = np.asarray([int(r["n_qubits"]) for r in rows])
    errors = np.asarray([1000 * abs(float(r["error_hartree"])) for r in rows])
    chemical = 1000 * provenance["chemical_accuracy_hartree"]
    assert len(rows) == 115 and np.all((errors > 0) & (errors < chemical))
    assert all(r["passes_chemical_accuracy"] == "True" for r in rows)
    initial = np.asarray([1000 * abs(float(r["baseline_error_hartree"])) for r in rows])
    assert np.all(np.isfinite(initial) & (initial > 0) & (initial < chemical))
    assert provenance["baseline_energy_iteration"] == 0
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
        }
    )

    def energy(ax):
        if heldout:
            # The 18--22-qubit checks are out of sample. Mark their region
            # behind the existing diamonds without hiding training points,
            # quartile boxes, or full-range whiskers.
            ax.axvspan(
                17,
                23.1,
                facecolor=(0.80, 0.80, 0.80, 0.08),
                edgecolor=(0.40, 0.40, 0.40, 0.10),
                hatch="/",
                linewidth=0,
                zorder=-5,
            )
        series = [
            ("Tight baseline", initial, "#D8D8D8", "#777777", "#666666", -0.38),
            ("ELOTE", errors, "#B7E36A", "#3E7E00", "#365A42", 0.38),
        ]
        for label, values, fill, edge, point, offset in series:
            for q in np.unique(sizes):
                e = values[sizes == q]
                position = q + offset
                width = 0.58
                if len(e) > 2:
                    ax.boxplot(
                        [e],
                        positions=[position],
                        widths=width,
                        whis=(0, 100),
                        showfliers=False,
                        patch_artist=True,
                        manage_ticks=False,
                        boxprops={
                            "facecolor": fill,
                            "edgecolor": edge,
                            "linewidth": 1.5,
                            "alpha": 0.60,
                        },
                        medianprops={"color": "#212121", "linewidth": 2.2},
                        whiskerprops={"color": edge, "linewidth": 1.4},
                        capprops={"color": edge, "linewidth": 1.4},
                        zorder=2,
                    )
                else:
                    # With only two Hamiltonians, show both points, their
                    # full range and median without a quartile box.
                    ax.vlines(position, e.min(), e.max(), color=edge, lw=1.4, zorder=2)
                    ax.hlines(
                        [e.min(), e.max()],
                        position - width * 0.25,
                        position + width * 0.25,
                        color=edge,
                        lw=1.4,
                        zorder=2,
                    )
                    ax.hlines(
                        np.median(e),
                        position - width * 0.5,
                        position + width * 0.5,
                        color="#212121",
                        lw=2.2,
                        zorder=4,
                    )
                x = position + np.random.default_rng(300 + int(q)).uniform(
                    width * 0.14, width * 0.41, len(e)
                )
                ax.scatter(x, e, s=18, marker="o", c=point, alpha=0.40, linewidths=0, zorder=3)
        if heldout:
            for q in sorted({int(row["n_qubits"]) for row in heldout}):
                cases = sorted(
                    (row for row in heldout if int(row["n_qubits"]) == q),
                    key=lambda row: row["case_id"],
                )
                offsets = np.linspace(-0.11, 0.11, len(cases)) if len(cases) > 1 else [0.0]
                for label, _, fill, edge, point, offset in series:
                    kind = "baseline" if label == "Tight baseline" else "elote"
                    values = [float(row[f"{kind}_error_mha"]) for row in cases]
                    position = q + offset
                    if len(values) >= 3:
                        ax.boxplot(
                            [values],
                            positions=[position],
                            widths=0.58,
                            whis=(0, 100),
                            showfliers=False,
                            patch_artist=True,
                            manage_ticks=False,
                            boxprops={
                                "facecolor": fill,
                                "edgecolor": edge,
                                "linewidth": 1.5,
                                "alpha": 0.60,
                            },
                            medianprops={"color": "#212121", "linewidth": 2.2},
                            whiskerprops={"color": edge, "linewidth": 1.4},
                            capprops={"color": edge, "linewidth": 1.4},
                            zorder=2,
                        )
                    for row, jitter in zip(cases, offsets, strict=True):
                        value = float(row[f"{kind}_error_mha"])
                        accurate = value < chemical
                        ax.scatter(
                            position + jitter,
                            value,
                            s=18,
                            marker="o",
                            c=point if accurate else "#9C2727",
                            alpha=0.40 if accurate else 0.90,
                            linewidths=0,
                            zorder=3 if accurate else 6,
                        )
            ax.axvline(17, color=".60", linestyle=":", linewidth=1.3, zorder=0)
        ax.axhline(chemical, color="#8C2727", ls="--", lw=1.4)
        ax.annotate(
            r"$\varepsilon_{\rm chem}$",
            xy=(0.02, chemical),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, 7),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color="#8C2727",
            fontsize=18,
        )
        ax.set_yscale("log")
        largest_heldout_error = max(
            (float(row[f"{kind}_error_mha"]) for row in heldout for kind in ("baseline", "elote")),
            default=0.0,
        )
        ax.set_ylim(1e-5, max(4.1, 1.35 * largest_heldout_error))
        ax.set_xlim(2.9, 23.1 if heldout else 17.1)
        if heldout:
            ax.text(
                10,
                0.045,
                "Training",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="#646464",
                fontsize=15,
            )
            ax.text(
                20.1,
                0.045,
                "Testing",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="#545454",
                fontsize=15,
            )
        ticks = np.unique(np.concatenate((sizes, [r["n_qubits"] for r in heldout])))
        ax.set_xticks(ticks, [rf"${n}$" for n in ticks])
        ax.set_xlabel("Qubits")
        ax.set_ylabel(r"$|E_{\mathcal P}-E|$ (mHa)")
        handles = [
            Patch(
                facecolor=fill,
                edgecolor=edge,
                alpha=0.60,
                label="Baseline" if heldout and label == "Tight baseline" else label,
            )
            for label, _, fill, edge, _, _ in series
        ]
        if not heldout:
            handles.append(Line2D([], [], color="#212121", lw=2.2, label="Median"))
        return handles

    assert len({r["case_id"] for r in rows}) == 115
    assert data["seed_ids"] == ["seed-01", "seed-02", "seed-03", "seed-04", "seed-05"]
    assert set(data["case_ids"]) == {r["case_id"] for r in rows}
    iterations = np.asarray(data["iterations"])
    last_iteration = int(iterations[-1])
    assert last_iteration > 0
    assert np.array_equal(iterations, np.arange(last_iteration + 1))
    molecule_means = np.asarray(data["molecule_mean_relative_t95"])
    seed_means = np.asarray(data["seed_mean_relative_t95"])
    assert molecule_means.shape == (115, last_iteration + 1)
    assert seed_means.shape == (5, last_iteration + 1)
    np.testing.assert_allclose(molecule_means.mean(axis=0), data["mean_relative_t95"])
    np.testing.assert_allclose(seed_means.mean(axis=0), data["mean_relative_t95"])
    np.testing.assert_allclose(seed_means.min(axis=0), data["min_relative_t95"])
    np.testing.assert_allclose(seed_means.max(axis=0), data["max_relative_t95"])
    iterations, seed_means, best, proposals, search_report = search_data(source)
    fig, axes = plt.subplots(
        1, 2, figsize=(15.8, 5.8), width_ratios=(1, 1), layout="constrained"
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    search_handles = search(axes[0], iterations, seed_means, best, proposals)
    energy_handles = energy(axes[1])
    # Separate one-row legends retain the Figure 2 printed font size. The
    # caption defines the SD bars, allowing compact Pass/Fail labels in (a).
    axes[0].legend(
        handles=search_handles,
        labels=["Best policy", "Independent seeds", "Pass", "Fail"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=4,
        fontsize=21.3, handlelength=1.0, columnspacing=0.45, handletextpad=0.3,
    )
    axes[1].legend(
        handles=energy_handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=len(energy_handles),
        fontsize=21.3, handlelength=1.15, columnspacing=0.8, handletextpad=0.4,
    )
    for i, ax in enumerate(axes):
        ax.text(-0.07, 1.02, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=25, va="bottom")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False, which="both")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    preview = ROOT / "build/fig3_search_and_energy.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=300)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "panel_a": search_report,
                "data_sha256": sha(DATA),
                "case_count": len(rows),
                "seed_ids": data["seed_ids"],
                "last_plotted_iteration": int(iterations[-1]),
                "mean_relative_t95_at_endpoint": data["mean_relative_t95"][-1],
                "policy_seed": provenance["policy_seed"],
                "policy_iteration": provenance["policy_iteration"],
                "mean_energy_error_mha": float(errors.mean()),
                "max_energy_error_mha": float(errors.max()),
                "mean_baseline_energy_error_mha": float(initial.mean()),
                "max_baseline_energy_error_mha": float(initial.max()),
                "chemical_accuracy_mha": chemical,
                "heldout_paired_cases": len(heldout),
                "heldout_energy_statuses": {
                    status: sum(r["energy_status"] == status for r in heldout)
                    for status in ("both_pass", "accuracy_failed")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

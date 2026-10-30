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
from matplotlib.patches import Patch
import numpy as np

from _paths import data_path, output_path

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
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            fontsize=15,
            columnspacing=1.0,
            handlelength=1.4,
        )

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
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    ax = axes[0]
    for values in data["molecule_mean_relative_t95"]:
        (individual,) = ax.step(
            iterations, values, where="post", color=".60", alpha=0.40, lw=0.65, zorder=2
        )
    span = ax.fill_between(
        iterations,
        data["min_relative_t95"],
        data["max_relative_t95"],
        step="post",
        color="#B7E36A",
        alpha=0.40,
        lw=0,
        zorder=3,
    )
    (mean,) = ax.step(
        iterations, data["mean_relative_t95"], where="post", color="#3E7E00", lw=2.8, zorder=5
    )
    ax.axhline(1, color=".45", ls="--", lw=1.2, zorder=0)
    ax.set_xlim(0, last_iteration)
    ax.set_xticks(np.arange(0, last_iteration + 1, 10))
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Relative $T_{95}/T_{95}^{(0)}$")
    ax.legend(
        [mean, span, individual],
        ["Mean", "Min--max range", "Hamiltonians"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=15,
        handlelength=1.6,
        columnspacing=1.3,
    )
    energy(axes[1])
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

#!/usr/bin/env python3
"""Compare a frozen learned tolerance shape with first-order inverse sensitivities."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from _paper_plot import GRAPHITE, NVIDIA_DARK_GREEN, apply_paper_style, molecule_label


AMBER = "#BB7C24"
DERIVED_BLUE = "#3977B8"
SENSITIVITY_CUTOFF = 1e-8


def geometric_mean(values, weights):
    return float(np.exp(np.average(np.log(values), weights=weights)))


def allocation_shapes(rows, cutoff=SENSITIVITY_CUTOFF):
    """Remove each molecule's scalar precision to compare allocation SHAPES.

    Tiny/zero first-order sensitivities have an unbounded interior optimum;
    omit them explicitly, rather than inventing a finite-error regularizer.
    Occurrences count as separate log costs even when their tolerances are tied.
    """
    selected = [dict(row) for row in rows if min(row["angle_weight"], row["su2_weight"]) > cutoff]
    for case in sorted({row["case_id"] for row in selected}):
        group = [row for row in selected if row["case_id"] == case]
        weights = np.array([row["occurrences"] for row in group])
        epsilon = np.array([row["epsilon"] for row in group])
        angle = np.array([row["angle_weight"] for row in group])
        su2 = np.array([row["su2_weight"] for row in group])
        gm_e, gm_a, gm_s = (geometric_mean(a, weights) for a in (epsilon, angle, su2))
        budgets = {"angle": float(np.sum(weights * angle * epsilon)),
                   "su2": float(np.sum(weights * su2 * epsilon))}
        # Shared per-term epsilon: epsilon*_j = B/(L * mean_occurrence_s_j).
        # These optional absolute predictions use the learned policy's own
        # linearized budget; they do NOT inherit the paper's error certificate.
        for row in group:
            row["learned_shape"] = row["epsilon"] / gm_e
            row["angle_shape"] = gm_a / row["angle_weight"]
            row["su2_shape"] = gm_s / row["su2_weight"]
            for field in ("angle", "su2"):
                row[f"{field}_local_budget_matched_epsilon"] = (
                    budgets[field] / weights.sum() / row[f"{field}_weight"]
                )
    return selected


def fit_shape(rows, field):
    """Descriptive, equal-molecule-weight fit; not a theoretical power law."""
    xs, ys, ws = [], [], []
    for case in sorted({row["case_id"] for row in rows}):
        group = [row for row in rows if row["case_id"] == case]
        w = np.array([row["occurrences"] for row in group], dtype=float)
        w /= w.sum()
        x = np.log([[row["relative_coefficient"], row["support_size"]] for row in group])
        y = np.log([row[field] for row in group])
        x -= np.average(x, weights=w, axis=0)
        y -= np.average(y, weights=w)
        xs.extend(x)
        ys.extend(y)
        ws.extend(w)
    x, y, w = np.array(xs), np.array(ys), np.array(ws)
    coefficients = np.linalg.lstsq(x * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    error = np.sum(w * (y - x @ coefficients) ** 2)
    total = np.sum(w * y**2)
    return {
        "coefficient_exponent": float(coefficients[0]),
        "support_exponent": float(coefficients[1]),
        "within_molecule_log_r_squared": float(1 - error / total),
        "interpretation": "descriptive fitted shape, equal molecule weight, in-sample",
    }


def summarize(rows, selected):
    result = {
        "terms_total": len(rows), "terms_plotted": len(selected),
        "terms_omitted_near_zero_sensitivity": len(rows) - len(selected),
        "dimensionless_sensitivity_cutoff": SENSITIVITY_CUTOFF,
        "normalization": "occurrence-weighted geometric mean within each molecule",
        "theoretical_model": "equal log costs, additive first-order absolute error, independent occurrences",
        "sensitivity_models": {},
    }
    for model in ("angle", "su2"):
        correlation = spearmanr(
            [r["learned_shape"] for r in selected], [r[f"{model}_shape"] for r in selected]
        ).statistic
        individual = []
        for case in {r["case_id"] for r in selected}:
            group = [r for r in selected if r["case_id"] == case]
            individual.append(spearmanr(
                [r["learned_shape"] for r in group], [r[f"{model}_shape"] for r in group]
            ).statistic)
        result["sensitivity_models"][model] = {
            "pooled_normalized_spearman": float(correlation),
            "median_within_molecule_spearman": float(np.nanmedian(individual)),
            "inverse_sensitivity_shape_fit": fit_shape(selected, f"{model}_shape"),
        }
    result["learned_shape_fit"] = fit_shape(selected, "learned_shape")
    result["cutoff_checks"] = {}
    for cutoff in (1e-6, 1e-8, 1e-10):
        group = allocation_shapes(rows, cutoff)
        result["cutoff_checks"][str(cutoff)] = {
            "retained": len(group),
            "angle_spearman": float(spearmanr(
                [r["learned_shape"] for r in group], [r["angle_shape"] for r in group]
            ).statistic),
            "su2_spearman": float(spearmanr(
                [r["learned_shape"] for r in group], [r["su2_shape"] for r in group]
            ).statistic),
        }
    return result


def binned_curve(ax, rows, field, *, color, label, linestyle="-", shade=False):
    x = np.array([row["relative_coefficient"] for row in rows])
    y = np.array([row[field] for row in rows])
    edges = np.geomspace(x.min() * 0.999, x.max() * 1.001, 16)
    centers, low, median, high = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        values = y[(x >= a) & (x < b)]
        if values.size < 8:
            continue
        centers.append(np.sqrt(a * b))
        lower, middle, upper = np.quantile(values, [0.25, 0.5, 0.75])
        low.append(lower)
        median.append(middle)
        high.append(upper)
    ax.plot(centers, median, color=color, linestyle=linestyle, label=label, linewidth=2.4)
    if shade:
        ax.fill_between(centers, low, high, color=color, alpha=0.10, linewidth=0)


def save(fig, output, name):
    for extension in ("pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)


def overlay_coordinates(rows, *, response_model="angle", normalization="shape"):
    """Use common physical baseline scales or separately centered shapes."""
    x = np.array([row["relative_coefficient"] for row in rows])
    if normalization == "baseline":
        baseline = np.array([row["baseline_epsilon"] for row in rows])
        return (x, np.array([row["derived_epsilon"] for row in rows]) / baseline,
                np.array([row["epsilon"] for row in rows]) / baseline)
    if normalization != "shape":
        raise ValueError("unknown overlay normalization")
    for case in sorted({row["case_id"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["case_id"] == case]
        coefficients = x[indices]
        weights = np.array([rows[i]["occurrences"] for i in indices])
        x[indices] = coefficients / geometric_mean(coefficients, weights)
    field = "derived_shape" if "derived_shape" in rows[0] else f"{response_model}_shape"
    return (x, np.array([row[field] for row in rows]),
            np.array([row["learned_shape"] for row in rows]))


def overlay_allocations(rows, output, *, response_model="angle", normalization="shape"):
    """Compare actual bounded allocations, or diagnostic normalized shapes."""
    x, derived, learned = overlay_coordinates(rows, response_model=response_model,
                                               normalization=normalization)
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(rows), size=min(len(rows), 12000), replace=False)
    fig, ax = plt.subplots(figsize=(10.8, 8.0), layout="constrained")
    for y, color, label in (
        (derived, DERIVED_BLUE, "Derived three-axis model" if response_model == "su2" else "Derived angle model"),
        (learned, NVIDIA_DARK_GREEN, "Agent"),
    ):
        ax.scatter(x[sampled], y[sampled], s=10, color=color,
                   alpha=.23, linewidths=0, rasterized=True, label=label)
    if normalization == "baseline":
        ax.axhline(1, color=GRAPHITE, linestyle="--", linewidth=1.4, alpha=.8,
                   label=r"Uniform baseline $\epsilon_0$")
    ax.set(xscale="log", yscale="log",
           xlabel=r"Relative coefficient $|c_j|/c_{\max}$" if normalization == "baseline"
           else r"Relative coefficient $|c_j|/\mathrm{GM}(|c|)$",
           ylabel="Relative synthesis error tolerance\n" + (r"$\epsilon_j/\epsilon_0$"
           if normalization == "baseline" else r"$\epsilon_j/\mathrm{GM}(\epsilon)$"))
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(30)
    ax.tick_params(axis="both", which="major", labelsize=26)
    legend = ax.legend(loc="lower left" if normalization == "baseline" else "upper right",
                       bbox_to_anchor=(.02, .08) if normalization == "baseline" else None,
                       markerscale=3, fontsize=24)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    ax.text(.03, .96, r"Higher $\epsilon$ = less precision", transform=ax.transAxes,
            fontsize=19, ha="left", va="top")
    ax.grid(which="major", alpha=.12, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, "sensitivity_overlay")


def figures(rows, summary, output, *, overlay_only=False, overlay_rows=None,
            response_model="angle", normalization="shape"):
    apply_paper_style()
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 23, "axes.titlesize": 22,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 17})
    overlay_allocations(overlay_rows if overlay_rows is not None else rows, output,
                        response_model=response_model, normalization=normalization)
    if overlay_only:
        return
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.8), layout="constrained")
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(rows), size=min(len(rows), 12000), replace=False)
    for off, color, label in ((0, GRAPHITE, r"$I/Z$"), (1, NVIDIA_DARK_GREEN, r"Contains $X/Y$")):
        dots = [rows[i] for i in sampled if rows[i]["off_diagonal"] == off]
        axes[0].scatter(
            [r["relative_coefficient"] for r in dots], [r["angle_weight"] for r in dots],
            s=7, color=color, alpha=0.25, linewidths=0, rasterized=True, label=label,
        )
    axes[0].set(xlabel=r"Relative coefficient $|c_j|/c_{\max}$",
                ylabel=r"Angle sensitivity $\Delta t\,s_j$", title="Ground-input approximation")
    axes[0].legend(loc="lower right", markerscale=3)
    for field, color, label, line in (
        ("learned_shape", NVIDIA_DARK_GREEN, "Learned", "-"),
        ("angle_shape", GRAPHITE, "Inverse angle response", "--"),
        ("su2_shape", AMBER, r"Inverse $\mathrm{SU}(2)$ response", ":"),
    ):
        binned_curve(axes[1], rows, field, color=color, label=label, linestyle=line, shade=True)
    axes[1].axhline(1, color="#B0B0B0", linewidth=1, zorder=0)
    axes[1].set(xlabel=r"Relative coefficient $|c_j|/c_{\max}$",
                ylabel=r"Normalized tolerance shape", title="Binned medians and quartiles")
    axes[1].legend(loc="lower left")
    dots = [rows[i] for i in sampled]
    axes[2].scatter(
        [r["angle_shape"] for r in dots], [r["learned_shape"] for r in dots],
        s=8, color=NVIDIA_DARK_GREEN, alpha=0.25, linewidths=0, rasterized=True,
    )
    xmin = min(r["angle_shape"] for r in rows)
    xmax = max(r["angle_shape"] for r in rows)
    axes[2].plot([xmin, xmax], [xmin, xmax], "--", color=GRAPHITE, linewidth=1.6)
    axes[2].set_ylim(min(r["learned_shape"] for r in rows) / 1.5,
                     max(r["learned_shape"] for r in rows) * 1.5)
    rho = summary["sensitivity_models"]["angle"]["pooled_normalized_spearman"]
    axes[2].text(0.04, 0.96, rf"$\rho_{{\mathrm{{S}}}}={rho:.2f}$", va="top", transform=axes[2].transAxes)
    axes[2].set(xlabel="Inverse-angle tolerance shape", ylabel="Learned tolerance shape",
                title="Trend comparison")
    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", alpha=0.12, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0, 1.04, rf"\textbf{{({chr(97 + i)})}}", transform=ax.transAxes,
                fontsize=21, va="bottom")
    save(fig, output, "sensitivity_comparison")

    representatives = (
        ("0265_", "H2"), ("0364_", "LiH"), ("0390_", "OH"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), layout="constrained")
    for ax, (prefix, molecule) in zip(axes, representatives):
        group = [r for r in rows if r["case_id"].startswith(prefix)]
        for field, color, label, marker in (
            ("angle_shape", GRAPHITE, "Inverse angle", "^"),
            ("su2_shape", AMBER, r"Inverse $\mathrm{SU}(2)$", "s"),
            ("learned_shape", NVIDIA_DARK_GREEN, "Learned", "o"),
        ):
            ax.scatter([r["relative_coefficient"] for r in group], [r[field] for r in group],
                       s=20, color=color, marker=marker, alpha=0.6, linewidths=0,
                       label=label, rasterized=True)
        ax.axhline(1, color="#B0B0B0", linewidth=1, zorder=0)
        ax.set(xscale="log", yscale="log", xlabel=r"$|c_j|/c_{\max}$",
               title=f"{molecule_label(molecule)}, {group[0]['qubits']:g} qubits")
        ax.grid(which="major", alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Normalized tolerance shape")
    axes[0].legend(loc="upper right")
    save(fig, output, "sensitivity_molecules")

    # Partial-regression view: remove the fitted support dependence and each
    # molecule intercept before showing the remaining coefficient trend.
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), layout="constrained")
    for ax, field, fit, color, title in (
        (axes[0], "angle_shape", summary["sensitivity_models"]["angle"]["inverse_sensitivity_shape_fit"],
         GRAPHITE, "Inverse angle response"),
        (axes[1], "learned_shape", summary["learned_shape_fit"], NVIDIA_DARK_GREEN, "Learned allocation"),
    ):
        coefficient_power, support_power = fit["coefficient_exponent"], fit["support_exponent"]
        xx, yy = [], []
        for case in sorted({row["case_id"] for row in rows}):
            group = [row for row in rows if row["case_id"] == case]
            weights = np.array([row["occurrences"] for row in group])
            coefficients = np.array([row["relative_coefficient"] for row in group])
            support = np.array([row["support_size"] for row in group])
            x = coefficients / geometric_mean(coefficients, weights)
            y = np.array([row[field] for row in group])
            y /= (support / geometric_mean(support, weights)) ** support_power
            xx.extend(x)
            yy.extend(y)
        ax.scatter(np.array(xx)[sampled], np.array(yy)[sampled], s=8, color=color,
                   alpha=0.2, linewidths=0, rasterized=True)
        grid = np.geomspace(min(xx), max(xx), 200)
        ax.plot(grid, grid ** coefficient_power, color=color, linewidth=2.4,
                label=rf"$a={coefficient_power:.2f},\ b={support_power:.2f}$")
        ax.text(.96, .94, rf"$R^2={fit['within_molecule_log_r_squared']:.3f}$",
                transform=ax.transAxes, va="top", ha="right")
        ax.set(xscale="log", yscale="log", title=title,
               xlabel=r"$|c_j|/\mathrm{GM}(|c|)$")
        ax.legend(loc="lower left")
        ax.grid(which="major", alpha=.12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Support-adjusted tolerance shape")
    save(fig, output, "sensitivity_power_law_fits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlay-only", action="store_true",
                        help="Only render the common-axis calculated/agent allocation overlay")
    parser.add_argument("--compilation-dir", type=Path,
                        help="Use the concrete bounded tolerances from compile_angle_allocation.py")
    parser.add_argument("--normalization", choices=("shape", "baseline"), default="shape",
                        help="shape: geometric means; baseline: c_max and sealed uniform epsilon_0")
    args = parser.parse_args()
    if args.normalization == "baseline" and not args.compilation_dir:
        parser.error("baseline normalization requires a concrete --compilation-dir")
    with gzip.open(args.data, "rt", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        rows.append({
            key: (value if key in {"case_id", "label"} else float(value))
            for key, value in row.items()
        })
    manifest = json.loads(args.manifest.read_text())
    if len(rows) != manifest["terms"] or len({r["case_id"] for r in rows}) != manifest["molecules"]:
        raise ValueError("data table and manifest disagree")
    selected = allocation_shapes(rows)
    summary = summarize(rows, selected)
    summary["policy_sha256"] = manifest["policy_sha256"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "allocation_comparison.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    overlay_rows, response_model = None, "angle"
    if args.compilation_dir:
        compilation = json.loads((args.compilation_dir / "synthesis_comparison.json").read_text())
        if compilation.get("status") != "complete" or compilation["policy_sha256"] != manifest["policy_sha256"]:
            raise ValueError("compilation comparison uses a different agent policy")
        response_model = compilation.get("response_model", "angle")
        if response_model not in {"angle", "su2"}:
            raise ValueError("unknown compiled response model")
        compiled_table = args.compilation_dir / "compiled_term_comparison.csv.gz"
        if hashlib.sha256(compiled_table.read_bytes()).hexdigest() != compilation["compiled_table_sha256"]:
            raise ValueError("compiled comparison table checksum mismatch")
        with gzip.open(compiled_table, "rt") as handle:
            overlay_rows = [
                {key: (value if key in {"case_id", "label"} else float(value))
                 for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        if len(overlay_rows) != len(rows) or {(r["case_id"], r["label"]) for r in overlay_rows} != {
            (r["case_id"], r["label"]) for r in rows
        }:
            raise ValueError("compiled comparison differs from the full Hamiltonian term cohort")
    summary["overlay"] = {
        "terms": len(overlay_rows) if overlay_rows is not None else len(selected),
        "derived_allocation": f"bounded, matched agent {response_model} budget" if overlay_rows is not None
        else "uncapped inverse angle response shape",
        "response_model": response_model,
        "normalization": args.normalization,
        "trend_curves": False,
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(selected, summary, output, overlay_only=args.overlay_only, overlay_rows=overlay_rows,
            response_model=response_model, normalization=args.normalization)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Explain the frozen Figure 2 ELOTE without another scientific evaluation.

Snapshot mode checks the seed-05/round-78 Figure 2 export against recorded
tolerances and the archived policy, then exports compact, self-contained plot
data.  Replay mode renders that compact export without reading the original
snapshot or a live campaign.  Observed allocations and conditional responses
are distinguished: this policy has no independent angle dependence.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import runpy
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from _paper_plot import GRAPHITE, NVIDIA_GREEN, apply_paper_style


PINNED_POLICY_SHA256 = "966d815a5dcbace3b9b5538ea450393a5207ebfaf228f99c2decbd4bae6e46d5"
BLUE = "#5B829A"
PLOT_DATA_SCHEMA = "elote-understanding-plot-data-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def factors(x, w, z, xy, W, c_ref, diagonal):
    """Exact dimensionless factors, before the policy's final epsilon clamp."""
    coefficient = 1 + 1.2 * x**0.3
    support = 1 + 0.12 * (w - 1) / W
    character = np.where(
        diagonal,
        (1 + 1.2 * x) * (1 + 0.18 * z / W * (1 + 0.2 * x)),
        1 + 0.3 * xy / W * (1 + 0.2 * x),
    )
    system = 1 + 0.12 * (c_ref / max(1, W))**0.4
    return coefficient, support, character, system


def verify(snapshot):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["seed"] == "seed-05" and manifest["iteration"] == 78
    assert manifest["policy_sha256"] == PINNED_POLICY_SHA256
    for name, digest in manifest["files"].items():
        assert sha(snapshot / name) == digest, name
    molecules = read_rows(snapshot / "molecules.csv")
    rotations = read_rows(snapshot / "rotations.csv.gz")
    assert len(molecules) == manifest["case_count"] == 115
    assert len(rotations) == manifest["unique_rotation_rows"]
    best = [row for row in rotations if row["stage"] == "best"]
    by_case = defaultdict(list)
    for row in best:
        by_case[row["case_id"]].append(row)
    policy_class = runpy.run_path(str(snapshot / "best_policy.py"))["ProductFormula"]
    max_factored_error = max_method_error = 0.0
    clamped_rows = 0
    summaries = []
    for case, rows in by_case.items():
        W = max(int(row["support"]) for row in rows)
        references = [abs(float(row["coefficient"])) /
                      float(row["coefficient_ratio_policy_reference"])
                      for row in rows if float(row["coefficient"]) != 0]
        c_ref = max(references)
        assert all(math.isclose(value, c_ref, rel_tol=1e-14) for value in references)
        epsilon0 = float(rows[0]["epsilon0"])
        formula = policy_class.__new__(policy_class)
        formula.max_abs_coeff, formula.max_support, formula.epsilon = c_ref, W, epsilon0
        for row in rows:
            c, w = float(row["coefficient"]), int(row["support"])
            label = row["label"]
            x = abs(c) / c_ref
            z, xy = label.count("Z"), label.count("X") + label.count("Y")
            values = factors(x, w, z, xy, W, c_ref, bool(int(row["diagonal"])))
            raw = epsilon0 * math.prod(float(value) for value in values)
            expected = min(1 - 1e-15, max(1e-15, raw))
            clamped_rows += expected != raw
            recorded = float(row["epsilon"])
            method = formula.synthesis_epsilon({"coefficient": c, "label": label},
                                              float(row["pauli_angle"]))
            max_factored_error = max(max_factored_error, abs(expected / recorded - 1))
            max_method_error = max(max_method_error, abs(method / recorded - 1))
            row["reference_c"] = c_ref
            row["reference_W"] = W
            row["x"] = x
        summaries.append({"case_id": case, "c_ref": c_ref, "W": W,
                          "cmax_over_cref": max(abs(float(r["coefficient"])) for r in rows) / c_ref,
                          "system_factor": 1 + 0.12 * (c_ref / W)**0.4})
    assert max_factored_error < 1e-12 and max_method_error < 1e-12
    assert clamped_rows == 0
    assert all(int(row["baseline_n_steps"]) == int(row["best_n_steps"]) for row in molecules)

    # A zero total coefficient in a host fermionic decomposition need not mean
    # its separately scheduled block contributions vanish individually.
    baseline_angles = defaultdict(list)
    best_angles = {}
    counts = {}
    for stage in ("baseline", "best"):
        rows = [row for row in rotations if row["stage"] == stage]
        occurrences = sum(int(row["multiplicity_per_step"]) for row in rows)
        zero = sum(int(row["multiplicity_per_step"]) for row in rows if float(row["pauli_angle"]) == 0)
        full = sum(int(row["multiplicity_per_step"]) * int(row["n_steps"]) for row in rows)
        full_zero = sum(int(row["multiplicity_per_step"]) * int(row["n_steps"])
                        for row in rows if float(row["pauli_angle"]) == 0)
        counts[stage] = {"distinct_label_angle_rows": len(rows),
                         "pooled_one_step_entries": occurrences,
                         "pooled_one_step_zero_angles": zero,
                         "pooled_one_step_nonzero_angles": occurrences - zero,
                         "full_evolution_entries": full,
                         "full_evolution_zero_angles": full_zero,
                         "full_evolution_nonzero_angles": full - full_zero}
        for row in rows:
            key = row["case_id"], row["label"]
            if stage == "baseline":
                baseline_angles[key].append(float(row["pauli_angle"]) * int(row["multiplicity_per_step"]))
            else:
                assert int(row["multiplicity_per_step"]) == 1 and key not in best_angles
                best_angles[key] = float(row["pauli_angle"])
    assert set(baseline_angles) == set(best_angles)
    angle_residual = max(abs(math.fsum(value) - best_angles[key])
                         for key, value in baseline_angles.items())
    assert angle_residual < 1e-14

    summary = {
        "schema": "elote-understanding-v1", "source_snapshot": str(snapshot),
        "policy_sha256": PINNED_POLICY_SHA256, "seed": "seed-05", "iteration": 78,
        "case_count": len(molecules), "checked_best_rotation_rows": len(best),
        "maximum_relative_factorization_error": max_factored_error,
        "maximum_relative_policy_method_error": max_method_error,
        "clamped_best_rows": clamped_rows,
        "formula": {
            "epsilon_j_over_epsilon0": "(1+1.2*x_j^0.3)*(1+0.12*(w_j-1)/W)*D_j*G_H",
            "x_j": "abs(c_j)/c_ref; c_ref is the largest absolute Pauli contribution in the verified fermionic blocks, before aggregation",
            "w_j": "number of nonidentity letters in P_j",
            "z_j": "number of Z letters in P_j",
            "q_j": "number of X or Y letters in P_j",
            "W": "maximum Pauli support in the verified block map; equal to observed maximum support for all 115 cases",
            "D_j_diagonal": "(1+1.2*x_j)*(1+0.18*z_j/W*(1+0.2*x_j))",
            "D_j_off_diagonal": "1+0.3*q_j/W*(1+0.2*x_j)",
            "G_H": "1+0.12*(c_ref/max(1,W))^0.4, with coefficients represented numerically in Hartree",
            "clamp": "epsilon_j=min(1-1e-15,max(1e-15,epsilon0*all_factors)); inactive for every recorded best row",
            "direct_angle_factor": 1,
        },
        "schedule": {
            "basis_change": "identity (empty Clifford frame)",
            "order": "flat lexicographic first-order Lie product formula, one signed coefficient times step_time per label",
            "groups_method": "defined but not called by build_step; does not create active grouping",
            "r_unchanged_all_cases": True,
            "r_range": [min(int(row["best_n_steps"]) for row in molecules),
                        max(int(row["best_n_steps"]) for row in molecules)],
            "same_case_label_set": True,
            "same_case_label_count": len(best_angles),
            "maximum_summed_signed_angle_difference_rad": angle_residual,
            "zero_angle_interpretation": "Zero total aggregated coefficients yield identity rotations; the policy does not threshold-prune nonzero coefficients. Zero T synthesis counts cannot be inferred from this export.",
            "counts": counts,
        },
        "epsilon_ratio_range": [min(float(row["epsilon_ratio"]) for row in best),
                                max(float(row["epsilon_ratio"]) for row in best)],
        "coefficient_reference_ratio_range": [min(row["cmax_over_cref"] for row in summaries),
                                              max(row["cmax_over_cref"] for row in summaries)],
        "source_files": manifest["files"], "per_case": summaries,
    }
    return best, summary


def conditional_arrays(coefficient_range, angle_range):
    """All theoretical line coordinates, separated from observed scatter data."""
    W, w = 16, 4
    xx = np.geomspace(max(1e-7, float(coefficient_range[0])), float(coefficient_range[1]), 300)
    ww = np.arange(1, 17)
    aa = np.geomspace(float(angle_range[0]), float(angle_range[1]), 200)
    result = {"coefficient_grid": xx, "weight_grid": ww,
              "angle_grid": aa, "angle_multiplier": np.ones_like(aa)}
    for diag, kind in ((False, "off_diagonal"), (True, "diagonal")):
        fs = factors(xx, w, w if diag else 0, 0 if diag else w, W, 1., diag)
        result[f"coefficient_response_{kind}"] = fs[0] * fs[1] * fs[2]
        fs = factors(.1, ww, ww if diag else 0, 0 if diag else ww, W, 1., diag)
        result[f"weight_response_{kind}"] = fs[0] * fs[1] * fs[2]
    return result


def compact_arrays(best):
    plotted = [row for row in best if float(row["pauli_angle"]) != 0]
    x = np.array([row["x"] for row in plotted], dtype=np.float64)
    angles = np.array([abs(float(row["rz_angle"])) for row in plotted], dtype=np.float64)
    arrays = {
        "scatter_x": x,
        "scatter_epsilon_ratio": np.array([float(row["epsilon_ratio"]) for row in plotted], dtype=np.float64),
        "scatter_diagonal": np.array([bool(int(row["diagonal"])) for row in plotted], dtype=np.bool_),
        "coefficient_range": np.array([x.min(), x.max()], dtype=np.float64),
        "angle_range": np.array([angles.min(), angles.max()], dtype=np.float64),
    }
    arrays.update(conditional_arrays(arrays["coefficient_range"], arrays["angle_range"]))
    return arrays


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_compact(path, arrays, summary):
    metadata = {
        "schema": PLOT_DATA_SCHEMA, "policy_sha256": PINNED_POLICY_SHA256,
        "source_scientific_verification": summary,
        "scatter_semantics": "One point per nonzero (case, Pauli label, angle); original ordering and duplicate coordinates retained. No sampling or multiplicity reweighting.",
        "arrays": {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                          "sha256": array_digest(value)} for name, value in arrays.items()},
    }
    np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **arrays)


def load_compact(path):
    """Check replay coordinates; do not claim to rerun archived science checks."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["schema"] == PLOT_DATA_SCHEMA
        assert metadata["policy_sha256"] == PINNED_POLICY_SHA256
        assert set(archive.files) == {"metadata_json", *metadata["arrays"]}
        arrays = {name: archive[name] for name in metadata["arrays"]}
    for name, value in arrays.items():
        expected = metadata["arrays"][name]
        assert value.dtype.str == expected["dtype"] and list(value.shape) == expected["shape"], name
        assert array_digest(value) == expected["sha256"], name
        assert np.isfinite(value).all(), name
    x, epsilon, diagonal = [arrays[name] for name in
                            ("scatter_x", "scatter_epsilon_ratio", "scatter_diagonal")]
    assert x.ndim == 1 and epsilon.shape == diagonal.shape == x.shape
    assert diagonal.dtype == np.bool_ and np.all(x > 0) and np.all(epsilon > 0)
    assert np.array_equal(arrays["coefficient_range"], [x.min(), x.max()])
    assert len(x) == metadata["source_scientific_verification"]["schedule"]["counts"]["best"]["pooled_one_step_nonzero_angles"]
    expected_curves = conditional_arrays(arrays["coefficient_range"], arrays["angle_range"])
    for name, value in expected_curves.items():
        assert np.array_equal(arrays[name], value), f"conditional formula changed: {name}"
    summary = metadata["source_scientific_verification"]
    assert summary["case_count"] == 115 and summary["policy_sha256"] == PINNED_POLICY_SHA256
    assert summary["seed"] == "seed-05" and summary["iteration"] == 78
    return arrays, summary


def draw(arrays, output):
    apply_paper_style()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 24, "axes.titlesize": 23,
                         "xtick.labelsize": 20, "ytick.labelsize": 20,
                         "legend.fontsize": 19})
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.3), layout="constrained")
    ratio = arrays["scatter_epsilon_ratio"]
    x = arrays["scatter_x"]
    diagonal = arrays["scatter_diagonal"]
    for mask, color, label in ((~diagonal, NVIDIA_GREEN, r"Contains $X/Y$"),
                               (diagonal, BLUE, r"$I/Z$ only")):
        axes[0, 0].scatter(x[mask], ratio[mask], color=color, alpha=0.25,
                           s=6, linewidths=0, rasterized=True, label=label)
    axes[0, 0].axhline(1, color=GRAPHITE, linestyle="--", linewidth=1.6)
    axes[0, 0].set(xscale="log", xlabel=r"Coefficient $x_j=|c_j|/c_{\rm ref}$",
                   ylabel=r"Tolerance $\epsilon_j/\epsilon_0$", title="Observed allocation")
    legend = axes[0, 0].legend(loc="upper left", markerscale=2.5)
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    # Full relative allocation with system factor divided out.  All remaining
    # feature values are fixed explicitly, making these conditional slices.
    xx = arrays["coefficient_grid"]
    for diag, color, label in ((False, NVIDIA_GREEN, r"$q=4$ ($X/Y$)"),
                               (True, BLUE, r"$z=4$ ($I/Z$)")):
        kind = "diagonal" if diag else "off_diagonal"
        axes[0, 1].plot(xx, arrays[f"coefficient_response_{kind}"], color=color, label=label)
    axes[0, 1].set(xscale="log", xlabel=r"Coefficient $x_j=|c_j|/c_{\rm ref}$",
                   ylabel=r"Tolerance $\epsilon_j/(\epsilon_0G_H)$", title=r"Fixed $w_j=4$, $W=16$")
    axes[0, 1].legend(loc="upper left")

    ww = arrays["weight_grid"]
    for diag, color, label in ((False, NVIDIA_GREEN, r"$q=w_j$ ($X/Y$)"),
                               (True, BLUE, r"$z=w_j$ ($I/Z$)")):
        kind = "diagonal" if diag else "off_diagonal"
        axes[1, 0].plot(ww, arrays[f"weight_response_{kind}"], color=color, label=label)
    axes[1, 0].set(xlabel=r"Pauli weight $w_j$", ylabel=r"Tolerance $\epsilon_j/(\epsilon_0G_H)$",
                   title=r"Fixed $x_j=0.1$, $W=16$")
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1, 0].legend(loc="upper left")

    axes[1, 1].plot(arrays["angle_grid"], arrays["angle_multiplier"], color=NVIDIA_GREEN)
    axes[1, 1].set(xscale="log", xlabel=r"Rotation angle $|\theta_j|$ (rad)",
                   ylabel=r"Independent angle multiplier", ylim=(.94, 1.11),
                   title="Fixed coefficient, weight, character")
    axes[1, 1].set_yticks([1.0], [r"$1$"])
    axes[1, 1].text(.5, .75, "No direct angle dependence", transform=axes[1, 1].transAxes,
                      ha="center", va="center", fontsize=22, color=GRAPHITE)
    for i, ax in enumerate(axes.flat):
        ax.grid(False, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-.13, 1.05, rf"({chr(97+i)})", transform=ax.transAxes, fontsize=26, va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"elote_policy_features.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="verify the full frozen scientific export and create compact plot data")
    source.add_argument("--replay", type=Path, help="regenerate from compact elote_policy_plot_data.npz without the original snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path, output = (args.snapshot or args.replay).resolve(), args.output.resolve()
    if output == input_path or output.is_relative_to(input_path):
        raise ValueError("output must not overwrite frozen input")
    if args.snapshot:
        best, scientific_summary = verify(input_path)
        arrays = compact_arrays(best)
    else:
        arrays, scientific_summary = load_compact(input_path)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "elote_policy_plot_data.npz"
    if data_path == input_path:
        raise ValueError("replay output must not overwrite its input data file")
    if args.snapshot:
        save_compact(data_path, arrays, scientific_summary)
    else:
        # Preserve the verified archive itself, including its ZIP metadata, so
        # a replay retains the same portable-data checksum as its source.
        data_path.write_bytes(input_path.read_bytes())
    # Both entry points must round-trip precisely through the public replay
    # format before drawing. This checks every plotted point and line coordinate.
    replay_arrays, _ = load_compact(data_path)
    assert set(replay_arrays) == set(arrays)
    assert all(np.array_equal(value, replay_arrays[name]) for name, value in arrays.items())
    draw(replay_arrays, output)
    summary = json.loads(json.dumps(scientific_summary))
    summary["verification_mode"] = "full_snapshot" if args.snapshot else "compact_plot_replay"
    summary["verification_scope"] = (
        "Rechecked all recorded allocations, archived policy method, and schedule sums; compact coordinates round-trip exactly."
        if args.snapshot else
        "Rechecked hashes, shapes, formula response curves and exact compact-coordinate round-trip. Full scientific checks are preserved from the original frozen-snapshot export, not rerun."
    )
    summary["plot_data"] = {"file": data_path.name, "sha256": sha(data_path),
                            "bytes": data_path.stat().st_size, "scatter_points": len(arrays["scatter_x"]),
                            "all_plot_arrays_round_trip_exactly": True}
    summary["script_sha256"] = sha(__file__)
    summary["output_files"] = {name: sha(output / name) for name in
                               ("elote_policy_features.pdf", "elote_policy_features.png")}
    (output / "policy_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "caption.tex").write_text(
        "Structure of the shared ELOTE used in Figure~2 (seed~5, mutation~78). "
        "(a) Recorded relative synthesis tolerances across all 115 Hamiltonians, "
        "excluding zero-angle identity rotations. The coefficient reference is "
        "the largest Pauli contribution in the verified fermionic block decomposition, "
        "before summing repeated labels. (b,c) Conditional responses of the exact "
        "learned rule, after dividing out the Hamiltonian-wide factor $G_H$. "
        "The fixed parameters are stated above each panel; the diagonal case uses "
        "$z_j=w_j$ and the off-diagonal case uses $q_j=w_j$. These slices isolate "
        "effects that are correlated in the observed data. (d) Holding these "
        "features fixed, the tolerance has no independent dependence on rotation "
        "angle. Its observed angle trend in Figure~2 arises from correlated "
        "Hamiltonian coefficients and step times. Larger $\\epsilon_j$ permits "
        "looser synthesis.\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_case"}, indent=2))


if __name__ == "__main__":
    main()

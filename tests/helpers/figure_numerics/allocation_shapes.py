"""Numerical validation extracted from archived plot_policy_sensitivity.py; no plotting frontend."""

from __future__ import annotations


import numpy as np


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
        budgets = {
            "angle": float(np.sum(weights * angle * epsilon)),
            "su2": float(np.sum(weights * su2 * epsilon)),
        }
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


def overlay_coordinates(rows, *, response_model="angle", normalization="shape"):
    """Use common physical baseline scales or separately centered shapes."""
    x = np.array([row["relative_coefficient"] for row in rows])
    if normalization == "baseline":
        baseline = np.array([row["baseline_epsilon"] for row in rows])
        return (
            x,
            np.array([row["derived_epsilon"] for row in rows]) / baseline,
            np.array([row["epsilon"] for row in rows]) / baseline,
        )
    if normalization != "shape":
        raise ValueError("unknown overlay normalization")
    for case in sorted({row["case_id"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["case_id"] == case]
        coefficients = x[indices]
        weights = np.array([rows[i]["occurrences"] for i in indices])
        x[indices] = coefficients / geometric_mean(coefficients, weights)
    field = "derived_shape" if "derived_shape" in rows[0] else f"{response_model}_shape"
    return (
        x,
        np.array([row[field] for row in rows]),
        np.array([row["learned_shape"] for row in rows]),
    )

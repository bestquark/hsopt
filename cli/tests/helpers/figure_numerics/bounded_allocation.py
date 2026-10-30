"""Numerical validation extracted from archived compile_angle_allocation.py; no plotting frontend."""

from __future__ import annotations


import numpy as np


def bounded_allocation(weights, response, agent):
    """Minimize sum m log(1/eps) at the agent's linearized error budget.

    eps_j = min(cap, C/g_j), with cap=max(agent_eps). A zero response receives
    cap; it is neither discarded nor assigned unbounded synthesis tolerance.
    g_j is the mean selected response magnitude over the term's occurrences.
    """
    weights, response, agent = (np.asarray(a, dtype=float) for a in (weights, response, agent))
    if not (weights.shape == response.shape == agent.shape) or weights.ndim != 1:
        raise ValueError("allocation vectors must be matching one-dimensional arrays")
    if not all(np.isfinite(a).all() for a in (weights, response, agent)):
        raise ValueError("allocation data must be finite")
    if np.any(weights <= 0) or np.any(response < 0) or np.any(agent <= 0) or np.any(agent >= 1):
        raise ValueError("invalid multiplicity, response, or tolerance")
    cap = float(agent.max())
    budget = float(np.sum(weights * response * agent))
    lower, upper = 0.0, cap * float(response.max())
    epsilon = np.full_like(response, cap)
    for _ in range(100):
        scale = 0.5 * (lower + upper)
        np.divide(scale, response, out=epsilon, where=response > 0)
        np.minimum(epsilon, cap, out=epsilon)
        used = float(np.sum(weights * response * epsilon))
        if used > budget:
            upper = scale
        else:
            lower = scale
    scale = lower
    np.divide(scale, response, out=epsilon, where=response > 0)
    np.minimum(epsilon, cap, out=epsilon)
    used = float(np.sum(weights * response * epsilon))
    if not np.isclose(used, budget, rtol=1e-11, atol=1e-28) or np.any(epsilon <= 0):
        raise ValueError("bounded allocation failed to match the first-order budget")
    old_cost = float(np.sum(weights * np.log(1 / agent)))
    new_cost = float(np.sum(weights * np.log(1 / epsilon)))
    if new_cost > old_cost + 1e-8 * max(1.0, old_cost):
        raise ValueError("derived allocation increased its own convex log-cost model")
    return epsilon.copy(), {
        "epsilon_cap": cap,
        "scale": scale,
        "agent_dimensionless_budget": budget,
        "derived_dimensionless_budget": used,
        "terms_at_cap": int(np.count_nonzero(epsilon >= cap * (1 - 1e-12))),
        "agent_log_cost": old_cost,
        "derived_log_cost": new_cost,
    }

#!/usr/bin/env python3
"""Figure 4: ten-reaction T95 comparison on the exact-ground-input branch.

Inputs are the immutable baseline and transfer validation runs, plus five
completed ELOTE-Delta trajectories. One complete shared policy is selected by
its suite-wide geometric-mean score; no per-reaction cherry-picking is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    GRAPHITE, NVIDIA_GREEN, REACTION_LABELS, REACTION_ORDER,
    apply_paper_style, output_file, read_json, save_figure,
)


EXPECTED_CASES = frozenset(REACTION_ORDER)
BLUE = "#2675A6"


def checked_result(path: Path) -> tuple[dict, dict[str, int]]:
    result = read_json(path)
    if (
        result.get("status") != "ok"
        or result.get("passes_chemical_accuracy") is not True
        or result.get("failed_case_ids")
        or result.get("unresolved_case_ids")
    ):
        raise ValueError(f"result is not a complete chemically accurate suite: {path}")
    metrics = result["case_metrics"]
    values = {row["case_id"]: int(row["t95"]) for row in metrics}
    if set(values) != EXPECTED_CASES or len(metrics) != len(EXPECTED_CASES):
        raise ValueError(f"result does not contain all ten distinct reactions: {path}")
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"nonpositive T95 in {path}")
    if not all(row["passes_chemical_accuracy"] for row in metrics):
        raise ValueError(f"a reaction failed chemical accuracy in {path}")
    return result, values


def check_run_contract(run: Path) -> None:
    suite = read_json(run / "suite.json")
    if (
        suite.get("kind") != "difference"
        or suite.get("energy_target") != "ground_branch"
        or suite.get("difference_error_contract") != "error_cancellation"
        or {row["id"] for row in suite["cases"]} != EXPECTED_CASES
    ):
        raise ValueError(f"run has a different scientific contract or membership: {run}")


def load(baseline: Path, transfer: Path, delta: Path) -> dict:
    baseline, transfer, delta = baseline.resolve(), transfer.resolve(), delta.resolve()
    check_run_contract(baseline)
    check_run_contract(transfer)
    base_result, base = checked_result(baseline / "iterations/0000/result.json")
    transfer_baseline, transfer_base = checked_result(transfer / "iterations/0000/result.json")
    if base != transfer_base:
        raise ValueError("transferred ELOTE was compared with a different tight baseline")
    transfer_result, elote = checked_result(transfer / "iterations/0001/result.json")
    seeds = sorted(delta.glob("seed-*"))
    if len(seeds) != 5:
        raise ValueError("Figure 4 requires all five independent ELOTE-Delta seeds")
    candidates = []
    for seed in seeds:
        check_run_contract(seed)
        manifest = read_json(seed / "manifest.json")
        state = read_json(seed / "state.json")
        if manifest.get("iterations_target") != 50 or state.get("iterations_completed") != 50:
            raise ValueError(f"ELOTE-Delta seed has not completed 50 mutations: {seed}")
        seed_baseline, seed_base = checked_result(seed / "iterations/0000/result.json")
        if seed_base != base:
            raise ValueError(f"ELOTE-Delta seed has a different tight baseline: {seed}")
        result, costs = checked_result(seed / "incumbent_result.json")
        candidate = seed / "incumbent.py"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        candidates.append((float(result["score"]), seed.name, result, costs, candidate))
    score, winner, delta_result, delta_costs, candidate = min(candidates, key=lambda x: x[:2])
    if not math.isfinite(score):
        raise ValueError("nonfinite winning ELOTE-Delta score")
    return {
        "order": list(REACTION_ORDER),
        "baseline_t95": base,
        "elote_t95": elote,
        "elote_delta_t95": delta_costs,
        "elote_delta_seed": winner,
        "elote_delta_score": score,
        "elote_delta_candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "source_results": {
            "baseline": str(baseline / "iterations/0000/result.json"),
            "elote": str(transfer / "iterations/0001/result.json"),
            "elote_delta": str(delta / winner / "incumbent_result.json"),
        },
        "all_seed_scores": {name: value for value, name, *_ in candidates},
    }


def plot(summary: dict, destination: Path) -> None:
    apply_paper_style(compact=True)
    order = summary["order"]
    theta = np.linspace(0.0, 2.0 * np.pi, len(order), endpoint=False)
    closed = np.r_[theta, theta[0]]
    stages = (
        ("baseline_t95", r"Tight PF2", GRAPHITE, "--", "o"),
        ("elote_t95", r"ELOTE", NVIDIA_GREEN, "-", "s"),
        ("elote_delta_t95", r"ELOTE-$\Delta$", BLUE, "-", "D"),
    )
    values = np.array([[summary[key][name] for name in order] for key, *_ in stages], dtype=float)
    radial = np.log10(values)
    lower = max(0, math.floor(float(radial.min()) - 0.3))
    upper = math.ceil(float(radial.max()) + 0.3)
    fig, ax = plt.subplots(figsize=(9.0, 8.4), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(lower, upper)
    ax.set_xticks(theta)
    ax.set_xticklabels([REACTION_LABELS[name] for name in order], fontsize=13)
    ax.tick_params(axis="x", pad=12)
    ticks = np.arange(math.ceil(lower), upper + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf"$10^{{{int(tick)}}}$" for tick in ticks], fontsize=12)
    ax.set_rlabel_position(86)
    ax.grid(alpha=0.18, linewidth=0.8)
    ax.spines["polar"].set_color("#A0A0A0")
    for index, (_, label, color, linestyle, marker) in enumerate(stages):
        ax.plot(closed, np.r_[radial[index], radial[index, 0]], color=color,
                linestyle=linestyle, marker=marker, markersize=5.5,
                linewidth=2.3, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3,
              columnspacing=1.3, handlelength=2.0)
    fig.text(0.5, 0.035, r"Compiled QPE cost $T_{95}$ (logarithmic radial scale)",
             ha="center", va="center", fontsize=16)
    save_figure(fig, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--transfer-run", required=True, type=Path)
    parser.add_argument("--delta-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = load(args.baseline_run, args.transfer_run, args.delta_root)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output_file(output, "reaction_elote_delta_radial.pdf")
    plot(summary, path)
    output_file(output, "reaction_elote_delta_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

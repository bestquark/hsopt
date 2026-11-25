#!/usr/bin/env python3
"""Figure 4: ten-reaction T95 comparison on the exact-ground-input branch.

Inputs are the immutable baseline and transfer validation runs, plus five
completed δ-ELOTE trajectories. One complete shared policy is selected by
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
    GRAPHITE,
    NVIDIA_GREEN,
    REACTION_LABELS,
    REACTION_ORDER,
    apply_paper_style,
    output_file,
    read_json,
    save_figure,
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
        or suite.get("baseline") != "tight"
        or suite.get("policy_profile") != "fermionic_nested"
        or suite.get("energy_target") != "ground_branch"
        or suite.get("difference_error_contract") != "error_cancellation"
        or {row["id"] for row in suite["cases"]} != EXPECTED_CASES
    ):
        raise ValueError(f"run has a different scientific contract or membership: {run}")


def reaction_widths(run: Path) -> dict[str, int]:
    """Maximum species-register width for each separately evaluated reaction."""
    suite = read_json(run / "suite.json")
    widths = {}
    for case in suite["cases"]:
        source = run / case["source"]
        reaction = read_json(source)
        qubits = []
        for term in reaction["terms"]:
            problem = read_json(source.parent / term["problem"])
            qubits.append(int(problem["n_qubits"]))
        if not qubits or min(qubits) < 1:
            raise ValueError(f"invalid species widths for {case['id']}")
        widths[case["id"]] = max(qubits)
    if set(widths) != EXPECTED_CASES:
        raise ValueError("reaction widths do not cover the ten Figure 4 cases")
    return widths


def load(baseline: Path, transfer: Path, delta: Path) -> dict:
    baseline, transfer, delta = baseline.resolve(), transfer.resolve(), delta.resolve()
    check_run_contract(baseline)
    check_run_contract(transfer)
    base_result, base = checked_result(baseline / "iterations/0000/result.json")
    transfer_baseline, transfer_base = checked_result(transfer / "iterations/0000/result.json")
    if base != transfer_base:
        raise ValueError("transferred ELOTE was compared with a different tight baseline")
    _, transfer_elote = checked_result(transfer / "iterations/0001/result.json")
    transferred_policy = transfer / "iterations/0001/candidate.py"
    transfer_sha256 = hashlib.sha256(transferred_policy.read_bytes()).hexdigest()
    seeds = sorted(delta.glob("seed-*"))
    if len(seeds) != 5:
        raise ValueError("Figure 4 requires all five independent δ-ELOTE seeds")
    candidates = []
    production_elote = None
    for seed in seeds:
        check_run_contract(seed)
        manifest = read_json(seed / "manifest.json")
        state = read_json(seed / "state.json")
        if manifest.get("iterations_target") != 50 or state.get("iterations_completed") != 50:
            raise ValueError(f"δ-ELOTE seed has not completed 50 mutations: {seed}")
        if (
            manifest.get("initial_policy_kind") != "warm_start"
            or manifest.get("initial_policy_sha256") != transfer_sha256
        ):
            raise ValueError(f"δ-ELOTE seed did not start from transferred ELOTE: {seed}")
        seed_baseline, seed_base = checked_result(seed / "iterations/0000/result.json")
        if production_elote is None:
            production_elote = seed_base
        elif seed_base != production_elote:
            raise ValueError(f"δ-ELOTE seeds disagree on the A100 ELOTE starting cost: {seed}")
        if any(abs(seed_base[case] / transfer_elote[case] - 1) > 0.01 for case in EXPECTED_CASES):
            raise ValueError(f"A100 and L4 ELOTE costs differ by over 1%: {seed}")
        result, costs = checked_result(seed / "incumbent_result.json")
        candidate = seed / "incumbent.py"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        candidates.append((float(result["score"]), seed.name, result, costs, candidate))
    score, winner, delta_result, delta_costs, candidate = min(candidates, key=lambda x: x[:2])
    if not math.isfinite(score):
        raise ValueError("nonfinite winning δ-ELOTE score")
    return {
        "order": list(REACTION_ORDER),
        "reaction_max_qubits": reaction_widths(baseline),
        "baseline_t95": base,
        "elote_t95": production_elote,
        "elote_delta_t95": delta_costs,
        "elote_delta_seed": winner,
        "elote_delta_score": score,
        "elote_delta_candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "source_results": {
            "baseline": str(baseline / "iterations/0000/result.json"),
            "elote": str(seeds[0] / "iterations/0000/result.json"),
            "elote_l4_validation": str(transfer / "iterations/0001/result.json"),
            "elote_delta": str(delta / winner / "incumbent_result.json"),
        },
        "all_seed_scores": {name: value for value, name, *_ in candidates},
    }


def plot(summary: dict, destination: Path) -> None:
    apply_paper_style(compact=True)
    widths = summary["reaction_max_qubits"]
    sorted_order = sorted(
        summary["order"], key=lambda name: (widths[name], summary["baseline_t95"][name])
    )
    order = sorted_order[::2] + sorted_order[::-2]
    theta = np.linspace(0.0, 2.0 * np.pi, len(order), endpoint=False)
    closed = np.r_[theta, theta[0]]
    stages = (
        ("baseline_t95", r"Tight baseline", GRAPHITE, "--", "o"),
        ("elote_t95", r"ELOTE", NVIDIA_GREEN, "-", "s"),
        ("elote_delta_t95", r"$\delta$-ELOTE", BLUE, "-", "D"),
    )
    values = np.array([[summary[key][name] for name in order] for key, *_ in stages], dtype=float)
    radial = np.log10(values)
    lower = max(0, math.floor(float(radial.min())) - 0.5)
    upper = math.ceil(float(radial.max())) - 0.5
    if upper - float(radial.max()) < 0.15:
        upper += 1
    fig, ax = plt.subplots(figsize=(16, 13.2), subplot_kw={"projection": "polar"})
    ax.set_position([0.235, 0.17, 0.53, 0.64])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(lower, upper)
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    ticks = np.arange(math.ceil(lower), math.floor(upper) + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf"$10^{{{int(tick)}}}$" for tick in ticks], fontsize=20)
    ax.set_rlabel_position(76)
    ax.grid(color="#D8D8D8", linewidth=0.8)
    ax.spines["polar"].set_color("#A8A8A8")
    for angle, name in zip(theta, order):
        x, y = math.sin(angle), math.cos(angle)
        ha = "left" if x > 0.25 else "right" if x < -0.25 else "center"
        va = "bottom" if y > 0.25 else "top" if y < -0.25 else "center"
        ax.text(
            0.5 + 0.61 * x,
            0.5 + 0.61 * y,
            REACTION_LABELS[name] + "\n" + rf"$n_{{\max}}={widths[name]}$",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=23,
            clip_on=False,
        )
    for index, (_, label, color, linestyle, marker) in enumerate(stages):
        ax.plot(
            closed,
            np.r_[radial[index], radial[index, 0]],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=6,
            linewidth=2.3,
            label=label,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.36),
        ncol=3,
        fontsize=22,
        columnspacing=1.4,
        handlelength=2.0,
    )
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

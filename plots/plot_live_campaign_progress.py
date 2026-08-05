#!/usr/bin/env python3
"""Render one isolated paper-figure set per HSOpt campaign."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    GRAPHITE,
    LIGHT_GRAPHITE,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    REACTION_LABELS,
    REACTION_ORDER,
    apply_paper_style,
)


CHEMICAL_ACCURACY_MHA = 1.5936014376405157
CAMPAIGN_ORDER = ("cpu", "gpu", "omni", "tight")
CAMPAIGN_STYLE = {
    "cpu": {
        "label": r"CPU loose",
        "short_label": r"CPU",
        "baseline": "loose",
        "color": GRAPHITE,
        "fill": LIGHT_GRAPHITE,
        "linestyle": "-",
    },
    "gpu": {
        "label": r"A100 loose",
        "short_label": r"A100",
        "baseline": "loose",
        "color": NVIDIA_DARK_GREEN,
        "fill": NVIDIA_GREEN,
        "linestyle": "--",
    },
    "omni": {
        "label": r"Omni L4 loose",
        "short_label": r"Omni L4",
        "baseline": "loose",
        "color": "#B77900",
        "fill": "#F2C14E",
        "linestyle": ":",
    },
    "tight": {
        "label": r"Omni L4 tight",
        "short_label": r"Omni L4 tight",
        "baseline": "tight",
        "color": "#6D5BD0",
        "fill": "#B8AEE8",
        "linestyle": "-.",
    },
}


@dataclass(frozen=True)
class LaneTrace:
    instance: str
    replicate: str
    improvement_percent: np.ndarray
    error_fraction: np.ndarray
    accepted: int


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("campaigns"), dict):
        raise ValueError("dashboard snapshot has no campaign mapping")
    return payload


def lane_trace(lane: dict[str, Any]) -> LaneTrace | None:
    history = {
        int(row["iteration"]): row
        for row in lane.get("history") or []
        if isinstance(row, dict) and isinstance(row.get("iteration"), int)
    }
    baseline = history.get(0)
    initial_score = finite_float(lane.get("initial_score"))
    if baseline is None or initial_score is None or initial_score <= 0.0:
        return None
    initial_error = finite_float(baseline.get("error_mha"))
    if initial_error is None:
        return None

    completed = int(lane.get("iterations_completed") or 0)
    incumbent_score = initial_score
    incumbent_error = initial_error
    improvements = [0.0]
    errors = [max(incumbent_error / CHEMICAL_ACCURACY_MHA, 1.0e-10)]
    accepted = 0
    for iteration in range(1, completed + 1):
        record = history.get(iteration)
        if record is None:
            break
        if record.get("accepted"):
            candidate_score = finite_float(record.get("score"))
            candidate_error = finite_float(record.get("error_mha"))
            if candidate_score is not None:
                incumbent_score = candidate_score
            if candidate_error is not None:
                incumbent_error = candidate_error
            accepted += 1
        improvements.append(100.0 * (initial_score - incumbent_score) / initial_score)
        errors.append(max(incumbent_error / CHEMICAL_ACCURACY_MHA, 1.0e-10))
    return LaneTrace(
        instance=str(lane.get("instance") or lane.get("name") or "lane"),
        replicate=str(lane.get("replicate") or "run"),
        improvement_percent=np.asarray(improvements, dtype=float),
        error_fraction=np.asarray(errors, dtype=float),
        accepted=accepted,
    )


def traces(
    snapshot: dict[str, Any],
    campaign_id: str,
    family: str,
) -> list[LaneTrace]:
    campaign = (snapshot.get("campaigns") or {}).get(campaign_id) or {}
    output = []
    for lane in campaign.get("lanes") or []:
        if lane.get("family") != family or int(lane.get("iterations_completed") or 0) <= 0:
            continue
        trace = lane_trace(lane)
        if trace is not None:
            output.append(trace)
    return output


def pointwise_summary(
    lane_traces: list[LaneTrace],
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maximum = max((getattr(trace, field).size for trace in lane_traces), default=0)
    x_values = []
    lower = []
    center = []
    upper = []
    counts = []
    for iteration in range(maximum):
        values = np.asarray(
            [
                getattr(trace, field)[iteration]
                for trace in lane_traces
                if getattr(trace, field).size > iteration
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        x_values.append(iteration)
        lower.append(np.quantile(values, 0.25))
        center.append(np.median(values))
        upper.append(np.quantile(values, 0.75))
        counts.append(values.size)
    return tuple(
        np.asarray(values, dtype=float)
        for values in (x_values, lower, center, upper, counts)
    )


def save_pair(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=240, facecolor="white")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def plot_accuracy(
    snapshot: dict[str, Any],
    output_dir: Path,
    campaign_id: str,
) -> None:
    apply_paper_style(compact=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    style = CAMPAIGN_STYLE[campaign_id]
    lane_traces = traces(snapshot, campaign_id, "abs_energy")
    for trace in lane_traces[:100]:
        x = np.arange(trace.error_fraction.size)
        ax.plot(x, trace.error_fraction, color=style["color"], alpha=0.09, linewidth=0.65)
    x, lower, center, upper, counts = pointwise_summary(
        lane_traces,
        "error_fraction",
    )
    if x.size:
        ax.fill_between(x, lower, upper, color=style["fill"], alpha=0.28, linewidth=0)
        ax.plot(
            x,
            center,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.5,
        )
        final_iteration = int(x[-1])
        final_count = int(counts[-1])
        note = rf"$n_{{{final_iteration}}}={final_count}$"
    else:
        note = r"no completed proposals"
    ax.axhspan(0.0, 1.0, color="#C7E596", alpha=0.14, zorder=0)
    ax.axhline(1.0, color="#C00000", linestyle="-.", linewidth=1.45)
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_ylim(1.0e-5, 3.0)
    ax.set_xlabel(r"Autoresearch iteration")
    ax.set_ylabel(r"$|\Delta E^{\star}|/\varepsilon_{\mathrm{chem}}$")
    ax.set_title(style["label"])
    ax.text(
        0.97,
        0.06,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=GRAPHITE,
    )
    ax.text(
        0.98,
        0.92,
        r"chemical accuracy",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#C00000",
    )
    ax.grid(True, which="major", axis="y", color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)
    save_pair(fig, output_dir, f"live_{campaign_id}_figure2_accuracy_progress")


def plot_improvement(
    snapshot: dict[str, Any],
    output_dir: Path,
    campaign_id: str,
) -> None:
    apply_paper_style(compact=True)
    fig, (ax_hist, ax_trace) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.55),
        width_ratios=(0.92, 1.18),
        layout="constrained",
    )
    style = CAMPAIGN_STYLE[campaign_id]
    lane_traces = traces(snapshot, campaign_id, "abs_energy")
    values = np.asarray(
        [trace.improvement_percent[-1] for trace in lane_traces],
        dtype=float,
    )
    maximum = max(1.0, float(np.max(values)) if values.size else 1.0)
    bins = np.linspace(0.0, max(5.0, 1.03 * maximum), 16)
    if values.size:
        ax_hist.hist(
            values,
            bins=bins,
            histtype="stepfilled",
            color=style["fill"],
            edgecolor=style["color"],
            linewidth=1.6,
            alpha=0.75,
            label=rf"$n={values.size}$",
        )
    ax_hist.set_xlabel(r"Current $T_{95}$ reduction (\%)")
    ax_hist.set_ylabel(r"Independent lanes")
    ax_hist.set_title(r"(a) Current distribution")
    ax_hist.grid(True, axis="y", color="0.90", linewidth=0.65)
    ax_hist.set_axisbelow(True)
    ax_hist.legend(loc="upper right")

    x, lower, center, upper, _ = pointwise_summary(
        lane_traces,
        "improvement_percent",
    )
    if x.size:
        ax_trace.fill_between(
            x,
            lower,
            upper,
            color=style["fill"],
            alpha=0.18,
            linewidth=0,
        )
        ax_trace.plot(
            x,
            center,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            label=style["short_label"],
        )
    ax_trace.axhline(0.0, color="0.75", linewidth=1.0)
    ax_trace.set_xlim(0, 50)
    ax_trace.set_xlabel(r"Autoresearch iteration")
    ax_trace.set_ylabel(r"Running-best $T_{95}$ reduction (\%)")
    ax_trace.set_title(r"(b) Median and interquartile range")
    ax_trace.grid(True, axis="y", color="0.90", linewidth=0.65)
    ax_trace.set_axisbelow(True)
    ax_trace.legend(loc="upper left")
    fig.suptitle(style["label"])
    save_pair(fig, output_dir, f"live_{campaign_id}_figure3_t95_improvement")


def reaction_instance_reductions(
    lane_traces: list[LaneTrace],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for trace in lane_traces:
        grouped.setdefault(trace.instance, []).append(float(trace.improvement_percent[-1]))
    return {key: float(np.median(values)) for key, values in grouped.items()}


def plot_reactions(
    snapshot: dict[str, Any],
    output_dir: Path,
    campaign_id: str,
) -> None:
    apply_paper_style(compact=True)
    style = CAMPAIGN_STYLE[campaign_id]
    lane_traces = traces(snapshot, campaign_id, "diff_energy")
    reductions = reaction_instance_reductions(lane_traces)
    available = set(reductions)
    ordered = [key for key in REACTION_ORDER if key in available]
    ordered.extend(sorted(available - set(ordered)))

    fig, (ax_bar, ax_trace) = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.35),
        width_ratios=(1.18, 1.0),
        layout="constrained",
    )
    y = np.arange(len(ordered), dtype=float)
    if ordered:
        values = [reductions.get(key, np.nan) for key in ordered]
        ax_bar.barh(
            y,
            values,
            height=0.62,
            color=style["fill"],
            edgecolor=style["color"],
            linewidth=1.0,
        )
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([REACTION_LABELS.get(key, key.replace("_", r"\_")) for key in ordered])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(r"Current $T_{95}$ reduction (\%)")
    ax_bar.set_title(r"(a) Reaction policies")
    ax_bar.grid(True, axis="x", color="0.90", linewidth=0.65)
    ax_bar.set_axisbelow(True)
    if not ordered:
        ax_bar.text(0.5, 0.5, r"No reaction proposals completed", transform=ax_bar.transAxes, ha="center")

    x, lower, center, upper, _ = pointwise_summary(
        lane_traces,
        "improvement_percent",
    )
    if x.size:
        ax_trace.fill_between(
            x,
            lower,
            upper,
            color=style["fill"],
            alpha=0.18,
            linewidth=0,
        )
        ax_trace.plot(
            x,
            center,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            label=style["short_label"],
        )
    ax_trace.axhline(0.0, color="0.75", linewidth=1.0)
    ax_trace.set_xlim(0, 50)
    ax_trace.set_xlabel(r"Autoresearch iteration")
    ax_trace.set_ylabel(r"Running-best $T_{95}$ reduction (\%)")
    ax_trace.set_title(r"(b) Partial trajectories")
    ax_trace.grid(True, axis="y", color="0.90", linewidth=0.65)
    ax_trace.set_axisbelow(True)
    if x.size:
        ax_trace.legend(loc="upper left")
    else:
        ax_trace.text(
            0.5,
            0.5,
            r"No reaction proposals completed",
            transform=ax_trace.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=GRAPHITE,
        )
    fig.suptitle(style["label"])
    save_pair(fig, output_dir, f"live_{campaign_id}_figure4_reaction_progress")


def write_statistics(
    snapshot: dict[str, Any],
    output_dir: Path,
    campaign_ids: tuple[str, ...],
) -> None:
    campaigns = {}
    for campaign_id in campaign_ids:
        campaign = (snapshot.get("campaigns") or {}).get(campaign_id) or {}
        absolute = traces(snapshot, campaign_id, "abs_energy")
        reactions = traces(snapshot, campaign_id, "diff_energy")
        lanes = campaign.get("lanes") or []
        campaigns[campaign_id] = {
            "label": CAMPAIGN_STYLE[campaign_id]["label"].replace("\\", ""),
            "baseline_variant": CAMPAIGN_STYLE[campaign_id]["baseline"],
            "absolute_lanes_started": len(absolute),
            "absolute_iterations": sum(
                int(row.get("iterations_completed") or 0)
                for row in lanes
                if row.get("family") == "abs_energy"
            ),
            "absolute_accepted": sum(trace.accepted for trace in absolute),
            "current_absolute_median_reduction_percent": (
                float(np.median([trace.improvement_percent[-1] for trace in absolute]))
                if absolute
                else None
            ),
            "reaction_lanes_started": len(reactions),
            "reaction_iterations": sum(
                int(row.get("iterations_completed") or 0)
                for row in lanes
                if row.get("family") == "diff_energy"
            ),
            "reaction_accepted": sum(trace.accepted for trace in reactions),
            "current_reaction_median_reduction_percent": (
                float(np.median([trace.improvement_percent[-1] for trace in reactions]))
                if reactions
                else None
            ),
        }
    payload = {
        "schema": 1,
        "generated_at": snapshot.get("generated_at")
        or datetime.now(timezone.utc).isoformat(),
        "partial_campaign_snapshot": True,
        "campaigns": campaigns,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "live_progress_statistics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        action="append",
        choices=CAMPAIGN_ORDER,
        help="campaign to render; repeat as needed (default: all four)",
    )
    args = parser.parse_args()
    snapshot = load_snapshot(args.snapshot.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    campaign_ids = tuple(args.campaign or CAMPAIGN_ORDER)
    for campaign_id in campaign_ids:
        plot_accuracy(snapshot, output_dir, campaign_id)
        plot_improvement(snapshot, output_dir, campaign_id)
        plot_reactions(snapshot, output_dir, campaign_id)
    write_statistics(snapshot, output_dir, campaign_ids)


if __name__ == "__main__":
    main()

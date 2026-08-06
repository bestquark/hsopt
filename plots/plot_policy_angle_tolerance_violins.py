#!/usr/bin/env python3
"""Plot initial-versus-best rotation distributions for every HSOpt experiment."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    GRAPHITE,
    LIGHT_GRAPHITE,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    apply_paper_style,
    external_output_dir,
)
from hsopt.evaluation.rotation_diagnostics import (
    comparison_histograms,
    rotation_samples,
)


@dataclass(frozen=True)
class ExperimentComparison:
    family: str
    instance: str
    replicate: str
    lane_dir: Path
    source_path: Path
    initial_policy: Path
    current_policy: Path
    initial_t95: float
    current_t95: float
    iterations_completed: int
    iterations_target: int

    @property
    def reduction_percent(self) -> float:
        return 100.0 * (self.initial_t95 - self.current_t95) / self.initial_t95

    @property
    def complete(self) -> bool:
        return self.iterations_completed >= self.iterations_target


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def score(result: dict[str, Any]) -> float | None:
    for key in ("score", "confidence_adjusted_t_count", "t_count"):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def select_experiments(
    campaign_root: Path,
    *,
    expected_baseline: str | None,
    require_complete: bool,
) -> tuple[dict[str, Any], list[ExperimentComparison]]:
    campaign_root = campaign_root.expanduser().resolve()
    campaign = read_json(campaign_root / "campaign.json")
    baseline = campaign.get("baseline_variant")
    if expected_baseline is not None and baseline != expected_baseline:
        raise ValueError(
            f"campaign baseline is {baseline!r}, expected {expected_baseline!r}"
        )
    selected: dict[tuple[str, str], ExperimentComparison] = {}
    for row in campaign.get("lanes") or []:
        if not isinstance(row, dict) or not row.get("lane_dir"):
            continue
        lane_dir = campaign_root / str(row["lane_dir"])
        manifest_path = lane_dir / "manifest.json"
        state_path = lane_dir / "state.json"
        baseline_path = lane_dir / "baseline_result.json"
        current_path = lane_dir / "incumbent_result.json"
        if not all(
            path.is_file()
            for path in (manifest_path, state_path, baseline_path, current_path)
        ):
            continue
        manifest = read_json(manifest_path)
        state = read_json(state_path)
        initial_result = read_json(baseline_path)
        current_result = read_json(current_path)
        initial_t95 = score(initial_result)
        current_t95 = score(current_result)
        if initial_t95 is None or initial_t95 <= 0.0 or current_t95 is None:
            continue
        if current_result.get("status") != "ok" or not current_result.get(
            "passes_chemical_accuracy"
        ):
            continue
        iterations_completed = int(state.get("iterations_completed") or 0)
        iterations_target = int(manifest.get("iterations_target") or campaign.get("iterations") or 50)
        if require_complete and iterations_completed < iterations_target:
            continue
        family = str(manifest.get("family") or "custom")
        instance = str(manifest.get("instance") or lane_dir.parent.name)
        comparison = ExperimentComparison(
            family=family,
            instance=instance,
            replicate=str(manifest.get("replicate") or lane_dir.name),
            lane_dir=lane_dir,
            source_path=Path(str(manifest["source"])).expanduser().resolve(),
            initial_policy=lane_dir / "initial_pf.py",
            current_policy=lane_dir / "incumbent.py",
            initial_t95=initial_t95,
            current_t95=current_t95,
            iterations_completed=iterations_completed,
            iterations_target=iterations_target,
        )
        key = (family, instance)
        incumbent = selected.get(key)
        if incumbent is None or (comparison.current_t95, comparison.replicate) < (
            incumbent.current_t95,
            incumbent.replicate,
        ):
            selected[key] = comparison
    return campaign, [selected[key] for key in sorted(selected)]


def _draw_violin(
    ax: plt.Axes,
    histogram: dict[str, list[float]],
    *,
    label: str,
) -> None:
    edges = np.asarray(histogram["log10_edges"], dtype=float)
    centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    stages = (
        (1.0, np.asarray(histogram["initial_density"], dtype=float), LIGHT_GRAPHITE),
        (0.0, np.asarray(histogram["current_density"], dtype=float), NVIDIA_GREEN),
    )
    for y, density, color in stages:
        width = 0.34 * density
        ax.fill_between(
            centers,
            y - width,
            y + width,
            color=color,
            alpha=0.72,
            linewidth=0.9,
            edgecolor=GRAPHITE if y else NVIDIA_DARK_GREEN,
        )
        ax.plot(centers, np.full_like(centers, y), color=GRAPHITE, linewidth=0.7, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yticks([0.0, 1.0], [r"Current best", r"Initial"])
    ax.set_ylim(-0.48, 1.48)
    ax.set_xlabel(label)
    ax.grid(True, axis="x", which="major", color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_experiment(
    comparison: ExperimentComparison,
    *,
    output_dir: Path,
    baseline_variant: str,
) -> dict[str, Any]:
    initial = rotation_samples(comparison.initial_policy, comparison.source_path)
    current = rotation_samples(comparison.current_policy, comparison.source_path)
    histograms = comparison_histograms(initial, current)
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.45), layout="constrained")
    _draw_violin(axes[0], histograms["angles"], label=r"Rotation angle, $|\theta|$ (rad)")
    _draw_violin(
        axes[1],
        histograms["tolerances"],
        label=r"Synthesis tolerance, $\epsilon_{\mathrm{synth}}$",
    )
    axes[1].tick_params(axis="y", labelleft=False)
    progress = (
        r"complete"
        if comparison.complete
        else rf"current at {comparison.iterations_completed}/{comparison.iterations_target}"
    )
    fig.suptitle(
        rf"{baseline_variant} baseline; {progress}; "
        rf"$\Delta T_{{95}}={comparison.reduction_percent:.2f}\%$"
    )

    experiment_dir = output_dir / comparison.family / comparison.instance
    experiment_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = experiment_dir / "angle_tolerance_violin.pdf"
    png_path = experiment_dir / "angle_tolerance_violin.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    metadata = {
        "schema": 1,
        "family": comparison.family,
        "instance": comparison.instance,
        "replicate": comparison.replicate,
        "baseline_variant": baseline_variant,
        "status": "complete" if comparison.complete else "partial",
        "iterations_completed": comparison.iterations_completed,
        "iterations_target": comparison.iterations_target,
        "initial_t95": comparison.initial_t95,
        "current_best_t95": comparison.current_t95,
        "relative_t95_improvement_percent": comparison.reduction_percent,
        "relative_t95_definition": (
            "100 * (initial_t95 - current_best_t95) / initial_t95"
        ),
        "rotation_distributions": histograms,
        "figure_pdf": pdf_path.name,
        "figure_png": png_path.name,
    }
    metadata_path = experiment_dir / "angle_tolerance_violin.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", choices=("loose", "tight"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    campaign_root = args.campaign_root.expanduser().resolve()
    output_dir = external_output_dir(args.output_dir, campaign_root)
    campaign, comparisons = select_experiments(
        campaign_root,
        expected_baseline=args.baseline,
        require_complete=args.require_complete,
    )
    baseline = str(campaign.get("baseline_variant") or args.baseline or "unknown")
    records = [
        plot_experiment(comparison, output_dir=output_dir, baseline_variant=baseline)
        for comparison in comparisons
    ]
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "campaign_root": str(campaign_root),
                "baseline_variant": baseline,
                "experiments": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

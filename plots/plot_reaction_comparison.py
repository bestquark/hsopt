#!/usr/bin/env python3
"""Recreate the ten-reaction accuracy and T95 comparison figure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from _paper_plot import (
    CHEMICAL_ACCURACY_HARTREE,
    GRAPHITE,
    LIGHT_GRAPHITE,
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    REACTION_LABELS,
    REACTION_ORDER,
    apply_paper_style,
    as_float,
    as_int,
    external_output_dir,
    output_file,
    read_json,
    save_figure,
    strict_export,
)


OUTPUT_NAME = "reaction_cancellation_budget.pdf"


@dataclass(frozen=True)
class ReactionRecord:
    key: str
    label: str
    equal_share_error_fraction: float
    reaction_aware_error_fraction: float
    equal_share_t95_millions: float
    reaction_aware_t95_millions: float

    @property
    def decrease_percent(self) -> float:
        return (
            100.0
            * (self.equal_share_t95_millions - self.reaction_aware_t95_millions)
            / self.equal_share_t95_millions
        )


def _selection(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{record.get('reaction_key')} lacks {name} selection")
    return value


def load_records(export_dir: Path, *, expected_reactions: int) -> list[ReactionRecord]:
    payload = read_json(export_dir / "reaction_summary.json")
    by_key = {str(row["reaction_key"]): row for row in payload["reactions"]}
    if len(by_key) != expected_reactions:
        raise ValueError(
            f"reaction summary contains {len(by_key)} reactions, expected {expected_reactions}"
        )
    ordered_keys = [key for key in REACTION_ORDER if key in by_key]
    ordered_keys.extend(sorted(set(by_key) - set(ordered_keys)))

    records: list[ReactionRecord] = []
    for key in ordered_keys:
        row = by_key[key]
        equal_share = _selection(row, "equal_share")
        reaction_aware = _selection(row, "reaction_aware")
        equal_t95 = 1.0e-6 * as_float(
            equal_share["confidence_adjusted_t_gate_count"],
            label=f"{key} equal-share T95",
        )
        reaction_t95 = 1.0e-6 * as_float(
            reaction_aware["confidence_adjusted_t_gate_count"],
            label=f"{key} reaction-aware T95",
        )
        if equal_t95 <= 0.0 or reaction_t95 <= 0.0:
            raise ValueError(f"{key} has a nonpositive T95")
        records.append(
            ReactionRecord(
                key=key,
                label=REACTION_LABELS.get(
                    key,
                    str(row.get("reaction", key)).replace("_", r"\_"),
                ),
                equal_share_error_fraction=as_float(
                    equal_share["absolute_reaction_error_hartree"],
                    label=f"{key} equal-share error",
                )
                / CHEMICAL_ACCURACY_HARTREE,
                reaction_aware_error_fraction=as_float(
                    reaction_aware["absolute_reaction_error_hartree"],
                    label=f"{key} reaction-aware error",
                )
                / CHEMICAL_ACCURACY_HARTREE,
                equal_share_t95_millions=equal_t95,
                reaction_aware_t95_millions=reaction_t95,
            )
        )
    return records


def plot(records: list[ReactionRecord], output_path: Path) -> None:
    apply_paper_style(compact=True)
    fig, (ax_error, ax_t95) = plt.subplots(
        1,
        2,
        figsize=(9.6, 5.35),
        width_ratios=[1.05, 1.0],
        layout="constrained",
    )
    y = np.arange(len(records), dtype=float)
    height = 0.34
    labels = [record.label for record in records]

    ax_error.barh(
        y - height / 2,
        [record.equal_share_error_fraction for record in records],
        height,
        color=LIGHT_GRAPHITE,
        edgecolor="white",
        linewidth=0.7,
        label=r"equal-share seed",
        zorder=3,
    )
    ax_error.barh(
        y + height / 2,
        [record.reaction_aware_error_fraction for record in records],
        height,
        color=NVIDIA_GREEN,
        edgecolor="white",
        linewidth=0.7,
        label=r"reaction-aware seed",
        zorder=3,
    )
    ax_error.axvline(
        1.0,
        color=NVIDIA_DARK_GREEN,
        linestyle="--",
        linewidth=1.8,
        zorder=4,
    )
    max_error = max(
        1.0,
        *(record.equal_share_error_fraction for record in records),
        *(record.reaction_aware_error_fraction for record in records),
    )
    ax_error.set_xlim(0.0, 1.06 * max_error)
    ax_error.set_xlabel(r"$|\Delta E_{\mathrm{rxn}}|/\varepsilon_{\mathrm{chem}}$")
    ax_error.set_yticks(y)
    ax_error.set_yticklabels(labels)
    ax_error.invert_yaxis()
    ax_error.grid(True, axis="x", color="0.86", linewidth=0.7)
    ax_error.set_axisbelow(True)
    ax_error.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        handlelength=1.3,
        columnspacing=0.9,
    )

    equal_counts = [record.equal_share_t95_millions for record in records]
    reaction_counts = [record.reaction_aware_t95_millions for record in records]
    ax_t95.barh(
        y - height / 2,
        equal_counts,
        height,
        color=LIGHT_GRAPHITE,
        edgecolor="white",
        linewidth=0.7,
        label=r"equal-share seed",
        zorder=3,
    )
    bars = ax_t95.barh(
        y + height / 2,
        reaction_counts,
        height,
        color=NVIDIA_GREEN,
        edgecolor="white",
        linewidth=0.7,
        label=r"reaction-aware seed",
        zorder=3,
    )
    ax_t95.set_xlabel(r"$T_{95}$ (millions)")
    ax_t95.set_yticks(y)
    ax_t95.set_yticklabels([])
    ax_t95.invert_yaxis()
    max_count = max(equal_counts + reaction_counts)
    ax_t95.set_xlim(0.0, 1.25 * max_count)
    ax_t95.grid(True, axis="x", color="0.86", linewidth=0.7)
    ax_t95.set_axisbelow(True)
    gains = [record.decrease_percent for record in records]
    for bar, record in zip(bars, records):
        ax_t95.text(
            max(record.equal_share_t95_millions, record.reaction_aware_t95_millions)
            + 0.018 * max_count,
            bar.get_y() + bar.get_height() / 2,
            rf"{record.decrease_percent:.1f}\%",
            ha="left",
            va="center",
            fontsize=10,
            color=GRAPHITE,
        )
    ax_t95.text(
        0.98,
        0.97,
        rf"mean: {np.mean(gains):.1f}\%",
        transform=ax_t95.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=GRAPHITE,
    )
    ax_t95.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        handlelength=1.3,
        columnspacing=0.9,
    )
    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_dir, manifest = strict_export(args.export_dir, "diff_energy")
    output_dir = external_output_dir(args.output_dir, export_dir)
    records = load_records(
        export_dir,
        expected_reactions=as_int(manifest["instance_count"]),
    )
    plot(records, output_file(output_dir, OUTPUT_NAME))


if __name__ == "__main__":
    main()

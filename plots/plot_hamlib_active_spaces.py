#!/usr/bin/env python3
"""Recreate the HamLib active-space census and selected campaign cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from _paper_plot import (
    NVIDIA_DARK_GREEN,
    NVIDIA_GREEN,
    apply_paper_style,
    as_int,
    external_output_dir,
    output_file,
    read_csv,
    read_json,
    save_figure,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "baseline/molecules.json"
OUTPUT_NAME = "hamlib_molecular_dataset_active_spaces.pdf"


def counts_from_csv(path: Path) -> Counter[int]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    field = "nqubits" if "nqubits" in rows[0] else "n_qubits"
    if field not in rows[0]:
        raise ValueError(f"{path} must contain nqubits or n_qubits")
    return Counter(as_int(row[field], label="qubit count") for row in rows)


def counts_from_full_census(path: Path) -> Counter[int]:
    if path.suffix.lower() == ".csv":
        return counts_from_csv(path)
    payload: dict[str, Any] = read_json(path)
    counts = payload.get("active_space_counts_by_nqubits")
    if isinstance(counts, dict):
        return Counter({int(key): int(value) for key, value in counts.items()})
    rows = payload.get("instances") or payload.get("lanes") or payload.get("campaign_lanes")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain active_space_counts_by_nqubits or a lane list")
    return Counter(
        as_int(row.get("nqubits", row.get("n_qubits")), label="qubit count") for row in rows
    )


def plot(
    selected_counts: Counter[int],
    full_counts: Counter[int] | None,
    output_path: Path,
) -> None:
    apply_paper_style()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 24,
                         "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18})
    all_qubits = sorted(set(selected_counts) | set(full_counts or {}))
    x = np.asarray(all_qubits, dtype=float)
    selected = np.asarray([selected_counts.get(value, 0) for value in all_qubits], dtype=float)
    total_selected = int(np.sum(selected))

    fig, ax = plt.subplots(figsize=(12.6, 4.9), layout="constrained")
    if full_counts is not None:
        full = np.asarray([full_counts.get(value, 0) for value in all_qubits], dtype=float)
        ax.bar(
            x,
            full,
            width=1.65,
            color="#BFC7BA",
            edgecolor="#8D9588",
            linewidth=0.65,
            label=f"HamLib ({sum(full_counts.values())})",
            zorder=1,
        )
    ax.bar(
        x,
        selected,
        width=1.65,
        color=NVIDIA_GREEN,
        edgecolor=NVIDIA_DARK_GREEN,
        linewidth=0.8,
        label=f"Current campaign ({total_selected})",
        zorder=2,
    )

    cutoff = max(selected_counts)
    ax.axvline(
        cutoff,
        color=NVIDIA_DARK_GREEN,
        linestyle="--",
        linewidth=1.7,
        zorder=3,
    )
    maximum = max(
        max(selected),
        max(full_counts.values()) if full_counts is not None else 0,
    )
    y_top = 20 * np.ceil(1.10 * maximum / 20)
    ax.text(
        cutoff + 2,
        0.82 * y_top,
        rf"$n_q\leq {cutoff}$",
        ha="left",
        va="top",
        color=NVIDIA_DARK_GREEN,
        fontsize=21,
    )
    ax.set_xlabel(r"Mapped qubits, $n_q$")
    ax.set_ylabel(r"Active-space instances")
    ax.set_xlim(min(all_qubits) - 2, max(all_qubits) + 3)
    ax.set_ylim(0.0, y_top)
    if max(all_qubits) <= 20:
        ax.set_xticks(np.arange(min(all_qubits), max(all_qubits) + 1, 2))
    else:
        ax.set_xticks(
            sorted({cutoff, *(value for value in (4, 16, 32, 48, 64, 80, 100, 120)
                              if min(x) <= value <= max(x))})
        )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(False, which="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=2 if full_counts is not None else 1)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Selected cohort CSV with nqubits or n_qubits, e.g. a frozen molecules.csv.",
    )
    parser.add_argument(
        "--full-census",
        type=Path,
        help=(
            "Optional full HamLib census JSON/CSV. JSON may contain "
            "active_space_counts_by_nqubits or a lane list."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = counts_from_csv(args.selected_manifest)
    full = counts_from_full_census(args.full_census) if args.full_census else None
    if full is not None and any(selected[key] > full.get(key, 0) for key in selected):
        raise ValueError("selected census is not a subset of the supplied full census")
    read_roots = [args.selected_manifest]
    if args.full_census is not None:
        read_roots.append(args.full_census)
    output_dir = external_output_dir(args.output_dir, *read_roots)
    output_path = output_file(output_dir, OUTPUT_NAME)
    plot(selected, full, output_path)
    inputs = [args.selected_manifest, *([args.full_census] if args.full_census else [])]
    metadata = {
        "selected_count": sum(selected.values()),
        "selected_counts_by_qubits": dict(sorted(selected.items())),
        "full_active_space_count": sum(full.values()) if full is not None else None,
        "full_counts_by_qubits": dict(sorted(full.items())) if full is not None else None,
        "cutoff_qubits": max(selected),
        "inputs": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

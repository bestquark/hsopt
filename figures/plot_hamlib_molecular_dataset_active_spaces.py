#!/usr/bin/env python3
"""Plot the HamLib molecular census with training and held-out size regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


from _paths import data_path, output_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = data_path("data/hamlib_molecular_dataset_active_spaces.json")
HAMLIB_DATA_ROOT = Path(
    os.environ.get(
        "HAMLIB_DATA_ROOT",
        str(REPO_ROOT.parent / "data/hamlib"),
    )
)
A100_CAMPAIGN_MANIFEST = Path(
    os.environ.get(
        "A100_CAMPAIGN_MANIFEST",
        str(REPO_ROOT.parent / "data/campaign_manifest.json"),
    )
)
OUTPUT_PATH = output_path("hamlib_molecular_dataset_active_spaces.pdf")

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK_GREEN = "#3E7E00"
GRAPHITE = "#4D4D4D"
LIGHT_GRAPHITE = "#BFC7BA"
INK = "#1F2937"
MAPPING_PREFIX_RE = re.compile(r"^/ham_(BK|JW|parity)")
TRAINING_MAX_QUBITS = 16
TEST_MAX_QUBITS = 22
SNAPSHOT_SCHEMA_VERSION = 1
LANE_SNAPSHOT_FIELDS = (
    "lane_id",
    "molecule",
    "nqubits",
    "estimated_state_workspace_gib",
    "estimated_sector_workspace_gib",
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 26,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "figure.titlesize": 28,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.35,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.frameon": False,
            "hatch.linewidth": 4.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def canonical_dataset(dataset: str) -> str:
    return MAPPING_PREFIX_RE.sub("/ham", dataset)


def load_hamlib_active_space_counts(data_root: Path) -> Counter[int]:
    groups: dict[tuple[str, str, str], int] = {}
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel_path = str(csv_path.relative_to(data_root))
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                nqubits_text = row.get("nqubits")
                if not nqubits_text:
                    continue
                key = (
                    rel_path,
                    row["File"],
                    canonical_dataset(row["Dataset"]),
                )
                groups.setdefault(key, int(float(nqubits_text)))
    return Counter(groups.values())


def load_campaign_lanes(manifest_path: Path) -> list[dict[str, object]]:
    with manifest_path.open() as handle:
        return list(json.load(handle)["lanes"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_csv_tree(data_root: Path) -> tuple[int, str]:
    csv_paths = sorted(data_root.rglob("*.csv"))
    digest = hashlib.sha256()
    for csv_path in csv_paths:
        digest.update(csv_path.relative_to(data_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(csv_path)))
    return len(csv_paths), digest.hexdigest()


def build_snapshot(data_root: Path, manifest_path: Path) -> dict[str, object]:
    total_counts = load_hamlib_active_space_counts(data_root)
    campaign_lanes = load_campaign_lanes(manifest_path)
    csv_file_count, csv_tree_sha256 = sha256_csv_tree(data_root)
    reduced_lanes = [
        {field: lane[field] for field in LANE_SNAPSHOT_FIELDS} for lane in campaign_lanes
    ]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": {
            "hamlib_csv_file_count": csv_file_count,
            "hamlib_csv_tree_sha256": csv_tree_sha256,
            "campaign_manifest_sha256": sha256_file(manifest_path),
            "campaign_manifest_lane_count": len(campaign_lanes),
        },
        "active_space_counts_by_nqubits": {
            str(nqubits): count for nqubits, count in sorted(total_counts.items())
        },
        "campaign_lanes": reduced_lanes,
    }


def write_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_snapshot(path: Path) -> tuple[Counter[int], list[dict[str, object]]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported HamLib snapshot schema in {path}")
    total_counts = Counter(
        {
            int(nqubits): int(count)
            for nqubits, count in payload["active_space_counts_by_nqubits"].items()
        }
    )
    return total_counts, list(payload["campaign_lanes"])


def training_lanes(
    lanes: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    return [lane for lane in lanes if int(lane["nqubits"]) <= TRAINING_MAX_QUBITS]


def count_by_nqubits(lanes: Iterable[dict[str, object]]) -> Counter[int]:
    return Counter(int(lane["nqubits"]) for lane in lanes)


def plot(snapshot_path: Path, *, refresh_snapshot: bool) -> None:
    apply_style()
    if refresh_snapshot:
        write_snapshot(
            snapshot_path,
            build_snapshot(HAMLIB_DATA_ROOT, A100_CAMPAIGN_MANIFEST),
        )
    total_counts, campaign_lanes = load_snapshot(snapshot_path)
    selected_lanes = training_lanes(campaign_lanes)
    selected_counts = count_by_nqubits(selected_lanes)

    x_values = np.asarray(sorted(total_counts), dtype=float)
    total_heights = np.asarray([total_counts[int(x)] for x in x_values], dtype=float)
    selected_heights = np.asarray(
        [selected_counts.get(int(x), 0) for x in x_values],
        dtype=float,
    )
    test_mask = (x_values > TRAINING_MAX_QUBITS) & (x_values <= TEST_MAX_QUBITS)
    test_count = int(np.sum(total_heights[test_mask]))

    fig, ax = plt.subplots(figsize=(12.6, 4.9), layout="constrained")
    # Testing bars replace the gray census bars at the same sizes, so their
    # translucent green fill is rendered against white, as in Figure 2(b).
    # They include all 78 census instances, independently of baseline
    # countability or completed energy checks.
    ax.bar(
        x_values[~test_mask],
        total_heights[~test_mask],
        width=1.65,
        color=LIGHT_GRAPHITE,
        edgecolor="#8E9789",
        linewidth=1.1,
        alpha=0.78,
        label="_nolegend_",
        zorder=1,
    )
    ax.bar(
        x_values,
        selected_heights,
        width=1.65,
        color=NVIDIA_GREEN,
        edgecolor=NVIDIA_GREEN,
        linewidth=1.1,
        alpha=1.0,
        label=rf"Training (${len(selected_lanes)}$)",
        zorder=2,
    )
    ax.bar(
        x_values[test_mask],
        total_heights[test_mask],
        width=1.65,
        color=to_rgba(NVIDIA_GREEN, 0.18),
        edgecolor=to_rgba(NVIDIA_GREEN, 0.38),
        hatch="/",
        linewidth=0.8,
        label=rf"Testing (${test_count}$)",
        zorder=2,
    )

    ax.axvline(
        TRAINING_MAX_QUBITS,
        color=NVIDIA_DARK_GREEN,
        linestyle="--",
        linewidth=2.1,
        zorder=4,
        label="_nolegend_",
    )

    ax.axvline(
        TEST_MAX_QUBITS,
        color=GRAPHITE,
        linestyle="--",
        linewidth=2.1,
        zorder=4,
        label="_nolegend_",
    )
    ax.text(
        TRAINING_MAX_QUBITS - 0.8,
        100.0,
        rf"$n\le {TRAINING_MAX_QUBITS}$",
        color=NVIDIA_DARK_GREEN,
        ha="right",
        va="center",
        fontsize=22,
    )
    # Rotate the inequality so it stays wholly inside the six-qubit test strip
    # while retaining the full 4--120-qubit census on a linear axis.
    ax.text(
        (TRAINING_MAX_QUBITS + TEST_MAX_QUBITS) / 2,
        98.0,
        rf"${TRAINING_MAX_QUBITS}<n\le {TEST_MAX_QUBITS}$",
        color=GRAPHITE,
        ha="center",
        va="center",
        rotation=90,
        fontsize=18,
    )

    ax.set_xlabel(r"Mapped qubits, $n$")
    ax.set_ylabel(r"Active-space instances")
    ax.set_xlim(2.0, 123.0)
    ax.set_ylim(0.0, 120.0)
    ax.set_xticks([4, 16, 22, 32, 48, 64, 80, 100, 120])
    ax.set_yticks(np.arange(0, 121.0, 20.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        borderpad=0.35,
        handlelength=1.6,
        columnspacing=0.9,
    )
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)

    print(f"active-space instances: {sum(total_counts.values())}")
    print(f"snapshot: {snapshot_path}")
    print(f"training Hamiltonians: {len(selected_lanes)}")
    print(f"training qubit cutoff: {TRAINING_MAX_QUBITS}")
    print(f"training counts: {dict(sorted(selected_counts.items()))}")
    print(f"held-out census instances: {test_count}")
    print(f"held-out qubit range: {TRAINING_MAX_QUBITS} < n <= {TEST_MAX_QUBITS}")
    print(f"wrote {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the checked-in HamLib active-space census snapshot."
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help=(
            "rebuild the snapshot from HAMLIB_DATA_ROOT and A100_CAMPAIGN_MANIFEST before plotting"
        ),
    )
    args = parser.parse_args()
    plot(args.snapshot, refresh_snapshot=args.refresh_snapshot)


if __name__ == "__main__":
    main()

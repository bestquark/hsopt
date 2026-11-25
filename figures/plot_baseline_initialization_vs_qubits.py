#!/usr/bin/env python3
"""Plot the actual iteration-zero baselines of the 115-Hamiltonian campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.mlab import GaussianKDE


from _paths import output_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "results/paper/elote115-paper/trajectories/trajectories.jsonl.gz"
DEFAULT_OUTPUT = output_path("baseline_initialization_vs_qubits.pdf")
EXPECTED_QUBIT_COUNTS = {
    4: 20,
    6: 2,
    8: 24,
    10: 11,
    12: 25,
    14: 9,
    16: 24,
}

BASELINE_GRAY = "#888888"
BASELINE_DARK_GRAY = "#555555"


@dataclass(frozen=True)
class BaselineRecord:
    case_id: str
    n_qubits: int
    n_steps: int
    rotation_tolerance: float
    target_time: float
    lambda_hartree: float
    bound_policy_sha256: str


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_records(path: Path) -> list[BaselineRecord]:
    """Read measured starting policies and require agreement across all five seeds."""
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][path.name]
    seeds = {}
    source_hashes = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "round" or row["iteration"] != 0:
                continue
            assert row["result"]["passes_chemical_accuracy"]
            source_hashes.add(row["candidate_sha256"])
            records = []
            for case, measurement in sorted(row["measurements"].items()):
                policy = measurement["policy"]
                phase = measurement["phase_encoding"]
                assert measurement["status"] == "ok" and measurement["passes_chemical_accuracy"]
                assert policy["formula_order"] == 2
                assert policy["formula_name"] == "symmetric_commuting_group_splitting"
                assert policy["synthesis_policy"] == "uniform"
                assert policy["n_steps"] == measurement["n_steps"]
                assert math.isclose(measurement["target_time"] * phase["energy_scale_hartree"], 1, rel_tol=1e-12)
                epsilon = float(policy["uniform_epsilon"])
                assert math.isfinite(epsilon) and 0 < epsilon < 1
                records.append(BaselineRecord(
                    case_id=case,
                    n_qubits=int(measurement["n_qubits"]),
                    n_steps=int(measurement["n_steps"]),
                    rotation_tolerance=epsilon,
                    target_time=float(measurement["target_time"]),
                    lambda_hartree=float(phase["energy_scale_hartree"]),
                    bound_policy_sha256=row["bound_policy_sha256"][case + ".py"],
                ))
            assert row["seed"] not in seeds
            seeds[row["seed"]] = records
    assert set(seeds) == {f"seed-{i:02d}" for i in range(1, 6)}
    assert len(source_hashes) == 1
    records = seeds["seed-01"]
    assert all(other == records for other in seeds.values()), "Starting parameters differ by seed"
    counts = Counter(record.n_qubits for record in records)
    assert len(records) == 115 and dict(sorted(counts.items())) == EXPECTED_QUBIT_COUNTS
    return records


def quartiles_by_qubit(
    records: list[BaselineRecord],
    *,
    metric: str,
    qubit_counts: list[int],
) -> tuple[list[float], list[float], list[float]]:
    first_quartiles: list[float] = []
    medians: list[float] = []
    third_quartiles: list[float] = []
    for n_qubits in qubit_counts:
        values = np.asarray(
            [getattr(record, metric) for record in records if record.n_qubits == n_qubits],
            dtype=float,
        )
        first_quartile, median, third_quartile = np.quantile(values, [0.25, 0.5, 0.75])
        first_quartiles.append(float(first_quartile))
        medians.append(float(median))
        third_quartiles.append(float(third_quartile))
    return first_quartiles, medians, third_quartiles


def plot(records: list[BaselineRecord], output_path: Path) -> None:
    apply_style()
    qubit_counts = sorted({record.n_qubits for record in records})
    panels = (("n_steps", r"Trotter repetitions, $r$"),
              ("rotation_tolerance", r"Synthesis tolerance, $\epsilon_0$"))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True, layout="constrained")
    for panel_label, ax, (metric, label) in zip(("a", "b"), axes, panels):
        q1, medians, q3 = quartiles_by_qubit(records, metric=metric, qubit_counts=qubit_counts)
        for nq in qubit_counts:
            values = np.asarray([getattr(row, metric) for row in records if row.n_qubits == nq])
            if metric == "n_steps":
                # Offset equal integer counts horizontally without altering r.
                for value in sorted(set(values)):
                    count = int(np.sum(values == value))
                    offsets = np.linspace(-0.52, 0.52, count) if count > 1 else np.zeros(1)
                    ax.scatter(nq + offsets, np.full(count, value), s=23,
                               color=BASELINE_GRAY, alpha=0.8, linewidths=0, zorder=2)
            else:
                transformed = np.log10(values)
                if len(values) > 1 and np.ptp(transformed) > np.finfo(float).eps:
                    grid = np.linspace(transformed.min(), transformed.max(), 256)
                    density = GaussianKDE(transformed, bw_method=0.45)(grid)
                    half_width = 0.62 * density / density.max()
                    ax.fill_betweenx(10**grid, nq - half_width, nq + half_width,
                                     facecolor=BASELINE_GRAY, edgecolor=BASELINE_DARK_GRAY,
                                     linewidth=1.25, alpha=0.34, zorder=2)
        ax.vlines(qubit_counts, q1, q3, color=BASELINE_DARK_GRAY, linewidth=2.5, zorder=3)
        ax.scatter(qubit_counts, medians, marker="D", s=46, facecolor=BASELINE_DARK_GRAY,
                   edgecolor="white", linewidth=0.7, zorder=4)
        if metric == "n_steps":
            ax.set_ylim(0.75, 3.3)
            ax.set_yticks([1, 2, 3])
        else:
            ax.set_yscale("log")
            ax.margins(y=0.10)
        ax.set_xticks(qubit_counts)
        ax.set_xlabel(r"Qubits, $n_q$")
        ax.set_ylabel(label)
        ax.set_xlim(min(qubit_counts) - 0.85, max(qubit_counts) + 0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(rf"\textbf{{({panel_label})}}", loc="left", pad=4, fontsize=18)
    axes[0].legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=5.5,
               color=BASELINE_GRAY, label="Hamiltonians"),
        Line2D([], [], color=BASELINE_DARK_GRAY, linewidth=2.5, marker="D",
               markersize=6, label="Median and IQR"),
    ], loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = load_records(args.data)
    plot(records, args.output)
    provenance = {
        "source": str(args.data.relative_to(REPO_ROOT)) if args.data.is_relative_to(REPO_ROOT) else str(args.data),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "iteration": 0, "cases": len(records), "seeds_verified_identical": 5,
        "profile": "nested_second_order_tight",
        "physical_time": "1 / sum(abs(nonidentity Pauli coefficients))",
        "records": [asdict(row) for row in records],
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.output} from {len(records)} Hamiltonians")


if __name__ == "__main__":
    main()

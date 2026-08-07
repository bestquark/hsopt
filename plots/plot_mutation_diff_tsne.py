#!/usr/bin/env python3
"""Embed archived HSOpt code mutations with t-SNE.

Each resolved proposal is represented by the changed lines in the unified diff
between the proposal-time incumbent and ``candidate.py``.  Character n-gram
TF-IDF features define a cosine-distance matrix; acceptance labels are used
only after embedding to color the scatter plot.

Run with scikit-learn available, for example:

    uv run --with scikit-learn python plots/plot_mutation_diff_tsne.py \
        --history-json /path/to/mutation-history.json \
        --output-dir /path/to/output \
        --basename omni_tight_mutation_diff_tsne
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
    raise SystemExit(
        "scikit-learn and its runtime dependencies are required; run this script with "
        "`uv run --with scikit-learn python plots/plot_mutation_diff_tsne.py ...`. "
        f"Import failed: {exc}"
    ) from exc


@dataclass(frozen=True)
class MutationDiff:
    """One decided mutation and the proposal-time source diff used for embedding."""

    lane: str
    instance: str
    replicate: str
    iteration: int
    incumbent_iteration: int
    status: str
    accepted: bool
    acceptance_reason: str
    n_qubits: int | None
    passes_chemical_accuracy: bool | None
    energy_error_mha: float | None
    previous_score: float | None
    candidate_score: float | None
    added_lines: int
    removed_lines: int
    changed_lines: int
    diff_sha256: str
    diff_text: str
    display_diff: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _history_from_campaign(campaign_root: Path) -> dict[str, Any]:
    from hsopt.view.mutation_history import build_mutation_history

    return build_mutation_history(campaign_root)


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def changed_line_diff(before: str, after: str) -> tuple[str, int, int]:
    """Return an auditable changed-lines-only unified diff.

    File headers, hunk coordinates, and unchanged context are removed so the
    embedding measures code edits rather than lane-independent diff metadata.
    Addition/deletion direction is retained as explicit ``ADD``/``DEL`` tokens.
    """

    changed: list[str] = []
    added = 0
    removed = 0
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="incumbent.py",
        tofile="candidate.py",
        n=0,
        lineterm="",
    )
    for line in lines:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            changed.append(f"ADD {line[1:]}")
            added += 1
        elif line.startswith("-"):
            changed.append(f"DEL {line[1:]}")
            removed += 1
    if not changed:
        changed.append("NO_CHANGE")
    return "\n".join(changed), added, removed


def contextual_diff(
    before: str,
    after: str,
    *,
    incumbent_iteration: int,
    candidate_iteration: int,
) -> str:
    """Return the human-facing unified diff with two context lines."""

    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"incumbent@{incumbent_iteration:04d}.py",
        tofile=f"candidate@{candidate_iteration:04d}.py",
        n=2,
        lineterm="",
    )
    payload = "\n".join(lines)
    return payload or "# No textual change was recorded."


def collect_mutation_diffs(history: dict[str, Any]) -> list[MutationDiff]:
    """Reconstruct every decided proposal against its proposal-time incumbent."""

    lanes = history.get("lanes")
    if not isinstance(lanes, list):
        raise ValueError("mutation history must contain a lanes list")

    records: list[MutationDiff] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id", ""))
        baseline = lane.get("baseline_source")
        if not isinstance(baseline, str):
            raise ValueError(f"lane {lane_id!r} has no baseline source")
        incumbent_source = baseline
        incumbent_iteration = 0
        cells = lane.get("iterations")
        if not isinstance(cells, list):
            raise ValueError(f"lane {lane_id!r} has no iterations list")

        for cell in sorted(
            (item for item in cells if isinstance(item, dict)),
            key=lambda item: int(item.get("iteration", 0)),
        ):
            status = str(cell.get("status", "pending"))
            if status not in {"accepted", "rejected"}:
                continue
            iteration = int(cell["iteration"])
            candidate = cell.get("candidate_source")
            if not isinstance(candidate, str):
                raise ValueError(
                    f"lane {lane_id!r} iteration {iteration} is decided but has no candidate source"
                )
            decision = cell.get("decision") if isinstance(cell.get("decision"), dict) else {}
            result = cell.get("result") if isinstance(cell.get("result"), dict) else {}
            diff_text, added, removed = changed_line_diff(incumbent_source, candidate)
            record = MutationDiff(
                lane=lane_id,
                instance=str(lane.get("instance", "")),
                replicate=str(lane.get("replicate", "")),
                iteration=iteration,
                incumbent_iteration=incumbent_iteration,
                status=status,
                accepted=status == "accepted",
                acceptance_reason=str(decision.get("acceptance_reason", "")),
                n_qubits=_optional_int(result.get("n_qubits")),
                passes_chemical_accuracy=(
                    bool(result.get("passes_chemical_accuracy"))
                    if "passes_chemical_accuracy" in result
                    else None
                ),
                energy_error_mha=_optional_float(
                    result.get(
                        "absolute_energy_error_mha",
                        result.get("contract_error_mha"),
                    )
                ),
                previous_score=_optional_float(decision.get("previous_incumbent_score")),
                candidate_score=_optional_float(decision.get("candidate_score")),
                added_lines=added,
                removed_lines=removed,
                changed_lines=added + removed,
                diff_sha256=hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
                diff_text=diff_text,
                display_diff=contextual_diff(
                    incumbent_source,
                    candidate,
                    incumbent_iteration=incumbent_iteration,
                    candidate_iteration=iteration,
                ),
            )
            records.append(record)
            if record.accepted:
                incumbent_source = candidate
                incumbent_iteration = iteration

    if len(records) < 3:
        raise ValueError("at least three decided mutations are required for t-SNE")
    return records


def embed_diffs(
    records: list[MutationDiff],
    *,
    perplexity: float,
    seed: int,
    max_features: int,
    min_document_frequency: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the embedding, cosine distances, and reproducibility metadata."""

    documents = [record.diff_text for record in records]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        min_df=min_document_frequency,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    try:
        features = vectorizer.fit_transform(documents)
    except ValueError:
        if min_document_frequency == 1:
            raise
        vectorizer.set_params(min_df=1)
        features = vectorizer.fit_transform(documents)
        min_document_frequency = 1
    if features.shape[1] < 2:
        raise ValueError("the archived diffs do not contain enough distinct features")

    distances = cosine_distances(features).astype(np.float32, copy=False)
    distances = np.clip((distances + distances.T) * 0.5, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    effective_perplexity = min(float(perplexity), max(1.0, (len(records) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        metric="precomputed",
        init="random",
        learning_rate="auto",
        max_iter=1_000,
        random_state=seed,
        method="barnes_hut",
        angle=0.5,
    ).fit_transform(distances)

    metadata = {
        "representation": "changed unified-diff lines with ADD/DEL direction",
        "vectorizer": "character n-gram TF-IDF",
        "ngram_range": [3, 5],
        "min_document_frequency": min_document_frequency,
        "maximum_features": max_features,
        "features": int(features.shape[1]),
        "distance_metric": "cosine",
        "tsne_perplexity": effective_perplexity,
        "tsne_iterations": 1_000,
        "random_seed": seed,
    }
    return embedding, distances, metadata


def separation_metrics(
    records: list[MutationDiff],
    distances: np.ndarray,
    embedding: np.ndarray,
    *,
    seed: int,
    permutation_samples: int,
) -> dict[str, float | int | None]:
    """Compute descriptive diagnostics without using labels in the embedding."""

    labels = np.asarray([record.accepted for record in records], dtype=np.int8)
    accepted = int(labels.sum())
    rejected = int(labels.size - accepted)
    metrics: dict[str, float | int | None] = {
        "samples": int(labels.size),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / labels.size,
        "silhouette_cosine_original_space": None,
        "silhouette_euclidean_tsne_space": None,
        "ten_neighbor_same_label_rate": None,
        "ten_neighbor_macro_same_label_rate": None,
        "ten_neighbor_random_label_baseline": None,
        "accepted_neighbor_acceptance_rate": None,
        "accepted_neighbor_random_baseline": None,
        "accepted_neighbor_enrichment_ratio": None,
        "accepted_neighbor_permutation_p_value": None,
        "rejected_neighbor_rejection_rate": None,
        "rejected_neighbor_random_baseline": None,
        "neighbor_permutation_samples": permutation_samples,
        "trustworthiness_ten_neighbors": None,
    }
    if accepted == 0 or rejected == 0:
        return metrics

    metrics["silhouette_cosine_original_space"] = float(
        silhouette_score(distances, labels, metric="precomputed")
    )
    metrics["silhouette_euclidean_tsne_space"] = float(silhouette_score(embedding, labels))
    neighbors = min(10, labels.size - 1)
    neighbor_indices = np.argsort(distances, axis=1)[:, 1 : neighbors + 1]
    same = labels[neighbor_indices] == labels[:, None]
    metrics["ten_neighbor_same_label_rate"] = float(same.mean())
    accepted_same = float(same[labels == 1].mean())
    rejected_same = float(same[labels == 0].mean())
    metrics["ten_neighbor_macro_same_label_rate"] = 0.5 * (accepted_same + rejected_same)
    accepted_rate = accepted / labels.size
    metrics["accepted_neighbor_acceptance_rate"] = accepted_same
    metrics["accepted_neighbor_random_baseline"] = accepted_rate
    metrics["accepted_neighbor_enrichment_ratio"] = accepted_same / accepted_rate
    metrics["rejected_neighbor_rejection_rate"] = rejected_same
    metrics["rejected_neighbor_random_baseline"] = 1.0 - accepted_rate
    metrics["ten_neighbor_random_label_baseline"] = float(
        accepted_rate**2 + (1.0 - accepted_rate) ** 2
    )
    metrics["trustworthiness_ten_neighbors"] = float(
        trustworthiness(distances, embedding, n_neighbors=neighbors, metric="precomputed")
    )
    if permutation_samples:
        random = np.random.default_rng(seed)
        exceedances = 0
        for _ in range(permutation_samples):
            shuffled = random.permutation(labels)
            shuffled_rate = float(shuffled[neighbor_indices][shuffled == 1].mean())
            exceedances += shuffled_rate >= accepted_same
        metrics["accepted_neighbor_permutation_p_value"] = (exceedances + 1) / (
            permutation_samples + 1
        )
    return metrics


def write_records(
    output_dir: Path,
    basename: str,
    records: list[MutationDiff],
    embedding: np.ndarray,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{basename}.csv"
    jsonl_path = output_dir / f"{basename}_diffs.jsonl"
    fields = [
        field for field in asdict(records[0]) if field not in {"diff_text", "display_diff"}
    ] + ["tsne_x", "tsne_y"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, (x, y) in zip(records, embedding, strict=True):
            row = asdict(record)
            row.pop("diff_text")
            row.pop("display_diff")
            row.update({"tsne_x": float(x), "tsne_y": float(y)})
            writer.writerow(row)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record, (x, y) in zip(records, embedding, strict=True):
            row = asdict(record)
            row.update({"tsne_x": float(x), "tsne_y": float(y)})
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return csv_path, jsonl_path


def write_interactive_json(
    output_dir: Path,
    basename: str,
    records: list[MutationDiff],
    embedding: np.ndarray,
    *,
    source_generated_at: object,
    method: dict[str, Any],
    diagnostics: dict[str, float | int | None],
) -> Path:
    """Write the browser payload atomically so dashboard polling is safe."""

    points = []
    for record, (x, y) in zip(records, embedding, strict=True):
        row = asdict(record)
        row.pop("diff_text")
        previous = record.previous_score
        candidate = record.candidate_score
        score_change = (
            previous - candidate if previous is not None and candidate is not None else None
        )
        score_change_percent = (
            100.0 * score_change / previous
            if score_change is not None and previous not in {None, 0.0}
            else None
        )
        row.update(
            {
                "tsne_x": float(x),
                "tsne_y": float(y),
                "score_change": score_change,
                "score_change_percent": score_change_percent,
            }
        )
        points.append(row)
    payload = {
        "schema": 1,
        "generated_at": source_generated_at,
        "labels_used_for_embedding": False,
        "method": method,
        "diagnostics": diagnostics,
        "points": points,
    }
    path = output_dir / f"{basename}.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def plot_embedding(
    records: list[MutationDiff],
    embedding: np.ndarray,
    *,
    output_dir: Path,
    basename: str,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from _paper_plot import GRAPHITE, NVIDIA_GREEN, RED, apply_paper_style

    apply_paper_style(compact=True)
    labels = np.asarray([record.accepted for record in records], dtype=bool)
    accepted = int(labels.sum())
    rejected = int(labels.size - accepted)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), layout="constrained")
    ax.scatter(
        embedding[~labels, 0],
        embedding[~labels, 1],
        s=19,
        c=RED,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        embedding[labels, 0],
        embedding[labels, 1],
        s=30,
        c=NVIDIA_GREEN,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.35,
        rasterized=True,
        zorder=2,
    )
    ax.set_xlabel(r"$t$-SNE 1")
    ax.set_ylabel(r"$t$-SNE 2")
    ax.grid(color="#D8D8D8", linewidth=0.6, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=NVIDIA_GREEN,
                markeredgecolor="white",
                markersize=8,
                label=rf"Accepted ($n={accepted}$)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=RED,
                markeredgecolor="none",
                alpha=0.6,
                markersize=8,
                label=rf"Rejected ($n={rejected}$)",
            ),
        ],
        loc="best",
    )
    ax.tick_params(colors=GRAPHITE)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history-json", type=Path, help="snapshot produced by hsopt-view")
    source.add_argument("--campaign-root", type=Path, help="read a live campaign directly")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="mutation_diff_tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--min-document-frequency", type=int, default=2)
    parser.add_argument("--permutation-samples", type=int, default=1_000)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="write coordinates, diffs, metrics, and browser JSON without rendering figures",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.perplexity <= 0:
        raise ValueError("perplexity must be positive")
    if args.max_features < 2:
        raise ValueError("max-features must be at least two")
    if args.min_document_frequency < 1:
        raise ValueError("min-document-frequency must be positive")
    if args.permutation_samples < 0:
        raise ValueError("permutation-samples cannot be negative")

    if args.history_json is not None:
        history = _read_json(args.history_json.expanduser().resolve())
        source = str(args.history_json.expanduser().resolve())
    else:
        campaign_root = args.campaign_root.expanduser().resolve()
        history = _history_from_campaign(campaign_root)
        source = str(campaign_root)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_mutation_diffs(history)
    embedding, distances, method = embed_diffs(
        records,
        perplexity=args.perplexity,
        seed=args.seed,
        max_features=args.max_features,
        min_document_frequency=args.min_document_frequency,
    )
    diagnostics = separation_metrics(
        records,
        distances,
        embedding,
        seed=args.seed,
        permutation_samples=args.permutation_samples,
    )
    csv_path, jsonl_path = write_records(output_dir, args.basename, records, embedding)
    interactive_path = write_interactive_json(
        output_dir,
        args.basename,
        records,
        embedding,
        source_generated_at=history.get("generated_at"),
        method=method,
        diagnostics=diagnostics,
    )
    pdf_path = None
    png_path = None
    if not args.data_only:
        pdf_path, png_path = plot_embedding(
            records,
            embedding,
            output_dir=output_dir,
            basename=args.basename,
        )
    metrics_path = output_dir / f"{args.basename}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": source,
                "source_generated_at": history.get("generated_at"),
                "labels_used_for_embedding": False,
                "method": method,
                "diagnostics": diagnostics,
                "outputs": {
                    "coordinates_csv": str(csv_path),
                    "diffs_jsonl": str(jsonl_path),
                    "interactive_json": str(interactive_path),
                    "figure_pdf": str(pdf_path) if pdf_path is not None else None,
                    "figure_png": str(png_path) if png_path is not None else None,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "written",
                "samples": len(records),
                "accepted": diagnostics["accepted"],
                "rejected": diagnostics["rejected"],
                "figure": str(png_path) if png_path is not None else None,
                "interactive_json": str(interactive_path),
                "metrics": str(metrics_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

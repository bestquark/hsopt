"""Scientific lane summaries, independent of execution resources."""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
from ..sandbox.mutation_retry import mutation_readiness


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def lane_summary(lane_dirs: list[Path]) -> tuple[dict[str, Any], list[Path]]:
    states = []
    ready = []
    by_family: dict[str, Counter[str]] = {}
    for lane_dir in lane_dirs:
        manifest = _read_json(lane_dir / "manifest.json")
        state = _read_json(lane_dir / "state.json")
        target = int(manifest["iterations_target"])
        completed = int(state["iterations_completed"])
        if completed >= target:
            status = "complete"
        elif state.get("pending_kind") == "mutation":
            status = "mutating"
        elif state.get("pending_kind") == "replay_validation_failed":
            status = "needs_attention"
        elif state.get("pending_task_id") is not None:
            status = "waiting_evaluation"
        elif (readiness := mutation_readiness(lane_dir)) is not None:
            status = str(readiness["status"])
        else:
            status = "ready"
            ready.append(lane_dir)
        states.append((manifest, state, status))
        by_family.setdefault(str(manifest["family"]), Counter())[status] += 1

    ready.sort(
        key=lambda lane_dir: (
            int(_read_json(lane_dir / "state.json")["iterations_completed"]),
            str(_read_json(lane_dir / "state.json").get("updated_at", "")),
            str(lane_dir),
        )
    )
    status_counts = Counter(status for _, _, status in states)
    return (
        {
            "total": len(states),
            "iterations_completed": sum(
                int(state["iterations_completed"]) for _, state, _ in states
            ),
            "iterations_target": sum(
                int(manifest["iterations_target"]) for manifest, _, _ in states
            ),
            "accepted_mutations": sum(
                int(state.get("accepted_mutations", 0)) for _, state, _ in states
            ),
            "by_status": dict(sorted(status_counts.items())),
            "by_family": {
                family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())
            },
        },
        ready,
    )

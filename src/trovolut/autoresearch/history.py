"""Experiment tracking for the autoresearch loop.

Records each optimization iteration's metrics, code changes, and scores
for analysis and leaderboard display.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExperimentRecord:
    """A single experiment result."""

    iteration: int = 0
    combined_score: float = -100.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    all_satisfied: bool = False
    code_hash: str = ""
    timestamp: float = field(default_factory=_time.time)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class History:
    """Tracks all experiments in an autoresearch run.

    Provides persistence (JSON save/load), best-result tracking,
    and leaderboard display.
    """

    def __init__(self):
        self.records: List[ExperimentRecord] = []

    def record(self, rec: ExperimentRecord) -> None:
        """Add an experiment record."""
        self.records.append(rec)

    def best(self) -> Optional[ExperimentRecord]:
        """Return the record with the highest combined_score."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.combined_score)

    def leaderboard(self, top_k: int = 10) -> List[ExperimentRecord]:
        """Return the top-k records by combined_score."""
        sorted_records = sorted(
            self.records, key=lambda r: r.combined_score, reverse=True
        )
        return sorted_records[:top_k]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the history."""
        if not self.records:
            return {"n_experiments": 0}

        best = self.best()
        scores = [r.combined_score for r in self.records]
        satisfied = [r for r in self.records if r.all_satisfied]

        return {
            "n_experiments": len(self.records),
            "n_satisfied": len(satisfied),
            "best_score": best.combined_score if best else None,
            "best_iteration": best.iteration if best else None,
            "mean_score": sum(scores) / len(scores),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [r.to_dict() for r in self.records],
            "summary": self.summary(),
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "History":
        """Load history from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        history = cls()
        for rec_dict in data.get("records", []):
            history.records.append(ExperimentRecord.from_dict(rec_dict))
        return history

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        best = self.best()
        best_str = f", best={best.combined_score:.4f}" if best else ""
        return f"History(n={len(self.records)}{best_str})"


__all__ = [
    "ExperimentRecord",
    "History",
]

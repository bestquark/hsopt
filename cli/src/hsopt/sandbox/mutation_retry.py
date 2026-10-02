"""Durable retry limits for agent failures; independent of scientific acceptance."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MAX_MUTATION_ATTEMPTS = 3


class AgentRuntimeError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, category: str = "inference_api"):
        super().__init__(message)
        self.retryable = retryable
        self.category = category


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def campaign_block_path(lane: Path) -> Path | None:
    manifest = _read(lane / "manifest.json")
    queue = manifest.get("queue_dir")
    return Path(queue).parent / "supervisor" / "mutation_runtime_block.json" if queue else None


def mutation_readiness(lane: Path) -> dict[str, Any] | None:
    block = campaign_block_path(lane)
    if block is not None and block.is_file():
        return {"status": "needs_attention", "mutation_block": str(block), **_read(block)}
    retry = _read(lane / "mutation_retry.json")
    if retry.get("blocked"):
        return {**retry, "status": "needs_attention"}
    if float(retry.get("retry_after_epoch", 0)) > time.time():
        return {**retry, "status": "mutation_cooldown"}
    return None


def record_mutation_failure(lane: Path, exc: Exception, *, iteration: int) -> None:
    if not (lane / "manifest.json").is_file():
        return
    old = _read(lane / "mutation_retry.json")
    count = int(old.get("attempts", 0)) + 1 if old.get("iteration") == iteration else 1
    retryable = bool(getattr(exc, "retryable", True))
    category = str(getattr(exc, "category", "mutation_or_evaluation"))
    blocked = not retryable or count >= MAX_MUTATION_ATTEMPTS
    payload = {
        "schema": 1,
        "iteration": iteration,
        "attempts": count,
        "max_attempts": MAX_MUTATION_ATTEMPTS,
        "blocked": blocked,
        "retryable": retryable,
        "category": category,
        "error": str(exc)[:600],
        "retry_after_epoch": None if blocked else time.time() + 30 * (2 ** (count - 1)),
        "updated_at_epoch": time.time(),
    }
    _write(lane / "mutation_retry.json", payload)
    if not retryable and category == "inference_api":
        block = campaign_block_path(lane)
        if block is not None:
            _write(
                block,
                {
                    **payload,
                    "lane": str(lane),
                    "instruction": "Repair inference configuration, then explicitly resume mutations. Evaluation collection continues.",
                },
            )


def clear_mutation_retry(lane: Path) -> None:
    (lane / "mutation_retry.json").unlink(missing_ok=True)


def resume_mutations(path: Path, *, reason: str) -> dict[str, Any]:
    """Explicit operator recovery; retain the old gate as an audit event."""
    if not reason.strip():
        raise ValueError("a recovery reason is required")
    if (path / "campaign.json").is_file():
        lanes = [path / row["lane_dir"] for row in _read(path / "campaign.json")["lanes"]]
        gate = path / "supervisor" / "mutation_runtime_block.json"
        log = path / "supervisor" / "mutation_recovery.jsonl"
    elif (path / "manifest.json").is_file():
        lanes = [path]
        gate = None
        log = path / "mutation_recovery.jsonl"
    else:
        raise ValueError("expected a campaign root or lane directory")
    paths = [lane / "mutation_retry.json" for lane in lanes]
    if gate is not None:
        paths.append(gate)
    cleared = []
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        for item in paths:
            if item.is_file():
                stream.write(
                    json.dumps(
                        {
                            "at_epoch": time.time(),
                            "reason": reason,
                            "path": str(item),
                            "previous": _read(item),
                        }
                    )
                    + "\n"
                )
                stream.flush()
                item.unlink()
                cleared.append(str(item))
    return {"status": "resumed", "cleared": cleared, "recovery_log": str(log)}

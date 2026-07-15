from __future__ import annotations

import ast
import csv
import copy
import hashlib
import html
import importlib
import json
import math
import os
import re
import shlex
import sqlite3
import stat
import subprocess
import sys
import time
import types
import zipfile
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


HARTREE_PER_KCAL_MOL = 1.0 / 627.5094740631
CHEMICAL_ACCURACY_HARTREE = HARTREE_PER_KCAL_MOL
DEFAULT_CLUSTER_ROOT = os.environ.get("HSOPT_HAMLIB_CLUSTER_ROOT")
DEFAULT_HAMLIB_DATA_ROOT = os.environ.get("HSOPT_HAMLIB_DATA_ROOT")
DEFAULT_CODEX_BIN = "npx -y @openai/codex@0.144.0"
PAPER_MODEL = "gpt-5.5"
PAPER_REASONING_EFFORT = "high"
PAPER_CODEX_CLI_VERSION = "0.144.0"
PAPER_CODEX_EXEC_FLAGS = (
    "exec",
    "--sandbox",
    "workspace-write",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "--json",
)
CAMPAIGN_ID_DOMAIN = b"hsopt-hamlib121-campaign-v1\0"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_HAMLIB_ARCHIVES = 83
CANONICAL_HAMLIB_SOURCE_MANIFEST_BYTES = 72_200
CANONICAL_HAMLIB_SOURCE_MANIFEST_SHA256 = (
    "ff7f9acce5f5140d65f84ebe33df92276c9058d13a8de4a0ed254b7ad868e489"
)
CANONICAL_HAMLIB_ZIP_SHA256_MANIFEST_BYTES = 9_383
CANONICAL_HAMLIB_ZIP_SHA256_MANIFEST_SHA256 = (
    "2a5958a86a6adaa30bd98b325701830f618f7663ddcd925509213ad3ae187def"
)
_VERIFIED_ARCHIVE_CACHE: set[tuple[str, int, int, int, int, int, str]] = set()
_VERIFIED_ARCHIVE_MEMBER_CACHE: set[tuple[tuple[str, int, int, int, int, int, str], str]] = set()
RUNNER_SCRIPT_FILES = (
    "build_hamlib_zip_checksums.py",
    "campaign_common.py",
    "collect_iteration.py",
    "evaluate_worker.py",
    "export_replicate_results.py",
    "finalize_replicates_when_complete.py",
    "mutate_iteration.py",
    "prepare_campaign.py",
    "prebuild_cluster_environments.py",
    "render_telemetry.py",
    "replicate_initial_baseline.py",
    "rescue_pending_eval_partitions.py",
    "submit_eval_batch.py",
    "supervise_streaming_batches.py",
    "verify_hamlib_archives.py",
)
RUNNER_SLURM_FILES = ("evaluate_batch.sbatch", "evaluate_batch_cpu.sbatch")
TOKEN_RE = re.compile(r"([XYZ])(\d+)")
ALLOWED_TERM_ORDER_STRATEGIES = {
    "input",
    "label",
    "coefficient_desc",
    "coefficient_asc",
    "support_then_coeff",
    "support_desc_then_coeff",
    "z_then_mixed",
    "mixed_then_z",
    "z_count_desc",
    "xy_count_desc",
    "pauli_family_then_coeff",
    "coefficient_bands",
    "alternating_large_small",
    "custom",
}
SAFE_POLICY_IMPORTS: dict[str, set[str] | None] = {
    "__future__": {"annotations"},
    "dataclasses": {"asdict", "dataclass", "field", "fields", "replace"},
    "typing": {
        "Any",
        "Callable",
        "Dict",
        "Iterable",
        "List",
        "Mapping",
        "Sequence",
        "Tuple",
    },
    "math": None,
}
FORBIDDEN_POLICY_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "quit",
    "setattr",
    "type",
    "vars",
    "__import__",
}


@dataclass(frozen=True)
class Lane:
    lane_id: str
    index: int
    collection: str
    file: str
    dataset: str
    mapping: str
    active_space_key: str
    molecule: str
    nqubits: int
    electrons: int
    terms: int
    one_norm: float
    exact_fit: bool
    fit_reason: str
    sector_dim: int
    estimated_state_workspace_gib: float
    estimated_sector_workspace_gib: float


@dataclass(frozen=True)
class EvalBatchRecord:
    batch_id: str
    iteration: int
    lane_ids: list[str]
    submission_key: str
    job_name: str
    job_id: str | None
    state: str


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def derive_campaign_id(preparation_nonce: str) -> str:
    if SHA256_RE.fullmatch(preparation_nonce) is None:
        raise ValueError("preparation nonce must be exactly 32 random bytes encoded as hex")
    return hashlib.sha256(CAMPAIGN_ID_DOMAIN + bytes.fromhex(preparation_nonce)).hexdigest()


def validate_absolute_posix_path(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or any(ch in value for ch in ("\n", "\r", "\0")):
        raise ValueError(f"{field} must be a nonempty absolute POSIX path without newline or NUL")
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/"):
        raise ValueError(f"{field} must be an absolute POSIX path below the filesystem root")
    parts = value.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts) or path.as_posix() != value:
        raise ValueError(f"{field} must be a canonical absolute POSIX path")
    return value


def campaign_paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    return {
        "root": root,
        "manifests": root / "manifests",
        "lanes": root / "lanes",
        "iterations": root / "iterations",
        "telemetry": root / "telemetry",
        "logs": root / "logs",
        "slurm": root / "slurm",
        "db": root / "campaign.sqlite",
        "identity": root / "manifests" / "campaign_identity.json",
    }


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = campaign_paths(root)
    for key in ("manifests", "lanes", "iterations", "telemetry", "logs", "slurm"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def db_connect(root: Path) -> sqlite3.Connection:
    paths = ensure_dirs(root)
    conn = sqlite3.connect(paths["db"])
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lanes (
          lane_id TEXT PRIMARY KEY,
          manifest_index INTEGER NOT NULL,
          collection TEXT NOT NULL,
          file TEXT NOT NULL,
          dataset TEXT NOT NULL,
          molecule TEXT NOT NULL,
          nqubits INTEGER NOT NULL,
          electrons INTEGER NOT NULL,
          terms INTEGER NOT NULL,
          one_norm REAL NOT NULL,
          exact_fit INTEGER NOT NULL,
          fit_reason TEXT NOT NULL,
          sector_dim INTEGER NOT NULL,
          estimated_state_workspace_gib REAL NOT NULL,
          estimated_sector_workspace_gib REAL NOT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          iteration INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'prepared',
          best_t_gates_qpe INTEGER,
          best_total_error_hartree REAL,
          best_score REAL,
          best_iteration INTEGER,
          best_policy_path TEXT,
          current_policy_path TEXT,
          current_config_json TEXT,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluations (
          lane_id TEXT NOT NULL,
          iteration INTEGER NOT NULL,
          status TEXT NOT NULL,
          score REAL,
          total_t_gates_qpe INTEGER,
          total_error_hartree REAL,
          qpe_bits INTEGER,
          qpe_unitary_applications INTEGER,
          pf_error_bound_hartree REAL,
          synthesis_error_bound_hartree REAL,
          qpe_grid_spacing_hartree REAL,
          config_json TEXT,
          result_json TEXT,
          created_at TEXT NOT NULL,
          PRIMARY KEY(lane_id, iteration)
        );

        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          kind TEXT NOT NULL,
          lane_id TEXT,
          iteration INTEGER,
          message TEXT NOT NULL,
          payload_json TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_batches (
          batch_id TEXT PRIMARY KEY,
          iteration INTEGER NOT NULL CHECK(iteration >= 0),
          lane_ids_json TEXT NOT NULL,
          submission_key TEXT NOT NULL UNIQUE,
          job_name TEXT NOT NULL UNIQUE,
          job_id TEXT UNIQUE,
          state TEXT NOT NULL CHECK(state IN ('prepared', 'submitted', 'collected')),
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          submitted_at TEXT,
          collected_at TEXT
        );
        """
    )
    lane_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(lanes)")}
    if "pending_iteration" not in lane_columns:
        conn.execute("ALTER TABLE lanes ADD COLUMN pending_iteration INTEGER")
    if "eval_retry_count" not in lane_columns:
        conn.execute("ALTER TABLE lanes ADD COLUMN eval_retry_count INTEGER NOT NULL DEFAULT 0")
    evaluation_columns = {
        "passes_accuracy": "INTEGER",
        "accepted": "INTEGER",
        "incumbent_iteration": "INTEGER",
        "incumbent_t_gates_qpe": "REAL",
        "relative_gain_percent": "REAL",
        "decision_reason": "TEXT",
        "policy_path": "TEXT",
        "mutation_summary": "TEXT",
    }
    existing_columns = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(evaluations)").fetchall()
    }
    for name, declaration in evaluation_columns.items():
        if name not in existing_columns:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {name} {declaration}")
    conn.commit()


def eval_submission_identity(
    *,
    cluster_root: str,
    iteration: int,
    batch_id: str,
    policy_snapshot: str,
    lane_ids: list[str],
    campaign_id: str = "",
    campaign_manifest_sha256: str = "",
    policy_sha256s: list[str] | None = None,
    execution_configuration: dict[str, Any] | None = None,
    runner_provenance: dict[str, Any] | None = None,
) -> tuple[str, str]:
    policy_sha256s = [] if policy_sha256s is None else policy_sha256s
    if policy_sha256s and len(policy_sha256s) != len(lane_ids):
        raise ValueError("policy SHA-256 list must align exactly with lane IDs")
    execution_json = json.dumps(
        execution_configuration or {}, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    runner_json = json.dumps(
        runner_provenance or {}, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    submission_key = hashlib.sha256(
        (
            f"{campaign_id}|{campaign_manifest_sha256}|{cluster_root}|{iteration}|{batch_id}|"
            f"{policy_snapshot}|{execution_json}|{runner_json}|"
            + "\n".join(
                f"{lane_id}:{policy_sha256}"
                for lane_id, policy_sha256 in zip(lane_ids, policy_sha256s, strict=False)
            )
            + ("\n".join(lane_ids) if not policy_sha256s else "")
        ).encode("utf-8")
    ).hexdigest()[:12]
    return submission_key, f"hpf_{submission_key}"


def _eval_batch_record(row: sqlite3.Row) -> EvalBatchRecord:
    try:
        lane_ids_raw = json.loads(str(row["lane_ids_json"]))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"batch {row['batch_id']}: malformed lane_ids_json") from exc
    if (
        not isinstance(lane_ids_raw, list)
        or not lane_ids_raw
        or any(not isinstance(lane_id, str) or not lane_id for lane_id in lane_ids_raw)
        or len(set(lane_ids_raw)) != len(lane_ids_raw)
    ):
        raise RuntimeError(f"batch {row['batch_id']}: lane_ids_json must be a nonempty unique list")
    job_id = None if row["job_id"] is None else str(row["job_id"])
    if job_id is not None and re.fullmatch(r"\d+", job_id) is None:
        raise RuntimeError(f"batch {row['batch_id']}: invalid Slurm job ID {job_id!r}")
    state = str(row["state"])
    if state not in {"prepared", "submitted", "collected"}:
        raise RuntimeError(f"batch {row['batch_id']}: invalid state {state!r}")
    if state == "submitted" and job_id is None:
        raise RuntimeError(f"batch {row['batch_id']}: submitted batch has no job ID")
    return EvalBatchRecord(
        batch_id=str(row["batch_id"]),
        iteration=int(row["iteration"]),
        lane_ids=list(lane_ids_raw),
        submission_key=str(row["submission_key"]),
        job_name=str(row["job_name"]),
        job_id=job_id,
        state=state,
    )


def register_eval_batch(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    iteration: int,
    lane_ids: list[str],
    submission_key: str,
    job_name: str,
) -> EvalBatchRecord:
    if iteration < 0:
        raise ValueError(f"batch iteration must be nonnegative, got {iteration}")
    if not lane_ids or len(set(lane_ids)) != len(lane_ids) or any(not item for item in lane_ids):
        raise ValueError("batch lane IDs must be a nonempty unique list")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id) is None:
        raise ValueError(f"invalid batch ID {batch_id!r}")
    if re.fullmatch(r"[0-9a-f]{12}", submission_key) is None:
        raise ValueError(f"invalid submission key {submission_key!r}")
    if job_name != f"hpf_{submission_key}":
        raise ValueError("job name does not match submission key")

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute(
            "SELECT * FROM eval_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if existing is not None:
            record = _eval_batch_record(existing)
            expected = (iteration, lane_ids, submission_key, job_name)
            actual = (
                record.iteration,
                record.lane_ids,
                record.submission_key,
                record.job_name,
            )
            if actual != expected:
                raise RuntimeError(f"batch {batch_id}: persisted submission intent changed")
            if record.state == "collected":
                raise RuntimeError(f"batch {batch_id}: cannot reopen a collected batch")
            conn.commit()
            return record

        requested = set(lane_ids)
        for row in conn.execute(
            "SELECT * FROM eval_batches WHERE state!='collected' ORDER BY batch_id"
        ).fetchall():
            other = _eval_batch_record(row)
            overlap = requested.intersection(other.lane_ids)
            if overlap:
                raise RuntimeError(
                    f"batch {batch_id}: lanes already belong to open batch {other.batch_id}: "
                    f"{sorted(overlap)}"
                )

        placeholders = ",".join("?" for _ in lane_ids)
        lane_rows = conn.execute(
            f"SELECT lane_id, active, pending_iteration FROM lanes WHERE lane_id IN ({placeholders})",
            lane_ids,
        ).fetchall()
        by_id = {str(row["lane_id"]): row for row in lane_rows}
        missing_or_inactive = [
            lane_id
            for lane_id in lane_ids
            if lane_id not in by_id or not bool(by_id[lane_id]["active"])
        ]
        if missing_or_inactive:
            raise RuntimeError(f"batch {batch_id}: unknown or inactive lanes {missing_or_inactive}")
        pending_conflicts = [
            lane_id
            for lane_id in lane_ids
            if by_id[lane_id]["pending_iteration"] is not None
            and int(by_id[lane_id]["pending_iteration"]) != iteration
        ]
        if pending_conflicts:
            raise RuntimeError(
                f"batch {batch_id}: lanes have a different pending iteration {pending_conflicts}"
            )

        timestamp = now_iso()
        conn.execute(
            """
            INSERT INTO eval_batches(
              batch_id, iteration, lane_ids_json, submission_key, job_name, job_id,
              state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, 'prepared', ?, ?)
            """,
            (
                batch_id,
                iteration,
                json.dumps(lane_ids, separators=(",", ":")),
                submission_key,
                job_name,
                timestamp,
                timestamp,
            ),
        )
        conn.executemany(
            """
            UPDATE lanes
            SET pending_iteration=?, status='submitting', updated_at=?
            WHERE lane_id=?
            """,
            [(iteration, timestamp, lane_id) for lane_id in lane_ids],
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    row = conn.execute("SELECT * FROM eval_batches WHERE batch_id=?", (batch_id,)).fetchone()
    assert row is not None
    return _eval_batch_record(row)


def mark_eval_batch_submitted(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    job_id: str,
) -> EvalBatchRecord:
    if re.fullmatch(r"\d+", job_id) is None:
        raise ValueError(f"invalid Slurm job ID {job_id!r}")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM eval_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"batch {batch_id}: submission intent is missing")
        record = _eval_batch_record(row)
        if record.state == "collected":
            raise RuntimeError(f"batch {batch_id}: cannot submit a collected batch")
        if record.job_id is not None and record.job_id != job_id:
            raise RuntimeError(
                f"batch {batch_id}: Slurm job changed from {record.job_id} to {job_id}"
            )
        timestamp = now_iso()
        conn.execute(
            """
            UPDATE eval_batches
            SET job_id=?, state='submitted', submitted_at=COALESCE(submitted_at, ?), updated_at=?
            WHERE batch_id=?
            """,
            (job_id, timestamp, timestamp, batch_id),
        )
        conn.executemany(
            "UPDATE lanes SET status='evaluating', updated_at=? WHERE lane_id=?",
            [(timestamp, lane_id) for lane_id in record.lane_ids],
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    updated = conn.execute("SELECT * FROM eval_batches WHERE batch_id=?", (batch_id,)).fetchone()
    assert updated is not None
    return _eval_batch_record(updated)


def load_open_eval_batches(conn: sqlite3.Connection) -> list[EvalBatchRecord]:
    records = [
        _eval_batch_record(row)
        for row in conn.execute(
            "SELECT * FROM eval_batches WHERE state!='collected' ORDER BY iteration, batch_id"
        ).fetchall()
    ]
    owners: dict[str, str] = {}
    for record in records:
        for lane_id in record.lane_ids:
            if lane_id in owners:
                raise RuntimeError(
                    f"lane {lane_id} belongs to open batches {owners[lane_id]} and {record.batch_id}"
                )
            owners[lane_id] = record.batch_id
    return records


def validate_result_submission_receipt(
    conn: sqlite3.Connection,
    *,
    lane_id: str,
    iteration: int,
    result: dict[str, Any],
    expected_batch_id: str | None = None,
) -> None:
    """Bind a collected result to the exact durable Slurm submission receipt."""
    if result.get("baseline_replication") is not None:
        if iteration != 0:
            raise RuntimeError(f"{lane_id}/iter {iteration}: invalid replicated baseline result")
        return
    execution = result.get("execution_configuration")
    if not isinstance(execution, dict):
        raise RuntimeError(f"{lane_id}/iter {iteration}: execution identity is missing")
    result_batch_id = execution.get("batch_id")
    if not isinstance(result_batch_id, str) or not result_batch_id:
        raise RuntimeError(f"{lane_id}/iter {iteration}: result batch ID is missing")
    batch_id = expected_batch_id or result_batch_id
    row = conn.execute("SELECT * FROM eval_batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"{lane_id}/iter {iteration}: durable batch {batch_id!r} is missing")
    record = _eval_batch_record(row)
    actual = {
        "batch_id": result_batch_id,
        "iteration": iteration,
        "lane_member": lane_id in record.lane_ids,
        "submission_key": str(execution.get("submission_key", "")),
        "job_name": str(execution.get("job_name", "")),
        "job_id": str(execution.get("slurm_job_id", "")),
    }
    expected = {
        "batch_id": record.batch_id,
        "iteration": record.iteration,
        "lane_member": True,
        "submission_key": record.submission_key,
        "job_name": record.job_name,
        "job_id": record.job_id,
    }
    if record.state not in {"submitted", "collected"} or actual != expected:
        raise RuntimeError(
            f"{lane_id}/iter {iteration}: result does not match durable Slurm receipt: "
            f"actual={actual}, expected={expected}, batch_state={record.state}"
        )


def mark_eval_batch_collected(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    successful_lane_ids: list[str],
    retry_lane_ids: list[str],
) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM eval_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"batch {batch_id}: submission intent is missing")
        record = _eval_batch_record(row)
        if record.state != "submitted":
            raise RuntimeError(
                f"batch {batch_id}: only a submitted batch can be collected, got {record.state}"
            )
        outcomes = successful_lane_ids + retry_lane_ids
        if len(set(outcomes)) != len(outcomes) or set(outcomes) != set(record.lane_ids):
            raise RuntimeError(f"batch {batch_id}: lane outcomes do not match submitted lanes")
        timestamp = now_iso()
        conn.executemany(
            """
            UPDATE lanes
            SET pending_iteration=NULL, eval_retry_count=0, updated_at=?
            WHERE lane_id=?
            """,
            [(timestamp, lane_id) for lane_id in successful_lane_ids],
        )
        conn.executemany(
            """
            UPDATE lanes
            SET pending_iteration=?, status='eval_retry',
                eval_retry_count=eval_retry_count + 1, updated_at=?
            WHERE lane_id=?
            """,
            [(record.iteration, timestamp, lane_id) for lane_id in retry_lane_ids],
        )
        conn.execute(
            """
            UPDATE eval_batches
            SET state='collected', collected_at=COALESCE(collected_at, ?), updated_at=?
            WHERE batch_id=?
            """,
            (timestamp, timestamp, batch_id),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def log_event(
    conn: sqlite3.Connection,
    *,
    kind: str,
    message: str,
    lane_id: str | None = None,
    iteration: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO events(created_at, kind, lane_id, iteration, message, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
        (now_iso(), kind, lane_id, iteration, message, json.dumps(payload or {}, sort_keys=True)),
    )
    conn.commit()


def lane_slug(row: dict[str, str], index: int) -> str:
    stem = Path(row["file"]).stem.replace("_ham", "")
    dataset = row.get("dataset", "").strip("/").replace("/", "_").replace("-", "m")
    raw = f"{index:04d}_{row['collection']}_{row['molecule']}_{stem}_{dataset}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:180]


def read_manifest_rows(path: Path, max_qubits: int = 32) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [row for row in rows if int(float(row["nqubits"])) <= max_qubits]
    rows.sort(key=lambda row: (row["collection"], row["molecule"], row["file"], row["dataset"]))
    return rows


def effective_electrons(row: dict[str, str]) -> int:
    text = row.get("effective_electrons_for_algorithm", "").strip()
    if text and text.lower() != "none":
        return int(float(text))
    return int(float(row["nqubits"])) // 2


def fit_lane(
    row: dict[str, str], index: int, *, memory_gib: float, workspace_factor: float
) -> Lane:
    nqubits = int(float(row["nqubits"]))
    electrons = effective_electrons(row)
    sector_dim = math.comb(nqubits, electrons) if 0 <= electrons <= nqubits else 0
    state_gib = workspace_factor * (2**nqubits) * 16.0 / (1024**3)
    sector_gib = workspace_factor * (sector_dim**2) * 16.0 / (1024**3)
    exact_fit = state_gib <= memory_gib and sector_gib <= memory_gib
    if exact_fit:
        reason = "fits_state_and_sector_dense_estimates"
    elif state_gib > memory_gib and sector_gib > memory_gib:
        reason = "state_and_sector_dense_estimates_exceed_memory"
    elif state_gib > memory_gib:
        reason = "statevector_estimate_exceeds_memory"
    else:
        reason = "sector_dense_baseline_estimate_exceeds_memory"
    return Lane(
        lane_id=lane_slug(row, index),
        index=index,
        collection=row["collection"],
        file=row["file"],
        dataset=row["dataset"],
        mapping=row.get("mapping", ""),
        active_space_key=row.get("active_space_key", ""),
        molecule=row["molecule"],
        nqubits=nqubits,
        electrons=electrons,
        terms=int(float(row["terms"])),
        one_norm=float(row["one_norm"]),
        exact_fit=exact_fit,
        fit_reason=reason,
        sector_dim=sector_dim,
        estimated_state_workspace_gib=state_gib,
        estimated_sector_workspace_gib=sector_gib,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)


def validate_campaign_identity(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = campaign_paths(campaign_root)
    manifest_path = paths["manifests"] / "campaign_manifest.json"
    if not manifest_path.is_file() or not paths["identity"].is_file():
        raise RuntimeError(f"{campaign_root.name}: missing campaign manifest or identity")
    manifest = load_json(manifest_path)
    identity = load_json(paths["identity"])
    expected_keys = {"schema_version", "campaign_id", "campaign_manifest_sha256"}
    if set(identity) != expected_keys or identity.get("schema_version") != 1:
        raise RuntimeError(f"{campaign_root.name}: malformed campaign identity")
    nonce = manifest.get("preparation_nonce")
    campaign_id = manifest.get("campaign_id")
    if (
        not isinstance(nonce, str)
        or not isinstance(campaign_id, str)
        or derive_campaign_id(nonce) != campaign_id
        or identity.get("campaign_id") != campaign_id
        or identity.get("campaign_manifest_sha256") != manifest_sha256(manifest_path)
    ):
        raise RuntimeError(f"{campaign_root.name}: campaign identity or manifest hash drift")
    return manifest, identity


def hamlib_archive_key(lane: Lane | dict[str, Any]) -> str:
    collection = lane.collection if isinstance(lane, Lane) else str(lane["collection"])
    file_name = lane.file if isinstance(lane, Lane) else str(lane["file"])
    return f"{collection}:{file_name.replace('.hdf5', '.zip')}"


def parse_hamlib_zip_checksums(
    path: Path,
    lanes: list[Lane] | list[dict[str, Any]],
) -> list[dict[str, str]]:
    expected = {hamlib_archive_key(lane) for lane in lanes}
    if len(expected) != EXPECTED_HAMLIB_ARCHIVES:
        raise RuntimeError(
            f"canonical active lanes must reference {EXPECTED_HAMLIB_ARCHIVES} ZIP archives, "
            f"found {len(expected)}"
        )
    records: dict[str, dict[str, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n\0]+)", raw)
        if match is None:
            raise RuntimeError(
                f"{path.name}:{line_number}: expected '<sha256><two spaces><relative ZIP path>'"
            )
        digest, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError(f"{path.name}:{line_number}: unsafe/noncanonical relative path")
        if len(relative.parts) < 2 or relative.suffix != ".zip":
            raise RuntimeError(f"{path.name}:{line_number}: expected collection-relative ZIP path")
        key = f"{relative.parts[0]}:{relative.name}"
        if key not in expected:
            raise RuntimeError(f"{path.name}:{line_number}: archive is not in HamLib121: {key}")
        if key in records:
            raise RuntimeError(f"{path.name}:{line_number}: duplicate archive key {key}")
        records[key] = {"archive_key": key, "relative_path": relative_text, "sha256": digest}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        raise RuntimeError(
            f"HamLib checksum manifest must bind exactly {EXPECTED_HAMLIB_ARCHIVES} archives; "
            f"found {len(records)}, missing={missing[:10]}"
        )
    return [records[key] for key in sorted(records)]


def validate_canonical_hamlib_zip_checksum_manifest(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("canonical HamLib ZIP checksum manifest is missing or unsafe")
    size = path.stat(follow_symlinks=False).st_size
    digest = sha256_file(path)
    if (
        size != CANONICAL_HAMLIB_ZIP_SHA256_MANIFEST_BYTES
        or digest != CANONICAL_HAMLIB_ZIP_SHA256_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "canonical HamLib ZIP checksum manifest drift: "
            f"expected {CANONICAL_HAMLIB_ZIP_SHA256_MANIFEST_BYTES} bytes/"
            f"{CANONICAL_HAMLIB_ZIP_SHA256_MANIFEST_SHA256}, found {size} bytes/{digest}"
        )


def validate_canonical_hamlib_source_manifest(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("canonical HamLib source manifest is missing or unsafe")
    size = path.stat(follow_symlinks=False).st_size
    digest = sha256_file(path)
    if (
        size != CANONICAL_HAMLIB_SOURCE_MANIFEST_BYTES
        or digest != CANONICAL_HAMLIB_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "canonical HamLib source manifest drift: "
            f"expected {CANONICAL_HAMLIB_SOURCE_MANIFEST_BYTES} bytes/"
            f"{CANONICAL_HAMLIB_SOURCE_MANIFEST_SHA256}, found {size} bytes/{digest}"
        )


def validate_hamlib_archive(
    *, data_root: Path, lane: Lane | dict[str, Any], archive_records: list[dict[str, str]]
) -> tuple[Path, dict[str, str]]:
    key = hamlib_archive_key(lane)
    matches = [record for record in archive_records if record.get("archive_key") == key]
    if len(matches) != 1:
        raise RuntimeError(f"{key}: expected exactly one bound HamLib archive record")
    record = matches[0]
    digest = str(record.get("sha256", ""))
    relative_text = str(record.get("relative_path", ""))
    if SHA256_RE.fullmatch(digest) is None:
        raise RuntimeError(f"{key}: invalid bound SHA-256")
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{key}: unsafe bound archive path")
    path = data_root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{key}: bound archive is missing, a symlink, or not a regular file")
    metadata = path.stat(follow_symlinks=False)
    cache_key = (
        str(path.resolve()),
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        digest,
    )
    if cache_key not in _VERIFIED_ARCHIVE_CACHE:
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(
                f"{key}: HamLib ZIP SHA-256 mismatch: expected {digest}, found {actual}"
            )
        _VERIFIED_ARCHIVE_CACHE.add(cache_key)
    expected_member = lane.file if isinstance(lane, Lane) else str(lane["file"])
    member_cache_key = (cache_key, expected_member)
    if member_cache_key not in _VERIFIED_ARCHIVE_MEMBER_CACHE:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
        if len(members) != 1 or members[0].is_dir():
            raise RuntimeError(
                f"{key}: expected exactly one regular ZIP member, found {len(members)}"
            )
        member = members[0]
        if (
            PurePosixPath(member.filename).name != expected_member
            or not member.filename.endswith(".hdf5")
            or member.flag_bits & 0x1
            or member.compress_type != zipfile.ZIP_DEFLATED
        ):
            raise RuntimeError(
                f"{key}: ZIP member must be the unencrypted deflated {expected_member} HDF5 file"
            )
        _VERIFIED_ARCHIVE_MEMBER_CACHE.add(member_cache_key)
    return path, {"archive_key": key, "relative_path": relative_text, "sha256": digest}


def archive_record_for_lane(
    manifest: dict[str, Any], lane: Lane | dict[str, Any]
) -> dict[str, str]:
    key = hamlib_archive_key(lane)
    matches = [
        record
        for record in manifest.get("hamlib_archives", [])
        if isinstance(record, dict) and record.get("archive_key") == key
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{key}: campaign manifest archive binding is missing or ambiguous")
    record = matches[0]
    normalized = {
        "archive_key": str(record.get("archive_key", "")),
        "relative_path": str(record.get("relative_path", "")),
        "sha256": str(record.get("sha256", "")),
    }
    if SHA256_RE.fullmatch(normalized["sha256"]) is None:
        raise RuntimeError(f"{key}: campaign manifest archive digest is invalid")
    return normalized


def validate_evaluation_result_provenance(
    *,
    campaign_root: Path,
    lane: Lane | dict[str, Any],
    iteration: int,
    result: dict[str, Any],
    policy_path: Path,
) -> None:
    manifest, identity = validate_campaign_identity(campaign_root)
    runner_provenance = validate_runner_provenance(campaign_root)
    lane_id = lane.lane_id if isinstance(lane, Lane) else str(lane["lane_id"])
    expected_identity = {
        "lane_id": lane_id,
        "iteration": iteration,
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
        "hamlib_archive": archive_record_for_lane(manifest, lane),
        "runner_provenance": runner_provenance,
    }
    drift = {
        key: {"actual": result.get(key), "expected": value}
        for key, value in expected_identity.items()
        if result.get(key) != value
    }
    if drift:
        raise RuntimeError(f"{lane_id}/iter {iteration}: result identity/input drift: {drift}")
    if policy_path.is_symlink() or not policy_path.is_file():
        raise RuntimeError(f"{lane_id}/iter {iteration}: evaluated policy snapshot is unsafe")
    policy_hash = sha256_file(policy_path)
    if result.get("policy_sha256") != policy_hash:
        raise RuntimeError(f"{lane_id}/iter {iteration}: evaluated policy bytes changed")
    initial_policy = campaign_root / "lanes" / lane_id / "initial_simple_pf.py"
    if initial_policy.is_symlink() or not initial_policy.is_file():
        raise RuntimeError(f"{lane_id}: initial policy snapshot is unsafe")
    if result.get("initial_policy_sha256") != sha256_file(initial_policy):
        raise RuntimeError(f"{lane_id}/iter {iteration}: initial policy provenance drift")
    baseline_path = campaign_root / "baselines" / f"{lane_id}.json"
    if baseline_path.is_symlink() or not baseline_path.is_file():
        raise RuntimeError(f"{lane_id}/iter {iteration}: baseline cache is missing or unsafe")
    if result.get("baseline_cache_sha256") != sha256_file(baseline_path):
        raise RuntimeError(f"{lane_id}/iter {iteration}: baseline cache bytes changed")
    baseline = load_json(baseline_path)
    for key in (
        "baseline_schema_version",
        "baseline_status",
        "baseline_backend",
        "baseline_accelerator",
        "hamlib_archive",
        "initial_policy_sha256",
        "gpu_preflight",
        "baseline_execution_configuration",
        "baseline_runner_provenance",
    ):
        if result.get(key) != baseline.get(key):
            raise RuntimeError(f"{lane_id}/iter {iteration}: result/baseline cache {key} drift")
    if (
        baseline.get("baseline_schema_version") != 2
        or baseline.get("baseline_status") != "computed"
        or baseline.get("baseline_backend") != "cupynumeric"
        or baseline.get("baseline_accelerator") != "NVIDIA A100"
        or baseline.get("baseline_runner_provenance") != runner_provenance
        or not isinstance(baseline.get("gpu_preflight"), dict)
        or baseline["gpu_preflight"].get("accelerator_family") != "NVIDIA A100 80GB"
        or baseline["gpu_preflight"].get("allocated_gpu_count") != 1
        or float(baseline["gpu_preflight"].get("gpu_memory_mib", 0.0)) < 80_000
        or not str(baseline["gpu_preflight"].get("gpu_uuid", "")).startswith("GPU-")
        or baseline["gpu_preflight"].get("array_backend_module") not in {"cupynumeric", "cunumeric"}
        or not str(baseline["gpu_preflight"].get("nvidia_smi_path", "")).startswith("/")
        or (
            "cupynumeric"
            not in str(baseline["gpu_preflight"].get("preflight_array_type", "")).lower()
            and "cunumeric"
            not in str(baseline["gpu_preflight"].get("preflight_array_type", "")).lower()
        )
        or (
            "cupynumeric" not in str(baseline.get("baseline_array_type", "")).lower()
            and "cunumeric" not in str(baseline.get("baseline_array_type", "")).lower()
        )
    ):
        raise RuntimeError(f"{lane_id}/iter {iteration}: baseline is not verified A100 evidence")
    execution = result.get("execution_configuration")
    if not isinstance(execution, dict) or execution.get("schema_version") != 1:
        raise RuntimeError(f"{lane_id}/iter {iteration}: execution configuration is missing")
    replicated = result.get("baseline_replication")
    if replicated is not None:
        if iteration != 0 or not isinstance(replicated, dict):
            raise RuntimeError(f"{lane_id}/iter {iteration}: invalid baseline replication marker")
        for field in ("source_campaign_id", "source_manifest_sha256", "source_result_sha256"):
            if SHA256_RE.fullmatch(str(replicated.get(field, ""))) is None:
                raise RuntimeError(f"{lane_id}: malformed baseline replication {field}")
        if baseline.get("baseline_computed_campaign_id") != replicated.get(
            "source_campaign_id"
        ) or baseline.get("baseline_computed_manifest_sha256") != replicated.get(
            "source_manifest_sha256"
        ):
            raise RuntimeError(f"{lane_id}: replicated baseline source binding drift")
        return
    archive_preflight = campaign_root / "manifests" / "hamlib_archive_preflight.json"
    if archive_preflight.is_symlink() or not archive_preflight.is_file():
        raise RuntimeError(f"{lane_id}/iter {iteration}: archive preflight receipt is missing")
    if execution.get("hamlib_archive_preflight_sha256") != sha256_file(archive_preflight):
        raise RuntimeError(f"{lane_id}/iter {iteration}: archive preflight receipt drift")
    expected_execution = (
        ("gpu_baseline", "cupynumeric", True, "initial")
        if iteration == 0
        else ("cpu_policy", "numpy", False, "current")
    )
    actual_execution = (
        execution.get("execution_mode"),
        execution.get("array_backend"),
        execution.get("compute_baseline"),
        execution.get("policy_snapshot"),
    )
    if actual_execution != expected_execution:
        raise RuntimeError(
            f"{lane_id}/iter {iteration}: execution mode drift: "
            f"actual={actual_execution}, expected={expected_execution}"
        )
    batch_id = execution.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise RuntimeError(f"{lane_id}/iter {iteration}: execution batch ID is missing")
    submission_key = str(execution.get("submission_key", ""))
    job_name = str(execution.get("job_name", ""))
    slurm_job_id = str(execution.get("slurm_job_id", ""))
    if (
        re.fullmatch(r"[0-9a-f]{12}", submission_key) is None
        or job_name != f"hpf_{submission_key}"
        or re.fullmatch(r"\d+", slurm_job_id) is None
        or not execution.get("slurm_account")
        or not execution.get("slurm_partition")
    ):
        raise RuntimeError(f"{lane_id}/iter {iteration}: incomplete Slurm execution identity")
    batch_manifest = (
        campaign_root
        / "iterations"
        / f"iter_{iteration:04d}"
        / "batches"
        / batch_id
        / "batch_manifest.json"
    )
    if batch_manifest.is_symlink() or not batch_manifest.is_file():
        raise RuntimeError(f"{lane_id}/iter {iteration}: batch manifest is missing or unsafe")
    if result.get("batch_manifest_sha256") != sha256_file(batch_manifest):
        raise RuntimeError(f"{lane_id}/iter {iteration}: batch manifest bytes changed")
    batch = load_json(batch_manifest)
    batch_execution = batch.get("execution_configuration")
    expected_batch_execution = {
        "execution_mode": execution.get("execution_mode"),
        "array_backend": execution.get("array_backend"),
        "compute_baseline": execution.get("compute_baseline"),
        "policy_snapshot": execution.get("policy_snapshot"),
        "account": execution.get("slurm_account"),
        "partition": execution.get("slurm_partition"),
        "legate_fbmem": execution.get("legate_fbmem"),
        "uv_base_dir": execution.get("uv_base_dir"),
        "uv_cache_dir": execution.get("uv_cache_dir"),
        "uv_python_install_dir": execution.get("uv_python_install_dir"),
        "hamlib_archive_preflight_sha256": execution.get("hamlib_archive_preflight_sha256"),
    }
    if (
        batch.get("campaign_id") != manifest["campaign_id"]
        or batch.get("campaign_manifest_sha256") != identity["campaign_manifest_sha256"]
        or batch.get("iteration") != iteration
        or batch.get("batch_id") != batch_id
        or batch.get("submission_key") != execution.get("submission_key")
        or batch.get("job_name") != execution.get("job_name")
        or batch.get("runner_provenance") != runner_provenance
        or not isinstance(batch.get("policy_sha256s"), dict)
        or batch["policy_sha256s"].get(lane_id) != policy_hash
        or not isinstance(batch_execution, dict)
        or any(batch_execution.get(key) != value for key, value in expected_batch_execution.items())
    ):
        raise RuntimeError(f"{lane_id}/iter {iteration}: batch intent/result provenance drift")


def runner_bundle_paths(runner_root: Path) -> list[Path]:
    relative_paths = [
        Path(".python-version"),
        Path("pyproject.toml"),
        Path("manifests/active_lanes.csv"),
        Path("manifests/hamlib_electronic_active_space_instances_le32_qubits.csv"),
        Path("manifests/hamlib_zip_sha256.txt"),
    ]
    relative_paths.extend(Path("scripts") / name for name in RUNNER_SCRIPT_FILES)
    relative_paths.extend(Path("slurm") / name for name in RUNNER_SLURM_FILES)
    paths = [runner_root / relative for relative in relative_paths]
    missing = [path.relative_to(runner_root).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"runner provenance source files are missing: {missing}")
    for path in paths:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"runner provenance source must be a real regular file: "
                f"{path.relative_to(runner_root).as_posix()}"
            )
    for directory_name, expected_suffix in (("scripts", ".py"), ("slurm", ".sbatch")):
        directory = runner_root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(
                f"runner provenance directory is missing or unsafe: {directory_name}"
            )
        expected = {
            path.relative_to(runner_root).as_posix() for path in paths if path.parent == directory
        }
        actual: set[str] = set()
        for item in directory.iterdir():
            relative = item.relative_to(runner_root).as_posix()
            if item.name == "__pycache__":
                if (runner_root / "manifests" / "campaign_manifest.json").is_file():
                    raise RuntimeError(f"forbidden runner bytecode tree: {relative}")
                continue
            if item.is_symlink() or not item.is_file() or item.suffix != expected_suffix:
                raise RuntimeError(f"undeclared or unsafe runner provenance entry: {relative}")
            actual.add(relative)
        if actual != expected:
            raise RuntimeError(
                f"runner provenance {directory_name} closure changed: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    return paths


def build_runner_provenance(runner_root: Path) -> dict[str, Any]:
    runner_root = runner_root.resolve()
    bundle = hashlib.sha256()
    paths = runner_bundle_paths(runner_root)
    for path in paths:
        relative = path.relative_to(runner_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        bundle.update(len(relative).to_bytes(8, "big"))
        bundle.update(relative)
        bundle.update(len(payload).to_bytes(8, "big"))
        bundle.update(payload)
    lock_path = runner_root / "uv.lock"
    if not lock_path.is_file():
        raise RuntimeError("runner provenance source file is missing: uv.lock")
    return {
        "schema_version": 1,
        "runner_bundle_sha256": bundle.hexdigest(),
        "uv_lock_sha256": sha256_file(lock_path),
    }


def ensure_runner_provenance(*, campaign_root: Path, runner_root: Path) -> dict[str, Any]:
    expected = build_runner_provenance(runner_root)
    path = campaign_root / "manifests" / "runner_provenance.json"
    if path.is_file():
        actual = load_json(path)
        if actual != expected:
            raise RuntimeError("runner provenance changed after campaign preparation")
    else:
        if has_mutation_evidence(campaign_root):
            raise RuntimeError("refusing to backfill missing runner provenance after mutations")
        write_json(path, expected)
    return expected


def validate_runner_provenance(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "manifests" / "runner_provenance.json"
    if not path.is_file():
        raise RuntimeError(f"{campaign_root.name}: missing runner_provenance.json")
    recorded = load_json(path)
    actual = build_runner_provenance(campaign_root)
    if recorded != actual:
        raise RuntimeError(f"{campaign_root.name}: runner or uv.lock differs from provenance")
    return recorded


def parse_codex_cli_version(output: str) -> str:
    match = re.search(
        r"(?:codex-cli\s+|OpenAI\s+Codex\s+v)(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?)",
        output,
    )
    if match is None:
        raise RuntimeError(f"could not parse Codex CLI version from {output.strip()!r}")
    return match.group(1)


def ensure_private_npm_cache(campaign_root: Path) -> Path:
    npm_cache = campaign_root / ".codex_npm"
    try:
        npm_cache.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    metadata = npm_cache.lstat()
    if (
        npm_cache.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise RuntimeError("campaign npm cache must be an owner-only real directory")
    return npm_cache


def probe_codex_cli_version(codex_bin: str, *, npm_cache: Path | None = None) -> str:
    command = shlex.split(codex_bin)
    if not command:
        raise ValueError("Codex command must not be empty")
    env = os.environ.copy()
    if npm_cache is not None:
        env["npm_config_cache"] = str(npm_cache)
        env["npm_config_update_notifier"] = "false"
    completed = subprocess.run(
        [*command, "--version"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Codex version probe failed: {detail}")
    return parse_codex_cli_version("\n".join((completed.stdout, completed.stderr)))


def probe_node_version() -> str:
    completed = subprocess.run(
        ["node", "--version"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Node.js version probe failed: {detail}")
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+)", completed.stdout.strip())
    if match is None:
        raise RuntimeError(f"could not parse Node.js version from {completed.stdout.strip()!r}")
    version = match.group(1)
    if int(version.split(".", 1)[0]) != 24:
        raise RuntimeError(f"paper autoresearch requires Node.js 24.x, found {version}")
    return version


def has_mutation_evidence(campaign_root: Path) -> bool:
    if next(campaign_root.glob("iterations/iter_*/mutations/*/after_simple_pf.py"), None):
        return True
    database = campaign_root / "campaign.sqlite"
    if not database.is_file():
        return False
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        try:
            count = conn.execute("SELECT COUNT(*) FROM evaluations WHERE iteration>0").fetchone()[0]
        except sqlite3.OperationalError:
            return False
    return bool(count)


def ensure_autoresearch_provenance(
    *,
    campaign_root: Path,
    model: str,
    reasoning_effort: str,
    codex_bin: str,
) -> dict[str, Any]:
    runner = validate_runner_provenance(campaign_root)
    path = campaign_root / "manifests" / "autoresearch_provenance.json"
    if not path.is_file() and has_mutation_evidence(campaign_root):
        raise RuntimeError("refusing to backfill missing provenance after mutation evidence exists")
    npm_cache = ensure_private_npm_cache(campaign_root)
    codex_cli_version = probe_codex_cli_version(codex_bin, npm_cache=npm_cache)
    node_version = probe_node_version()
    expected = {
        "schema_version": 1,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "codex_cli_version": codex_cli_version,
        "node_version": node_version,
        "npm_cache": ".codex_npm",
        "codex_exec_flags": list(PAPER_CODEX_EXEC_FLAGS),
        "runner_bundle_sha256": runner["runner_bundle_sha256"],
        "uv_lock_sha256": runner["uv_lock_sha256"],
    }
    if path.is_file():
        actual = load_json(path)
        if actual != expected:
            raise RuntimeError(
                "autoresearch provenance drifted; model, reasoning, Codex CLI, runner, "
                "and lock must remain fixed within a campaign"
            )
    else:
        write_json(path, expected)
    return expected


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def initial_policy_parameters(lane: Lane) -> dict[str, Any]:
    """Return the standard Reiher-style error-budget PF baseline."""
    budget = CHEMICAL_ACCURACY_HARTREE / 3.0
    tau = math.pi / float(lane.one_norm)
    qpe_bits = math.ceil(math.log2((2.0 * float(lane.one_norm)) / budget))
    qpe_unitary_applications = (1 << qpe_bits) - 1
    nonidentity_terms = max(1, int(lane.terms) - 1)

    def candidate(order: int) -> dict[str, Any]:
        if order == 1:
            n_steps = max(1, math.ceil(qpe_unitary_applications * math.pi**2 / (tau * budget)))
            rotations_per_base = max(1, nonidentity_terms * n_steps)
            pf_error_bound_hartree = qpe_unitary_applications * math.pi**2 / (tau * n_steps)
        elif order == 2:
            n_steps = max(
                1, math.ceil(math.sqrt(qpe_unitary_applications * math.pi**3 / (tau * budget)))
            )
            rotations_per_base = max(1, (2 * nonidentity_terms - 1) * n_steps)
            pf_error_bound_hartree = (
                qpe_unitary_applications * math.pi**3 / (tau * n_steps * n_steps)
            )
        else:
            raise ValueError(f"unsupported formula order {order}")
        uniform_epsilon = budget * tau / float(qpe_unitary_applications * rotations_per_base)
        t_per_rotation_bound = 4.0 * math.log2(1.0 / uniform_epsilon) + 11.0
        t_proxy = qpe_unitary_applications * rotations_per_base * t_per_rotation_bound
        return {
            "formula_order": order,
            "n_steps": n_steps,
            "rotations_per_base": rotations_per_base,
            "uniform_epsilon": uniform_epsilon,
            "t_per_rotation_bound": t_per_rotation_bound,
            "deterministic_t_proxy": t_proxy,
            "pf_error_bound_hartree": pf_error_bound_hartree,
        }

    pf1 = candidate(1)
    pf2 = candidate(2)
    chosen = min((pf1, pf2), key=lambda item: float(item["deterministic_t_proxy"]))
    return {
        **chosen,
        "baseline_source": "Reiher2017_equal_error_budget",
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "qpe_error_budget_hartree": budget,
        "pf_error_budget_hartree": budget,
        "synthesis_error_budget_hartree": budget,
        "tau": tau,
        "qpe_bits": qpe_bits,
        "qpe_unitary_applications": qpe_unitary_applications,
        "nonidentity_terms": nonidentity_terms,
        "pf1_n_steps": pf1["n_steps"],
        "pf1_uniform_epsilon": pf1["uniform_epsilon"],
        "pf1_deterministic_t_proxy": pf1["deterministic_t_proxy"],
        "pf2_n_steps": pf2["n_steps"],
        "pf2_uniform_epsilon": pf2["uniform_epsilon"],
        "pf2_deterministic_t_proxy": pf2["deterministic_t_proxy"],
        "term_order_strategy": "coefficient_desc",
    }


def policy_template(lane: Lane) -> str:
    initial = initial_policy_parameters(lane)
    return f'''"""Editable product-formula policy for {lane.lane_id}.

Mutators should edit only this file. The evaluator supplies Hamiltonian-derived
term metadata to choose_synthesis_epsilon: label, coefficient, support_size,
x_count, y_count, and z_count. Keep choose_synthesis_epsilon independent of
step_index so the scaled evaluator can aggregate repeated QPE rotations.

Standard initial policy:
- Reiher-style equal split of chemical accuracy into QPE, product-formula, and
  synthesis budgets;
- order {initial["formula_order"]} product formula selected by a deterministic
  T-count proxy over PF1/PF2;
- {initial["n_steps"]} repetitions per base QPE unitary;
- coefficient-descending deterministic term order;
- uniform synthesis epsilon chosen so qpe_apps * sum(epsilon_j) / tau spends
  the synthesis error budget.

This is a conservative standard baseline. The optimizer may improve it by
redistributing synthesis precision, changing term order, or finding a cheaper
accuracy-passing product-formula schedule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


INITIAL_BASELINE_SOURCE = "{initial["baseline_source"]}"
INITIAL_CHEMICAL_ACCURACY_HARTREE = {initial["chemical_accuracy_hartree"]:.17e}
INITIAL_QPE_ERROR_BUDGET_HARTREE = {initial["qpe_error_budget_hartree"]:.17e}
INITIAL_PF_ERROR_BUDGET_HARTREE = {initial["pf_error_budget_hartree"]:.17e}
INITIAL_SYNTHESIS_ERROR_BUDGET_HARTREE = {initial["synthesis_error_budget_hartree"]:.17e}
INITIAL_TAU = {initial["tau"]:.17e}
INITIAL_QPE_BITS = {initial["qpe_bits"]}
INITIAL_QPE_UNITARY_APPLICATIONS = {initial["qpe_unitary_applications"]}
INITIAL_NONIDENTITY_TERMS = {initial["nonidentity_terms"]}
INITIAL_PF_ERROR_BOUND_HARTREE = {initial["pf_error_bound_hartree"]:.17e}
INITIAL_ROTATIONS_PER_BASE = {initial["rotations_per_base"]}
INITIAL_T_PER_ROTATION_BOUND = {initial["t_per_rotation_bound"]:.17e}
INITIAL_DETERMINISTIC_T_PROXY = {initial["deterministic_t_proxy"]:.17e}
INITIAL_PF1_N_STEPS = {initial["pf1_n_steps"]}
INITIAL_PF1_UNIFORM_EPSILON = {initial["pf1_uniform_epsilon"]:.17e}
INITIAL_PF1_DETERMINISTIC_T_PROXY = {initial["pf1_deterministic_t_proxy"]:.17e}
INITIAL_PF2_N_STEPS = {initial["pf2_n_steps"]}
INITIAL_PF2_UNIFORM_EPSILON = {initial["pf2_uniform_epsilon"]:.17e}
INITIAL_PF2_DETERMINISTIC_T_PROXY = {initial["pf2_deterministic_t_proxy"]:.17e}


@dataclass(frozen=True)
class ProductFormulaConfig:
    name: str = "{lane.lane_id}_pf{initial["formula_order"]}_r{initial["n_steps"]}_reiher_equal_budget"
    formula_order: int = {initial["formula_order"]}
    n_steps: int = {initial["n_steps"]}
    zero_angle_threshold: float = 0.0
    uniform_epsilon: float = {initial["uniform_epsilon"]:.17e}
    term_order_strategy: str = "{initial["term_order_strategy"]}"
    initial_baseline_source: str = INITIAL_BASELINE_SOURCE
    initial_chemical_accuracy_hartree: float = INITIAL_CHEMICAL_ACCURACY_HARTREE
    initial_qpe_error_budget_hartree: float = INITIAL_QPE_ERROR_BUDGET_HARTREE
    initial_pf_error_budget_hartree: float = INITIAL_PF_ERROR_BUDGET_HARTREE
    initial_synthesis_error_budget_hartree: float = INITIAL_SYNTHESIS_ERROR_BUDGET_HARTREE
    initial_tau: float = INITIAL_TAU
    initial_qpe_bits: int = INITIAL_QPE_BITS
    initial_qpe_unitary_applications: int = INITIAL_QPE_UNITARY_APPLICATIONS
    initial_nonidentity_terms: int = INITIAL_NONIDENTITY_TERMS
    initial_pf_error_bound_hartree: float = INITIAL_PF_ERROR_BOUND_HARTREE
    initial_rotations_per_base: int = INITIAL_ROTATIONS_PER_BASE
    initial_t_per_rotation_bound: float = INITIAL_T_PER_ROTATION_BOUND
    initial_deterministic_t_proxy: float = INITIAL_DETERMINISTIC_T_PROXY
    initial_pf1_n_steps: int = INITIAL_PF1_N_STEPS
    initial_pf1_uniform_epsilon: float = INITIAL_PF1_UNIFORM_EPSILON
    initial_pf1_deterministic_t_proxy: float = INITIAL_PF1_DETERMINISTIC_T_PROXY
    initial_pf2_n_steps: int = INITIAL_PF2_N_STEPS
    initial_pf2_uniform_epsilon: float = INITIAL_PF2_UNIFORM_EPSILON
    initial_pf2_deterministic_t_proxy: float = INITIAL_PF2_DETERMINISTIC_T_PROXY


def get_method_config() -> ProductFormulaConfig:
    return ProductFormulaConfig()


def choose_synthesis_epsilon(
    term: Dict[str, object],
    alpha: float,
    step_index: int,
    config: ProductFormulaConfig,
) -> float:
    _ = (term, alpha, step_index)
    return float(config.uniform_epsilon)


def order_product_formula_terms(
    terms: list[Dict[str, object]],
    config: ProductFormulaConfig,
) -> list[Dict[str, object]]:
    """Optional hook used when term_order_strategy == "custom".

    Return the same Hamiltonian terms in a deterministic order. Do not drop,
    duplicate, or alter labels, coefficients, keys, or derived metadata.
    """
    _ = config
    return list(terms)


def describe_policy() -> Dict[str, object]:
    return asdict(get_method_config())
'''


def validate_policy_ast(source: str, *, filename: str) -> ast.Module:
    """Reject policy source with filesystem, process, network, or reflection capabilities."""
    tree = ast.parse(source, filename=filename, mode="exec")
    allowed_top_level = (
        ast.Assign,
        ast.AnnAssign,
        ast.ClassDef,
        ast.Expr,
        ast.FunctionDef,
        ast.Import,
        ast.ImportFrom,
    )
    for statement in tree.body:
        if not isinstance(statement, allowed_top_level):
            raise ValueError(
                f"policy module-level {type(statement).__name__} statements are not allowed"
            )
        if isinstance(statement, ast.Expr) and not (
            isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)
        ):
            raise ValueError("policy module-level expressions must be docstrings")
        if isinstance(statement, ast.FunctionDef) and statement.decorator_list:
            raise ValueError("policy function decorators are not allowed")
        if isinstance(statement, ast.ClassDef):
            if statement.bases or statement.keywords:
                raise ValueError("policy classes cannot inherit or declare metaclasses")
            for decorator in statement.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not isinstance(target, ast.Name) or target.id != "dataclass":
                    raise ValueError("only the dataclass policy class decorator is allowed")

    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            raise ValueError("async policy code is not allowed")
        if isinstance(
            node, (ast.Global, ast.Nonlocal, ast.Delete, ast.While, ast.Yield, ast.YieldFrom)
        ):
            raise ValueError(f"policy {type(node).__name__} statements are not allowed")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_POLICY_NAMES or node.id.startswith("__"):
                raise ValueError(f"policy name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"private/dunder policy attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in SAFE_POLICY_IMPORTS:
                    raise ValueError(f"policy import {alias.name!r} is not allowed")
                if alias.name != "math":
                    raise ValueError(
                        f"policy module import {alias.name!r} is not allowed; "
                        "import only explicit names"
                    )
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module not in SAFE_POLICY_IMPORTS:
                raise ValueError(f"policy import from {node.module!r} is not allowed")
            allowed_names = SAFE_POLICY_IMPORTS[node.module]
            for alias in node.names:
                if (
                    alias.name == "*"
                    or alias.name.startswith("_")
                    or (allowed_names is not None and alias.name not in allowed_names)
                ):
                    raise ValueError(f"policy import {node.module}.{alias.name} is not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_POLICY_NAMES or node.func.id.startswith("__"):
                raise ValueError(f"policy call {node.func.id!r} is not allowed")
            if node.func.id == "range":
                for argument in node.args:
                    if (
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, int)
                        and abs(argument.value) > 100_000
                    ):
                        raise ValueError(
                            "policy range literal exceeds the deterministic work limit"
                        )
    return tree


def _safe_policy_import(
    name: str,
    globals_dict: dict[str, Any] | None = None,
    locals_dict: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] | list[str] = (),
    level: int = 0,
):
    del globals_dict, locals_dict
    if level != 0 or name not in SAFE_POLICY_IMPORTS:
        raise ImportError(f"policy import {name!r} is not allowed")
    if not fromlist and name != "math":
        raise ImportError(f"policy module import {name!r} is not allowed")
    allowed_names = SAFE_POLICY_IMPORTS[name]
    if allowed_names is not None and any(item not in allowed_names for item in fromlist):
        raise ImportError(f"policy import list from {name!r} is not allowed")
    return importlib.import_module(name)


def load_policy_module(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ImportError(f"policy must be a real regular file: {path}")
    source = path.read_text(encoding="utf-8")
    tree = validate_policy_ast(source, filename=str(path))
    module_name = f"_hsopt_policy_{hashlib.sha256(source.encode('utf-8')).hexdigest()}"
    module = types.ModuleType(module_name)
    safe_builtins = {
        "Exception": Exception,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "object": object,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__build_class__": __build_class__,
        "__import__": _safe_policy_import,
    }
    module.__dict__.update(
        {
            "__builtins__": safe_builtins,
            "__file__": str(path),
            "__package__": None,
        }
    )
    code = compile(tree, str(path), "exec")
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__, module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    return module


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return {key: getattr(config, key) for key in config.__dataclass_fields__}
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError(f"unsupported config object: {type(config)!r}")


def locate_zip(data_root: Path, lane: Lane | dict[str, Any]) -> Path:
    collection = lane.collection if isinstance(lane, Lane) else lane["collection"]
    file_name = lane.file if isinstance(lane, Lane) else lane["file"]
    zip_name = file_name.replace(".hdf5", ".zip")
    if collection == "hydrogen_data":
        matches = list((data_root / "hydrogen_data").rglob(zip_name))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one match for {zip_name}, found {len(matches)}")
        return matches[0]
    return data_root / collection / zip_name


def read_hamlib_hdf5_text_inline(zip_path: Path, dataset: str) -> tuple[str, dict[str, Any]]:
    import h5py

    with zipfile.ZipFile(zip_path) as archive:
        hdf5_names = [name for name in archive.namelist() if name.endswith(".hdf5")]
        if len(hdf5_names) != 1:
            raise ValueError(f"expected one HDF5 file in {zip_path}, found {hdf5_names}")
        payload = archive.read(hdf5_names[0])
    with h5py.File(io.BytesIO(payload), "r") as handle:
        ds = handle[dataset]
        attrs = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in ds.attrs.items()
        }
        text = ds[()].decode("utf-8")
    return text, attrs


def read_hamlib_hdf5_text_subprocess(zip_path: Path, dataset: str) -> tuple[str, dict[str, Any]]:
    venv = os.environ.get("VIRTUAL_ENV")
    python_bin = Path(venv) / "bin" / "python" if venv else Path(sys.executable)
    code = r"""
import io
import json
import sys
import zipfile

import h5py


def clean(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


zip_path = sys.argv[1]
dataset = sys.argv[2]
with zipfile.ZipFile(zip_path) as archive:
    hdf5_names = [name for name in archive.namelist() if name.endswith(".hdf5")]
    if len(hdf5_names) != 1:
        raise ValueError(f"expected one HDF5 file in {zip_path}, found {hdf5_names}")
    payload = archive.read(hdf5_names[0])
with h5py.File(io.BytesIO(payload), "r") as handle:
    ds = handle[dataset]
    attrs = {key: clean(value) for key, value in ds.attrs.items()}
    text = ds[()].decode("utf-8")
print(json.dumps({"text": text, "attrs": attrs}))
"""
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    completed = subprocess.run(
        [str(python_bin), "-c", code, str(zip_path), dataset],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    payload = json.loads(completed.stdout)
    return str(payload["text"]), dict(payload["attrs"])


def read_hamlib_hdf5_text(zip_path: Path, dataset: str) -> tuple[str, dict[str, Any]]:
    if os.environ.get("CAMPAIGN_HDF5_SUBPROCESS") == "1":
        return read_hamlib_hdf5_text_subprocess(zip_path, dataset)
    try:
        return read_hamlib_hdf5_text_inline(zip_path, dataset)
    except Exception:
        return read_hamlib_hdf5_text_subprocess(zip_path, dataset)


def parse_hamlib_terms(
    zip_path: Path, dataset: str, nqubits: int, imag_tol: float = 1e-6
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    text, attrs = read_hamlib_hdf5_text(zip_path, dataset)

    identity = 0.0
    terms: list[dict[str, Any]] = []
    max_imag = 0.0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("+"):
            line = line[:-1].strip()
        coeff_text, pauli_text = line.split(" [", 1)
        coeff = complex(coeff_text.strip().strip("()"))
        max_imag = max(max_imag, abs(coeff.imag))
        if abs(coeff.imag) > imag_tol:
            raise ValueError(
                f"imaginary HamLib coefficient {coeff.imag:.3e} exceeds {imag_tol:.3e}"
            )
        pauli_body = pauli_text.rstrip("]").strip()
        if not pauli_body:
            identity += float(coeff.real)
            continue
        label = ["I"] * nqubits
        for pauli, index_text in TOKEN_RE.findall(pauli_body):
            label[int(index_text)] = pauli
        label_text = "".join(label)
        coefficient = float(coeff.real)
        if abs(coefficient) <= 1e-15:
            continue
        terms.append(
            {
                "label": label_text,
                "coefficient": coefficient,
                "support_size": sum(ch != "I" for ch in label_text),
                "x_count": label_text.count("X"),
                "y_count": label_text.count("Y"),
                "z_count": label_text.count("Z"),
            }
        )
    terms.sort(key=lambda item: item["label"])
    return identity, terms, {"hdf5_attrs": attrs, "max_abs_imag_coeff": max_imag}


def term_family(term: dict[str, Any]) -> tuple[int, int, int]:
    x_count = int(term["x_count"])
    y_count = int(term["y_count"])
    z_count = int(term["z_count"])
    xy_count = x_count + y_count
    if xy_count == 0:
        family = 0
    elif z_count == 0:
        family = 1
    else:
        family = 2
    return family, xy_count, z_count


def validate_custom_order(original: list[dict[str, Any]], ordered: Any) -> list[dict[str, Any]]:
    if not isinstance(ordered, (list, tuple)):
        raise ValueError("order_product_formula_terms must return a list or tuple")
    ordered_list = list(ordered)
    if len(ordered_list) != len(original):
        raise ValueError(
            f"order_product_formula_terms returned {len(ordered_list)} terms; expected {len(original)}"
        )
    for term in ordered_list:
        if not isinstance(term, dict):
            raise ValueError("order_product_formula_terms must return term dictionaries")
    try:
        original_terms = Counter(
            json.dumps(term, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for term in original
        )
        ordered_terms = Counter(
            json.dumps(term, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for term in ordered_list
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "order_product_formula_terms must return JSON-serializable Hamiltonian term dictionaries"
        ) from exc
    if ordered_terms != original_terms:
        raise ValueError(
            "order_product_formula_terms must preserve the exact Hamiltonian term dictionaries; "
            "only their order may change"
        )
    return ordered_list


def order_terms(
    terms: list[dict[str, Any]],
    strategy: str,
    policy_module: Any | None = None,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    if strategy == "custom":
        hook = getattr(policy_module, "order_product_formula_terms", None)
        if not callable(hook):
            raise ValueError(
                'term_order_strategy="custom" requires order_product_formula_terms(terms, config)'
            )
        immutable_original = copy.deepcopy(terms)
        hook_input = copy.deepcopy(terms)
        validated = validate_custom_order(immutable_original, hook(hook_input, config))
        return copy.deepcopy(validated)
    if strategy == "input":
        return list(terms)
    if strategy == "label":
        return sorted(terms, key=lambda t: t["label"])
    if strategy == "coefficient_desc":
        return sorted(terms, key=lambda t: (-abs(float(t["coefficient"])), t["label"]))
    if strategy == "coefficient_asc":
        return sorted(terms, key=lambda t: (abs(float(t["coefficient"])), t["label"]))
    if strategy == "support_then_coeff":
        return sorted(
            terms, key=lambda t: (int(t["support_size"]), -abs(float(t["coefficient"])), t["label"])
        )
    if strategy == "support_desc_then_coeff":
        return sorted(
            terms,
            key=lambda t: (-int(t["support_size"]), -abs(float(t["coefficient"])), t["label"]),
        )
    if strategy == "z_then_mixed":
        return sorted(
            terms,
            key=lambda t: (
                (t["x_count"] + t["y_count"]) != 0,
                -t["z_count"],
                -abs(float(t["coefficient"])),
                t["label"],
            ),
        )
    if strategy == "mixed_then_z":
        return sorted(
            terms,
            key=lambda t: (
                (t["x_count"] + t["y_count"]) == 0,
                -(t["x_count"] + t["y_count"]),
                -abs(float(t["coefficient"])),
                t["label"],
            ),
        )
    if strategy == "z_count_desc":
        return sorted(
            terms,
            key=lambda t: (
                -int(t["z_count"]),
                int(t["x_count"]) + int(t["y_count"]),
                -abs(float(t["coefficient"])),
                t["label"],
            ),
        )
    if strategy == "xy_count_desc":
        return sorted(
            terms,
            key=lambda t: (
                -(int(t["x_count"]) + int(t["y_count"])),
                -abs(float(t["coefficient"])),
                t["label"],
            ),
        )
    if strategy == "pauli_family_then_coeff":
        return sorted(
            terms, key=lambda t: (*term_family(t), -abs(float(t["coefficient"])), t["label"])
        )
    if strategy == "coefficient_bands":

        def band_key(term: dict[str, Any]) -> tuple[float, int, int, int, str]:
            coefficient = abs(float(term["coefficient"]))
            band = math.floor(math.log10(coefficient)) if coefficient > 0.0 else -99
            family, xy_count, z_count = term_family(term)
            return (
                band,
                family,
                -int(term["support_size"]),
                -(xy_count + z_count),
                str(term["label"]),
            )

        return sorted(terms, key=band_key)
    if strategy == "alternating_large_small":
        ordered = sorted(terms, key=lambda t: (abs(float(t["coefficient"])), t["label"]))
        out: list[dict[str, Any]] = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            out.append(ordered[right])
            right -= 1
            if left <= right:
                out.append(ordered[left])
                left += 1
        return out
    raise ValueError(f"unsupported term_order_strategy {strategy!r}")


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                command, cwd=cwd, env=merged_env, stdout=handle, stderr=subprocess.STDOUT, text=True
            )
            return int(proc.returncode)
    proc = subprocess.run(command, cwd=cwd, env=merged_env, text=True)
    return int(proc.returncode)


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))

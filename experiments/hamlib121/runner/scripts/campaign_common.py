from __future__ import annotations

import csv
import copy
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import zipfile
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HARTREE_PER_KCAL_MOL = 1.0 / 627.5094740631
CHEMICAL_ACCURACY_HARTREE = HARTREE_PER_KCAL_MOL
DEFAULT_CLUSTER_ROOT = os.environ.get("HSOPT_HAMLIB_CLUSTER_ROOT")
DEFAULT_HAMLIB_DATA_ROOT = os.environ.get("HSOPT_HAMLIB_DATA_ROOT")
DEFAULT_CODEX_BIN = "npx -y @openai/codex@0.144.0"
PAPER_MODEL = "gpt-5.5"
PAPER_REASONING_EFFORT = "high"
PAPER_CODEX_CLI_VERSION = "0.144.0"
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
) -> tuple[str, str]:
    submission_key = hashlib.sha256(
        (f"{cluster_root}|{iteration}|{batch_id}|{policy_snapshot}|" + "\n".join(lane_ids)).encode(
            "utf-8"
        )
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


def runner_bundle_paths(runner_root: Path) -> list[Path]:
    relative_paths = [
        Path(".python-version"),
        Path("pyproject.toml"),
        Path("manifests/active_lanes.csv"),
        Path("manifests/hamlib_electronic_active_space_instances_le32_qubits.csv"),
    ]
    relative_paths.extend(
        path.relative_to(runner_root) for path in sorted((runner_root / "scripts").glob("*.py"))
    )
    relative_paths.extend(
        path.relative_to(runner_root) for path in sorted((runner_root / "slurm").glob("*.sbatch"))
    )
    paths = [runner_root / relative for relative in relative_paths]
    missing = [path.relative_to(runner_root).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"runner provenance source files are missing: {missing}")
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


def probe_codex_cli_version(codex_bin: str) -> str:
    command = shlex.split(codex_bin)
    if not command:
        raise ValueError("Codex command must not be empty")
    completed = subprocess.run(
        [*command, "--version"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Codex version probe failed: {detail}")
    return parse_codex_cli_version("\n".join((completed.stdout, completed.stderr)))


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
    codex_cli_version = probe_codex_cli_version(codex_bin)
    expected = {
        "schema_version": 1,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "codex_cli_version": codex_cli_version,
        "runner_bundle_sha256": runner["runner_bundle_sha256"],
        "uv_lock_sha256": runner["uv_lock_sha256"],
    }
    path = campaign_root / "manifests" / "autoresearch_provenance.json"
    if path.is_file():
        actual = load_json(path)
        if actual != expected:
            raise RuntimeError(
                "autoresearch provenance drifted; model, reasoning, Codex CLI, runner, "
                "and lock must remain fixed within a campaign"
            )
    else:
        if has_mutation_evidence(campaign_root):
            raise RuntimeError(
                "refusing to backfill missing provenance after mutation evidence exists"
            )
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


def load_policy_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"policy_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import policy {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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

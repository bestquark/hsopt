#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-runner-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import (  # noqa: E402
    SHA256_RE,
    campaign_lock,
    candidate_path,
    expected_lane_ids,
    ensure_campaign_runner_runtime,
    exact_flat_regular_file_names,
    iteration_dir,
    mutation_dir,
    now_iso,
    read_json,
    require_campaign,
    sha256_file,
    validate_absolute_posix_path,
    validate_iteration,
    write_json,
    write_state,
)


REMOTE_CLAIM_SCHEMA_VERSION = 1
REMOTE_CLAIM_NAME = ".hsopt-campaign-claim.json"
REMOTE_CLAIM_LOCK_NAME = ".hsopt-campaign-claim.lock"


def _campaign_identity(manifest: dict[str, Any], state: dict[str, Any]) -> tuple[str, str]:
    campaign_id = str(manifest.get("campaign_id", ""))
    manifest_hash = str(state.get("manifest_sha256", ""))
    if SHA256_RE.fullmatch(campaign_id) is None or SHA256_RE.fullmatch(manifest_hash) is None:
        raise RuntimeError("campaign ID or manifest hash is not a SHA-256 digest")
    return campaign_id, manifest_hash


def _remote_claim_payload(*, campaign_id: str, manifest_sha256: str) -> str:
    if SHA256_RE.fullmatch(campaign_id) is None:
        raise ValueError("campaign_id must be a SHA-256 digest")
    if SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("manifest_sha256 must be a SHA-256 digest")
    return (
        json.dumps(
            {
                "campaign_id": campaign_id,
                "manifest_sha256": manifest_sha256,
                "schema_version": REMOTE_CLAIM_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _remote_claim_command(*, cluster_root: str, campaign_id: str, manifest_sha256: str) -> str:
    """Return a lock-protected, idempotent first-claim command for a remote root."""
    validate_absolute_posix_path(cluster_root, "cluster_root")
    payload = _remote_claim_payload(campaign_id=campaign_id, manifest_sha256=manifest_sha256)
    root = cluster_root.rstrip("/") or "/"
    claim = f"{root}/{REMOTE_CLAIM_NAME}" if root != "/" else f"/{REMOTE_CLAIM_NAME}"
    lock = f"{root}/{REMOTE_CLAIM_LOCK_NAME}" if root != "/" else f"/{REMOTE_CLAIM_LOCK_NAME}"
    return (
        "set -eu; umask 077; "
        f"root={shlex.quote(root)}; claim={shlex.quote(claim)}; lock={shlex.quote(lock)}; "
        'if [ -L "$root" ]; then echo "campaign root is a symlink" >&2; exit 41; fi; '
        'if [ -e "$root" ] && [ ! -d "$root" ]; then '
        'echo "campaign root is not a directory" >&2; exit 42; fi; '
        'mkdir -p "$root"; '
        'if [ -L "$lock" ] || { [ -e "$lock" ] && [ ! -f "$lock" ]; }; then '
        'echo "campaign claim lock is unsafe" >&2; exit 47; fi; '
        'exec 9>"$lock"; flock -x 9; '
        'if [ -L "$claim" ]; then echo "campaign claim is a symlink" >&2; exit 43; fi; '
        'tmp="$root/.hsopt-campaign-claim.tmp.$$"; trap \'rm -f "$tmp"\' EXIT HUP INT TERM; '
        f'if ! (set -C; printf %s {shlex.quote(payload)} >"$tmp"); then '
        'echo "campaign claim temp path is unsafe" >&2; exit 48; fi; '
        'if [ -e "$claim" ]; then '
        'if [ ! -f "$claim" ] || ! cmp -s "$tmp" "$claim"; then '
        'echo "remote campaign claim mismatch" >&2; exit 44; fi; '
        "else "
        'if [ -e "$root/.venv" ] && '
        '{ [ ! -d "$root/.venv" ] || [ -L "$root/.venv" ]; }; then '
        'echo "prebuilt .venv is not a real directory" >&2; exit 45; fi; '
        'stale=$(find "$root" -mindepth 1 -maxdepth 1 '
        '! -name "$' + "{lock##*/}" + '" ! -name "$' + "{tmp##*/}" + '" '
        "! -name '.venv' -print -quit); "
        'if [ -n "$stale" ]; then '
        'echo "refusing to claim non-empty unclaimed campaign root: $stale" >&2; exit 46; fi; '
        'mv "$tmp" "$claim"; '
        "fi; "
        'cmp -s "$tmp" "$claim" 2>/dev/null || [ ! -e "$tmp" ]'
    )


def claim_remote_campaign(
    *,
    sync_host: str,
    cluster_root: str,
    campaign_id: str,
    manifest_sha256: str,
    attempts: int,
    timeout_seconds: int,
) -> None:
    run_with_retries(
        [
            "ssh",
            sync_host,
            _remote_claim_command(
                cluster_root=cluster_root,
                campaign_id=campaign_id,
                manifest_sha256=manifest_sha256,
            ),
        ],
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )


def run_with_retries(
    command: list[str],
    *,
    attempts: int,
    timeout_seconds: int,
    retry_delay_seconds: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"attempt {attempt}: timeout after {timeout_seconds}s")
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            failures.append(f"attempt {attempt}: exit {exc.returncode}: {detail}")
        if attempt < attempts:
            time.sleep(retry_delay_seconds * attempt)
    raise RuntimeError(f"command failed after {attempts} attempts: {'; '.join(failures)}")


def _export_safe(value: str, field: str) -> str:
    if not value or any(character in value for character in (",", "\n", "\r", "\x00")):
        raise ValueError(f"{field} cannot be empty or contain comma/newline/NUL")
    return value


def _batch_identity(
    *,
    campaign_id: str,
    manifest_sha256: str,
    cluster_root: str,
    iteration: int,
    batch_id: str,
    attempt: int,
    lane_ids: list[str],
    root: Path,
) -> tuple[str, str]:
    identity = {
        "campaign_id": campaign_id,
        "manifest_sha256": manifest_sha256,
        "cluster_root": cluster_root,
        "iteration": iteration,
        "batch_id": batch_id,
        "attempt": attempt,
        "candidates": [
            {
                "lane_id": lane_id,
                "policy_sha256": sha256_file(candidate_path(root, iteration, lane_id)),
            }
            for lane_id in lane_ids
        ],
    }
    digest = hashlib.sha256(
        b"hsopt-reactions10-submission-v1\0"
        + json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, f"rx_{digest[:32]}"


def _write_batch_mirror(root: Path, batch: dict[str, Any]) -> None:
    path = (
        iteration_dir(root, int(batch["iteration"]))
        / "batches"
        / str(batch["batch_id"])
        / "batch.json"
    )
    write_json(path, batch)


def _validate_candidates(root: Path, manifest: dict[str, Any], iteration: int) -> None:
    for lane_id in expected_lane_ids(manifest):
        candidate = candidate_path(root, iteration, lane_id)
        record_path = mutation_dir(root, iteration, lane_id) / "record.json"
        if not candidate.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"immutable candidate/record missing for {lane_id}")
        if exact_flat_regular_file_names(candidate.parent, "immutable candidate directory") != [
            "simple_pf.py"
        ]:
            raise RuntimeError(f"candidate directory is not exact for {lane_id}")
        record = read_json(record_path)
        if (
            record.get("lane_id") != lane_id
            or record.get("iteration") != iteration
            or record.get("status") != "candidate_ready"
            or record.get("candidate_policy_sha256") != sha256_file(candidate)
        ):
            raise RuntimeError(f"candidate/record drift for {lane_id}")


def validate_hamiltonian_verification(
    root: Path, manifest: dict[str, Any], state: dict[str, Any]
) -> None:
    path = root / "baselines" / "hamiltonian_verification.json"
    if not path.is_file():
        raise RuntimeError(
            "run runner/verify_hamiltonians.py successfully before submitting evaluations"
        )
    payload = read_json(path)
    rows = payload.get("rows")
    expected = manifest["inputs"]["problem_cache_files"]
    if (
        payload.get("all_verified") is not True
        or payload.get("hamiltonian_count") != 21
        or payload.get("manifest_sha256") != state["manifest_sha256"]
        or not isinstance(rows, list)
        or len(rows) != 21
        or state.get("hamiltonian_verification_sha256") != sha256_file(path)
    ):
        raise RuntimeError("Hamiltonian verification report is incomplete or stale")
    seen: set[str] = set()
    for row in rows:
        npz_name = str(row.get("cache_file", ""))
        json_name = str(Path(npz_name).with_suffix(".json"))
        numeric_fields = (
            "ground_energy_delta_hartree",
            "max_spectrum_delta_hartree",
            "cached_ground_vector_residual",
            "evaluator_ground_state_outside_sector_norm",
            "evaluator_ground_state_residual",
            "evaluator_recomputed_ground_eigenspace_overlap",
            "evaluator_cached_ground_eigenspace_overlap",
            "pauli_dense_max_abs_residual_hartree",
            "pauli_dense_relative_frobenius_residual",
        )
        try:
            metrics = {field: float(row.get(field, float("nan"))) for field in numeric_fields}
        except (TypeError, ValueError):
            metrics = {field: float("nan") for field in numeric_fields}
        if (
            npz_name in seen
            or row.get("verified") is not True
            or row.get("cache_sha256") != expected.get(npz_name)
            or row.get("metadata_sha256") != expected.get(json_name)
            or not all(math.isfinite(value) for value in metrics.values())
            or abs(metrics["ground_energy_delta_hartree"]) > 1.0e-10
            or metrics["max_spectrum_delta_hartree"] > 1.0e-10
            or metrics["cached_ground_vector_residual"] > 1.0e-8
            or metrics["evaluator_ground_state_outside_sector_norm"] > 1.0e-10
            or metrics["evaluator_ground_state_residual"] > 1.0e-8
            or metrics["evaluator_recomputed_ground_eigenspace_overlap"] < 1.0 - 1.0e-8
            or metrics["evaluator_cached_ground_eigenspace_overlap"] < 1.0 - 1.0e-8
            or metrics["pauli_dense_max_abs_residual_hartree"] > 1.0e-10
            or metrics["pauli_dense_relative_frobenius_residual"] > 1.0e-10
        ):
            raise RuntimeError("Hamiltonian verification row/hash mismatch")
        seen.add(npz_name)
    if seen != {name for name in expected if name.endswith(".npz")}:
        raise RuntimeError("Hamiltonian verification does not cover the fixed cache")


def _initial_batches(lane_ids: list[str], batch_size: int) -> list[list[str]]:
    batch_count = max(1, math.ceil(len(lane_ids) / batch_size))
    return [lane_ids[index::batch_count] for index in range(batch_count)]


def prepare_batch_attempts(
    *,
    root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    iteration: int,
    execution_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cluster_root = str(manifest["execution"]["cluster_root"])
    campaign_id, manifest_hash = _campaign_identity(manifest, state)
    batch_size = int(manifest["execution"]["batch_size"])
    prefix = f"iter{iteration:04d}_batch"
    batches = [
        batch
        for batch_id, batch in sorted(state["batches"].items())
        if batch_id.startswith(prefix) and int(batch.get("iteration", -1)) == iteration
    ]
    if not batches:
        for index, lane_ids in enumerate(_initial_batches(expected_lane_ids(manifest), batch_size)):
            batch_id = f"{prefix}{index:02d}"
            batch = {
                "batch_id": batch_id,
                "iteration": iteration,
                "campaign_id": campaign_id,
                "manifest_sha256": manifest_hash,
                "lane_ids": lane_ids,
                "state": "prepared",
                "retry_lane_ids": lane_ids,
                "retry_count": 0,
                "attempts": [],
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
            state["batches"][batch_id] = batch
            batches.append(batch)
    covered = [lane_id for batch in batches for lane_id in batch["lane_ids"]]
    for batch in batches:
        if batch.get("campaign_id") != campaign_id or batch.get("manifest_sha256") != manifest_hash:
            raise RuntimeError(f"persisted campaign identity drift for {batch['batch_id']}")
        for attempt_record in batch.get("attempts", []):
            attempt_number = int(attempt_record.get("attempt", -1))
            active_lane_ids = list(attempt_record.get("lane_ids", []))
            if (
                attempt_record.get("campaign_id") != campaign_id
                or attempt_record.get("manifest_sha256") != manifest_hash
                or attempt_number <= 0
                or not active_lane_ids
                or not set(active_lane_ids).issubset(set(batch["lane_ids"]))
            ):
                raise RuntimeError(f"persisted attempt identity drift for {batch['batch_id']}")
            expected_key, expected_job_name = _batch_identity(
                campaign_id=campaign_id,
                manifest_sha256=manifest_hash,
                cluster_root=cluster_root,
                iteration=iteration,
                batch_id=str(batch["batch_id"]),
                attempt=attempt_number,
                lane_ids=active_lane_ids,
                root=root,
            )
            if (
                attempt_record.get("submission_key") != expected_key
                or attempt_record.get("job_name") != expected_job_name
            ):
                raise RuntimeError(f"persisted submission identity drift for {batch['batch_id']}")
    if covered != [
        lane_id
        for group in _initial_batches(expected_lane_ids(manifest), batch_size)
        for lane_id in group
    ]:
        # Compare sets as well as uniqueness; ordering inside each persisted batch is separately hashed.
        if len(covered) != len(set(covered)) or set(covered) != set(expected_lane_ids(manifest)):
            raise RuntimeError("persisted batch partition does not cover each campaign lane once")

    pending: list[dict[str, Any]] = []
    effective_execution = dict(execution_spec or {})
    for batch in batches:
        batch_state = str(batch["state"])
        if batch_state == "collected":
            continue
        if batch_state == "submitted":
            continue
        if batch_state not in {"prepared", "retry_ready"}:
            raise RuntimeError(f"invalid batch state {batch_state!r} for {batch['batch_id']}")
        attempts = list(batch.get("attempts", []))
        if attempts and attempts[-1].get("state") == "prepared":
            attempt_record = attempts[-1]
            if (
                attempt_record.get("campaign_id") != campaign_id
                or attempt_record.get("manifest_sha256") != manifest_hash
                or (
                    execution_spec is not None
                    and attempt_record.get("execution") != effective_execution
                )
            ):
                raise RuntimeError(
                    f"campaign identity/execution drift for prepared attempt {batch['batch_id']}"
                )
        else:
            active_lane_ids = list(batch.get("retry_lane_ids") or batch["lane_ids"])
            if not active_lane_ids or not set(active_lane_ids).issubset(set(batch["lane_ids"])):
                raise RuntimeError(f"invalid retry lane set for {batch['batch_id']}")
            attempt_number = len(attempts) + 1
            submission_key, job_name = _batch_identity(
                campaign_id=campaign_id,
                manifest_sha256=manifest_hash,
                cluster_root=cluster_root,
                iteration=iteration,
                batch_id=str(batch["batch_id"]),
                attempt=attempt_number,
                lane_ids=active_lane_ids,
                root=root,
            )
            relative_lane_file = (
                Path("iterations")
                / f"iter_{iteration:04d}"
                / "batches"
                / str(batch["batch_id"])
                / f"attempt_{attempt_number:02d}"
                / "lane_ids.txt"
            )
            lane_file = root / relative_lane_file
            lane_file.parent.mkdir(parents=True, exist_ok=True)
            lane_file.write_text("\n".join(active_lane_ids) + "\n", encoding="utf-8")
            attempt_record = {
                "attempt": attempt_number,
                "campaign_id": campaign_id,
                "manifest_sha256": manifest_hash,
                "lane_ids": active_lane_ids,
                "submission_key": submission_key,
                "job_name": job_name,
                "lane_ids_path": relative_lane_file.as_posix(),
                "state": "prepared",
                "job_id": None,
                "created_at": now_iso(),
                "submitted_at": None,
                "execution": effective_execution,
            }
            attempts.append(attempt_record)
            batch["attempts"] = attempts
            batch["state"] = "prepared"
            batch["updated_at"] = now_iso()
        pending.append({"batch": batch, "attempt": attempt_record})
        _write_batch_mirror(root, batch)
    write_state(root, state)
    return pending


def _remote_submission_command(
    *,
    cluster_root: str,
    cluster_python: str,
    campaign_id: str,
    manifest_sha256: str,
    batch: dict[str, Any],
    attempt: dict[str, Any],
    account: str,
    partition: str,
    workers: int,
    cpus_per_task: int,
    memory: str,
    time_limit: str,
) -> str:
    validate_absolute_posix_path(cluster_root, "cluster_root")
    validate_absolute_posix_path(cluster_python, "cluster_python")
    claim_payload = _remote_claim_payload(campaign_id=campaign_id, manifest_sha256=manifest_sha256)
    if (
        attempt.get("campaign_id") != campaign_id
        or attempt.get("manifest_sha256") != manifest_sha256
        or batch.get("campaign_id") != campaign_id
        or batch.get("manifest_sha256") != manifest_sha256
    ):
        raise RuntimeError("batch/attempt campaign identity drift before submission")
    submission_key = str(attempt["submission_key"])
    job_name = str(attempt["job_name"])
    receipt_dir = f"{cluster_root.rstrip('/')}/receipts/{campaign_id}"
    receipt = f"{receipt_dir}/{submission_key}.jobid"
    lock = f"{receipt_dir}/{submission_key}.lock"
    claim = f"{cluster_root.rstrip('/')}/{REMOTE_CLAIM_NAME}"
    lane_file = f"{cluster_root.rstrip('/')}/{attempt['lane_ids_path']}"
    export_values = {
        "CAMPAIGN_ROOT": cluster_root,
        "CAMPAIGN_PYTHON": cluster_python,
        "ITERATION": str(batch["iteration"]),
        "BATCH_ID": str(batch["batch_id"]),
        "LANE_IDS_FILE": lane_file,
        "EVALUATOR_WORKERS": str(workers),
    }
    exports = "ALL," + ",".join(
        f"{key}={_export_safe(value, key)}" for key, value in export_values.items()
    )
    script = f"{cluster_root.rstrip('/')}/runner/slurm/evaluate_batch.sbatch"
    output = f"{cluster_root.rstrip('/')}/logs/{job_name}-%j.out"
    error = f"{cluster_root.rstrip('/')}/logs/{job_name}-%j.err"
    sbatch = " ".join(
        [
            "sbatch",
            "--parsable",
            f"--job-name={shlex.quote(job_name)}",
            f"--account={shlex.quote(account)}",
            f"--partition={shlex.quote(partition)}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={shlex.quote(memory)}",
            f"--time={shlex.quote(time_limit)}",
            f"--output={shlex.quote(output)}",
            f"--error={shlex.quote(error)}",
            f"--export={shlex.quote(exports)}",
            shlex.quote(script),
        ]
    )
    # The exact campaign claim is checked on the submit host before receipts or
    # Slurm state are touched. This also proves the sync and submit hosts share
    # the same claimed filesystem root.
    return (
        "set -eu; umask 077; "
        f"root={shlex.quote(cluster_root)}; claim={shlex.quote(claim)}; "
        'if [ -L "$root" ] || [ ! -d "$root" ] || [ -L "$claim" ] || '
        '[ ! -f "$claim" ]; then echo "remote campaign claim missing or unsafe" >&2; exit 51; fi; '
        f'if ! printf %s {shlex.quote(claim_payload)} | cmp -s - "$claim"; then '
        'echo "remote campaign claim mismatch" >&2; exit 52; fi; '
        f"mkdir -p {shlex.quote(receipt_dir)} {shlex.quote(cluster_root + '/logs')}; "
        f"exec 9>{shlex.quote(lock)}; flock -x 9; "
        f"receipt={shlex.quote(receipt)}; job_name={shlex.quote(job_name)}; "
        'if [ -s "$receipt" ]; then cat "$receipt"; exit 0; fi; '
        "existing=$(squeue -h -n \"$job_name\" -o '%A|%j' 2>/dev/null | "
        "awk -F'|' -v n=\"$job_name\" '$2==n {print $1; exit}'); "
        'if [ -z "$existing" ]; then '
        'existing=$(sacct -n -X --name "$job_name" --format=JobIDRaw,JobName -P 2>/dev/null | '
        "awk -F'|' -v n=\"$job_name\" '$2==n && $1 ~ /^[0-9]+$/ {print $1; exit}'); fi; "
        'if [ -z "$existing" ]; then '
        f"existing=$({sbatch} | cut -d';' -f1); fi; "
        "case \"$existing\" in ''|*[!0-9]*) echo 'invalid Slurm job id' >&2; exit 3;; esac; "
        'tmp="$receipt.tmp.$$"; printf \'%s\\n\' "$existing" >"$tmp"; mv "$tmp" "$receipt"; '
        'cat "$receipt"'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync and idempotently submit one reaction evaluation iteration."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--sync-host")
    parser.add_argument("--submit-host")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--command-attempts", type=int, default=3)
    parser.add_argument("--sync-timeout-seconds", type=int, default=600)
    parser.add_argument("--submit-timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    validate_iteration(args.iteration)
    if not args.sync_host:
        parser.error("--sync-host is required")
    missing = [
        name
        for name, value in (
            ("--submit-host", args.submit_host),
            ("--account", args.account),
            ("--partition", args.partition),
        )
        if not value
    ]
    if missing:
        parser.error("required execution provenance: " + ", ".join(missing))
    if min(args.workers, args.cpus_per_task, args.command_attempts) <= 0:
        parser.error("workers, cpus-per-task, and command-attempts must be positive")

    root = args.campaign_root.resolve()
    with campaign_lock(root, "orchestration"):
        manifest, state = require_campaign(root)
        ensure_campaign_runner_runtime(root, Path(__file__), manifest)
        if int(state["next_iteration"]) != args.iteration:
            raise RuntimeError(
                f"iteration sequence violation: state expects {state['next_iteration']}, got {args.iteration}"
            )
        validate_hamiltonian_verification(root, manifest, state)
        _validate_candidates(root, manifest, args.iteration)
        execution_spec = {
            "sync_host": str(args.sync_host),
            "submit_host": str(args.submit_host),
            "account": str(args.account),
            "partition": str(args.partition),
            "workers": int(args.workers),
            "cpus_per_task": int(args.cpus_per_task),
            "memory": str(args.memory),
            "time_limit": str(args.time_limit),
            "cluster_root": str(manifest["execution"]["cluster_root"]),
            "cluster_python": str(manifest["execution"]["cluster_python"]),
        }
        pending = prepare_batch_attempts(
            root=root,
            manifest=manifest,
            state=state,
            iteration=args.iteration,
            execution_spec=execution_spec,
        )
        cluster_root = str(manifest["execution"]["cluster_root"])
        cluster_python = str(manifest["execution"]["cluster_python"])
        validate_absolute_posix_path(cluster_root, "cluster_root")
        validate_absolute_posix_path(cluster_python, "cluster_python")
        campaign_id, manifest_hash = _campaign_identity(manifest, state)
        claim_remote_campaign(
            sync_host=str(args.sync_host),
            cluster_root=cluster_root,
            campaign_id=campaign_id,
            manifest_sha256=manifest_hash,
            attempts=args.command_attempts,
            timeout_seconds=args.sync_timeout_seconds,
        )
        run_with_retries(
            [
                "rsync",
                "-ah",
                "--partial",
                "--exclude",
                ".locks/",
                "--exclude",
                "__pycache__/",
                "--exclude",
                ".venv/",
                "--exclude",
                "logs/*.out",
                "--exclude",
                "logs/*.err",
                f"{root}/",
                f"{args.sync_host}:{cluster_root}/",
            ],
            attempts=args.command_attempts,
            timeout_seconds=args.sync_timeout_seconds,
        )
        if args.sync_only:
            print(json.dumps({"iteration": args.iteration, "synced": True, "submitted": False}))
            return

        submitted: list[dict[str, Any]] = []
        for item in pending:
            batch = item["batch"]
            attempt = item["attempt"]
            remote = _remote_submission_command(
                cluster_root=cluster_root,
                cluster_python=cluster_python,
                campaign_id=campaign_id,
                manifest_sha256=manifest_hash,
                batch=batch,
                attempt=attempt,
                account=str(args.account),
                partition=str(args.partition),
                workers=args.workers,
                cpus_per_task=args.cpus_per_task,
                memory=args.memory,
                time_limit=args.time_limit,
            )
            completed = run_with_retries(
                ["ssh", str(args.submit_host), remote],
                attempts=args.command_attempts,
                timeout_seconds=args.submit_timeout_seconds,
            )
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if not lines or re.fullmatch(r"[0-9]+", lines[-1]) is None:
                raise RuntimeError(
                    f"remote submission returned invalid receipt: {completed.stdout!r}"
                )
            job_id = lines[-1]
            attempt["state"] = "submitted"
            attempt["job_id"] = job_id
            attempt["submitted_at"] = now_iso()
            batch["state"] = "submitted"
            batch["retry_lane_ids"] = []
            batch["updated_at"] = now_iso()
            _write_batch_mirror(root, batch)
            write_state(root, state)
            submitted.append(
                {
                    "batch_id": batch["batch_id"],
                    "attempt": attempt["attempt"],
                    "job_id": job_id,
                    "lane_ids": attempt["lane_ids"],
                }
            )
            print(
                f"submitted {batch['batch_id']} attempt={attempt['attempt']} "
                f"lanes={len(attempt['lane_ids'])} job={job_id}",
                flush=True,
            )
        print(json.dumps({"iteration": args.iteration, "submitted": submitted}, sort_keys=True))


if __name__ == "__main__":
    main()

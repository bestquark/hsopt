#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections.abc import Callable

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-hamlib121-pycache-")
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
    db_connect,
    ensure_rsync_provenance,
    eval_submission_identity,
    init_db,
    log_event,
    mark_eval_batch_submitted,
    register_eval_batch,
    revalidate_rsync_binary,
    resolve_rsync_binary,
    rsync_proxy_jump_args,
    sha256_file,
    ssh_command,
    validate_absolute_posix_path,
    validate_campaign_identity,
    validate_existing_rsync_provenance,
    validate_manifests_parent_chain,
    validate_runner_provenance,
    validate_ssh_host,
    write_json,
)


REMOTE_CLAIM_NAME = ".hsopt-campaign-claim.json"
REMOTE_CLAIM_LOCK_NAME = ".hsopt-campaign-claim.lock"
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def validate_slurm_name(value: str, field: str) -> str:
    if not isinstance(value, str) or SAFE_NAME_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe Slurm name")
    return value


def slurm_export_safe(value: object, field: str) -> str:
    text = str(value)
    if not text or any(character in text for character in (",", "\n", "\r", "\0")):
        raise ValueError(f"{field} is unsafe for Slurm --export")
    return text


def slurm_export_assignment(name: str, value: object) -> str:
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None:
        raise ValueError(f"invalid Slurm export variable name: {name!r}")
    return f"{name}={slurm_export_safe(value, name)}"


def remote_claim_payload(*, campaign_id: str, manifest_sha256: str) -> str:
    if SHA256_RE.fullmatch(campaign_id) is None or SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("campaign claim requires SHA-256 campaign and manifest identities")
    return (
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": campaign_id,
                "campaign_manifest_sha256": manifest_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def remote_claim_command(*, cluster_root: str, campaign_id: str, manifest_sha256: str) -> str:
    validate_absolute_posix_path(cluster_root, "cluster_root")
    payload = remote_claim_payload(campaign_id=campaign_id, manifest_sha256=manifest_sha256)
    root = cluster_root.rstrip("/")
    claim = f"{root}/{REMOTE_CLAIM_NAME}"
    lock = f"{root}/{REMOTE_CLAIM_LOCK_NAME}"
    return (
        "set -eu; umask 077; "
        f"root={shlex.quote(root)}; claim={shlex.quote(claim)}; lock={shlex.quote(lock)}; "
        'if [ -L "$root" ] || { [ -e "$root" ] && [ ! -d "$root" ]; }; then '
        'echo "campaign root is unsafe" >&2; exit 41; fi; mkdir -p "$root"; '
        'if [ -L "$lock" ] || { [ -e "$lock" ] && [ ! -f "$lock" ]; }; then '
        'echo "campaign claim lock is unsafe" >&2; exit 42; fi; exec 9>"$lock"; flock -x 9; '
        'if [ -L "$claim" ]; then echo "campaign claim is a symlink" >&2; exit 43; fi; '
        'tmp="$root/.hsopt-campaign-claim.tmp.$$"; trap \'rm -f "$tmp"\' EXIT HUP INT TERM; '
        'if [ -L "$tmp" ] || [ -e "$tmp" ]; then '
        'echo "campaign claim temporary path is unsafe" >&2; exit 46; fi; '
        f'if ! (set -C; printf %s {shlex.quote(payload)} >"$tmp") 2>/dev/null; then '
        'echo "could not create campaign claim temporary file" >&2; exit 47; fi; '
        'if [ -e "$claim" ]; then if [ ! -f "$claim" ] || ! cmp -s "$tmp" "$claim"; then '
        'echo "remote campaign claim mismatch" >&2; exit 44; fi; else '
        'stale=$(find "$root" -mindepth 1 -maxdepth 1 '
        f'! -name {shlex.quote(REMOTE_CLAIM_LOCK_NAME)} ! -name "${{tmp##*/}}" '
        "! -name '.venv' ! -name '.venv_cpu' -print -quit); "
        'if [ -n "$stale" ]; then echo "refusing to claim non-empty campaign root: $stale" >&2; '
        'exit 45; fi; mv "$tmp" "$claim"; fi'
    )


def assert_remote_claim_command(
    *, cluster_root: str, campaign_id: str, manifest_sha256: str
) -> str:
    validate_absolute_posix_path(cluster_root, "cluster_root")
    payload = remote_claim_payload(campaign_id=campaign_id, manifest_sha256=manifest_sha256)
    root = cluster_root.rstrip("/")
    claim = f"{root}/{REMOTE_CLAIM_NAME}"
    return (
        f"root={shlex.quote(root)}; claim={shlex.quote(claim)}; "
        'if [ -L "$root" ] || [ ! -d "$root" ] || [ -L "$claim" ] || [ ! -f "$claim" ]; '
        'then echo "remote campaign claim missing or unsafe" >&2; exit 51; fi; '
        f'if ! printf %s {shlex.quote(payload)} | cmp -s - "$claim"; then '
        'echo "remote campaign claim mismatch" >&2; exit 52; fi; '
    )


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_lane_ids(path: Path, lane_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lane_ids) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"durable lane intent is unsafe: {path}")
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"durable lane intent changed: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def persist_json_intent(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"durable JSON intent is unsafe: {path}")
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"durable JSON intent changed: {path}")
        return
    write_json(path, payload)


def validate_rsync_binding(
    campaign_root: Path,
    provenance: dict[str, object],
) -> bool:
    """Validate the secondary binding marker used to detect a deleted receipt."""
    path = campaign_root / "manifests" / "rsync_binding.json"
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"campaign rsync binding marker is unsafe: {path}")
    expected = {"schema_version": 1, "rsync": provenance}
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("campaign rsync binding marker is unreadable") from exc
    if observed != expected:
        raise RuntimeError("campaign rsync binding marker drifted")
    return True


def campaign_has_rsync_or_batch_evidence(campaign_root: Path) -> bool:
    """Return whether a campaign has evidence that requires an existing receipt."""
    if (campaign_root / "manifests" / "rsync_binding.json").exists():
        return True
    if (campaign_root / "manifests" / "supervisor_provenance.json").exists():
        return True
    iterations = campaign_root / "iterations"
    return any(iterations.glob("iter_*/batches/*")) or any(iterations.glob("iter_*/results/*.json"))


def require_or_allow_new_rsync_binding(
    campaign_root: Path,
    provenance: dict[str, object],
) -> bool:
    """Return receipt presence, rejecting rebinding once activity is durable."""
    validate_rsync_binding(campaign_root, provenance)
    present = validate_existing_rsync_provenance(campaign_root, provenance)
    if not present and campaign_has_rsync_or_batch_evidence(campaign_root):
        raise RuntimeError(
            "campaign rsync provenance receipt is missing after transfer or batch activity"
        )
    return present


def run_with_retries(
    command: list[str],
    *,
    attempts: int,
    timeout_seconds: int,
    retry_delay_seconds: float = 2.0,
    input_text: str | None = None,
    before_attempt: Callable[[], object] | None = None,
) -> subprocess.CompletedProcess[str]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        if before_attempt is not None:
            before_attempt()
        try:
            return subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                input=input_text,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"attempt {attempt}: timed out after {timeout_seconds}s")
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            failures.append(f"attempt {attempt}: exit {exc.returncode}: {detail}")
        if attempt < attempts:
            time.sleep(retry_delay_seconds * attempt)
    raise RuntimeError(f"command failed after {attempts} attempts: {'; '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one streaming evaluation batch.")
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--lane-id", action="append", default=[])
    parser.add_argument("--lane-ids-file", type=Path)
    parser.add_argument("--all-active", action="store_true")
    parser.add_argument("--cluster-root", default=None)
    parser.add_argument("--hamlib-data-root", default=None)
    parser.add_argument("--account", default=os.environ.get("HSOPT_SLURM_ACCOUNT"))
    parser.add_argument("--partition", default=os.environ.get("HSOPT_GPU_PARTITION"))
    parser.add_argument("--cpu-partition", default=os.environ.get("HSOPT_CPU_PARTITION"))
    parser.add_argument("--sync-host", default=os.environ.get("HSOPT_SYNC_HOST"))
    parser.add_argument("--submit-host", default=os.environ.get("HSOPT_SUBMIT_HOST"))
    parser.add_argument(
        "--proxy-jump-host",
        help="Optional SSH config alias used as ProxyJump for sync and submit hosts.",
    )
    parser.add_argument(
        "--rsync-bin",
        default=os.environ.get("HSOPT_RSYNC_BIN"),
        help=(
            "Explicit local rsync executable. Defaults to HSOPT_RSYNC_BIN, then discovers "
            "a compatible rsync >= 3 with --protect-args support."
        ),
    )
    parser.add_argument("--legate-fbmem", default=None)
    parser.add_argument("--uv-base-dir", default=None)
    parser.add_argument("--uv-cache-dir", default=None)
    parser.add_argument("--uv-python-install-dir", default=None)
    parser.add_argument("--compute-baseline", action="store_true")
    parser.add_argument("--policy-snapshot", choices=("current", "initial"), default="current")
    parser.add_argument("--sync-timeout-seconds", type=int, default=300)
    parser.add_argument("--submit-timeout-seconds", type=int, default=60)
    parser.add_argument("--command-attempts", type=int, default=5)
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Sync the prepared batch but do not submit a Slurm job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print the batch plan without local writes, file transfer, SSH, "
            "or Slurm. The local rsync version/parser preflight still runs."
        ),
    )
    parser.add_argument(
        "--no-submit", dest="sync_only", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.dry_run and args.sync_only:
        parser.error("--dry-run and --sync-only are mutually exclusive")
    if args.iteration < 0:
        parser.error("--iteration must be nonnegative")
    if SAFE_NAME_RE.fullmatch(args.batch_id) is None:
        parser.error("--batch-id must contain only letters, digits, '.', '_', and '-'")
    if args.command_attempts < 1:
        parser.error("--command-attempts must be positive")
    if args.sync_timeout_seconds < 1 or args.submit_timeout_seconds < 1:
        parser.error("command timeouts must be positive")
    if args.legate_fbmem is not None and (re.fullmatch(r"[1-9][0-9]*", args.legate_fbmem) is None):
        parser.error("--legate-fbmem must be a positive integer number of MiB")
    for field, value in (
        ("uv_base_dir", args.uv_base_dir),
        ("uv_cache_dir", args.uv_cache_dir),
        ("uv_python_install_dir", args.uv_python_install_dir),
    ):
        if value is not None:
            validate_absolute_posix_path(value, field)

    root = validate_manifests_parent_chain(args.campaign_root)
    manifest, campaign_identity = validate_campaign_identity(root)
    runner_provenance = validate_runner_provenance(root)
    campaign_id = str(manifest["campaign_id"])
    campaign_manifest_sha256 = str(campaign_identity["campaign_manifest_sha256"])
    cluster_root = validate_absolute_posix_path(
        str(args.cluster_root or manifest["cluster_root"]), "cluster_root"
    )
    data_root = validate_absolute_posix_path(
        str(args.hamlib_data_root or manifest["hamlib_data_root"]), "hamlib_data_root"
    )
    archive_preflight_path = root / "manifests" / "hamlib_archive_preflight.json"
    archive_preflight_sha256: str | None = None
    if archive_preflight_path.exists():
        if archive_preflight_path.is_symlink() or not archive_preflight_path.is_file():
            raise RuntimeError("HamLib archive preflight receipt is unsafe")
        archive_preflight = json.loads(archive_preflight_path.read_text(encoding="utf-8"))
        expected_archive_keys = sorted(
            str(record["archive_key"]) for record in manifest.get("hamlib_archives", [])
        )
        expected_preflight = {
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_sha256": campaign_identity["campaign_manifest_sha256"],
            "runner_provenance": runner_provenance,
            "hamlib_zip_checksums_sha256": manifest["hamlib_zip_checksums_sha256"],
            "active_lanes_verified": 121,
            "archives_verified": 83,
            "archive_keys": expected_archive_keys,
        }
        drift = {
            key: {"actual": archive_preflight.get(key), "expected": value}
            for key, value in expected_preflight.items()
            if archive_preflight.get(key) != value
        }
        if archive_preflight.get("schema_version") != 1 or drift:
            raise RuntimeError(f"HamLib archive preflight receipt drift: {drift}")
        archive_preflight_sha256 = sha256_file(archive_preflight_path)
    elif not (args.dry_run or args.sync_only):
        raise RuntimeError(
            "actual submission requires manifests/hamlib_archive_preflight.json from a "
            "read-only verification on the bound cluster data root"
        )

    active_lane_ids = [lane["lane_id"] for lane in manifest["lanes"] if lane["exact_fit"]]
    lane_ids = list(args.lane_id)
    if args.all_active:
        lane_ids.extend(active_lane_ids)
    if args.lane_ids_file is not None:
        lane_ids.extend(
            line.strip()
            for line in args.lane_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    lane_ids = list(dict.fromkeys(lane_ids))
    if not lane_ids:
        raise ValueError("provide at least one --lane-id or --lane-ids-file entry")

    active_lane_id_set = set(active_lane_ids)
    missing = set(lane_ids) - active_lane_id_set
    if missing:
        raise ValueError(f"lane ids are not active in manifest: {sorted(missing)}")
    missing_baselines = [
        lane_id for lane_id in lane_ids if not (root / "baselines" / f"{lane_id}.json").exists()
    ]
    use_gpu_baseline = bool(args.compute_baseline)
    slurm_script = (
        "slurm/evaluate_batch.sbatch" if use_gpu_baseline else "slurm/evaluate_batch_cpu.sbatch"
    )
    submit_partition = args.partition if use_gpu_baseline else args.cpu_partition
    compute_baseline = bool(use_gpu_baseline)
    execution_mode = "gpu_baseline" if use_gpu_baseline else "cpu_policy"
    effective_legate_fbmem = args.legate_fbmem or ("70000" if use_gpu_baseline else None)
    if missing_baselines and not args.compute_baseline:
        raise RuntimeError(
            f"CPU policy batch is missing {len(missing_baselines)} verified baseline caches; "
            "establish the exact iteration-0 A100 baseline first"
        )
    if args.iteration == 0:
        if (
            lane_ids != active_lane_ids
            or len(lane_ids) != 121
            or not args.compute_baseline
            or args.policy_snapshot != "initial"
            or not use_gpu_baseline
        ):
            raise RuntimeError(
                "iteration-0 paper baseline must submit all canonical 121 lanes in manifest "
                "order with --compute-baseline --policy-snapshot initial on the GPU path"
            )
    elif args.compute_baseline:
        raise RuntimeError("baseline computation is permitted only for the exact iteration-0 batch")

    policy_paths = [
        root
        / "lanes"
        / lane_id
        / ("initial_simple_pf.py" if args.policy_snapshot == "initial" else "simple_pf.py")
        for lane_id in lane_ids
    ]
    unsafe_policies = [
        str(path) for path in policy_paths if path.is_symlink() or not path.is_file()
    ]
    if unsafe_policies:
        raise RuntimeError(f"batch policy snapshots are missing or unsafe: {unsafe_policies[:10]}")
    policy_sha256s = [sha256_file(path) for path in policy_paths]
    execution_configuration = {
        "schema_version": 1,
        "execution_mode": execution_mode,
        "array_backend": "cupynumeric" if use_gpu_baseline else "numpy",
        "compute_baseline": compute_baseline,
        "policy_snapshot": args.policy_snapshot,
        "account": args.account,
        "partition": submit_partition,
        "slurm_script": slurm_script,
        "legate_fbmem": effective_legate_fbmem,
        "uv_base_dir": args.uv_base_dir,
        "uv_cache_dir": args.uv_cache_dir,
        "uv_python_install_dir": args.uv_python_install_dir,
        "hamlib_archive_preflight_sha256": archive_preflight_sha256,
    }

    batch_rel = (
        Path("iterations")
        / f"iter_{args.iteration:04d}"
        / "batches"
        / args.batch_id
        / "lane_ids.txt"
    )
    batch_path = root / batch_rel
    missing_config: list[str] = []
    if not args.sync_host:
        missing_config.append("--sync-host or HSOPT_SYNC_HOST")
    if not args.account:
        missing_config.append("--account or HSOPT_SLURM_ACCOUNT")
    if not submit_partition:
        selected_flag = "--partition" if use_gpu_baseline else "--cpu-partition"
        missing_config.append(selected_flag)
    if not args.submit_host:
        missing_config.append("--submit-host or HSOPT_SUBMIT_HOST")
    if missing_config:
        parser.error("missing cluster configuration: " + ", ".join(missing_config))
    sync_host = validate_ssh_host(str(args.sync_host), "sync_host")
    submit_host = validate_ssh_host(str(args.submit_host), "submit_host")
    proxy_jump_host = (
        validate_ssh_host(str(args.proxy_jump_host), "proxy_jump_host")
        if args.proxy_jump_host is not None
        else None
    )
    try:
        rsync_provenance = resolve_rsync_binary(args.rsync_bin)
        require_or_allow_new_rsync_binding(root, rsync_provenance)
    except RuntimeError as exc:
        parser.error(str(exc))
    rsync_binary = str(rsync_provenance["binary"])
    account = validate_slurm_name(str(args.account), "account")
    submit_partition = validate_slurm_name(str(submit_partition), "partition")
    transport_configuration = {
        "sync_host": sync_host,
        "submit_host": submit_host,
        "proxy_jump_host": proxy_jump_host,
        "rsync": rsync_provenance,
    }
    submission_key, job_name = eval_submission_identity(
        cluster_root=cluster_root,
        iteration=args.iteration,
        batch_id=args.batch_id,
        policy_snapshot=args.policy_snapshot,
        lane_ids=lane_ids,
        campaign_id=campaign_id,
        campaign_manifest_sha256=campaign_manifest_sha256,
        policy_sha256s=policy_sha256s,
        execution_configuration=execution_configuration,
        runner_provenance=runner_provenance,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "account": args.account,
                    "batch_id": args.batch_id,
                    "cluster_root": cluster_root,
                    "execution_mode": execution_mode,
                    "hamlib_data_root": data_root,
                    "iteration": args.iteration,
                    "lane_ids": lane_ids,
                    "partition": submit_partition,
                    "slurm_script": slurm_script,
                    "submit_host": submit_host,
                    "sync_host": sync_host,
                    "proxy_jump_host": proxy_jump_host,
                    "rsync": rsync_provenance,
                    "campaign_id": campaign_id,
                    "campaign_manifest_sha256": campaign_manifest_sha256,
                    "submission_key": submission_key,
                    "job_name": job_name,
                    "policy_sha256s": dict(zip(lane_ids, policy_sha256s, strict=True)),
                    "execution_configuration": execution_configuration,
                    "runner_provenance": runner_provenance,
                },
                sort_keys=True,
            )
        )
        return
    ensure_rsync_provenance(root, rsync_provenance)
    persist_json_intent(
        root / "manifests" / "rsync_binding.json",
        {"schema_version": 1, "rsync": rsync_provenance},
    )
    if args.sync_only:
        run_with_retries(
            ssh_command(sync_host, "bash", "-s", proxy_jump_host=proxy_jump_host),
            attempts=args.command_attempts,
            timeout_seconds=args.submit_timeout_seconds,
            input_text=remote_claim_command(
                cluster_root=cluster_root,
                campaign_id=campaign_id,
                manifest_sha256=campaign_manifest_sha256,
            ),
        )
        run_with_retries(
            [
                rsync_binary,
                "-ah",
                "--partial",
                "--protect-args",
                *rsync_proxy_jump_args(proxy_jump_host),
                "--exclude",
                ".venv/",
                "--exclude",
                ".venv_cpu/",
                "--exclude",
                ".codex_npm/",
                "--exclude",
                ".supervisor.lock",
                "--exclude",
                "__pycache__/",
                "--exclude",
                "logs/*.out",
                "--exclude",
                "logs/*.err",
                f"{root}/",
                f"{sync_host}:{cluster_root}/",
            ],
            attempts=args.command_attempts,
            timeout_seconds=args.sync_timeout_seconds,
            before_attempt=lambda: revalidate_rsync_binary(rsync_provenance),
        )
        print(f"claimed and synced campaign staging root {cluster_root}; no batch intent created")
        return
    write_lane_ids(batch_path, lane_ids)
    batch_manifest = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "batch_id": args.batch_id,
        "iteration": args.iteration,
        "lane_ids": lane_ids,
        "policy_snapshot": args.policy_snapshot,
        "policy_sha256s": dict(zip(lane_ids, policy_sha256s, strict=True)),
        "submission_key": submission_key,
        "job_name": job_name,
        "execution_configuration": execution_configuration,
        "transport_configuration": transport_configuration,
        "runner_provenance": runner_provenance,
    }
    persist_json_intent(batch_path.parent / "batch_manifest.json", batch_manifest)
    conn = db_connect(root)
    init_db(conn)
    register_eval_batch(
        conn,
        batch_id=args.batch_id,
        iteration=args.iteration,
        lane_ids=lane_ids,
        submission_key=submission_key,
        job_name=job_name,
    )

    run_with_retries(
        ssh_command(sync_host, "bash", "-s", proxy_jump_host=proxy_jump_host),
        attempts=args.command_attempts,
        timeout_seconds=args.submit_timeout_seconds,
        input_text=remote_claim_command(
            cluster_root=cluster_root,
            campaign_id=campaign_id,
            manifest_sha256=campaign_manifest_sha256,
        ),
    )

    run_with_retries(
        [
            rsync_binary,
            "-ah",
            "--partial",
            "--protect-args",
            *rsync_proxy_jump_args(proxy_jump_host),
            "--exclude",
            ".venv/",
            "--exclude",
            ".venv_cpu/",
            "--exclude",
            ".codex_npm/",
            "--exclude",
            ".supervisor.lock",
            "--exclude",
            "__pycache__/",
            "--exclude",
            "logs/*.out",
            "--exclude",
            "logs/*.err",
            f"{root}/",
            f"{sync_host}:{cluster_root}/",
        ],
        attempts=args.command_attempts,
        timeout_seconds=args.sync_timeout_seconds,
        before_attempt=lambda: revalidate_rsync_binary(rsync_provenance),
    )
    cluster_lane_file = f"{cluster_root.rstrip('/')}/{batch_rel.as_posix()}"
    export_parts = [
        slurm_export_assignment(
            "PATH", "/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        slurm_export_assignment("CAMPAIGN_ROOT", cluster_root),
        slurm_export_assignment("ITERATION", args.iteration),
        slurm_export_assignment("BATCH_ID", args.batch_id),
        slurm_export_assignment("LANE_IDS_FILE", cluster_lane_file),
        slurm_export_assignment("HAMLIB_DATA_ROOT", data_root),
        slurm_export_assignment("COMPUTE_BASELINE", 1 if compute_baseline else 0),
        slurm_export_assignment("POLICY_SNAPSHOT", args.policy_snapshot),
        slurm_export_assignment("CAMPAIGN_ID", campaign_id),
        slurm_export_assignment("CAMPAIGN_MANIFEST_SHA256", campaign_manifest_sha256),
        slurm_export_assignment("CAMPAIGN_SUBMISSION_KEY", submission_key),
        slurm_export_assignment("CAMPAIGN_JOB_NAME", job_name),
        slurm_export_assignment("CAMPAIGN_EXECUTION_MODE", execution_mode),
        slurm_export_assignment("CAMPAIGN_SLURM_ACCOUNT", account),
        slurm_export_assignment("CAMPAIGN_SLURM_PARTITION", submit_partition),
        slurm_export_assignment("CAMPAIGN_HAMLIB_PREFLIGHT_SHA256", archive_preflight_sha256),
    ]
    if effective_legate_fbmem:
        export_parts.append(
            slurm_export_assignment("CAMPAIGN_LEGATE_FBMEM", effective_legate_fbmem)
        )
    if args.uv_base_dir:
        export_parts.append(slurm_export_assignment("CAMPAIGN_UV_BASE_DIR", args.uv_base_dir))
    if args.uv_cache_dir:
        export_parts.append(slurm_export_assignment("CAMPAIGN_UV_CACHE_DIR", args.uv_cache_dir))
    if args.uv_python_install_dir:
        export_parts.append(
            slurm_export_assignment("CAMPAIGN_UV_PYTHON_INSTALL_DIR", args.uv_python_install_dir)
        )
    export_vars = ",".join(export_parts)
    cluster_job_file = (
        f"{cluster_root.rstrip('/')}/iterations/iter_{args.iteration:04d}/batches/"
        f"{args.batch_id}/slurm_job_id_{submission_key}.txt"
    )
    remote = (
        "set -euo pipefail; umask 077; "
        + assert_remote_claim_command(
            cluster_root=cluster_root,
            campaign_id=campaign_id,
            manifest_sha256=campaign_manifest_sha256,
        )
        + f"job_file={shell_quote(cluster_job_file)}; "
        'job_dir="${job_file%/*}"; '
        'if [ -L "$job_dir" ] || [ ! -d "$job_dir" ]; then '
        'echo "submission directory is missing or unsafe" >&2; exit 53; fi; '
        'submit_lock="$job_file.lock"; '
        'if [ -L "$submit_lock" ] || { [ -e "$submit_lock" ] && [ ! -f "$submit_lock" ]; }; '
        'then echo "submission lock is unsafe" >&2; exit 54; fi; '
        'exec 8>"$submit_lock"; /usr/bin/flock -x 8; '
        f"job_name={shell_quote(job_name)}; "
        f"campaign_id={shell_quote(campaign_id)}; "
        f"manifest_sha={shell_quote(campaign_manifest_sha256)}; "
        f"submission_key={shell_quote(submission_key)}; "
        'intent_file="$job_file.submitting"; '
        'if [ -L "$job_file" ] || { [ -e "$job_file" ] && [ ! -f "$job_file" ]; }; then '
        'echo "submission receipt is unsafe" >&2; exit 55; fi; '
        'if [ -L "$intent_file" ] || { [ -e "$intent_file" ] && [ ! -f "$intent_file" ]; }; '
        'then echo "submission intent is unsafe" >&2; exit 56; fi; '
        'write_receipt() { local id="$1" tmp="$job_file.tmp.$$"; '
        'if [ -L "$tmp" ] || [ -e "$tmp" ]; then '
        'echo "submission receipt temporary path is unsafe" >&2; return 57; fi; '
        'printf "%s|%s|%s|%s|%s\\n" "$campaign_id" "$manifest_sha" '
        '"$submission_key" "$job_name" "$id" >"$tmp"; chmod 600 "$tmp"; '
        'mv "$tmp" "$job_file"; }; '
        'read_receipt() { local -a lines=(); mapfile -t lines <"$job_file"; '
        'if [ "${#lines[@]}" -ne 1 ]; then return 1; fi; '
        "local rid rmanifest rkey rname rjob extra; "
        'IFS="|" read -r rid rmanifest rkey rname rjob extra <<<"${lines[0]}"; '
        '[ -z "${extra:-}" ] && [ "$rid" = "$campaign_id" ] && '
        '[ "$rmanifest" = "$manifest_sha" ] && [ "$rkey" = "$submission_key" ] && '
        '[ "$rname" = "$job_name" ] && [[ "$rjob" =~ ^[0-9]+$ ]] && printf "%s\\n" "$rjob"; }; '
        f"cd {shell_quote(cluster_root)}; "
        "mkdir -p logs; "
        'if [ -s "$job_file" ]; then if ! read_receipt; then '
        'echo "submission receipt identity is invalid" >&2; exit 58; fi; exit 0; fi; '
        "created_intent=0; "
        'if [ ! -e "$intent_file" ]; then '
        'if ! (set -o noclobber; printf "%s|%s|%s|%s\\n" "$campaign_id" "$manifest_sha" '
        '"$submission_key" "$job_name" >"$intent_file") 2>/dev/null; then '
        'echo "could not persist pre-submit intent" >&2; exit 59; fi; created_intent=1; '
        'else intent=$(<"$intent_file"); '
        'if [ "$intent" != "$campaign_id|$manifest_sha|$submission_key|$job_name" ]; then '
        'echo "pre-submit intent identity is invalid" >&2; exit 60; fi; fi; '
        'SQUEUE=$(command -v squeue) || { echo "squeue is unavailable" >&2; exit 61; }; '
        'SACCT=$(command -v sacct) || { echo "sacct is unavailable" >&2; exit 61; }; '
        'SBATCH=$(command -v sbatch) || { echo "sbatch is unavailable" >&2; exit 61; }; '
        'if ! queue_rows=$("$SQUEUE" -h -u "$USER" -n "$job_name" -o "%A|%j"); then '
        'echo "squeue reconciliation failed" >&2; exit 62; fi; '
        'if ! account_rows=$("$SACCT" -n -X -u "$USER" --name "$job_name" '
        "--starttime 2000-01-01 --format=JobIDRaw,JobName%128 --parsable2); then "
        'echo "sacct reconciliation failed" >&2; exit 63; fi; '
        "declare -A ids=(); "
        'while IFS="|" read -r jid jname rest; do '
        'if [[ "$jid" =~ ^[0-9]+$ ]] && [ "$jname" = "$job_name" ]; then ids["$jid"]=1; fi; '
        'done <<<"$queue_rows"; '
        'while IFS="|" read -r jid jname rest; do '
        'if [[ "$jid" =~ ^[0-9]+$ ]] && [ "$jname" = "$job_name" ]; then ids["$jid"]=1; fi; '
        'done <<<"$account_rows"; '
        'if [ "${#ids[@]}" -gt 1 ]; then '
        'echo "multiple Slurm jobs match one durable submission intent" >&2; exit 64; fi; '
        'if [ "${#ids[@]}" -eq 1 ]; then for existing in "${!ids[@]}"; do :; done; '
        'write_receipt "$existing"; rm -f "$intent_file"; printf "%s\\n" "$existing"; exit 0; fi; '
        'if [ "$created_intent" -ne 1 ]; then '
        'echo "ambiguous prior submission has no Slurm receipt; refusing to resubmit" >&2; exit 65; fi; '
        f'submitted=$("$SBATCH" --parsable --job-name="$job_name" --account={shell_quote(account)} '
        f"--partition={shell_quote(submit_partition)} "
        f"--export={shell_quote(export_vars)} {shell_quote(slurm_script)}); "
        'job_id="${submitted%%;*}"; '
        'if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then '
        'echo "sbatch returned an invalid job ID: $submitted" >&2; exit 66; fi; '
        'write_receipt "$job_id"; rm -f "$intent_file"; printf "%s\\n" "$job_id"'
    )
    completed = run_with_retries(
        ssh_command(submit_host, "bash", "-s", proxy_jump_host=proxy_jump_host),
        attempts=args.command_attempts,
        timeout_seconds=args.submit_timeout_seconds,
        input_text=remote,
    )
    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("sbatch returned no job id")
    job_id = output_lines[-1].split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise RuntimeError(f"invalid sbatch job id: {output_lines[-1]!r}")

    mark_eval_batch_submitted(conn, batch_id=args.batch_id, job_id=job_id)
    log_event(
        conn,
        kind="submit-batch",
        iteration=args.iteration,
        message=f"submitted streaming batch {args.batch_id} job {job_id} ({execution_mode})",
        payload={
            "job_id": job_id,
            "job_name": job_name,
            "batch_id": args.batch_id,
            "lane_ids": lane_ids,
            "cluster_root": cluster_root,
            "compute_baseline": compute_baseline,
            "requested_compute_baseline": args.compute_baseline,
            "policy_snapshot": args.policy_snapshot,
            "missing_baselines": missing_baselines,
            "execution_mode": execution_mode,
            "partition": submit_partition,
            "account": account,
            "requested_partition": args.partition,
            "slurm_script": slurm_script,
            "sync_host": sync_host,
            "submit_host": submit_host,
            "proxy_jump_host": proxy_jump_host,
            "rsync": rsync_provenance,
            "legate_fbmem": effective_legate_fbmem,
            "uv_base_dir": args.uv_base_dir,
            "uv_cache_dir": args.uv_cache_dir,
            "uv_python_install_dir": args.uv_python_install_dir,
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "submission_key": submission_key,
            "policy_sha256s": dict(zip(lane_ids, policy_sha256s, strict=True)),
            "execution_configuration": execution_configuration,
            "runner_provenance": runner_provenance,
        },
    )
    print(job_id)


if __name__ == "__main__":
    main()

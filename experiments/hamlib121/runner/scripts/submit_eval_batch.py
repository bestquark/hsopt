#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from campaign_common import (
    db_connect,
    eval_submission_identity,
    init_db,
    log_event,
    mark_eval_batch_submitted,
    now_iso,
    register_eval_batch,
)


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_lane_ids(path: Path, lane_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lane_ids) + "\n"
    path.write_text(payload, encoding="utf-8")


def run_with_retries(
    command: list[str],
    *,
    attempts: int,
    timeout_seconds: int,
    retry_delay_seconds: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
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
        help="Validate and print the batch plan without local writes, rsync, SSH, or Slurm.",
    )
    parser.add_argument(
        "--no-submit", dest="sync_only", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.dry_run and args.sync_only:
        parser.error("--dry-run and --sync-only are mutually exclusive")

    root = args.campaign_root.resolve()
    manifest_path = root / "manifests" / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cluster_root = args.cluster_root or manifest["cluster_root"]
    data_root = args.hamlib_data_root or manifest["hamlib_data_root"]

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
    use_gpu_baseline = bool(args.compute_baseline and missing_baselines)
    slurm_script = (
        "slurm/evaluate_batch.sbatch" if use_gpu_baseline else "slurm/evaluate_batch_cpu.sbatch"
    )
    submit_partition = args.partition if use_gpu_baseline else args.cpu_partition
    compute_baseline = bool(use_gpu_baseline)
    execution_mode = "gpu_baseline" if use_gpu_baseline else "cpu_policy"

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
    if not args.sync_only:
        if not args.account:
            missing_config.append("--account or HSOPT_SLURM_ACCOUNT")
        if not submit_partition:
            selected_flag = "--partition" if use_gpu_baseline else "--cpu-partition"
            missing_config.append(selected_flag)
        if not args.submit_host:
            missing_config.append("--submit-host or HSOPT_SUBMIT_HOST")
    if missing_config:
        parser.error("missing cluster configuration: " + ", ".join(missing_config))
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
                    "submit_host": args.submit_host,
                    "sync_host": args.sync_host,
                },
                sort_keys=True,
            )
        )
        return
    submission_key, job_name = eval_submission_identity(
        cluster_root=cluster_root,
        iteration=args.iteration,
        batch_id=args.batch_id,
        policy_snapshot=args.policy_snapshot,
        lane_ids=lane_ids,
    )
    conn = db_connect(root)
    init_db(conn)
    tracked = conn.execute(
        "SELECT 1 FROM eval_batches WHERE batch_id=?", (args.batch_id,)
    ).fetchone()
    if tracked is not None:
        register_eval_batch(
            conn,
            batch_id=args.batch_id,
            iteration=args.iteration,
            lane_ids=lane_ids,
            submission_key=submission_key,
            job_name=job_name,
        )
    write_lane_ids(batch_path, lane_ids)

    run_with_retries(
        [
            "rsync",
            "-ah",
            "--partial",
            "--exclude",
            ".venv/",
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
            f"{args.sync_host}:{cluster_root}/",
        ],
        attempts=args.command_attempts,
        timeout_seconds=args.sync_timeout_seconds,
    )
    if args.sync_only:
        print(f"synced batch {args.batch_id} to {cluster_root}; not submitted")
        return

    cluster_lane_file = f"{cluster_root.rstrip('/')}/{batch_rel.as_posix()}"
    export_parts = [
        "ALL",
        f"CAMPAIGN_ROOT={cluster_root}",
        f"ITERATION={args.iteration}",
        f"BATCH_ID={args.batch_id}",
        f"LANE_IDS_FILE={cluster_lane_file}",
        f"HAMLIB_DATA_ROOT={data_root}",
        f"COMPUTE_BASELINE={1 if compute_baseline else 0}",
        f"POLICY_SNAPSHOT={args.policy_snapshot}",
    ]
    if args.legate_fbmem:
        export_parts.append(f"CAMPAIGN_LEGATE_FBMEM={args.legate_fbmem}")
    if args.uv_base_dir:
        export_parts.append(f"CAMPAIGN_UV_BASE_DIR={args.uv_base_dir}")
    if args.uv_cache_dir:
        export_parts.append(f"CAMPAIGN_UV_CACHE_DIR={args.uv_cache_dir}")
    if args.uv_python_install_dir:
        export_parts.append(f"CAMPAIGN_UV_PYTHON_INSTALL_DIR={args.uv_python_install_dir}")
    export_vars = ",".join(export_parts)
    cluster_job_file = (
        f"{cluster_root.rstrip('/')}/iterations/iter_{args.iteration:04d}/batches/"
        f"{args.batch_id}/slurm_job_id_{submission_key}.txt"
    )
    remote = (
        "set -eu; "
        f"job_file={shell_quote(cluster_job_file)}; "
        f"job_name={shell_quote(job_name)}; "
        f"cd {shell_quote(cluster_root)}; "
        f"mkdir -p logs && "
        'if [ -s "$job_file" ]; then cat "$job_file"; '
        "else existing=$(squeue -h -n \"$job_name\" -o '%A' | head -n 1); "
        'if [ -n "$existing" ]; then printf \'%s\\n\' "$existing" | tee "$job_file"; '
        f'else sbatch --parsable --job-name="$job_name" --account={shell_quote(args.account)} '
        f"--partition={shell_quote(submit_partition)} "
        f'--export={shell_quote(export_vars)} {shell_quote(slurm_script)} | tee "$job_file"; fi; fi'
    )
    completed = run_with_retries(
        ["ssh", args.submit_host, remote],
        attempts=args.command_attempts,
        timeout_seconds=args.submit_timeout_seconds,
    )
    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("sbatch returned no job id")
    job_id = output_lines[-1].split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise RuntimeError(f"invalid sbatch job id: {output_lines[-1]!r}")

    if tracked is not None:
        mark_eval_batch_submitted(conn, batch_id=args.batch_id, job_id=job_id)
    else:
        conn.executemany(
            "UPDATE lanes SET status='evaluating', updated_at=? WHERE lane_id=?",
            [(now_iso(), lane_id) for lane_id in lane_ids],
        )
        conn.commit()
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
            "account": args.account,
            "requested_partition": args.partition,
            "slurm_script": slurm_script,
            "sync_host": args.sync_host,
            "submit_host": args.submit_host,
            "legate_fbmem": args.legate_fbmem,
            "uv_base_dir": args.uv_base_dir,
            "uv_cache_dir": args.uv_cache_dir,
            "uv_python_install_dir": args.uv_python_install_dir,
        },
    )
    print(job_id)


if __name__ == "__main__":
    main()

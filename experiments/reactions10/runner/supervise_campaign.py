#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

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
    campaign_lock,
    ensure_campaign_runner_runtime,
    open_collect_transactions,
    proxy_jump_cli_args,
    read_json,
    require_campaign,
    validate_ssh_host,
)


def run_child(command: list[str], cwd: Path, *, allowed_codes: set[int] = {0}) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode not in allowed_codes:
        raise RuntimeError(f"campaign child exited {completed.returncode}: {' '.join(command)}")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume and supervise the fixed reaction campaign through iteration 50."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--sync-host", required=True)
    parser.add_argument("--submit-host", required=True)
    parser.add_argument(
        "--proxy-jump-host",
        help="Optional SSH config alias used as ProxyJump for sync and submit hosts.",
    )
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--max-concurrent-mutators", type=int, default=8)
    parser.add_argument("--mutation-timeout-seconds", type=int, default=1200)
    parser.add_argument("--evaluator-workers", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--evaluation-wait-timeout-seconds", type=int, default=18000)
    parser.add_argument("--max-evaluation-retries", type=int, default=3)
    parser.add_argument("--unbounded-evaluation-retries", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.max_concurrent_mutators,
            args.mutation_timeout_seconds,
            args.evaluator_workers,
            args.cpus_per_task,
            args.evaluation_wait_timeout_seconds,
        )
        <= 0
    ):
        parser.error("concurrency, timeout, worker, CPU, and polling values must be positive")
    if args.max_evaluation_retries < 0:
        parser.error("--max-evaluation-retries must be nonnegative")
    if args.evaluator_workers != 1:
        parser.error("one-A100 reaction jobs require --evaluator-workers 1")
    try:
        sync_host = validate_ssh_host(str(args.sync_host), "--sync-host")
        submit_host = validate_ssh_host(str(args.submit_host), "--submit-host")
        proxy_jump_host = (
            validate_ssh_host(str(args.proxy_jump_host), "--proxy-jump-host")
            if args.proxy_jump_host is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = args.campaign_root.resolve()
    runner = Path(__file__).resolve().parent
    with campaign_lock(root, "supervisor"):
        while True:
            manifest, state = require_campaign(root, allow_open_collect_transaction=True)
            ensure_campaign_runner_runtime(root, Path(__file__), manifest)
            open_transactions = open_collect_transactions(root)
            if open_transactions:
                if len(open_transactions) != 1:
                    raise RuntimeError(
                        f"multiple open collection transactions: {open_transactions}"
                    )
                transaction = read_json(open_transactions[0])
                run_child(
                    [
                        sys.executable,
                        str(runner / "collect_iteration.py"),
                        "--campaign-root",
                        str(root),
                        "--iteration",
                        str(transaction["iteration"]),
                        "--skip-rsync",
                    ]
                    + proxy_jump_cli_args(proxy_jump_host),
                    root,
                )
                continue
            manifest, state = require_campaign(root)
            next_iteration = state.get("next_iteration")
            if next_iteration is None:
                if int(state["current_collected_iteration"]) != 50:
                    raise RuntimeError("campaign ended before collecting iteration 50")
                print("reaction campaign complete: baseline + 50 proposals collected for 20 lanes")
                return
            iteration = int(next_iteration)
            if iteration > 0:
                run_child(
                    [
                        sys.executable,
                        str(runner / "mutate_iteration.py"),
                        "--campaign-root",
                        str(root),
                        "--iteration",
                        str(iteration),
                        "--max-concurrent",
                        str(args.max_concurrent_mutators),
                        "--timeout-seconds",
                        str(args.mutation_timeout_seconds),
                    ],
                    root,
                )

            _manifest, state = require_campaign(root)
            batches = [
                batch
                for batch in state["batches"].values()
                if int(batch.get("iteration", -1)) == iteration
            ]
            if not batches or any(
                batch.get("state") in {"prepared", "retry_ready"} for batch in batches
            ):
                retry_counts = [
                    int(batch.get("retry_count", 0))
                    for batch in batches
                    if batch.get("state") == "retry_ready"
                ]
                if (
                    retry_counts
                    and not args.unbounded_evaluation_retries
                    and max(retry_counts) > args.max_evaluation_retries
                ):
                    raise RuntimeError(
                        f"iteration {iteration} exceeded {args.max_evaluation_retries} "
                        "evaluation retries; inspect errors or pass --unbounded-evaluation-retries"
                    )
                run_child(
                    [
                        sys.executable,
                        str(runner / "submit_iteration.py"),
                        "--campaign-root",
                        str(root),
                        "--iteration",
                        str(iteration),
                        "--sync-host",
                        sync_host,
                        "--submit-host",
                        submit_host,
                        "--account",
                        args.account,
                        "--partition",
                        args.partition,
                        "--workers",
                        str(args.evaluator_workers),
                        "--cpus-per-task",
                        str(args.cpus_per_task),
                        "--memory",
                        args.memory,
                        "--time-limit",
                        args.time_limit,
                        "--submit-timeout-seconds",
                        str(args.evaluation_wait_timeout_seconds),
                    ]
                    + proxy_jump_cli_args(proxy_jump_host),
                    root,
                )

            collect_code = run_child(
                [
                    sys.executable,
                    str(runner / "collect_iteration.py"),
                    "--campaign-root",
                    str(root),
                    "--iteration",
                    str(iteration),
                    "--sync-host",
                    sync_host,
                ]
                + proxy_jump_cli_args(proxy_jump_host),
                root,
                allowed_codes={0, 2},
            )
            if collect_code == 2:
                print(
                    f"iteration {iteration}: retryable evaluator output; resubmitting", flush=True
                )


if __name__ == "__main__":
    main()

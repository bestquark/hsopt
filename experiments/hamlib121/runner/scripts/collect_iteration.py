#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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
    db_connect,
    ensure_dirs,
    init_db,
    log_event,
    mark_eval_batch_collected,
    now_iso,
    revalidate_rsync_binary,
    require_existing_rsync_provenance,
    resolve_rsync_binary,
    rsync_proxy_jump_args,
    ssh_command,
    validate_absolute_posix_path,
    validate_campaign_identity,
    validate_evaluation_result_provenance,
    validate_manifests_parent_chain,
    validate_result_submission_receipt,
    validate_ssh_host,
    write_json,
)
from render_telemetry import main as render_main  # noqa: E402


SQLITE_INT_MAX = 2**63 - 1


def sqlite_number(value):
    if isinstance(value, int) and abs(value) > SQLITE_INT_MAX:
        return float(value)
    return value


def proposal_policy_path(root: Path, iteration: int, lane_id: str) -> Path:
    if iteration == 0:
        initial_policy = root / "lanes" / lane_id / "initial_simple_pf.py"
        if initial_policy.is_file():
            return initial_policy
        return root / "iterations" / "iter_0001" / "mutations" / lane_id / "before_simple_pf.py"
    return (
        root / "iterations" / f"iter_{iteration:04d}" / "mutations" / lane_id / "after_simple_pf.py"
    )


def before_policy_path(root: Path, iteration: int, lane_id: str) -> Path | None:
    if iteration <= 0:
        return None
    return (
        root
        / "iterations"
        / f"iter_{iteration:04d}"
        / "mutations"
        / lane_id
        / "before_simple_pf.py"
    )


def mutation_summary(root: Path, iteration: int, lane_id: str) -> str:
    if iteration == 0:
        return "Measured-energy A100 baseline initializer."
    path = (
        root / "iterations" / f"iter_{iteration:04d}" / "mutations" / lane_id / "last_message.txt"
    )
    if not path.is_file():
        return ""
    summary = " ".join(path.read_text(encoding="utf-8", errors="replace").split())
    return summary[:1000]


def previous_incumbent(conn, lane_id: str, iteration: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT iteration, result_json, policy_path
        FROM evaluations
        WHERE lane_id=? AND accepted=1 AND iteration<?
        ORDER BY iteration DESC
        LIMIT 1
        """,
        (lane_id, iteration),
    ).fetchone()
    if row is None:
        return None
    result = json.loads(str(row["result_json"] or "{}"))
    total_t = result.get("total_t_gates_qpe")
    return {
        "iteration": int(row["iteration"]),
        "total_t_gates_qpe": None if total_t is None else int(total_t),
        "score": result.get("score"),
        "config": result.get("config", {}),
        "ground_state_energy_error_hartree": result.get("ground_state_energy_error_hartree"),
        "policy_path": row["policy_path"],
    }


def decide_candidate(
    *,
    iteration: int,
    result: dict[str, Any],
    incumbent: dict[str, Any] | None,
    proposal_path: Path,
) -> tuple[bool, str, float | None]:
    status = str(result.get("status", "error"))
    passes = bool(result.get("passes_chemical_accuracy_bound", False))
    total_t_raw = result.get("total_t_gates_qpe")
    total_t = None if total_t_raw is None else int(total_t_raw)

    relative_gain = None
    if incumbent is not None and total_t is not None and incumbent.get("total_t_gates_qpe"):
        incumbent_t = int(incumbent["total_t_gates_qpe"])
        relative_gain = 100.0 * (incumbent_t - total_t) / incumbent_t

    if status != "ok":
        return False, "evaluation_error", relative_gain
    if not passes:
        return False, "chemical_accuracy_failed", relative_gain
    if total_t is None or total_t <= 0:
        return False, "missing_t_gate_count", relative_gain
    if not proposal_path.is_file():
        return False, "missing_policy_snapshot", relative_gain
    if iteration == 0:
        return True, "initial_baseline", None
    if incumbent is None or incumbent.get("total_t_gates_qpe") is None:
        return False, "missing_accepted_baseline", relative_gain
    if total_t < int(incumbent["total_t_gates_qpe"]):
        return True, "strict_t_gate_improvement", relative_gain
    return False, "no_t_gate_improvement", relative_gain


def rsync_from_cluster(
    cluster_root: str,
    campaign_root: Path,
    iteration: int,
    sync_host: str,
    *,
    rsync_provenance: dict[str, Any],
    proxy_jump_host: str | None = None,
) -> None:
    cluster_root = validate_absolute_posix_path(cluster_root, "cluster_root")
    sync_host = validate_ssh_host(sync_host, "sync_host")
    revalidate_rsync_binary(rsync_provenance)
    rsync_binary = str(rsync_provenance["binary"])
    src = f"{sync_host}:{cluster_root.rstrip('/')}/iterations/iter_{iteration:04d}/results/"
    dst = campaign_root / "iterations" / f"iter_{iteration:04d}" / "results"
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            rsync_binary,
            "-ah",
            "--partial",
            "--protect-args",
            *rsync_proxy_jump_args(proxy_jump_host),
            src,
            f"{dst}/",
        ],
        check=True,
    )
    if iteration != 0:
        return

    baseline_dst = campaign_root / "baselines"
    probe = subprocess.run(
        ssh_command(sync_host, "bash", "-s", proxy_jump_host=proxy_jump_host),
        input=f"test -d {shlex.quote(cluster_root.rstrip('/') + '/baselines')}\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        revalidate_rsync_binary(rsync_provenance)
        baseline_dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                rsync_binary,
                "-ah",
                "--partial",
                "--protect-args",
                *rsync_proxy_jump_args(proxy_jump_host),
                f"{sync_host}:{cluster_root.rstrip('/')}/baselines/",
                f"{baseline_dst}/",
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one evaluated iteration from Lustre and update telemetry DB."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cluster-root", default=None)
    parser.add_argument("--sync-host", default=os.environ.get("HSOPT_SYNC_HOST"))
    parser.add_argument(
        "--proxy-jump-host",
        help="Optional SSH config alias used as ProxyJump for the sync host.",
    )
    parser.add_argument(
        "--rsync-bin",
        default=os.environ.get("HSOPT_RSYNC_BIN"),
        help=(
            "Explicit local rsync executable. Defaults to HSOPT_RSYNC_BIN, then discovers "
            "a compatible rsync >= 3 with --protect-args support."
        ),
    )
    parser.add_argument("--skip-rsync", action="store_true")
    parser.add_argument("--lane-id", action="append", default=[])
    parser.add_argument("--lane-ids-file", type=Path)
    parser.add_argument(
        "--batch-id",
        help=(
            "Close this durable submission record after collecting its exact result closure. "
            "Use this for the manually submitted iteration-0 baseline; the streaming "
            "supervisor closes its own batches."
        ),
    )
    parser.add_argument(
        "--expected-batch-id",
        help=(
            "Require every result to match this currently submitted batch, but leave the "
            "batch lifecycle transition to the streaming supervisor."
        ),
    )
    args = parser.parse_args()

    root = validate_manifests_parent_chain(args.campaign_root)
    if not args.skip_rsync and not args.sync_host:
        parser.error("pass --sync-host or set HSOPT_SYNC_HOST unless using --skip-rsync")
    proxy_jump_host = (
        validate_ssh_host(str(args.proxy_jump_host), "proxy_jump_host")
        if args.proxy_jump_host is not None
        else None
    )
    if not args.skip_rsync:
        try:
            rsync_provenance = resolve_rsync_binary(args.rsync_bin)
            require_existing_rsync_provenance(root, rsync_provenance)
        except RuntimeError as exc:
            parser.error(str(exc))
    manifest, _campaign_identity = validate_campaign_identity(root)
    lane_by_id = {str(lane["lane_id"]): lane for lane in manifest["lanes"] if lane.get("exact_fit")}
    cluster_root = args.cluster_root or manifest["cluster_root"]
    ensure_dirs(root)
    if not args.skip_rsync:
        rsync_from_cluster(
            cluster_root,
            root,
            args.iteration,
            args.sync_host,
            rsync_provenance=rsync_provenance,
            proxy_jump_host=proxy_jump_host,
        )

    conn = db_connect(root)
    init_db(conn)
    result_dir = root / "iterations" / f"iter_{args.iteration:04d}" / "results"
    result_paths = sorted(result_dir.glob("*.json"))
    selected_lane_ids = set(args.lane_id)
    if args.lane_ids_file is not None:
        selected_lane_ids.update(
            line.strip()
            for line in args.lane_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.batch_id is not None and args.expected_batch_id is not None:
        parser.error("--batch-id and --expected-batch-id are mutually exclusive")
    expected_batch_id = args.batch_id or args.expected_batch_id
    expected_batch = None
    if expected_batch_id is not None:
        row = conn.execute(
            "SELECT iteration, lane_ids_json, state FROM eval_batches WHERE batch_id=?",
            (expected_batch_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"batch {expected_batch_id}: durable submission intent is missing")
        if int(row["iteration"]) != args.iteration:
            raise RuntimeError(
                f"batch {expected_batch_id}: iteration {row['iteration']} does not match "
                f"requested iteration {args.iteration}"
            )
        batch_lane_ids = json.loads(str(row["lane_ids_json"]))
        if (
            not isinstance(batch_lane_ids, list)
            or not batch_lane_ids
            or any(not isinstance(lane_id, str) or not lane_id for lane_id in batch_lane_ids)
            or len(set(batch_lane_ids)) != len(batch_lane_ids)
        ):
            raise RuntimeError(f"batch {expected_batch_id}: malformed durable lane closure")
        if selected_lane_ids and selected_lane_ids != set(batch_lane_ids):
            raise RuntimeError(
                f"batch {expected_batch_id}: explicit lane selection differs from durable intent"
            )
        selected_lane_ids = set(batch_lane_ids)
        expected_batch = {
            "batch_id": expected_batch_id,
            "lane_ids": batch_lane_ids,
            "state": str(row["state"]),
        }
        allowed_states = {"submitted", "collected"} if args.batch_id is not None else {"submitted"}
        if expected_batch["state"] not in allowed_states:
            raise RuntimeError(
                f"batch {expected_batch_id}: cannot collect batch in state "
                f"{expected_batch['state']!r}"
            )
    if selected_lane_ids:
        result_paths = [path for path in result_paths if path.stem in selected_lane_ids]

    if expected_batch is not None:
        actual_result_ids = {path.stem for path in result_paths}
        expected_result_ids = set(expected_batch["lane_ids"])
        extra_result_ids = actual_result_ids - expected_result_ids
        missing_result_ids = expected_result_ids - actual_result_ids
        if extra_result_ids or (args.batch_id is not None and missing_result_ids):
            raise RuntimeError(
                f"batch {expected_batch_id}: result closure is incomplete or ambiguous: "
                f"missing={sorted(missing_result_ids)}, "
                f"extra={sorted(extra_result_ids)}"
            )

    ok_count = 0
    error_count = 0
    accepted_count = 0
    rejected_count = 0
    validated_status_by_lane: dict[str, str] = {}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        lane_id = result.get("lane_id")
        if not lane_id:
            raise RuntimeError(f"result {path.name!r} is missing its lane_id")
        if lane_id != path.stem:
            raise RuntimeError(
                f"result filename/lane identity mismatch: {path.name!r} contains {lane_id!r}"
            )
        status = result.get("status", "error")
        if lane_id not in lane_by_id:
            raise RuntimeError(f"result names unknown or inactive lane {lane_id!r}")
        validate_result_submission_receipt(
            conn,
            lane_id=str(lane_id),
            iteration=args.iteration,
            result=result,
            expected_batch_id=expected_batch_id,
        )
        validated_status_by_lane[str(lane_id)] = str(status)
        proposal_path = proposal_policy_path(root, args.iteration, lane_id)
        if status == "ok":
            validate_evaluation_result_provenance(
                campaign_root=root,
                lane=lane_by_id[lane_id],
                iteration=args.iteration,
                result=result,
                policy_path=proposal_path,
            )
        if status == "ok":
            ok_count += 1
        else:
            error_count += 1

        incumbent = previous_incumbent(conn, lane_id, args.iteration)
        accepted, decision_reason, relative_gain = decide_candidate(
            iteration=args.iteration,
            result=result,
            incumbent=incumbent,
            proposal_path=proposal_path,
        )
        accepted_count += int(accepted)
        rejected_count += int(not accepted)
        summary = mutation_summary(root, args.iteration, lane_id)
        result.update(
            {
                "accepted": accepted,
                "decision_reason": decision_reason,
                "relative_gain_percent": relative_gain,
                "incumbent_iteration": None if incumbent is None else incumbent["iteration"],
                "incumbent_t_gates_qpe": None
                if incumbent is None
                else incumbent["total_t_gates_qpe"],
                "mutation_summary": summary,
            }
        )
        write_json(path, result)

        config_json = json.dumps(result.get("config", {}), sort_keys=True)
        result_json = json.dumps(result, sort_keys=True)
        score = result.get("score")
        total_t = sqlite_number(result.get("total_t_gates_qpe"))
        total_error = result.get("ground_state_energy_error_hartree")
        conn.execute(
            """
            INSERT OR REPLACE INTO evaluations(
              lane_id, iteration, status, score, total_t_gates_qpe, total_error_hartree,
              qpe_bits, qpe_unitary_applications, pf_error_bound_hartree,
              synthesis_error_bound_hartree, qpe_grid_spacing_hartree, config_json,
              result_json, passes_accuracy, accepted, incumbent_iteration,
              incumbent_t_gates_qpe, relative_gain_percent, decision_reason,
              policy_path, mutation_summary, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lane_id,
                args.iteration,
                status,
                score,
                total_t,
                total_error,
                result.get("qpe_bits"),
                result.get("qpe_unitary_applications"),
                result.get("pf_error_bound_hartree"),
                result.get("synthesis_error_bound_hartree"),
                result.get("qpe_grid_spacing_hartree"),
                config_json,
                result_json,
                int(bool(result.get("passes_chemical_accuracy_bound", False))),
                int(accepted),
                None if incumbent is None else incumbent["iteration"],
                None if incumbent is None else sqlite_number(incumbent["total_t_gates_qpe"]),
                relative_gain,
                decision_reason,
                str(proposal_path),
                summary,
                now_iso(),
            ),
        )

        lane_row = conn.execute(
            "SELECT iteration FROM lanes WHERE lane_id=?", (lane_id,)
        ).fetchone()
        lane_iteration = -1 if lane_row is None else int(lane_row["iteration"])
        update_live_policy = args.iteration == 0 or lane_iteration <= args.iteration
        live_policy = root / "lanes" / lane_id / "simple_pf.py"
        restore_path = before_policy_path(root, args.iteration, lane_id)
        if update_live_policy and accepted:
            live_policy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(proposal_path, live_policy)
            conn.execute(
                """
                UPDATE lanes
                SET iteration=?, status='evaluated', best_t_gates_qpe=?, best_total_error_hartree=?,
                    best_score=?, best_iteration=?, best_policy_path=?, current_policy_path=?,
                    current_config_json=?, updated_at=?
                WHERE lane_id=?
                """,
                (
                    args.iteration,
                    total_t,
                    total_error,
                    score,
                    args.iteration,
                    str(proposal_path),
                    str(live_policy),
                    config_json,
                    now_iso(),
                    lane_id,
                ),
            )
        elif update_live_policy:
            if restore_path is not None and restore_path.is_file():
                live_policy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(restore_path, live_policy)
            incumbent_config = {} if incumbent is None else incumbent.get("config", {})
            conn.execute(
                """
                UPDATE lanes
                SET iteration=?, status=?, current_policy_path=?, current_config_json=?, updated_at=?
                WHERE lane_id=?
                """,
                (
                    args.iteration,
                    "evaluated" if status == "ok" else "error",
                    str(live_policy),
                    json.dumps(incumbent_config, sort_keys=True),
                    now_iso(),
                    lane_id,
                ),
            )
    conn.commit()
    if (
        args.batch_id is not None
        and expected_batch is not None
        and expected_batch["state"] != "collected"
    ):
        successful_lane_ids = [
            lane_id
            for lane_id in expected_batch["lane_ids"]
            if validated_status_by_lane[lane_id] == "ok"
        ]
        retry_lane_ids = [
            lane_id
            for lane_id in expected_batch["lane_ids"]
            if validated_status_by_lane[lane_id] != "ok"
        ]
        mark_eval_batch_collected(
            conn,
            batch_id=expected_batch["batch_id"],
            successful_lane_ids=successful_lane_ids,
            retry_lane_ids=retry_lane_ids,
        )
    log_event(
        conn,
        kind="collect",
        iteration=args.iteration,
        message=(
            f"collected iteration {args.iteration}: ok={ok_count}, error={error_count}, "
            f"accepted={accepted_count}, rejected={rejected_count}"
        ),
        payload={
            "ok": ok_count,
            "error": error_count,
            "accepted": accepted_count,
            "rejected": rejected_count,
        },
    )

    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["render_telemetry.py", "--campaign-root", str(root)]
        render_main()
    finally:
        sys.argv = old_argv
    print(
        f"collected iteration={args.iteration} ok={ok_count} error={error_count} "
        f"accepted={accepted_count} rejected={rejected_count}"
    )


if __name__ == "__main__":
    main()

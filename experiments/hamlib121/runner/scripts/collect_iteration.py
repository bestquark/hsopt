#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from campaign_common import db_connect, ensure_dirs, init_db, log_event, now_iso, write_json
from render_telemetry import main as render_main


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
        return "Reiher-style uniform-precision initializer."
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
        "total_error_bound_hartree": result.get("total_error_bound_hartree"),
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
    cluster_root: str, campaign_root: Path, iteration: int, sync_host: str
) -> None:
    src = f"{sync_host}:{cluster_root.rstrip('/')}/iterations/iter_{iteration:04d}/results/"
    dst = campaign_root / "iterations" / f"iter_{iteration:04d}" / "results"
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-ah", "--partial", src, f"{dst}/"], check=True)
    baseline_dst = campaign_root / "baselines"
    baseline_dst.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ssh", sync_host, f"test -d {cluster_root.rstrip()}/baselines"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        subprocess.run(
            [
                "rsync",
                "-ah",
                "--partial",
                f"{sync_host}:{cluster_root.rstrip('/')}/baselines/",
                f"{baseline_dst}/",
            ],
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one evaluated iteration from Lustre and update telemetry DB."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cluster-root", default=None)
    parser.add_argument("--sync-host", default=os.environ.get("HSOPT_SYNC_HOST"))
    parser.add_argument("--skip-rsync", action="store_true")
    parser.add_argument("--lane-id", action="append", default=[])
    parser.add_argument("--lane-ids-file", type=Path)
    args = parser.parse_args()

    root = args.campaign_root.resolve()
    paths = ensure_dirs(root)
    manifest = json.loads(
        (paths["manifests"] / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    cluster_root = args.cluster_root or manifest["cluster_root"]
    if not args.skip_rsync and not args.sync_host:
        parser.error("pass --sync-host or set HSOPT_SYNC_HOST unless using --skip-rsync")
    if not args.skip_rsync:
        rsync_from_cluster(cluster_root, root, args.iteration, args.sync_host)

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
    if selected_lane_ids:
        result_paths = [path for path in result_paths if path.stem in selected_lane_ids]

    ok_count = 0
    error_count = 0
    accepted_count = 0
    rejected_count = 0
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        lane_id = result.get("lane_id")
        if not lane_id:
            continue
        status = result.get("status", "error")
        if status == "ok":
            ok_count += 1
        else:
            error_count += 1

        proposal_path = proposal_policy_path(root, args.iteration, lane_id)
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
        total_error = result.get("total_error_bound_hartree")
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

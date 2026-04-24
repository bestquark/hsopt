from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import (
    DEFAULT_FAMILY,
    DEFAULT_MOLECULE,
    REPO_ROOT,
    discover_lanes,
    evaluate_command,
    next_iteration,
    repo_relative,
    resolve_lane,
    restore_command,
    shell_join,
    summarize_lane,
    track_command,
)


def resolve_from_args(args: argparse.Namespace, require_supported: bool = True):
    return resolve_lane(
        args.family,
        args.molecule,
        REPO_ROOT,
        require_supported=require_supported,
        target=getattr(args, "target", None),
    )


def build_agent_prompt(lane, target_iteration: int | None = None) -> str:
    target_iteration = lane.target_iteration if target_iteration is None else target_iteration
    last_iteration = next_iteration(lane) - 1
    resume_iteration = last_iteration + 1
    instruction = f"Read {repo_relative(lane.instruction_path)} first." if lane.instruction_path else ""
    baseline_clause = (
        "- snapshots are empty; archive the untouched baseline as iteration 0 before the first mutation\n"
        if last_iteration < 0
        else ""
    )
    return f"""Use the `$t-autoresearch` workflow for `{lane.key}`.

{instruction}

Current state:
- editable file: `{repo_relative(lane.editable_script)}`
- last archived iteration: {last_iteration}
- next iteration: {resume_iteration}
- target iteration: {target_iteration}
{baseline_clause}

Goal:
- minimize the evaluator's lower-is-better `score`
- in practice, reduce compiled `t_gate_count`
- preserve the fixed accuracy contract reported by the evaluator

Mutation loop:
1. Restore the best kept valid iteration before mutating if the live file is not already best.
2. Edit only `{repo_relative(lane.editable_script)}` unless the user explicitly changes scope.
3. Make one coherent precision-policy mutation.
4. Score it with:
   `{shell_join(track_command(lane, "<one-line technical mutation summary>"))}`
5. Read the JSON result before choosing the next mutation.
6. If the result is `discard` or `crash`, restore best with:
   `{shell_join(restore_command(lane))}`
7. Keep every scored proposal archived. Do not delete, renumber, or overwrite snapshots.
8. Leave the live file restored to the best valid version when stopping.
"""


def cmd_list(args: argparse.Namespace) -> int:
    rows = [summarize_lane(lane) for lane in discover_lanes(REPO_ROOT) if args.all or lane.supported]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'lane':<38} {'ok':<5} {'last':>5} {'best_T':>12} {'reduction':>10}  file")
    print("-" * 102)
    for row in rows:
        reduction = row["t_gate_reduction_pct"]
        reduction_text = "-" if reduction is None else f"{reduction:.2f}%"
        best_t = "-" if row["best_t_gate_count"] is None else f"{row['best_t_gate_count']:,}"
        print(
            f"{row['key']:<38} {str(row['supported']):<5} "
            f"{row['last_archived_iteration']:>5} {best_t:>12} {reduction_text:>10}  {row['editable_script']}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args, require_supported=False)
    payload = summarize_lane(lane)
    payload["evaluate_command"] = shell_join(evaluate_command(lane))
    payload["track_command"] = shell_join(track_command(lane, "<description>"))
    payload["restore_command"] = shell_join(restore_command(lane))
    print(json.dumps(payload, indent=2))
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    print(build_agent_prompt(lane, args.target_iteration))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    cmd = evaluate_command(lane, Path(args.script).resolve() if args.script else None)
    if args.print_only:
        print(shell_join(cmd))
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def cmd_track(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    cmd = track_command(lane, args.description, Path(args.script).resolve() if args.script else None)
    if args.print_only:
        print(shell_join(cmd))
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def cmd_restore(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    cmd = restore_command(lane, Path(args.script).resolve() if args.script else None)
    if args.print_only:
        print(shell_join(cmd))
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect trovolut T-gate autoresearch lanes.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List known T-gate optimization lanes.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument("--all", action="store_true", help="Include scaffolded lanes without runnable evaluators.")
    list_parser.set_defaults(func=cmd_list)

    status = sub.add_parser("status", help="Print lane status and canonical commands.")
    status.add_argument(
        "target",
        nargs="?",
        help="Optional molecule folder, e.g. examples/ft_protocols/pfs/h2 or pfs/h2.",
    )
    status.add_argument("--family", default=DEFAULT_FAMILY)
    status.add_argument("--molecule", default=DEFAULT_MOLECULE)
    status.set_defaults(func=cmd_status)

    prompt = sub.add_parser("prompt", help="Render the agent prompt for a lane.")
    prompt.add_argument(
        "target",
        nargs="?",
        help="Optional molecule folder, e.g. examples/ft_protocols/pfs/h2 or pfs/h2.",
    )
    prompt.add_argument("--family", default=DEFAULT_FAMILY)
    prompt.add_argument("--molecule", default=DEFAULT_MOLECULE)
    prompt.add_argument("--target-iteration", type=int)
    prompt.set_defaults(func=cmd_prompt)

    evaluate = sub.add_parser("evaluate", help="Run or print the fixed evaluator command.")
    evaluate.add_argument(
        "target",
        nargs="?",
        help="Optional molecule folder, e.g. examples/ft_protocols/pfs/h2 or pfs/h2.",
    )
    evaluate.add_argument("--family", default=DEFAULT_FAMILY)
    evaluate.add_argument("--molecule", default=DEFAULT_MOLECULE)
    evaluate.add_argument("--script")
    evaluate.add_argument("--print-only", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    track = sub.add_parser("track", help="Run or print one queued tracked iteration command.")
    track.add_argument(
        "target",
        nargs="?",
        help="Optional molecule folder, e.g. examples/ft_protocols/pfs/h2 or pfs/h2.",
    )
    track.add_argument("--family", default=DEFAULT_FAMILY)
    track.add_argument("--molecule", default=DEFAULT_MOLECULE)
    track.add_argument("--script")
    track.add_argument("--description", required=True)
    track.add_argument("--print-only", action="store_true")
    track.set_defaults(func=cmd_track)

    restore = sub.add_parser("restore", help="Run or print the restore-best command.")
    restore.add_argument(
        "target",
        nargs="?",
        help="Optional molecule folder, e.g. examples/ft_protocols/pfs/h2 or pfs/h2.",
    )
    restore.add_argument("--family", default=DEFAULT_FAMILY)
    restore.add_argument("--molecule", default=DEFAULT_MOLECULE)
    restore.add_argument("--script")
    restore.add_argument("--print-only", action="store_true")
    restore.set_defaults(func=cmd_restore)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

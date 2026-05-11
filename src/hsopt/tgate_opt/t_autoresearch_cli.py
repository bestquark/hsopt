from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import (
    DEFAULT_EVALUATOR,
    DEFAULT_FAMILY,
    DEFAULT_MOLECULE,
    REPO_ROOT,
    discover_lanes,
    evaluate_command,
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


def build_agent_prompt(
    lane,
    target_iteration: int | None = None,
    evaluator: str = DEFAULT_EVALUATOR,
    epsilon: float | None = None,
) -> str:
    target_iteration = lane.target_iteration if target_iteration is None else target_iteration
    eval_command = shell_join(evaluate_command(lane, evaluator=evaluator, epsilon=epsilon))
    return f"""Use the `$t-autoresearch` workflow for `{lane.key}`.

Current state:
- editable file: `{repo_relative(lane.editable_script)}`
- evaluator: `{evaluator}`
- target iterations: {target_iteration}

Goal:
- minimize the evaluator's lower-is-better `score`
- in practice, reduce compiled `t_gate_count`
- preserve the fixed accuracy contract reported by the selected evaluator

Mutation loop:
1. Edit only `{repo_relative(lane.editable_script)}` unless the user explicitly changes scope.
2. Make one coherent precision-policy mutation.
3. Score it with:
   `{eval_command}`
4. Read the JSON result before choosing the next mutation.
5. Keep campaign history in the external run directory used by the caller. The simplified `pfs/` tree contains only benchmark definitions and fixed evaluators.
"""


def cmd_list(args: argparse.Namespace) -> int:
    rows = [summarize_lane(lane) for lane in discover_lanes(REPO_ROOT) if args.all or lane.supported]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'lane':<16} {'ok':<5} file")
    print("-" * 86)
    for row in rows:
        print(f"{row['key']:<16} {str(row['supported']):<5} {row['editable_script']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args, require_supported=False)
    payload = summarize_lane(lane)
    payload["evaluate_energy_command"] = shell_join(evaluate_command(lane, evaluator="energy"))
    payload["evaluate_fidelity_command"] = shell_join(
        evaluate_command(lane, evaluator="fidelity", epsilon=args.epsilon)
    )
    payload["selected_evaluator_command"] = shell_join(
        evaluate_command(lane, evaluator=args.evaluator, epsilon=args.epsilon)
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    print(build_agent_prompt(lane, args.target_iteration, args.evaluator, args.epsilon))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    cmd = evaluate_command(
        lane,
        Path(args.script).resolve() if args.script else None,
        evaluator=args.evaluator,
        epsilon=args.epsilon,
    )
    if args.print_only:
        print(shell_join(cmd))
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def cmd_track(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    cmd = track_command(
        lane,
        args.description,
        Path(args.script).resolve() if args.script else None,
        evaluator=args.evaluator,
        epsilon=args.epsilon,
    )
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


def add_lane_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        nargs="?",
        help="Optional molecule folder, e.g. examples/ft_protocols/pfs/h2 or pfs/h2.",
    )
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE)


def add_evaluator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluator", choices=("energy", "fidelity"), default=DEFAULT_EVALUATOR)
    parser.add_argument("--epsilon", type=float, help="Allowed infidelity for --evaluator fidelity.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect hsopt T-gate autoresearch lanes.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List known T-gate optimization lanes.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument("--all", action="store_true", help="Include scaffolded lanes without runnable evaluators.")
    list_parser.set_defaults(func=cmd_list)

    status = sub.add_parser("status", help="Print lane status and canonical commands.")
    add_lane_args(status)
    add_evaluator_args(status)
    status.set_defaults(func=cmd_status)

    prompt = sub.add_parser("prompt", help="Render the agent prompt for a lane.")
    add_lane_args(prompt)
    add_evaluator_args(prompt)
    prompt.add_argument("--target-iteration", type=int)
    prompt.set_defaults(func=cmd_prompt)

    evaluate = sub.add_parser("evaluate", help="Run or print the fixed evaluator command.")
    add_lane_args(evaluate)
    add_evaluator_args(evaluate)
    evaluate.add_argument("--script")
    evaluate.add_argument("--print-only", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    track = sub.add_parser("track", help="Compatibility alias for one direct evaluator run.")
    add_lane_args(track)
    add_evaluator_args(track)
    track.add_argument("--script")
    track.add_argument("--description", default="")
    track.add_argument("--print-only", action="store_true")
    track.set_defaults(func=cmd_track)

    restore = sub.add_parser("restore", help="Compatibility no-op; simplified pfs has no in-tree snapshots.")
    add_lane_args(restore)
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

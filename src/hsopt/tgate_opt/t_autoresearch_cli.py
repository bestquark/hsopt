from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
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
from .execution import (
    EvaluationRequest,
    build_local_evaluation,
    build_slurm_submission,
    merged_environment,
)
from .mutation_agents import (
    NEMOTRON_MODEL,
    build_mutation_invocation,
    invocation_environment,
    validate_agent_lifecycle,
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
    return f"""Use the `$hsopt` workflow for `{lane.key}`.

Current state:
- editable file: `{repo_relative(lane.editable_script)}`
- evaluator: `{evaluator}`
- target iterations: {target_iteration}

Goal:
- minimize the evaluator's lower-is-better `score`
- in practice, reduce compiled `t_gate_count`
- preserve the fixed accuracy contract reported by the selected evaluator
- for product-formula lanes, use Hamiltonian-aware precision and ordering
  policies instead of only changing scalar config values

Mutation loop:
1. Edit only `{repo_relative(lane.editable_script)}` unless the user explicitly changes scope.
2. Inspect the editable product-formula API before editing. Prefer one coherent
   Hamiltonian-aware mutation:
   - structured synthesis precision in `ProductFormula.synthesis_epsilon(term, angle)`
   - term ordering, grouping, or recurrence changes in `ProductFormula.build_step`
   - repetition changes in `ProductFormula.repetitions`
3. Score it with:
   `{eval_command}`
4. Read the JSON result before choosing the next mutation.
5. If changing formula order or PF steps, check that `scheduled_rotation_count`,
   `nonzero_rotation_count`, and `t_gate_count` scale with the added rotations.
6. Keep campaign history in the external run directory used by the caller. The simplified `pfs/` tree contains only benchmark definitions and fixed evaluators.

Product-formula search guidance:
- `ProductFormula(problem)` receives a plain dict with `name`, `n_qubits`,
  `electrons`, `target_time`, `identity_coefficient`, and `terms`.
- Each term is a plain dict with `label`, `coefficient`, `support_size`,
  `x_count`, `y_count`, and `z_count`. `label` is an `n_qubits`-character
  I/X/Y/Z string.
- `repetitions()` supplies positive `n_steps`; the evaluator passes
  `step_time = problem["target_time"] / n_steps` to `build_step`.
- `build_step` returns `(term, angle)` pairs for one stationary step, where
  `angle` is the real `alpha` in `exp(+i * alpha * P)`.
- `synthesis_epsilon` receives a fresh trusted term dict plus that exact
  per-application angle and must return a finite PyGridSynth tolerance in
  `(0, 1)`. Use those fields to
  allocate looser precision to low-impact rotations and tighter precision where
  accuracy is likely to matter.
- The rotation schedule returned by `build_step` defines the formula. The
  policy may use any deterministic ordering, higher-order composition,
  grouping, or custom explicit stationary schedule.
- Do not drop terms, change Hamiltonian coefficients, or edit evaluator code.
  Record every seed and keep the policy small enough to audit and reproduce.
"""


def cmd_list(args: argparse.Namespace) -> int:
    rows = [
        summarize_lane(lane) for lane in discover_lanes(REPO_ROOT) if args.all or lane.supported
    ]
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
    script = Path(args.script).resolve() if args.script else lane.editable_script
    request = EvaluationRequest(
        evaluator=lane.root_dir / f"evaluate_{args.evaluator}.py",
        molecule=lane.molecule,
        script=script,
        epsilon=args.epsilon if args.evaluator == "fidelity" else None,
    )
    if args.transport == "local":
        invocation = build_local_evaluation(
            request,
            backend=args.backend,
            python=args.python or sys.executable,
            legate=args.legate,
        )
    else:
        required = {
            "--submit-host": args.submit_host,
            "--remote-repo-root": args.remote_repo_root,
            "--account": args.account,
            "--partition": args.partition,
            "--remote-python": args.remote_python,
            "--remote-legate": args.remote_legate,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Slurm evaluation requires: {', '.join(missing)}")
        invocation = build_slurm_submission(
            request,
            submit_host=args.submit_host,
            remote_repo_root=args.remote_repo_root,
            account=args.account,
            partition=args.partition,
            python=args.remote_python,
            legate=args.remote_legate,
            proxy_jump_host=args.proxy_jump_host,
            time_limit=args.time_limit,
            memory=args.memory,
            cpus_per_task=args.cpus_per_task,
        )
    if args.print_only:
        print(shell_join(invocation.command))
        if invocation.stdin:
            print(invocation.stdin, end="")
        return 0
    proc = subprocess.run(
        invocation.command,
        cwd=REPO_ROOT,
        env=merged_environment(invocation),
        input=invocation.stdin or None,
        text=bool(invocation.stdin),
        check=False,
    )
    return proc.returncode


def cmd_mutate(args: argparse.Namespace) -> int:
    lane = resolve_from_args(args)
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"mutation run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "work"
    state_dir = run_dir / "agent-state"
    work_dir.mkdir()
    state_dir.mkdir()
    before = lane.editable_script.read_bytes()
    candidate = work_dir / "simple_pf.py"
    candidate.write_bytes(before)
    (run_dir / "before_simple_pf.py").write_bytes(before)
    prompt = build_agent_prompt(
        lane,
        target_iteration=1,
        evaluator=args.evaluator,
        epsilon=args.epsilon,
    )
    prompt += "\nThis invocation prepares exactly one proposal. Edit simple_pf.py now.\n"
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    if args.agent == "opencode" and not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit(
            "OpenCode/Nemotron requires NVIDIA_API_KEY in the process environment; "
            "do not put the key in the repository or command line"
        )
    invocation = build_mutation_invocation(
        agent=args.agent,
        work_dir=work_dir,
        prompt=prompt,
        state_dir=state_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    stdout_path = run_dir / f"{invocation.agent}.jsonl"
    stderr_path = run_dir / f"{invocation.agent}.stderr"
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                invocation.command,
                cwd=work_dir,
                env=invocation_environment(invocation),
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"{invocation.agent} mutation timed out after {args.timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        raise SystemExit(f"{invocation.agent} mutation exited with status {completed.returncode}")
    lifecycle = validate_agent_lifecycle(invocation.agent, stdout_path)
    entries = list(work_dir.iterdir())
    if entries != [candidate] or candidate.is_symlink() or not candidate.is_file():
        raise SystemExit("isolated mutation workspace contains files other than simple_pf.py")
    after = candidate.read_bytes()
    if after == before:
        raise SystemExit(f"{invocation.agent} completed without changing simple_pf.py")
    compile(after.decode("utf-8"), str(candidate), "exec")
    (run_dir / "after_simple_pf.py").write_bytes(after)
    (run_dir / "last_message.txt").write_text(lifecycle.final_text + "\n", encoding="utf-8")
    if not args.no_apply:
        shutil.copy2(candidate, lane.editable_script)
    print(
        json.dumps(
            {
                "status": "mutated",
                "agent": invocation.agent,
                "model": invocation.model,
                "session_id": lifecycle.session_id,
                "edit_count": lifecycle.edit_count,
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
                "applied": not args.no_apply,
                "run_dir": str(run_dir),
            },
            sort_keys=True,
        )
    )
    return 0


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


def cmd_extrapolate_femoco(args: argparse.Namespace) -> int:
    from .extrapolate import run_extrapolate_cli

    return run_extrapolate_cli(args)


def add_lane_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        nargs="?",
        help="Optional legacy molecule folder; no legacy lanes ship on main.",
    )
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE)


def add_evaluator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluator", choices=("energy", "fidelity"), default=DEFAULT_EVALUATOR)
    parser.add_argument(
        "--epsilon", type=float, help="Allowed infidelity for --evaluator fidelity."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or inspect hsopt T-gate autoresearch lanes.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List known T-gate optimization lanes.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.add_argument(
        "--all", action="store_true", help="Include scaffolded lanes without runnable evaluators."
    )
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
    evaluate.add_argument("--transport", choices=("local", "slurm"), default="local")
    evaluate.add_argument("--backend", choices=("auto", "numpy", "cupynumeric"), default="auto")
    evaluate.add_argument("--python", help="Local evaluator Python; defaults to this environment.")
    evaluate.add_argument("--legate", help="Local Legate executable for --backend cupynumeric.")
    evaluate.add_argument("--submit-host")
    evaluate.add_argument("--proxy-jump-host")
    evaluate.add_argument("--remote-repo-root")
    evaluate.add_argument("--account")
    evaluate.add_argument("--partition")
    evaluate.add_argument("--remote-python")
    evaluate.add_argument("--remote-legate")
    evaluate.add_argument("--time-limit", default="04:00:00")
    evaluate.add_argument("--memory", default="64G")
    evaluate.add_argument("--cpus-per-task", type=int, default=8)
    evaluate.add_argument("--print-only", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    mutate = sub.add_parser("mutate", help="Create one isolated mutation with Codex or OpenCode.")
    add_lane_args(mutate)
    add_evaluator_args(mutate)
    mutate.add_argument("--agent", choices=("codex", "opencode"), default="codex")
    mutate.add_argument("--model", help=f"Defaults to {NEMOTRON_MODEL} for OpenCode.")
    mutate.add_argument("--reasoning-effort", default="high", help="Codex-only reasoning effort.")
    mutate.add_argument(
        "--run-dir", required=True, help="External directory for this mutation history."
    )
    mutate.add_argument("--timeout-seconds", type=int, default=1200)
    mutate.add_argument("--no-apply", action="store_true")
    mutate.set_defaults(func=cmd_mutate)

    track = sub.add_parser("track", help="Compatibility alias for one direct evaluator run.")
    add_lane_args(track)
    add_evaluator_args(track)
    track.add_argument("--script")
    track.add_argument("--description", default="")
    track.add_argument("--print-only", action="store_true")
    track.set_defaults(func=cmd_track)

    restore = sub.add_parser(
        "restore", help="Compatibility no-op; simplified pfs has no in-tree snapshots."
    )
    add_lane_args(restore)
    restore.add_argument("--script")
    restore.add_argument("--print-only", action="store_true")
    restore.set_defaults(func=cmd_restore)

    extrapolate = sub.add_parser(
        "extrapolate-femoco",
        help="Extrapolate observed T-gate gains to a FeMoco-sized active space.",
    )
    from .extrapolate import build_extrapolate_parser

    build_extrapolate_parser(extrapolate)
    extrapolate.set_defaults(func=cmd_extrapolate_femoco)

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

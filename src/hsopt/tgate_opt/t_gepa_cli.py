from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_FAMILY,
    REPO_ROOT,
    Lane,
    discover_lanes,
    next_iteration,
    normalize_family,
    resolve_lane,
    restore_command,
    shell_join,
    summarize_lane,
    track_command,
)
from .t_autoresearch_cli import build_agent_prompt


def default_seed_prompt() -> str:
    return """You are optimizing product-formula synthesis precision for lower compiled T count.

Focus on simple, auditable changes in `simple_pf.py`:
- use Hamiltonian term metadata in `choose_synthesis_epsilon`: Pauli label,
  coefficient magnitude, support size, Pauli-family counts, and actual
  per-application rotation angle
- group or reorder Pauli terms by label family, support size, coefficient
  bands, Z-only versus mixed X/Y/Z structure, or alternating large/small
  coefficients when the method exposes an ordering hook
- relax epsilons for low-impact terms first
- tighten only the term families or coefficient/angle bands implicated by
  failed accuracy diagnostics
- prefer broad policy changes over one-off cosmetic edits
- keep `n_steps` and `formula_order` fixed unless diagnostics show the product
  formula error itself is limiting

After each scored run, interpret `t_gate_count`, `passes_accuracy_constraint`,
`ground_state_energy_error_hartree` or `fidelity`, and the synthesis epsilon
histogram before choosing the next mutation.
"""


def default_examples(family: str = DEFAULT_FAMILY, smoke: bool = False) -> list[dict[str, Any]]:
    family = normalize_family(family)
    molecules = ["h2"] if smoke else ["h2", "bh", "lih", "dmds"]
    lanes = {lane.key: lane for lane in discover_lanes(REPO_ROOT)}
    examples: list[dict[str, Any]] = []
    for molecule in molecules:
        key = f"{family}:{molecule}"
        if key not in lanes or not lanes[key].supported:
            continue
        examples.append(
            {
                "id": key.replace(":", "_"),
                "family": family,
                "molecule": molecule,
                "target_iterations": 1 if smoke else 2,
                "max_agent_seconds": 900 if smoke else 1800,
            }
        )
    return examples


def write_default_dataset(path: Path, family: str, smoke: bool) -> list[dict[str, Any]]:
    examples = default_examples(family, smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2) + "\n")
    return examples


def load_examples(path: Path | None, family: str, smoke: bool) -> list[dict[str, Any]]:
    if path is None:
        return default_examples(family, smoke)
    return json.loads(path.read_text())


def candidate_text(candidate: str | dict[str, str]) -> str:
    if isinstance(candidate, dict):
        return candidate.get("prompt", "")
    return str(candidate)


def stable_trial_name(candidate: str, example: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps({"candidate": candidate, "example": example}, sort_keys=True).encode()
    ).hexdigest()[:12]
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}"
    return f"{example.get('id', 'example')}_{stamp}_{digest}"


def _copy_ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "runs",
        "eval_queue",
        "snapshots",
        "reset_archive",
        "gepa_results",
        "t_gepa_workspace",
        "hsgepa_workspace",
        "_archive",
    }
    result: set[str] = set()
    for name in names:
        path = Path(_dir) / name
        if name in ignored or name.endswith(".pyc") or name.endswith(".log"):
            result.add(name)
        elif path.is_dir() and (name.startswith(".track_") or "_track_" in name):
            result.add(name)
    return result


def prepare_trial_workspace(trial_root: Path, family: str) -> Path:
    repo = trial_root / "repo"
    repo.mkdir(parents=True, exist_ok=False)
    for filename in ["pyproject.toml", "uv.lock", "README.md", "AGENTS.md"]:
        source = REPO_ROOT / filename
        if source.exists():
            shutil.copy2(source, repo / filename)
    _ = family

    for relative in [
        Path("src/hsopt/__init__.py"),
        Path("src/hsopt/tgate_opt"),
        Path("examples/pfs"),
        Path("skills"),
    ]:
        source = REPO_ROOT / relative
        if not source.exists():
            continue
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_copy_ignore)
        else:
            shutil.copy2(source, destination)
    return repo


def archive_baseline(repo_root: Path, lane: Lane) -> dict[str, Any]:
    if next_iteration(lane) > 0:
        return {"skipped": True, "reason": "baseline already archived"}
    proc = subprocess.run(
        track_command(lane, "trial baseline before GEPA prompt evaluation"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    if proc.returncode == 0:
        try:
            payload["result"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return payload


def build_trial_agent_prompt(candidate_prompt: str, lane: Lane, target_iteration: int) -> str:
    base_prompt = build_agent_prompt(lane, target_iteration)
    return f"""{base_prompt}

Candidate strategy prompt being evaluated by `$hsgepa`:
---
{candidate_prompt.strip()}
---

Use the candidate strategy as the main search policy. Complete the requested
trial iterations, then stop with the live file restored to the best valid
version.
"""


def launch_agent(
    prompt: str,
    repo_root: Path,
    output_path: Path,
    model: str | None,
    max_seconds: int,
    dry_run: bool,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "agent_prompt.md").write_text(prompt)
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(repo_root),
        "-o",
        str(output_path),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    if dry_run:
        return {"dry_run": True, "command": shell_join(cmd), "prompt_file": str(output_path.parent / "agent_prompt.md")}
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=max_seconds, check=False)
    (output_path.parent / "agent_stdout.txt").write_text(proc.stdout)
    (output_path.parent / "agent_stderr.txt").write_text(proc.stderr)
    return {
        "dry_run": False,
        "command": shell_join(cmd),
        "returncode": proc.returncode,
        "duration_seconds": time.perf_counter() - start,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def score_trial(lane: Lane) -> tuple[float, dict[str, Any]]:
    summary = summarize_lane(lane)
    baseline_t = summary.get("baseline_t_gate_count")
    best_t = summary.get("best_t_gate_count")
    best_iteration = summary.get("best_iteration")
    if not isinstance(baseline_t, int) or baseline_t <= 0:
        reward = -0.25
        status = "missing_baseline"
    elif best_t is None or best_iteration is None:
        reward = -0.1
        status = "no_valid_result"
    else:
        reward = max(-0.05, (baseline_t - best_t) / baseline_t)
        status = "valid"
    side_info = {
        "scores": {
            "t_gate_reduction_fraction": reward,
            "best_t_gate_count": float(best_t) if best_t is not None else 0.0,
        },
        "Feedback": {
            "status": status,
            "summary": summary,
        },
    }
    return float(reward), side_info


def evaluate_candidate_prompt(
    candidate: str | dict[str, str],
    example: dict[str, Any],
    *,
    run_root: Path,
    model: str | None,
    dry_run: bool,
) -> tuple[float, dict[str, Any]]:
    prompt = candidate_text(candidate)
    if not prompt.strip():
        return -0.25, {"Feedback": {"status": "empty candidate prompt"}, "scores": {"t_gate_reduction_fraction": -0.25}}

    family = normalize_family(str(example.get("family", DEFAULT_FAMILY)))
    molecule = str(example.get("molecule", "h2"))
    target_iterations = int(example.get("target_iterations", 1))
    max_agent_seconds = int(example.get("max_agent_seconds", 900))
    trial_root = run_root / stable_trial_name(prompt, example)
    repo_root = prepare_trial_workspace(trial_root, family)
    lane = resolve_lane(family, molecule, repo_root=repo_root)

    baseline = {"dry_run": True} if dry_run else archive_baseline(repo_root, lane)
    next_after_baseline = 1 if dry_run else next_iteration(lane)
    target_iteration = next_after_baseline + max(0, target_iterations - 1)
    agent_prompt = build_trial_agent_prompt(prompt, lane, target_iteration)
    agent = launch_agent(
        agent_prompt,
        repo_root,
        trial_root / "agent" / "last_message.txt",
        model,
        max_agent_seconds,
        dry_run,
    )
    if dry_run:
        payload = {
            "trial_root": str(trial_root),
            "repo_root": str(repo_root),
            "example": example,
            "baseline": baseline,
            "agent": agent,
        }
        (trial_root / "dry_run.json").write_text(json.dumps(payload, indent=2))
        return 0.0, {"Feedback": payload, "scores": {"t_gate_reduction_fraction": 0.0}}

    restore = subprocess.run(restore_command(lane), cwd=repo_root, capture_output=True, text=True, check=False)
    reward, side_info = score_trial(lane)
    side_info["Input"] = {"example": example, "candidate_prompt": prompt}
    side_info["Generated Outputs"] = {
        "trial_root": str(trial_root),
        "agent": agent,
        "restore_returncode": restore.returncode,
        "restore_stdout_tail": restore.stdout[-2000:],
        "restore_stderr_tail": restore.stderr[-2000:],
    }
    (trial_root / "evaluation.json").write_text(json.dumps({"reward": reward, "side_info": side_info}, indent=2))
    return reward, side_info


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def cmd_seed_prompt(_args: argparse.Namespace) -> int:
    print(default_seed_prompt())
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    examples = write_default_dataset(Path(args.output), args.family, args.smoke)
    print(json.dumps({"output": args.output, "examples": examples}, indent=2))
    return 0


def cmd_evaluate_prompt(args: argparse.Namespace) -> int:
    prompt = Path(args.prompt_file).read_text()
    example = {
        "id": args.example_id or f"{args.family}_{args.molecule}",
        "family": args.family,
        "molecule": args.molecule,
        "target_iterations": args.target_iterations,
        "max_agent_seconds": args.max_agent_seconds,
    }
    reward, side_info = evaluate_candidate_prompt(
        {"prompt": prompt},
        example,
        run_root=Path(args.run_root).resolve(),
        model=args.model,
        dry_run=args.dry_run,
    )
    print(json.dumps({"reward": reward, "side_info": side_info}, indent=2))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    try:
        from gepa.optimize_anything import EngineConfig, GEPAConfig, optimize_anything
    except ImportError as exc:
        raise SystemExit(
            "GEPA is not installed. Install the optional dependency with `uv sync --extra gepa` "
            "or `uv pip install gepa`, then rerun `hsgepa optimize`."
        ) from exc

    seed = Path(args.seed_prompt).read_text() if args.seed_prompt else default_seed_prompt()
    dataset = load_examples(Path(args.dataset) if args.dataset else None, args.family, args.smoke)
    valset = load_examples(Path(args.valset), args.family, args.smoke) if args.valset else None
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    def evaluator(candidate: dict[str, str], example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        return evaluate_candidate_prompt(candidate, example, run_root=run_root / "trials", model=args.model, dry_run=False)

    result = optimize_anything(
        seed_candidate={"prompt": seed},
        evaluator=evaluator,
        dataset=dataset,
        valset=valset,
        objective=(
            "Optimize an agent strategy prompt that makes Codex reduce compiled T-gate counts "
            "in hsopt product-formula benchmarks while preserving the fixed evaluator accuracy contracts."
        ),
        background=(
            "Each evaluation runs the candidate prompt in an isolated copy of a benchmark lane. "
            "The only intended code mutation target is simple_pf.py. The reward is the valid fractional "
            "reduction in compiled t_gate_count from the trial baseline; invalid or missing results are penalized."
        ),
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(run_root / "gepa"),
                max_metric_calls=args.max_metric_calls,
                parallel=args.parallel,
                max_workers=args.max_workers,
                display_progress_bar=True,
                capture_stdio=True,
            )
        ),
    )
    summary = _jsonable(result)
    best_candidate = getattr(result, "best_candidate", None)
    if best_candidate is not None:
        summary["best_candidate"] = _jsonable(best_candidate)
    (run_root / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"run_root": str(run_root), "result_file": str(run_root / "result.json")}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize T-gate autoresearch prompts with GEPA optimize_anything.")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed-prompt", help="Print the default GEPA seed prompt.")
    seed.set_defaults(func=cmd_seed_prompt)

    dataset = sub.add_parser("dataset", help="Write a default train dataset JSON for T-gate prompt optimization.")
    dataset.add_argument(
        "--output",
        required=True,
        help="Explicit path for the generated dataset (keep campaign inputs outside the checkout).",
    )
    dataset.add_argument("--family", default=DEFAULT_FAMILY)
    dataset.add_argument("--smoke", action="store_true", help="Use only the cheapest H2 lane.")
    dataset.set_defaults(func=cmd_dataset)

    eval_prompt = sub.add_parser("evaluate-prompt", help="Evaluate one prompt candidate on one isolated T-gate trial.")
    eval_prompt.add_argument("--prompt-file", required=True)
    eval_prompt.add_argument("--family", default=DEFAULT_FAMILY)
    eval_prompt.add_argument("--molecule", default="h2")
    eval_prompt.add_argument("--example-id")
    eval_prompt.add_argument("--target-iterations", type=int, default=1)
    eval_prompt.add_argument("--max-agent-seconds", type=int, default=900)
    eval_prompt.add_argument(
        "--run-root",
        required=True,
        help="Explicit external directory for the isolated trial workspace.",
    )
    eval_prompt.add_argument("--model")
    eval_prompt.add_argument("--dry-run", action="store_true")
    eval_prompt.set_defaults(func=cmd_evaluate_prompt)

    optimize = sub.add_parser("optimize", help="Run GEPA optimize_anything over the T-gate agent prompt.")
    optimize.add_argument("--seed-prompt")
    optimize.add_argument("--dataset")
    optimize.add_argument("--valset")
    optimize.add_argument("--family", default=DEFAULT_FAMILY)
    optimize.add_argument("--smoke", action="store_true")
    optimize.add_argument(
        "--run-root",
        required=True,
        help="Explicit external directory for GEPA state, trial workspaces, and results.",
    )
    optimize.add_argument("--model", help="Optional Codex model used inside each trial agent launch.")
    optimize.add_argument("--max-metric-calls", type=int, default=12)
    optimize.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=False)
    optimize.add_argument("--max-workers", type=int, default=1)
    optimize.set_defaults(func=cmd_optimize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

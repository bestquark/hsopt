from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime

from model_registry import ACTIVE_MOLECULES, REPO_ROOT, SCRIPT_DIR, get_spec


RUN_LOG_ROOT = SCRIPT_DIR / "runs" / "codex_watchdog"


def next_iteration(molecule: str) -> int:
    spec = get_spec(molecule)
    if not spec.snapshots_dir.exists():
        return 0
    values = []
    for child in spec.snapshots_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            values.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(values, default=-1) + 1


def row_passes_constraint(row: dict[str, str]) -> bool:
    value = (row.get("passes_accuracy_constraint") or row.get("passes_fidelity_constraint") or "").strip().lower()
    return value in {"true", "1", "yes"}


def best_kept_score(molecule: str) -> tuple[int | None, float | None]:
    spec = get_spec(molecule)
    if not spec.results_path.exists():
        return None, None
    best_iteration = None
    best_score = None
    with spec.results_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            if not row_passes_constraint(row):
                continue
            try:
                iteration = int(row["iteration"])
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if best_score is None or score < best_score:
                best_iteration = iteration
                best_score = score
    return best_iteration, best_score


def archived_iteration_span(molecule: str) -> tuple[int, int]:
    nxt = next_iteration(molecule)
    if nxt <= 0:
        return 0, -1
    return 0, nxt - 1


def build_prompt(molecule: str, target_iteration: int) -> str:
    spec = get_spec(molecule)
    start_iter, last_iter = archived_iteration_span(molecule)
    resume_iter = last_iter + 1
    best_iter, _best_score = best_kept_score(molecule)

    if last_iter < 0:
        archive_clause = (
            "No iterations are archived yet, so you should begin from the untouched baseline "
            "and archive that baseline as iteration 0 before starting the mutation loop."
        )
    else:
        if best_iter is None:
            best_clause = "the live file should be kept in the current best valid state"
        else:
            best_clause = f"the live file has been restored to best iteration {best_iter}"
        archive_clause = (
            f"Iterations {start_iter} through {last_iter} are already archived, {best_clause}, "
            f"and you should resume at iteration {resume_iter}."
        )

    return (
        f"Read [{spec.instruction_path.name}]({spec.instruction_path}) and continue the existing "
        f"{spec.display_name} benchmark from the current snapshots. {archive_clause} Keep making "
        f"sequential one-mutation then one-scored-run progress until iteration {target_iteration}. "
        "Do not stop just because the queue is busy. If needed, use an attached persistent PTY/controller, "
        "but choose each next mutation only after reading the previous scored result. Do not use predefined "
        "sweeps, seed loops, or scripted batches of future mutations. Every mutation must still be scored "
            "with `queued_track_iteration.py` and archived, even when it fails the chemical-accuracy energy check or is worse than the "
        "current best. Failed and valid-but-worse proposals are useful data and must remain in the history. "
        "Use a precise `--description` each time because it is logged as the per-iteration autoresearch note. "
        "Leave the live file in the best valid state whenever you exit."
    )


def launch_codex(molecule: str, prompt: str, model: str | None) -> subprocess.CompletedProcess[str]:
    RUN_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = RUN_LOG_ROOT / f"{molecule}_{stamp}"
    prompt_file = log_prefix.with_suffix(".prompt.txt")
    stdout_file = log_prefix.with_suffix(".stdout.txt")
    stderr_file = log_prefix.with_suffix(".stderr.txt")
    last_message_file = log_prefix.with_suffix(".last_message.txt")
    prompt_file.write_text(prompt)

    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(REPO_ROOT),
        "-o",
        str(last_message_file),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    stdout_file.write_text(proc.stdout)
    stderr_file.write_text(proc.stderr)
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeatedly invoke `codex exec` until a small-molecule PF lane reaches the target iteration."
    )
    parser.add_argument("--molecule", required=True, choices=ACTIVE_MOLECULES)
    parser.add_argument("--target-iteration", type=int)
    parser.add_argument("--max-launches", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--model", default=None, help="Optional Codex model override.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = get_spec(args.molecule)
    target_iteration = args.target_iteration if args.target_iteration is not None else spec.target_iteration

    current_next = next_iteration(args.molecule)
    if current_next - 1 >= target_iteration:
        print(
            json.dumps(
                {
                    "done": True,
                    "molecule": args.molecule,
                    "last_iteration": current_next - 1,
                    "target_iteration": target_iteration,
                }
            )
        )
        return 0

    launches = 0
    consecutive_no_progress = 0
    while launches < args.max_launches:
        last_before = next_iteration(args.molecule) - 1
        if last_before >= target_iteration:
            break

        prompt = build_prompt(args.molecule, target_iteration)
        if args.dry_run:
            print(prompt)
            return 0

        proc = launch_codex(args.molecule, prompt, args.model)
        launches += 1
        last_after = next_iteration(args.molecule) - 1

        summary = {
            "launch": launches,
            "molecule": args.molecule,
            "returncode": proc.returncode,
            "last_iteration_before": last_before,
            "last_iteration_after": last_after,
            "target_iteration": target_iteration,
        }
        print(json.dumps(summary), flush=True)

        if last_after >= target_iteration:
            break

        if last_after <= last_before:
            consecutive_no_progress += 1
            if consecutive_no_progress >= 3:
                print(
                    json.dumps(
                        {
                            "done": False,
                            "reason": "no_progress",
                            "molecule": args.molecule,
                            "last_iteration": last_after,
                        }
                    ),
                    file=sys.stderr,
                )
                return 1
        else:
            consecutive_no_progress = 0

        time.sleep(args.sleep_seconds)

    final_last = next_iteration(args.molecule) - 1
    done = final_last >= target_iteration
    print(
        json.dumps(
            {
                "done": done,
                "molecule": args.molecule,
                "last_iteration": final_last,
                "target_iteration": target_iteration,
                "launches": launches,
            }
        )
    )
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shlex
import signal
import shutil
import stat
import sys
import tempfile
from pathlib import Path

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
    ALLOWED_SCHEDULE_STRATEGIES,
    ALLOWED_TERM_ORDER_STRATEGIES,
    DEFAULT_CODEX_BIN,
    PAPER_MODEL,
    PAPER_REASONING_EFFORT,
    PAPER_CODEX_EXEC_FLAGS,
    build_explicit_product_formula_schedule,
    db_connect,
    ensure_autoresearch_provenance,
    ensure_dirs,
    ensure_private_npm_cache,
    init_db,
    load_policy_module,
    log_event,
    now_iso,
    order_terms,
)
from render_telemetry import main as render_main  # noqa: E402


def prepare_codex_tools(root: Path) -> Path:
    tools_dir = root / ".codex_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    ensure_private_npm_cache(root)
    shim_text = f'#!/bin/sh\nexec {sys.executable!r} "$@"\n'
    for name in ("python", "python3"):
        python_shim = tools_dir / name
        if (
            not python_shim.exists()
            or python_shim.read_text(encoding="utf-8", errors="replace") != shim_text
        ):
            python_shim.write_text(shim_text, encoding="utf-8")
            python_shim.chmod(
                python_shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
    return tools_dir


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary policy path is unsafe: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_codex_lifecycle(path: Path) -> tuple[bool, str | None]:
    if path.is_symlink() or not path.is_file():
        return False, "Codex JSONL lifecycle output is missing or unsafe"
    event_types: list[str] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            event_type = payload.get("type")
            if not isinstance(event_type, str) or not event_type:
                return False, f"Codex JSONL line {line_number} has no event type"
            event_types.append(event_type)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid Codex JSONL lifecycle output: {exc}"
    if "thread.started" not in event_types or "turn.completed" not in event_types:
        return False, f"incomplete Codex lifecycle: {event_types}"
    failures = [
        event_type
        for event_type in event_types
        if event_type.endswith(".failed") or event_type.endswith(".error")
    ]
    if failures:
        return False, f"Codex lifecycle reported failure events: {failures}"
    return True, None


async def terminate_process_group(process_group_id: int) -> None:
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        await asyncio.sleep(0.05)
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    for _ in range(20):
        await asyncio.sleep(0.05)
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
    raise RuntimeError(f"mutation process group {process_group_id} survived SIGKILL")


def isolated_policy_path(work_dir: Path) -> Path:
    entries = list(work_dir.iterdir())
    if len(entries) != 1 or entries[0].name != "simple_pf.py":
        raise RuntimeError(
            "isolated mutation workspace closure changed: "
            f"{sorted(entry.name for entry in entries)}"
        )
    policy = entries[0]
    metadata = policy.lstat()
    if policy.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("isolated candidate policy is not a real regular file")
    return policy


def ensure_mutation_host_allowed() -> None:
    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if slurm_job_id:
        raise RuntimeError(
            f"Refusing to launch a mutation agent inside Slurm job {slurm_job_id}. "
            "Run mutation agents on Omni/login only; "
            "Slurm jobs must be evaluator/diagonalization jobs only."
        )


def load_lanes(conn, *, limit: int | None) -> list[dict]:
    query = "SELECT * FROM lanes WHERE active=1 ORDER BY nqubits, terms, lane_id"
    if limit:
        query += f" LIMIT {int(limit)}"
    return [dict(row) for row in conn.execute(query).fetchall()]


def validate_policy(policy_path: Path) -> tuple[bool, str | None]:
    try:
        module = load_policy_module(policy_path)
        config = module.get_method_config()
        formula_order = int(getattr(config, "formula_order"))
        n_steps = int(getattr(config, "n_steps"))
        zero_angle_threshold = float(getattr(config, "zero_angle_threshold", 0.0))
        uniform_epsilon = float(getattr(config, "uniform_epsilon"))
        term_order = str(getattr(config, "term_order_strategy"))
        schedule_strategy = str(getattr(config, "schedule_strategy", "stationary_trotter"))
        if formula_order not in (1, 2):
            return False, f"formula_order must be 1 or 2, got {formula_order}"
        if n_steps <= 0:
            return False, f"n_steps must be positive, got {n_steps}"
        if not math.isfinite(zero_angle_threshold) or zero_angle_threshold < 0.0:
            return (
                False,
                f"zero_angle_threshold must be finite and nonnegative, got {zero_angle_threshold}",
            )
        if not math.isfinite(uniform_epsilon) or uniform_epsilon <= 0.0:
            return False, f"uniform_epsilon must be positive and finite, got {uniform_epsilon}"
        if term_order not in ALLOWED_TERM_ORDER_STRATEGIES:
            return False, f"unsupported term_order_strategy {term_order!r}"
        if schedule_strategy not in ALLOWED_SCHEDULE_STRATEGIES:
            return False, f"unsupported schedule_strategy {schedule_strategy!r}"
        sample_terms = [
            {
                "label": "IXYZ",
                "coefficient": 0.03125,
                "support_size": 3,
                "x_count": 1,
                "y_count": 1,
                "z_count": 1,
            },
            {
                "label": "IIZZ",
                "coefficient": 1.0e-5,
                "support_size": 2,
                "x_count": 0,
                "y_count": 0,
                "z_count": 2,
            },
        ]
        _ = order_terms(sample_terms, term_order, module, config)
        if schedule_strategy != "stationary_trotter":
            _ = build_explicit_product_formula_schedule(
                terms=sample_terms,
                target_time=0.5,
                evaluator_seed=17,
                strategy=schedule_strategy,
                policy_module=module,
                config=config,
            )
        sample_term = {
            "label": "X0 Y1 Z2",
            "coefficient": 0.03125,
            "support_size": 3,
            "x_count": 1,
            "y_count": 1,
            "z_count": 1,
        }
        epsilon = float(module.choose_synthesis_epsilon(sample_term, 0.03125, 0, config))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            return False, f"choose_synthesis_epsilon returned invalid epsilon {epsilon}"
        return True, None
    except BaseException as exc:
        return False, f"{type(exc).__name__}: {exc}"


def load_accepted_incumbent(root: Path, lane_id: str, iteration: int) -> dict | None:
    conn = db_connect(root)
    init_db(conn)
    try:
        row = conn.execute(
            """
            SELECT result_json
            FROM evaluations
            WHERE lane_id=? AND accepted=1 AND iteration<?
            ORDER BY iteration DESC
            LIMIT 1
            """,
            (lane_id, iteration),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return json.loads(str(row["result_json"] or "{}"))


def compact_result_for_prompt(result: dict | None) -> dict:
    if not result:
        return {}
    fields = (
        "lane_id",
        "iteration",
        "accepted",
        "decision_reason",
        "passes_chemical_accuracy_bound",
        "total_t_gates_qpe",
        "ground_state_energy_hartree",
        "ground_state_energy_estimate_hartree",
        "ground_state_energy_error_hartree",
        "passes_energy_constraint",
        "chemical_accuracy_hartree",
        "base_unitary_t_gates",
        "propagation_wall_seconds",
        "analytic_total_error_bound_hartree",
        "pf_error_bound_hartree",
        "synthesis_error_bound_hartree",
        "zero_angle_error_bound_hartree",
        "qpe_grid_spacing_hartree",
        "qpe_bits",
        "qpe_unitary_applications",
        "scheduled_rotation_count_per_base_unitary",
        "nonzero_rotation_count_per_base_unitary",
        "unique_compilation_count",
        "mutation_summary",
    )
    compact = {key: result.get(key) for key in fields if key in result}
    compact["config"] = result.get("config", {})
    return compact


def mutation_prompt(
    *,
    lane: dict,
    iteration: int,
    incumbent_result: dict | None,
    previous_attempt: dict | None,
) -> str:
    incumbent = json.dumps(compact_result_for_prompt(incumbent_result), indent=2, sort_keys=True)
    previous = json.dumps(compact_result_for_prompt(previous_attempt), indent=2, sort_keys=True)
    allowed_orders = ", ".join(sorted(ALLOWED_TERM_ORDER_STRATEGIES))
    allowed_schedules = ", ".join(sorted(ALLOWED_SCHEDULE_STRATEGIES))
    return f"""You are mutating one HamLib product-formula policy for an hsopt-style campaign.

Lane:
- lane_id: {lane["lane_id"]}
- molecule: {lane["molecule"]}
- HamLib file: {lane["file"]}
- dataset: {lane["dataset"]}
- nqubits: {lane["nqubits"]}
- effective electrons: {lane["electrons"]}
- Pauli terms: {lane["terms"]}
- one_norm: {lane["one_norm"]}
- autoresearch iteration: {iteration}

Edit only `simple_pf.py` in the current directory. Make exactly one coherent policy mutation.

Policy API available in `simple_pf.py`:
- `ProductFormulaConfig` controls schedule family, PF1/PF2, step count, base epsilon, zero-angle threshold, and named term ordering.
- `choose_synthesis_epsilon(term, alpha, step_index, config)` is called for each nonzero rotation. `step_index` is the explicit schedule occurrence for nonstationary schedules, so tolerances may vary arbitrarily from one occurrence to another.
- `term` is Hamiltonian-derived metadata with at least `label`, `coefficient`, `support_size`, `x_count`, `y_count`, and `z_count`.
- `alpha` is the actual per-application PF rotation coefficient after step-time and PF half-step factors; `abs(alpha)` is often a better precision signal than the raw coefficient alone.
- Optional: define `order_product_formula_terms(terms, config)` and set `term_order_strategy = "custom"` to return the same Hamiltonian terms in a deterministic custom order. This is the hook for grouping or interleaving terms by coefficient bands, Pauli family, support/locality, Z-only vs mixed X/Y/Z structure, or other auditable Hamiltonian metadata. Do not drop, duplicate, or alter terms.
- Set `schedule_strategy` to `randomized_trotter` for evaluator-seeded independent term permutations at each Trotter step, or `qdrift` for coefficient-weighted sampling with replacement using `qdrift_samples` rotations.
- For a fully custom algorithm, set `schedule_strategy = "custom"` and define `build_product_formula_schedule(terms, target_time, evaluator_seed, config)`. Return dictionaries containing `term_index`, `alpha`, and optionally a positive `epsilon`. Terms may be reordered, repeated, or omitted; the evaluator seed makes randomized schedules reproducible and is not chosen by the agent.

Evaluation contract:
- The evaluator runs the actual compiled schedule on one A100 through Legate/cuPyNumeric. Stationary Trotter retains a compact one-slice fast path; randomized Trotter, qDRIFT, and custom schedules are propagated in full.
- It estimates the lowest resolved Hartree-Fock spectral peak and compares that measured energy with the exact fixed-electron-sector ground energy.
- Analytic PF/synthesis bounds are diagnostics only. They do not determine acceptance.

Allowed knobs:
- `formula_order`: 1 or 2.
- `n_steps`: positive integer. Increasing it increases scheduled rotations and T gates.
- `uniform_epsilon` plus structured logic inside `choose_synthesis_epsilon`.
- `zero_angle_threshold` when it safely removes negligible rotations.
- `term_order_strategy`: one of {allowed_orders}.
- `schedule_strategy`: one of {allowed_schedules}.
- Optional deterministic helper functions used by `choose_synthesis_epsilon`, `order_product_formula_terms`, or `build_product_formula_schedule`.

Hard constraints:
- Return strictly positive finite epsilons.
- Custom ordering must preserve exactly the same Pauli labels and coefficients.
- Custom schedules may use arbitrary ordering, repetition, omission, rotation angles, and per-occurrence tolerances, but may reference only supplied Hamiltonian term indices and must reproduce byte-for-byte for the same evaluator seed.
- Do not change the Hamiltonian, QPE parameters, evaluator formulas, chemical-accuracy bound, or pygridsynth behavior.
- Do not edit lane metadata, evaluator scripts, or files outside this directory.
- Keep the policy auditable; avoid random search or giant tables.
- Do not run cluster data searches, evaluator probes, pygridsynth probes, or Lustre scans during mutation; rely on the previous result JSON and edit the policy directly.
- If you run a local syntax check, use `python3`, not `python`.

Objective:
- Minimize QPE T gates while keeping the measured ground-state energy error within chemical accuracy.
- Do not default to scalar-only changes. Prefer Hamiltonian-aware precision policies in `choose_synthesis_epsilon`: coefficient/angle bands, support size, Pauli family, Z-only vs mixed terms, and combinations of these features.
- Use `n_steps` or `formula_order` when product-formula error is the limiting term; otherwise prefer synthesis precision and term-order changes that reduce compiled T gates.
- If the immediately previous proposal failed measured chemical accuracy, tighten the product-formula schedule or synthesis precision instead of trusting an analytic bound that happened to pass.
- Starting from the accepted incumbent, try reducing T gates by loosening epsilons for low-impact rotations, increasing zero-angle filtering only when justified, or changing term order/grouping.
- Name the policy to reflect the actual idea, not just the scalar values.

Current accepted incumbent result JSON:
```json
{incumbent}
```

Immediately previous proposal result JSON (it may have been rejected):
```json
{previous}
```

After editing, briefly summarize the mutation in your final response.
"""


async def run_one(
    *,
    root: Path,
    lane: dict,
    iteration: int,
    model: str,
    reasoning_effort: str,
    codex_bin: str,
    timeout_seconds: int,
    tools_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, bool, str | None]:
    async with semaphore:
        lane_dir = root / "lanes" / lane["lane_id"]
        iter_dir = root / "iterations" / f"iter_{iteration:04d}" / "mutations" / lane["lane_id"]
        iter_dir.mkdir(parents=True, exist_ok=True)
        policy_path = lane_dir / "simple_pf.py"
        before_path = iter_dir / "before_simple_pf.py"
        after_path = iter_dir / "after_simple_pf.py"
        prompt_path = iter_dir / "prompt.md"
        stdout_path = iter_dir / "codex.stdout"
        stderr_path = iter_dir / "codex.stderr"
        last_message = iter_dir / "last_message.txt"

        if policy_path.is_symlink() or not policy_path.is_file():
            raise RuntimeError(f"live policy is missing or unsafe: {policy_path}")
        before_bytes = policy_path.read_bytes()
        write_bytes_atomic(before_path, before_bytes)
        work_dir = Path(tempfile.mkdtemp(prefix=".candidate-work-", dir=iter_dir))
        work_policy = work_dir / "simple_pf.py"
        write_bytes_atomic(work_policy, before_bytes)

        previous_path = (
            root
            / "iterations"
            / f"iter_{iteration - 1:04d}"
            / "results"
            / f"{lane['lane_id']}.json"
        )
        previous_attempt = (
            json.loads(previous_path.read_text(encoding="utf-8"))
            if previous_path.exists()
            else None
        )
        incumbent_result = load_accepted_incumbent(root, lane["lane_id"], iteration)
        prompt = mutation_prompt(
            lane=lane,
            iteration=iteration,
            incumbent_result=incumbent_result,
            previous_attempt=previous_attempt,
        )
        prompt_path.write_text(prompt, encoding="utf-8")

        ensure_mutation_host_allowed()
        command = [
            *shlex.split(codex_bin),
            *PAPER_CODEX_EXEC_FLAGS,
            "-C",
            str(work_dir),
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "-o",
            str(last_message),
            prompt,
        ]
        env = os.environ.copy()
        env["npm_config_cache"] = str(root / ".codex_npm")
        env["npm_config_update_notifier"] = "false"
        env["PATH"] = f"{tools_dir}:{env.get('PATH', '')}"
        for output_path in (stdout_path, stderr_path, last_message):
            if output_path.is_symlink() or (output_path.exists() and not output_path.is_file()):
                raise RuntimeError(f"mutation output path is unsafe: {output_path}")
        last_message.unlink(missing_ok=True)
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as out,
                stderr_path.open("w", encoding="utf-8") as err,
            ):
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=out,
                    stderr=err,
                    env=env,
                    start_new_session=True,
                )
                try:
                    code = await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    code = 124
                    await terminate_process_group(proc.pid)
                    await proc.wait()
                else:
                    # Codex may have spawned descendants that outlive its leader. Tear down the
                    # entire isolated process group before inspecting or promoting any bytes.
                    await terminate_process_group(proc.pid)

            lifecycle_ok, lifecycle_error = validate_codex_lifecycle(stdout_path)
            closure_error: str | None = None
            try:
                candidate_path = isolated_policy_path(work_dir)
            except BaseException as exc:
                candidate_path = work_policy
                closure_error = f"{type(exc).__name__}: {exc}"
            candidate_bytes = (
                candidate_path.read_bytes()
                if candidate_path.is_file() and not candidate_path.is_symlink()
                else before_bytes
            )
            write_bytes_atomic(after_path, candidate_bytes)
            changed = candidate_bytes != before_bytes
            valid, validation_error = (
                validate_policy(candidate_path) if closure_error is None else (False, closure_error)
            )
            mutation_succeeded = (
                valid and lifecycle_ok and code == 0 and changed and closure_error is None
            )
            if code != 0:
                mutation_error = f"Codex exited with status {code}"
            elif not lifecycle_ok:
                mutation_error = lifecycle_error
            elif closure_error is not None:
                mutation_error = closure_error
            elif not valid:
                mutation_error = validation_error
            elif not changed:
                mutation_error = "Codex completed without changing simple_pf.py"
            else:
                mutation_error = None
            if mutation_succeeded:
                write_bytes_atomic(policy_path, candidate_bytes)
            return lane["lane_id"], int(code), mutation_succeeded, mutation_error
        finally:
            shutil.rmtree(work_dir, ignore_errors=False)


async def run_all(args) -> int:
    root = args.campaign_root.resolve()
    ensure_dirs(root)
    ensure_autoresearch_provenance(
        campaign_root=root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        codex_bin=args.codex_bin,
    )
    conn = db_connect(root)
    init_db(conn)
    lanes = load_lanes(conn, limit=args.limit)
    conn.executemany(
        "UPDATE lanes SET status='mutating', updated_at=? WHERE lane_id=?",
        [(now_iso(), lane["lane_id"]) for lane in lanes],
    )
    conn.commit()
    log_event(
        conn,
        kind="mutate",
        iteration=args.iteration,
        message=f"starting mutation wave for {len(lanes)} lanes",
    )

    tools_dir = prepare_codex_tools(root)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        run_one(
            root=root,
            lane=lane,
            iteration=args.iteration,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            codex_bin=args.codex_bin,
            timeout_seconds=args.timeout_seconds,
            tools_dir=tools_dir,
            semaphore=semaphore,
        )
        for lane in lanes
    ]
    failures = 0
    for future in asyncio.as_completed(tasks):
        lane_id, code, accepted, validation_error = await future
        status = "mutated" if accepted else "mutation_failed"
        if not accepted:
            failures += 1
        conn.execute(
            "UPDATE lanes SET status=?, updated_at=? WHERE lane_id=?", (status, now_iso(), lane_id)
        )
        conn.commit()
        print(
            json.dumps(
                {
                    "lane_id": lane_id,
                    "status": status,
                    "returncode": code,
                    "accepted": accepted,
                    "validation_error": validation_error,
                }
            ),
            flush=True,
        )
    log_event(
        conn,
        kind="mutate",
        iteration=args.iteration,
        message=f"finished mutation wave: lanes={len(lanes)}, failures={failures}",
        payload={"lanes": len(lanes), "failures": failures},
    )

    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["render_telemetry.py", "--campaign-root", str(root)]
        render_main()
    finally:
        sys.argv = old_argv
    return 1 if failures and args.fail_on_mutation_error else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one Codex mutation wave over active lanes."
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=os.environ.get("CODEX_MODEL", PAPER_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("CODEX_REASONING_EFFORT", PAPER_REASONING_EFFORT),
    )
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", DEFAULT_CODEX_BIN))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("CODEX_MUTATION_TIMEOUT_SECONDS", "1200")),
    )
    parser.add_argument("--fail-on-mutation-error", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_all(args)))


if __name__ == "__main__":
    main()

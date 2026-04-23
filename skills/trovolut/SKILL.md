---
name: trovolut
description: Run a `/loop`-style fixed-budget code-mutation campaign for resource-aware quantum simulation benchmarks. Use when Codex should scaffold a run directory, mutate one editable benchmark file for exactly `n` iterations, score each change through a fixed evaluator such as `evaluate.py`, preserve snapshots/logs, and optimize objectives like T-gate count, fidelity-penalized resource score, wall time, or related lower-is-better metrics.
---

# Trovolut

Use this skill when the job is not a one-off code edit but a full scored
mutation loop with a fixed iteration budget, explicit evaluation policy, and
reproducible run logs.

Invocation note:
- In Codex CLI and the IDE, invoke the skill as `$trovolut` or through `/skills`.
- The repo already uses `trovolut` as its package CLI, so the runtime fallback
  command is the script under `skills/trovolut/scripts/trovolut_cli.py`, not
  `uv run trovolut`.

Use the bundled runtime for any benchmark directory that exposes:

- one editable method file such as `simple_pf.py`, `initial_script.py`, `initial_program.py`, or `simple_dmrg.py`
- one evaluator file such as `evaluate.py`, `evaluator.py`, or `eval.py` that prints JSON with `score` or the manifest objective metric
- an optional `.trovolut.json` manifest when the benchmark wants custom objective text or metadata

The first worked resource-optimization benchmarks in this repo live under
`examples/ft_protocols/`. Each protocol directory contains an H2 benchmark with
a fixed chemistry instance, an editable method file, and an evaluator that
returns JSON with `score`, `t_gate_count`, and the relevant chemical-accuracy
proxy.

## First Step

Create or resume a timestamped run directory before mutating anything:

```bash
python3 skills/trovolut/scripts/trovolut_cli.py <num_iterations> [benchmark_dir_or_file] {additional instructions}
```

Examples:

```bash
python3 skills/trovolut/scripts/trovolut_cli.py 80 examples/ft_protocols/pfs/h2 "Search for lower-T product-formula synthesis policies under chemical accuracy."
python3 skills/trovolut/scripts/trovolut_cli.py 60 examples/ft_protocols/taylor_lcu/h2 "Tune Taylor order, segmentation, and loading precision without exceeding the proxy."
python3 skills/trovolut/scripts/trovolut_cli.py init-run 60 examples/ft_protocols/qsvt/h2 --evaluation-mode serialized "Explore QSVT precision allocation and polynomial-degree tradeoffs."
python3 skills/trovolut/scripts/trovolut_cli.py 50 . --source simple_pf.py --evaluator evaluate.py "Reduce the evaluator score without changing the scoring contract."
```

This creates `run_<timestamp>/` under the benchmark directory with:
- a copy of the editable baseline
- the copied evaluator as `_user_evaluate.py`
- `plan.md`
- `agent_prompt.md`
- `run.json`
- `status.json`
- local `snapshots/`, `logs/`, and `figs/`

After scaffolding, work inside the run directory.

## Loop Contract

1. Read `README.md`, `plan.md`, and `agent_prompt.md`.
2. Archive the untouched baseline first as iteration `0`.
3. Then do exactly the requested number of mutated outer iterations.
4. One outer iteration means: inspect the previous result, make one explicit code mutation, then run one scored evaluation.
5. Read the returned JSON before choosing the next mutation.
6. If the last scored result is `discard` or `crash`, restore the best kept iteration before continuing.
7. Keep the live file in the best valid state before stopping.

This is intended `/loop` behavior: one trajectory, one scored step at a time,
with full archival history.

## Evaluation Contract

- If no evaluator file is present, stop and say that scoring infrastructure is missing.
- If the editable source file cannot be inferred, point the runtime at it with `--source <path>`.
- Treat the evaluator as a fixed scoring contract, not as a mutation target.
- Inside a scaffolded run directory, also treat `_user_evaluate.py`, `run_eval.py`, `restore_best.py`, `status.py`, `plot.py`, `watchdog.py`, and `campaign.py` as fixed infrastructure unless the user explicitly asks to change the runtime itself.
- Mutate the benchmark method file named by the manifest or benchmark path. Only touch nearby support files when the benchmark genuinely requires a coupled change and the scoring semantics stay identical.
- Optimize the evaluator's returned objective, not an assumed proxy.
- Every scored iteration must include a short technical mutation summary naming the actual code change.
- Run scored evaluations only through the wrapped launcher so snapshots, watchdogs, and logs stay consistent.

## Required Commands

Run scored evaluations only through the wrapper:

```bash
python3 run_eval.py -- uv run python evaluate.py --description "<technical mutation summary>"
```

Check status:

```bash
uv run python status.py
```

Render local figures:

```bash
uv run python plot.py
```

Restore the best kept iteration:

```bash
uv run python restore_best.py
```

Optional run-level watchdog in a second terminal:

```bash
python3 watchdog.py
```

Optional relaunch loop:

```bash
python3 campaign.py --agent codex --search
python3 campaign.py --agent claude
```

## Search Behavior

- Be really creative about lowering the scored objective, especially when it encodes resource usage such as T-gate count or synthesis cost under a hard fidelity floor.
- Prefer structural ideas that plausibly change the score surface: grouped precision policies, better term ordering, altered decomposition structure, staged schedules, symmetry tying, or cleaner resource allocation heuristics.
- Once tiny tolerance tweaks plateau, stop burning iterations on them alone.
- Do not batch blind future evaluations, write hidden search drivers, or run offline probes outside the scorer.
- Keep the benchmark family intact. Improve the method inside the benchmark instead of rewriting the evaluator to make the score easier.

## Idea Research Mode

When the mutation space feels stale, do targeted online research before
spending many more scored iterations.

Use research only where it can materially change the next mutation:

- diagnose the current bottleneck first: synthesis precision allocation, product-formula ordering, decomposition overhead, state-prep cost, or weak constraint handling
- prefer primary sources: papers, official docs, and benchmark repos
- translate what you find into concrete mutations that fit the exact evaluator and fixed runtime budget
- reject ideas that require a different backend, heavy preprocessing, or a much larger runtime budget

For T-gate and resource-aware benchmarks, focus on ideas like:

- nonuniform synthesis tolerances tied to Pauli magnitude or term family
- grouping or reordering strategies that reduce expensive compiled rotations
- exact-zero pruning thresholds that stay safely inside the fidelity budget
- decomposition changes that reduce repeated basis changes or parity wiring
- schedule changes that improve resource use without violating the evaluator contract

If you research, keep the output mutation-ready:

1. current bottleneck
2. 3 to 5 ranked ideas
3. why each idea could help under the fixed budget
4. the smallest plausible code mutation for the top 1 or 2 ideas
5. links to the sources used

## Logging And Recovery

- `evaluate.py` appends each scored evaluation to `logs/evaluations.jsonl`.
- `status.py` reports how many mutated iterations are complete and whether the run hit its target.
- `python3 run_eval.py -- ...` uses the bundled heartbeat and watchdog scripts to kill stalled jobs.
- If a run stalls at the campaign level, use `python3 watchdog.py` inside the run directory to detect no-progress periods.
- `python3 campaign.py --agent codex|claude` keeps relaunching the agent until the mutation budget is exhausted or repeated launches make no progress.
- `logs/evaluations.jsonl` and each snapshot `metadata.json` store the technical mutation summary passed through `--description`.

## Scope

The first canonical Trovolut benchmark surfaces in this repo are:

- `examples/ft_protocols/pfs/h2`
- `examples/ft_protocols/taylor_lcu/h2`
- `examples/ft_protocols/qsvt/h2`
- `examples/ft_protocols/qubitization/h2`

These are small H2 resource-estimation examples. The first three estimate
direct Hamiltonian dynamics through product formulas, Taylor/LCU, and QSVT/QSP;
the fourth estimates qubitization-QPE energy readout. All enforce a fixed
chemical-accuracy proxy before rewarding lower T-gate count.

Prefer opening the benchmark directory itself and using the local editable file
plus `evaluate.py` when it exists. After scaffolding, continue inside the
generated `run_<timestamp>/` directory rather than editing the shared benchmark
file directly.

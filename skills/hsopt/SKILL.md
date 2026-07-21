---
name: hsopt
description: Run and organize sequential autoresearch-style T-gate optimization in hsopt. Use this whenever the user asks to reduce T gates, optimize synthesis precision, mutate product-formula precision policies, continue a lane, inspect T-count progress, manage HSOpt campaign storage, or run the repo's `/hsopt` workflow.
---

# HSOpt

Use this skill for direct T-gate optimization runs where an agent edits a
benchmark lane's `simple_pf.py`, scores the mutation, reads the result, and
continues one scored step at a time.

Invocation note:
- In Codex, use this skill as `$hsopt` or through `/skills`.
- OpenCode and other skill-aware clients may expose it as `/hsopt`.
- The repo CLI is `uv run hsopt ...`.

## Prerequisites

Treat the installed agent skill and the benchmark runtime as separate pieces.
Installing the skill supplies these instructions; it does not clone the fixed
benchmark files or install their Python dependencies. Direct evaluator runs
also require a local HSOpt checkout with `uv sync` completed. Run the commands
below from that checkout's root. Cloning the repository alone likewise does not
register this skill with the agent.

## First Step

Resolve the run storage before inspecting or mutating a lane. Prefer an
explicit absolute `HSOPT_DATA_ROOT`; otherwise use the caller's established
central HSOpt data directory. Never invent a campaign folder in the checkout,
`examples/pfs/`, the current working directory, or directly in the user's home
directory.

```text
$HSOPT_DATA_ROOT/
  paper/hamlib121/<family>_YYYYMMDD_<commit>/run_01 ... run_05
  paper/reactions10/<family>_YYYYMMDD_<commit>/
  development/<benchmark>/<campaign-id>/
  old_experiments/<benchmark>/<campaign-id>/
  scratch/preflight/<campaign-id>/
```

Use `paper` only for protocol-locked campaigns, `development` for exploratory
runs, `scratch/preflight` for disposable checks, and `old_experiments` for
inactive or superseded campaigns. Once a campaign starts, keep its ID and
directory contents together. Do not delete histories during cleanup; move the
whole inactive campaign to `old_experiments` after confirming that no local
supervisor or scheduler job still references it.

Then inspect lanes and pick the active benchmark:

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
```

Render the exact agent prompt for a lane:

```bash
uv run hsopt prompt examples/pfs/h2
```

The primary runnable family is `examples/pfs/`. The CLI also
accepts legacy shorthand targets such as `pfs/h2`.
Read [references/layout.md](references/layout.md) when locating the installed
skill, CLI runtime, benchmark definitions, or external campaign root.

## Loop Contract

1. Read the lane status and choose `--evaluator energy` or `--evaluator fidelity`.
2. Edit only the lane's `simple_pf.py` unless the user explicitly changes the
   benchmark framework.
3. Inspect the product-formula API exposed by that file before editing:
   `ProductFormulaConfig`, `choose_synthesis_epsilon`, term-order fields, and
   any custom order hook.
4. Make one coherent Hamiltonian-aware mutation: synthesis precision, term
   ordering/grouping, or PF order/step count when the error budget justifies it.
5. Run exactly one fixed evaluator command.
6. Read the JSON result before choosing the next mutation.
7. Preserve all detailed history in the selected external campaign directory.
8. Copy only completion-gated compact exports or a small summary PDF into the
   repository; never commit raw transcripts, caches, databases, environments,
   scheduler logs, or partial runs.

## Agent and Evaluator Choices

- Mutation and evaluation are independent choices. Use `--agent codex` or
  `--agent opencode`; use `--transport local` or `--transport slurm` for the
  evaluator.
- OpenCode defaults to pinned `opencode-ai@1.4.11` with
  `nvidia/nvidia/nemotron-3-super-v3`. Supply `NVIDIA_API_KEY` only through the
  process environment or a protected environment file. Never put it in a
  prompt, command line, campaign manifest, log, or repository file.
- Every mutation runs in an isolated directory containing only `simple_pf.py`.
  OpenCode uses deny-all permissions with only read/edit access to that file.
  Never pass `--auto`, `--yolo`, or any permission-bypass flag.
- Local evaluation uses NumPy by default when Legate is unavailable. Select
  cuPyNumeric explicitly on a local NVIDIA GPU with `--backend cupynumeric`.
- Slurm evaluation submits the same deterministic evaluator with
  `sbatch --wait`, one GPU per job, and no `squeue`/`sacct` polling. A durable
  supervisor may keep up to the campaign's declared job cap blocked in those
  wait calls and resume mutation only after each call returns.
- Never start recurring `nvidia-smi`, scheduler, or device-monitoring loops on
  a login node. A single GPU identity query inside the allocated job is enough.
- Preserve a nonzero mutation exit and its stderr as experiment evidence. Let
  the bounded retry policy resume that lane; do not silently skip it.

## Canonical Commands

Print the evaluator command:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy --print-only
uv run hsopt evaluate examples/pfs/h2 --evaluator fidelity --epsilon 1e-3 --print-only
```

Score one mutation directly:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy --transport local --backend numpy
uv run hsopt evaluate examples/pfs/h2 --evaluator energy \
  --transport local --backend cupynumeric
```

Create one OpenCode/Nemotron proposal with external history:

```bash
NVIDIA_API_KEY=... uv run hsopt mutate examples/pfs/h2 \
  --agent opencode --run-dir "$HSOPT_DATA_ROOT/development/pfs/h2-proposal-001"
```

Print a one-A100 Slurm handoff without submitting it:

```bash
uv run hsopt evaluate examples/pfs/h2 --transport slurm --print-only \
  --submit-host nvlogin --remote-repo-root /shared/hsopt \
  --account ACCOUNT --partition PARTITION \
  --remote-python /shared/hsopt/.venv/bin/python \
  --remote-legate /shared/hsopt/.venv/bin/legate
```

## Optimization Target

Optimize the evaluator's lower-is-better `score`. In practice, this means
lowering compiled `t_gate_count` while keeping the fixed accuracy contract true:

- `energy`: chemical-accuracy estimate from the lowest HF-reference spectral
  peak in the fixed-electron sector.
- `fidelity`: final-state infidelity below the selected epsilon budget.

Treat these as fixed infrastructure:

- `chemistry.py`
- `evaluate_energy.py`
- `evaluate_fidelity.py`
- molecule-local `evaluate.py` wrappers

## Search Behavior

Prefer simple precision policies that are easy to audit:

- Pauli label families (`XXYY`, `ZZ`, single-`Z`, etc.)
- support size
- coefficient magnitude
- per-application rotation angle magnitude, usually available as `alpha`
- grouped epsilon buckets
- targeted tightening for terms implicated by failed accuracy diagnostics
- product-formula term ordering when the benchmark exposes it, for example
  reordering by coefficient magnitude, support size, Pauli family, or a
  symmetric large/small interleaving before applying PF1/PF2 sweeps
- custom deterministic term-order hooks such as
  `order_product_formula_terms(terms, config)` when present or explicitly
  allowed by the benchmark
- evaluator-seeded randomized Trotter, qDRIFT, or a custom explicit schedule
  when the benchmark exposes `schedule_strategy`; custom schedule items identify
  a Hamiltonian `term_index`, an `alpha`, and may supply a per-occurrence `epsilon`

For product-formula lanes, do not default to scalar-only edits of
`formula_order`, `n_steps`, or `uniform_epsilon`. First check whether
`choose_synthesis_epsilon(term, alpha, step_index, config)` receives
Hamiltonian metadata such as Pauli label, coefficient, support size, and
Pauli-family counts. Explicit schedules may use `step_index` for arbitrary
per-occurrence tolerances. Use those features to loosen synthesis precision for
low-impact terms and tighten only where the error budget needs it.

Changing product-formula order or the number of PF steps is allowed only when
the method file already exposes those knobs or the user explicitly requests it.
When doing so, verify that the reported `nonzero_rotation_count`,
`scheduled_rotation_count`, and `t_gate_count` reflect the additional rotations
introduced by the higher-order formula or extra steps.

Avoid scripted sweeps, future-result batching, evaluator edits, chemistry
changes, and long stretches of tiny one-label tweaks after progress stalls.

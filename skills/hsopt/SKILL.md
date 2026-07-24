---
name: hsopt
description: Run sequential exact compiled-unitary T-gate autoresearch in hsopt. Use whenever the user asks to reduce T gates, optimize synthesis precision or product formulas, start or continue HamLib/reaction lanes, inspect exact-energy or T-count progress, or run the repo's HSOpt campaign workflow.
---

# HSOpt Exact Autoresearch

Use this skill for product-formula optimization where an isolated agent mutates
one lane policy, the fixed evaluator diagonalizes the compiled unitary, and the
lane advances only after reading that exact result.

Treat `skills/hsopt/` as the single source bundle for this workflow. The skill
instructions live in `SKILL.md`, supporting guidance lives in `references/`,
and the complete importable runtime lives in `scripts/hsopt/`. Direct evaluator
runs require a local HSOpt checkout with `uv sync` completed so that the
runtime's dependencies and CLI entry points are available.

## Required stage order

A production campaign has three strictly ordered stages:

1. **Paper bound and exact reference:** run `hsopt-paper-baseline-batch` once
   for each unique Hamiltonian. Compute the supplement's Eq. (19)
   double-commutator coefficient sum exactly for the Pauli decomposition and
   derive the molecule-specific PF2 Trotter number at the paper's 0.1 mHa
   reference error without using `E_exact`. Separately construct and
   diagonalize `U_exact = exp(+i H t)` once, caching `E_exact` and a normalized
   deterministic-anchor projection onto the complete fixed-electron ground
   eigenspace for the evaluator. Prefer the Hartree--Fock determinant as the
   anchor; if its ground support is below the fixed threshold, use the
   lowest-index fixed-electron determinant with resolvable ground support.
2. **Iteration 0:** evaluate the immutable paper baseline against the cached
   exact reference. This is a measurement, not a mutation.
3. **Iterations 1–50:** mutate the current external incumbent, evaluate the
   compiled candidate against the same cached reference, and accept only a
   feasible strict T-count improvement.

Never let campaign initialization or a candidate recompute `U_exact`: reuse
the exact reference embedded beside the sealed bound result. Never start
mutations before iteration 0 succeeds. Five HamLib replicates share one exact
reference per instance; reaction references are assembled from the
already-calibrated species and shared by both policy lanes.

## Baseline and mutation boundaries

Generate `initial_pf.py` only with `experiments/generate_initial_pf.py`. The
generator embeds the fixed paper contract; do not infer a policy or invent a
paper attribution.

The checked-in paper contract records:

- the Reiher, Wiebe, Svore, Wecker, and Troyer PNAS 2017 citation;
- symmetric second-order Trotter--Suzuki structure, never a PF1/PF2 choice;
- lexicographic term ordering;
- Eq. (19) double-commutator-bound `beta(H)` referenced to 0.1 mHa;
- `t=1`, the resulting integer step count and step time;
- the supplement's rigorous Eq. (32)/(34) calculus allocation with
  `alpha=8*pi`, `gamma=4`, and `delta_0=11`, adapted only to the benchmark's
  1 kcal/mol total target;
- the resulting uniform synthesis tolerance. Do not run PyGridSynth while
  generating `initial_pf.py`; iteration 0 performs the first direct synthesis
  and records the realized T count.

The current campaign baseline is `hamiltonian_bound`. It evaluates the full
nonnegative Eq. (19) Pauli sum deterministically; it does not substitute a
Monte Carlo estimate. The exact reference stored in the same sealed record is
used only for later evaluation and cannot influence the initial policy.

`initial_pf.py` is an immutable, concise executable policy. Its
`ProductFormula.repetitions()`, `build_step(terms, step_time)`, and
`synthesis_epsilon(term, angle)` methods are copied to the lane mutation
target. The rotation schedule returned by `build_step()` is authoritative:
the evaluator can apply it with ideal rotations or compile the same schedule
with PyGridSynth. Do not force a declared formula order; infer PF1/PF2 from the
actual schedule and record arbitrary recurrences as custom. The verbose bound
derivation remains in the sealed calibration JSON. Candidate policies may use
the restricted `import math` module; other imports and external capabilities
remain unavailable.

The agent-visible runtime contract is deliberately plain Python:

```text
problem = {
    "name": str,
    "n_qubits": int,
    "electrons": int,
    "target_time": float,
    "identity_coefficient": float,
    "terms": list[term],
    ...  # immutable source metadata may also be present
}

term = {
    "label": str,         # n_qubits characters from I, X, Y, Z
    "coefficient": float,
    "support_size": int,  # number of non-I characters
    "x_count": int,
    "y_count": int,
    "z_count": int,
}
```

`ProductFormula(problem)` receives a fresh dictionary and should treat the
benchmark description as read-only. `repetitions()` returns a positive integer
`n_steps`, after which the evaluator computes
`step_time = problem["target_time"] / n_steps`.
`build_step(terms, step_time)` receives fresh copies of the term dictionaries
and returns a nonempty sequence of `(term, angle)` pairs for one stationary
step. The pair's `term` may be a supplied dictionary or its Pauli-label string;
the evaluator resolves it against the trusted Hamiltonian. `angle` is the
finite real `alpha` in `exp(+i * alpha * P)`, so an ordinary first-order
occurrence uses `term["coefficient"] * step_time`. The exact returned schedule
is repeated `n_steps` times. For every pair,
`synthesis_epsilon(term, angle)` receives a fresh trusted term dictionary and
the same `alpha`; it must return a finite PyGridSynth operator-norm tolerance
strictly between zero and one.

Do not create or mutate an in-tree `optimized_pf.py`. External lane
directories contain `candidate.py`, `incumbent.py`, exact results, decisions,
and audit logs. The checked-in experiment tree has one explicit evaluator entry
point at each family root: `experiments/abs_energy/evaluate.py` and
`experiments/diff_energy/evaluate.py`. Each requires the coefficient-bearing
Hamiltonian/reaction and cached exact reference explicitly.

Do not modify Hamiltonians, geometries, basis sets, active spaces, target
times, localization, electron sectors, or evaluator semantics during a run.

## Scientific contract

The score is the compiled Clifford+T cost needed to observe the selected
product-formula QPE peak with 95% confidence, subject to chemical accuracy:

```text
U_PF = product_i compiled(exp(+i alpha_i P_i))
input = fixed Hartree--Fock determinant
E_PF = lowest unwrapped U_PF eigenphase with resolved HF weight
p_HF = squared HF overlap of that selected PF eigenstate
R_95 = ceil(log(0.05) / log(1 - p_HF))
acceptance = abs(E_PF - E_exact) <= chemical accuracy
score = R_95 * compiled T count per attempt
```

The fixed HF-weight threshold only suppresses numerical zero-weight
eigenvectors; it is not a physical `p_min`. Any resolved small probability
remains eligible and pays its correspondingly large repetition cost. The exact
ground energy remains the reference, but the exact ground-state vector is not
used to select the PF eigenphase.

For the stationary formulas exposed by the benchmark, construct the compiled
matrix for one full formula step, diagonalize it, and obtain the full compiled
formula through the exact spectral identity
`U_PF = U_step ** n_steps`. Multiplying each selected step eigenphase by
`n_steps` is therefore an exact evaluation of that compiled repeated product,
not an approximation or surrogate.

For a reaction, form the signed difference from independently compiled species
eigenphase energies and compare it with the cached exact signed difference.
Give each unique species an equal share of the five-percent failure budget;
the union bound then guarantees at least 95% confidence that all required
peaks have been observed. The reaction score is the sum of those
confidence-adjusted species costs. The candidate unitary is still constructed
and diagonalized in the full Hilbert space so Clifford+T synthesis leakage is
retained.

The dense implementation may operate in the fixed-electron sector where
appropriate, but it must actually construct and diagonalize the exact or
compiled unitary used by the reported observable.

## Production topology

- `abs_energy`: 91 HamLib instances (20 at 4, 2 at 6, 24 at 8, 11 at
  10, 25 at 12, and 9 at 14 qubits) × 5 independent mutation lanes.
- `diff_energy`: 10 reactions × 2 policy lanes.
- 50 mutations per lane, plus the separate iteration-0 baseline.
- Total: 475 lanes and 23,750 mutation evaluations.

Before preparing inputs, run `uv run hsopt-layout --check`. The 91
`experiments/abs_energy/*/instance.json` files and ten
`experiments/diff_energy/*/definition.json` files must match their compact
registries and source checksums exactly. Each of those 101 directories also
contains its generated immutable `initial_pf.py`. The checked-in policy must
be byte-identical to the copy generated under the external campaign data root;
coefficient-heavy Hamiltonians and sealed calibration/reference records remain
external.

Run OpenShell and OpenCode/Nemotron on persistent Omni infrastructure.
OpenCode receives the incumbent policy, prior result, the exact fixed evaluator
source, and that lane's coefficient-bearing Hamiltonian inputs. Only
`candidate.py` is writable. It may repeatedly read `candidate.py`,
`evaluate.py`, and the benchmark JSON context as needed but must complete
exactly one mutation. Preserve:

- raw `opencode.jsonl` and `openshell.log`;
- redacted `agent_actions.jsonl`;
- `opencode_metrics.json`, `openshell_metrics.json`, and `agent_audit.json`;
- the durable mutation result and every evaluator decision.

API credentials must enter through a protected environment file or process
environment. Never put them in prompts, command lines, manifests, logs, or the
repository.

## NVPark execution

Use the `$ssh` skill for cluster details. The CPU campaign is the canonical
mutation source. Its exact evaluator is cuPyNumeric on the CPU partition:

```text
account: nvr_qualg_lmbm
partition: cpu
qos: normal
nodes: 1
gpus: 0
cpus: 8
memory: 64G
Legate system memory: 61000 MiB
```

Use the pinned cuPyNumeric/Legate 25.11 CPU build and configure seven Legate
CPU workers; the eighth Slurm CPU is available to the runtime. This gives the
64 GB dense-unitary allocation a conventional 8 GB/core shape while supporting
the 14-qubit eigensolver's unitary, eigenvectors, and temporary workspace.

The A100 validation/replay allocation uses `polar4`, `normal`, one 80 GB A100,
2 CPUs, 16 GB of host memory, and a 60000 MiB Legate framebuffer. It runs the
same uploaded evaluator source and cuPyNumeric version. The evaluator accepts
up to 15 qubits, while the canonical 91-row HamLib cutoff is 14 qubits because
that is the next complete dataset boundary after the old 57-row, 10-qubit
subset.

The CPU campaign is the only campaign allowed to invoke OpenCode. Run the A100
supervisor with `--no-mutations` and replay the archived CPU candidate files in
the same order using `--replay-campaign-root`. Require policy hashes, T counts,
and the acceptance trajectory to match and surface any discrepancy. Dense
CPU/GPU eigensolvers need only agree within an explicit numerical tolerance;
do not promise bitwise-identical floating-point eigenvalues. Any CPU run must
use the `cpu` partition and request no GPU.

PyGridSynth runs on CPU in both placements. Tasks within a submitted batch run
sequentially so only one dense unitary is resident at a time. Do not request a
full multi-GPU node, excess CPUs, or excess RAM. `(Priority)` is scheduling
priority, not an OOM condition.

The persistent supervisor must durably record submission intent, recover a job
ID after a lost SSH response, collect terminal batches, and retry only tasks
without results.

## Canonical commands

Build the calibration manifest after preparing both input families. Run the
four `run` commands as separate CPU-partition Slurm jobs; they are resumable
and jointly compute each unique Hamiltonian exactly once:

```bash
uv run hsopt-paper-baseline-batch build-manifest \
  --abs-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --output "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json"

uv run hsopt-paper-baseline-batch run \
  "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --backend numpy --mode hamiltonian_bound \
  --shard-index 0 --shard-count 4

uv run hsopt-paper-baseline-batch audit \
  "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results"

uv run hsopt-paper-baseline-batch generate \
  "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
  --calibration-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/initial-policies" \
  --generator experiments/generate_initial_pf.py \
  --repository-root "$PWD"
```

Initialize a fresh external campaign only after all four shards pass audit:

```bash
uv run hsopt-campaign initialize \
  --hamlib-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --calibration-root "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --generator experiments/generate_initial_pf.py \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --hamlib-replicates 5 --iterations 50
```

Run one idempotent CPU supervisor cycle:

```bash
uv run hsopt-supervisor \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --repo-dir "$PWD" --once \
  --threshold 15 --max-active-jobs 4 --mutation-workers 16 \
  --max-difference-tasks-per-batch 3 \
  --mutation-timeout-seconds 300 \
  --partition cpu --qos normal --cpus 8 --memory 64G --walltime 24:00:00 \
  --backend cupynumeric --accelerator cpu \
  --runtime-python /path/to/pinned/cupynumeric-cpu-25.11/bin/python
```

Initialize the A100 campaign from scratch with the same inputs and generator,
verify all 475 generated `initial_pf.py` files against the CPU campaign, and
run the replay supervisor with:

```bash
uv run hsopt-campaign initialize \
  --hamlib-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --calibration-root "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --generator experiments/generate_initial_pf.py \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu91-v2" \
  --hamlib-replicates 5 --iterations 50

uv run hsopt-supervisor \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu91-v2" \
  --repo-dir "$PWD" --once \
  --threshold 15 --max-active-jobs 4 \
  --max-difference-tasks-per-batch 3 \
  --partition polar4 --qos normal --cpus 2 --memory 16G --walltime 04:00:00 \
  --backend cupynumeric --accelerator gpu \
  --runtime-python /path/to/pinned/cupynumeric-gpu-25.11/bin/python \
  --no-mutations \
  --replay-campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1"
```

Never point the CPU campaign at `polar4`.

After all lanes finish, generate both strict summaries:

```bash
uv run hsopt-export-exact \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --family abs_energy \
  --output-dir /path/to/final-abs-energy-export

uv run hsopt-reports --family abs_energy \
  --export-dir /path/to/final-abs-energy-export \
  --output-dir /path/to/final-abs-energy-reports

uv run hsopt-export-exact \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --family diff_energy \
  --output-dir /path/to/final-diff-energy-export

uv run hsopt-reports --family diff_energy \
  --export-dir /path/to/final-diff-energy-export \
  --output-dir /path/to/final-diff-energy-reports
```

Configure the four `--final-*-export-dir` and `--final-*-report-dir` supervisor
flags to run this completion gate automatically. `hsopt-reports` uses LaTeX
when available and its portable ReportLab renderer otherwise.

Run the cross-backend audit as an unattended completion gate:

```bash
uv run hsopt-exact-parity \
  --cpu-campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --gpu-campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu91-v2" \
  --output "$HSOPT_DATA_ROOT/exact/production-gpu91-v2/cross_backend_audit.json" \
  --wait --interval-seconds 60
```

The audit must pass before treating the A100 replay as reproducible. It checks
the complete lane grid, initial and candidate hashes, compiled synthesis
records, T counts, policies, decisions, replay validations, and exact energies
within `1e-10` Hartree. Preserve its per-stage, per-qubit CPU/A100 timing
summary with the final reports.

## Mutation behavior

Prefer coherent, auditable policies over blind sweeps:

- grouped synthesis tolerances by Pauli family, support, coefficient, or angle;
- deterministic term-order changes;
- PF order or step-count changes only when exposed by the method contract;
- targeted tightening after an exact-energy failure;
- loosening where exact results demonstrate margin.

Make exactly one proposal per lane at a time. Read the exact JSON result before
the next mutation. Preserve failed proposals and agent failures as evidence;
never silently skip or rewrite history.

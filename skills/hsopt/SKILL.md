---
name: hsopt
description: Run sequential exact compiled-unitary T-gate autoresearch in hsopt. Use whenever the user asks to reduce T gates, optimize synthesis precision or product formulas, start or continue HamLib/reaction lanes, inspect exact-energy or T-count progress, or run the repo's HSOpt campaign workflow.
---

# HSOpt Exact Autoresearch

Use this skill for product-formula optimization where an isolated agent mutates
one lane policy, the fixed evaluator diagonalizes the compiled unitary, and the
lane advances only after reading that exact result.

Treat the installed agent skill and the benchmark runtime as separate pieces.
The skill supplies workflow instructions; it does not clone the repository or
install its dependencies. Direct evaluator runs require a local HSOpt checkout
with `uv sync` completed.

## Required stage order

A production campaign has three strictly ordered stages:

1. **Exact reference:** construct and diagonalize
   `U_exact = exp(+i H t)` once for each distinct Hamiltonian reference. Cache
   `E_exact`, a normalized deterministic-anchor projection onto the complete
   fixed-electron ground eigenspace, source hashes, and diagnostics in
   `exact_reference.json`. Prefer the Hartree--Fock determinant as the anchor;
   if its ground support is below the fixed threshold, use the lowest-index
   fixed-electron determinant with resolvable ground support. This projection
   is basis-independent even when the ground energy is degenerate; never use
   an arbitrary eigensolver basis vector from a degenerate subspace.
2. **Iteration 0:** evaluate the immutable paper baseline against the cached
   exact reference. This is a measurement, not a mutation.
3. **Iterations 1–50:** mutate the current external incumbent, evaluate the
   compiled candidate against the same cached reference, and accept only a
   feasible strict T-count improvement.

Never let a candidate recompute `U_exact`. Never start mutations before
iteration 0 succeeds. Five HamLib replicates share one exact reference per
instance; two reaction policies share one exact reaction reference.

## Baseline and mutation boundaries

Generate `initial_pf.py` only with `examples/generate_initial_pf.py`. The
generator must load
`examples/paper_baselines/reiher_wiebe_svore_2017.json`; do not infer a policy
or invent a paper attribution.

The checked-in paper contract records:

- the Reiher, Wiebe, Svore, Wecker, and Troyer PNAS 2017 citation;
- symmetric second-order Trotter--Suzuki structure;
- lexicographic term ordering;
- explicit benchmark choices that the paper itself does not numerically fix.

`initial_pf.py` is immutable. Do not create or mutate an in-tree
`optimized_pf.py`. External lane directories contain `candidate.py`,
`incumbent.py`, exact results, decisions, and audit logs.

Do not modify Hamiltonians, geometries, basis sets, active spaces, target
times, localization, electron sectors, or evaluator semantics during a run.

## Scientific contract

The score is compiled Clifford+T count subject to chemical accuracy of the
target compiled-unitary eigenphase:

```text
U_PF = product_i compiled(exp(+i alpha_i P_i))
anchor = HF if resolvable, else lowest resolved fixed-electron determinant
target = normalize(project_ground(H) |anchor>)
E_PF = unwrapped eigenphase of the U_PF eigenstate with maximum target overlap
acceptance = abs(E_PF - E_exact) <= chemical accuracy
score = compiled T count
```

For a reaction, form the signed difference from independently compiled species
eigenphase energies and compare it with the cached exact signed difference.
The candidate unitary is still constructed and diagonalized in the full
Hilbert space so Clifford+T synthesis leakage is retained. Hartree--Fock
overlap is a QPE success-probability diagnostic and may be zero when the
deterministic fallback anchor is required; propagation of the HF state is not
the acceptance metric.

The dense implementation may operate in the fixed-electron sector where
appropriate, but it must actually construct and diagonalize the exact or
compiled unitary used by the reported observable.

## Production topology

- HamLib121 exact subset: 57 instances (at most 10 qubits) × 5 independent
  mutation lanes.
- reactions10: 10 reactions × 2 policy lanes.
- 50 mutations per lane, plus the separate iteration-0 baseline.
- Total: 305 lanes and 15,250 mutation evaluations.

Run OpenShell and OpenCode/Nemotron on persistent Omni infrastructure.
OpenCode receives only the incumbent policy and prior result, with deny-all
permissions except the allowed candidate read/edit operations. Preserve:

- raw `opencode.jsonl` and `openshell.log`;
- redacted `agent_actions.jsonl`;
- `opencode_metrics.json`, `openshell_metrics.json`, and `agent_audit.json`;
- the durable mutation result and every evaluator decision.

API credentials must enter through a protected environment file or process
environment. Never put them in prompts, command lines, manifests, logs, or the
repository.

## NVPark execution

Use the `$ssh` skill for cluster details. The canonical mutation evaluator is
cuPyNumeric running on the CPU partition:

```text
account: nvr_qualg_lmbm
partition: cpu
qos: normal
nodes: 1
gpus: 0
cpus: 2
memory: 16G
```

Use the pinned cuPyNumeric/Legate 25.11 CPU build and configure one Legate CPU
worker; the second Slurm CPU is available to the runtime. A CPU run must use
the `cpu` partition and must not request a GPU.

The A100 validation/replay allocation uses `polar4`, `normal`, one GPU, 2 CPUs,
and 16 GB. It runs the same uploaded evaluator source and cuPyNumeric version.
The CPU campaign is the only campaign allowed to invoke OpenCode. Run the A100
supervisor with `--no-mutations` and replay the archived CPU candidate files in
the same order using `--replay-campaign-root`. Require policy hashes, T counts,
and the acceptance trajectory to match and surface any discrepancy. Dense
CPU/GPU eigensolvers need only agree within an explicit numerical tolerance;
do not promise bitwise-identical floating-point eigenvalues.

PyGridSynth runs on CPU in both placements. Tasks within a submitted batch run
sequentially so only one dense unitary is resident at a time. Do not request a
full multi-GPU node, excess CPUs, or excess RAM. `(Priority)` is scheduling
priority, not an OOM condition.

The persistent supervisor must durably record submission intent, recover a job
ID after a lost SSH response, collect terminal batches, and retry only tasks
without results.

## Canonical commands

Initialize a fresh external campaign:

```bash
uv run hsopt-campaign initialize \
  --hamlib-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/hamlib57" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/reactions10" \
  --seed-validation experiments/reactions10/inputs/seed_validation.json \
  --generator examples/generate_initial_pf.py \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu-v2" \
  --hamlib-replicates 5 --iterations 50
```

Run one idempotent CPU supervisor cycle:

```bash
uv run hsopt-supervisor \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu-v2" \
  --repo-dir "$PWD" --once \
  --threshold 15 --max-active-jobs 4 --mutation-workers 8 \
  --mutation-timeout-seconds 300 \
  --partition cpu --qos normal --cpus 2 --memory 16G --walltime 12:00:00 \
  --backend cupynumeric --accelerator cpu \
  --runtime-python /path/to/pinned/cupynumeric-cpu-25.11/bin/python
```

Initialize the A100 campaign from scratch with the same inputs and generator,
verify all 305 generated `initial_pf.py` files against the CPU campaign, and
run it separately with:

```bash
uv run hsopt-supervisor \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu-v3" \
  --repo-dir "$PWD" \
  --threshold 15 --max-active-jobs 2 \
  --partition polar4 --qos normal --cpus 2 --memory 16G --walltime 03:00:00 \
  --backend cupynumeric --accelerator gpu \
  --runtime-python /path/to/pinned/cupynumeric-gpu-25.11/bin/python \
  --no-mutations \
  --replay-campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu-v2"
```

Never point the CPU campaign at `polar4`.

After all lanes finish, generate both strict summaries:

```bash
uv run hsopt-export-exact \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu-v2" \
  --family hamlib121 \
  --output-dir /path/to/final-hamlib-export

uv run hsopt-reports --family hamlib121 \
  --export-dir /path/to/final-hamlib-export \
  --output-dir experiments/hamlib121/results/final

uv run hsopt-export-exact \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu-v2" \
  --family reactions10 \
  --output-dir /path/to/final-reactions-export

uv run hsopt-reports --family reactions10 \
  --export-dir /path/to/final-reactions-export \
  --output-dir experiments/reactions10/results/final
```

Configure the four `--final-*-export-dir` and `--final-*-report-dir` supervisor
flags to run this completion gate automatically. `hsopt-reports` uses LaTeX
when available and its portable ReportLab renderer otherwise.

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

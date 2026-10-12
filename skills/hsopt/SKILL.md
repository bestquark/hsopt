---
name: hsopt
description: Run or inspect exact compiled-unitary T-gate autoresearch for HSOpt product formulas, HamLib absolute energies, and reaction-energy lanes.
---

# HSOpt product-formula autoresearch

The repository runtime is ordinary package code under `cli/src/hsopt/`; this skill defines the mutation contract.

## Mutation workspace

`candidate.py` is the only editable file. `evaluate.py`, benchmark JSON,
prior feedback, and `research_context.json` are read-only. The context contains
the supported API, Hamiltonian statistics, and this lane's recent proposals.

Before editing, successfully read `candidate.py`, `research_context.json`, and
all benchmark files in `required_reads`, including every reaction species.
The host verifies receipts before resuming the same session with editing enabled;
it may request up to two additional read-only turns within the original budget.
Reread `candidate.py` in the editing continuation because OpenCode resets its
file-read timestamps between processes. Page raw JSON when summaries are
insufficient; reading all of `evaluate.py` is optional.

Use rejected proposals and failures to choose a new hypothesis. Previously
scored policies and formatting-only changes are rejected before evaluation.
Optional literature/code/web searches are allowed. Use `read` for local files
and `webfetch` for HTTP(S) URLs. Make one coherent edit, explain it, then stop.
Never fabricate an evaluation. Prefer a complete plain Python `write` for
multi-block edits; retain four-space indentation and all four interface methods.

Only `import math` and the briefing's `available_builtins` are allowed.
Private attributes, generators, top-level helpers, `set`, `getattr`, `hasattr`,
`isinstance`, other imports, and external runtime access are unavailable.
Context statistics are not extra `problem` keys; inspect the actual JSON.

## Editable policy

Preserve this interface:

```python
class ProductFormula:
    def __init__(self, problem): ...
    def repetitions(self): ...
    def build_step(self, terms, step_time): ...
    def synthesis_epsilon(self, term, angle): ...
```

`problem` and `term` are dictionaries. Terms have `label`, `coefficient`,
`support_size`, and X/Y/Z counts.

`repetitions()` returns a positive integer. The evaluator sets
`step_time = problem["target_time"] / repetitions()`.
`target_time` is not an arbitrary fixed one: it is the physical time
`1 / sum(abs(term["coefficient"]))` from the immutable QPE phase encoding.
The exact unitary is `exp(+i * (H - c0 I) / lambda)`, and the principal phase
is decoded back to Hartree as `c0 + lambda * phase`.

`build_step()` returns a nonempty deterministic sequence of `(term, angle)`
pairs for one stationary step. `angle` is the real alpha in
`exp(+i * alpha * P)`. The evaluator resolves every returned term against the
trusted Hamiltonian and repeats the exact returned schedule. The method may
define PF1, PF2/Strang, a higher-order composition, grouping, reordered terms,
or another stationary recurrence; no declared `formula_order` is required.
It is called **once** per species: encode all stages in the schedule, with at most
16 times the Hamiltonian term count; an internal counter cannot alternate steps.

Angles may vary by label, coefficient, support, or species, including negative
substeps and corrections. The judge compiles `Rz(-2 * angle)` and scores energy;
custom schedules are not certified for full-unitary fidelity.

`synthesis_epsilon()` returns a finite CUDA-Q operator-norm tolerance
strictly between zero and one for each scheduled rotation. It may depend on the
term and angle.

Do not change the Hamiltonian, electron sector, target time, exact reference,
evaluator, or benchmark definition.

## Fixed score

The evaluator:
1. compiles scheduled rotations with CUDA-Q and solves the compiled-step spectrum;
   the GPU Lanczos backend requires stable eigenphases/HF weights and never
   falls back to full-unitary diagonalization;
2. repeats eigenphases exactly according to `repetitions()`;
3. scores the lowest HF-accessible energy against the cached exact ground energy;
4. assigns the remaining one-kcal/mol budget to QPE, using the paper bound
   `ceil(8*pi/(target_time*epsilon_QPE))` and enough attempts for 95% confidence.

For a reaction, independently evaluated species share the failure budget and
their confidence-adjusted T costs are summed. `individual_errors` requires
`sum_s |nu_s (E_pf,s - E_exact,s)| < epsilon_chem`; `error_cancellation` requires
`|sum_s nu_s (E_pf,s - E_exact,s)| < epsilon_chem`. Both values are recorded,
but lane identity, feasibility, history, exports, and plots use one contract.

Once a lane is feasible, accept only a strict T95 reduction that remains within
1 kcal/mol. Before feasibility, the loop may retain exact-error progress.

## Campaign invariants

- Select one generated PF2 baseline: `loose` uses the rigorous Eq. (19) beta
  bound; `tight` measures beta against cached exact energy at 0.1 mHa.
  Both use Eq. (34) allocation and the same fixed evaluator.
- Cache the exact reference once per Hamiltonian; reuse it in tight calibration,
  iteration 0, and every mutation. Measure the unchanged baseline at iteration 0.
- Keep both reaction contracts and loose/tight campaigns separate: histories,
  task IDs, mutation pools, queues, workers, incumbents, exports, and plots.
  Exact references and hardware locks may be shared; candidate histories may not.
- Expose only the selected baseline as `candidate.py`, never its sibling.
  Each worker accepts one baseline-tagged root and never alternates baseline queues.
- Mutate external incumbents sequentially for the iteration budget in the
  campaign manifest. Dataset size, replicates, and compute placement are selected
  by the deployment. Histories and transcripts stay external.
- Record `mutation_protocol.json` and context/prompt hashes. Label protocol
  changes in analyses. Permanent inference failures block campaign mutations;
  transient/invalid proposals get backoff and at most three attempts per iteration.
  Repair the cause, then explicitly run `hsopt-openshell resume --path <root>
  --reason <repair>`. Evaluation collection continues during a mutation block.
- The configured canonical campaign owns the mutation trace; validation replays
  archived candidates without another agent pass and requires numerical parity.

After complete matched loose/tight iteration-0 censuses, `hsopt-baseline-drop`
records `100 * (T95_loose - T95_tight) / T95_loose` per experiment/contract.
Require matching grids, hashes, contracts, backends, accelerators, placement,
and accuracy; never pool hardware.

Use `uv run`; see `cli/README.md` for agent setup and `references/layout.md` for layout.
The external runtime uses `hsopt-handoff` to dispatch evaluation requests,
deliver results, record acceptance, and resume mutations; see `docs/runtime.md`.
Compute configuration belongs to the deployment; scientific evaluators and acceptance stay fixed.

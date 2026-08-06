---
name: hsopt
description: Run or inspect exact compiled-unitary T-gate autoresearch for HSOpt product formulas, HamLib absolute energies, and reaction-energy lanes.
---

# HSOpt product-formula autoresearch

Use this skill for the HSOpt mutation loop. The repository runtime is ordinary
package code under `cli/src/hsopt/`; this file is only the compact agent contract.

## Mutation workspace

An isolated mutation workspace contains:

- `candidate.py`: the only editable file;
- `evaluate.py`: the read-only evaluator implementation;
- `problem.json`, or `reaction.json` plus referenced Hamiltonian JSON files:
  read-only benchmark inputs;
- immutable feedback from the preceding exact evaluation.

Read these files as often as useful. Optional literature/code/web searches are
allowed. `import math` is allowed in `candidate.py`; other imports and external
runtime access are not. Make exactly one coherent edit to `candidate.py`, then
stop. Never fabricate an evaluation.

## Editable policy

Preserve this interface:

```python
class ProductFormula:
    def __init__(self, problem): ...
    def repetitions(self): ...
    def build_step(self, terms, step_time): ...
    def synthesis_epsilon(self, term, angle): ...
```

`problem` and every `term` are plain dictionaries. A term includes its trusted
Pauli `label`, numerical `coefficient`, `support_size`, and X/Y/Z counts.

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

`synthesis_epsilon()` returns a finite PyGridSynth operator-norm tolerance
strictly between zero and one for each scheduled rotation. It may depend on the
term and angle.

Inspect `evaluate.py` and the concrete JSON inputs rather than guessing the
runtime types or coefficients. Do not change the Hamiltonian, electron sector,
target time, exact reference, evaluator, or benchmark definition.

## Fixed score

The evaluator:

1. compiles every scheduled Pauli rotation with PyGridSynth;
2. constructs and diagonalizes the actual compiled one-step unitary;
3. repeats its eigenphase exactly according to `repetitions()`;
4. selects the lowest-energy peak with numerically resolved weight from the
   fixed Hartree--Fock input;
5. compares it with the cached exact ground energy; and
6. assigns the remaining one-kcal/mol budget to QPE precision and multiplies
   the directly compiled one-evolution T count by the paper's
   `ceil(8*pi/(target_time*epsilon_QPE))` full-evolution query bound; and
7. multiplies that QPE-run cost by enough attempts for 95% confidence,
   accounting for both HF peak weight and the paper's one-half
   phase-estimation success guarantee.

For a reaction, the signed energy is assembled from independently evaluated
species. The five-percent failure budget is shared across unique species and
their confidence-adjusted T costs are summed.

Once a lane is feasible, accept only a strict T95 reduction that remains within
1 kcal/mol. Before feasibility, the loop may retain exact-error progress.

## Campaign invariants

- Select exactly one generated paper PF2 baseline per campaign: `loose` uses
  the rigorous Eq. (19) beta upper bound; `tight` uses the numeric beta measured
  against the cached exact energy at 0.1 mHa. Both use the same Eq. (34)
  allocation and evaluator.
- Compute the exact reference once per unique Hamiltonian and reuse it during
  tight calibration, iteration 0, and every mutation evaluation.
- The mutation sandbox receives only the selected policy as `candidate.py`; it
  must not receive or read the sibling baseline initializer.
- Keep loose and tight supervisors, mutation pools, queues, evaluator workers,
  and output roots separate. Each evaluator worker accepts exactly one
  baseline-tagged campaign root; a hardware lock may coordinate a shared GPU,
  but a worker must never alternate between baseline queues.
- Iteration 0 measures the immutable generated baseline.
- Iterations 1--50 mutate the external incumbent sequentially.
- The production grid is 82 absolute-energy instances times three lanes, plus
  ten reactions times two lanes: 266 lanes and 13,300 mutations.
- Candidate histories, evaluator results, OpenCode/OpenShell transcripts,
  tool metrics, and sandbox rejections remain in the external campaign root.
- The canonical A100 campaign creates the single mutation trace. Any CPU
  validation campaign replays those exact archived candidates without invoking
  the agent and is audited for numerical parity.

After the independent iteration-0 census exists for complete same-placement
loose and tight baselines, run `hsopt-baseline-drop`. It writes one
`baseline_t_gate_drop_percent.txt` number per checked-in experiment folder,
using `100 * (T95_loose - T95_tight) / T95_loose`. Require identical census
grids, source and reference hashes, evaluator contracts, backends,
accelerators, placement labels, and chemical-accuracy passes; never pool CPU,
A100, and L4 observations.

For campaign orchestration, invoke the installed commands (`hsopt-campaign`,
`hsopt-supervisor`, `hsopt-export-exact`, and `hsopt-reports`) through `uv run`;
do not execute package files by path. OpenShell host setup and telemetry use
`hsopt-openshell` and are documented in `cli/README.md`. For repository layout,
read `references/layout.md`; for Slurm resources use the current repository
`README.md` and the `$ssh` skill. Treat `cli/src/hsopt/` and family evaluators
as fixed infrastructure during a campaign.

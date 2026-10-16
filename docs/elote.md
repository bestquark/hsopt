# ELOTE

**Error-Limited Optimized Trotter Evolution** names a learned, saved protocol.
HSOpt is the framework that searches for it. An ELOTE includes its Hamiltonian
representation and grouping, ordered splitting stages, repetition rule,
rotation angles, optional basis change, and rotation-synthesis tolerance rule.
The Python policy is the algorithm; the resolved per-molecule recipe in each
evaluation records the concrete groups, stages, angles and tolerances.

For commuting groups (H_g=\sum_{j\in g}c_jP_j), a splitting defines one step

\[
S(\Delta t)=\prod_{\ell}^{\text{application order}}
  \prod_{j\in g_\ell} e^{i a_\ell c_j\Delta t P_j},
\qquad \Delta t=t/r.
\]

`build_step(terms, step_time)` can return the historical flat `(term, angle)`
list, or explicitly describe commuting groups and their splitting:

```python
return {
    "groups": [[(term, term["coefficient"] * step_time) for term in group]
               for group in groups],
    "stages": [(0, 0.5), (1, 1.0), (0, 0.5)],
    "merge_adjacent": True,
}
```

The example is Strang splitting of **two** groups. The host checks commutation
inside each group, expands every stage, and optionally combines consecutive
rotations of the same Pauli before synthesis. Negative composition weights,
reordered groups and mixed precision are allowed. Grouping does not automatically
give joint synthesis of a block: these commuting exponentials still use individual
Pauli gadgets. Every physical rotation is charged, including repeated occurrences
of a cached synthesis result. No accuracy or formal-order claim follows from a
policy's label; the compiled circuit must pass the fixed numerical evaluator.

An optional `basis_change()` returns application-ordered `("H", q)`, `("S", q)`,
`("SDG", q)` and `("CX", control, target)` gates. The evaluator supplies signed
terms of (CHC^\dagger), then evaluates (C^\dagger S_C^r C) against the original
input and reference. Literal synthesis matrices, including transverse errors and
global phase, are conjugated back exactly. These are Clifford changes of frame;
general orbital/Givens rotations and low-rank integral factorizations require a
separate validated representation and cost model. They are not silently free.

## Nested starting protocol

`experiments/initial_nested_pf.py` uses the published greedy **disjoint-spin-orbital** nesting idea
from [Reiher et al., Appendix E.2.b](https://arxiv.org/pdf/1605.03590), symmetric
second-order splitting, and uniform synthesis precision. The host recovers
Hermitian number-conserving one-/two-body blocks from a Jordan–Wigner Hamiltonian
with OpenFermion, checks their Pauli reconstruction, and retains roundoff
corrections explicitly. Unsupported encodings or noncommuting block expansions
are rejected. The identity shift, trusted Hamiltonian, phase encoding and exact
reference are unchanged. The agent may edit the grouping, splitting, repetitions,
angles, Clifford frame and tolerance equation.
If OpenFermion's default symbolic pruning loses small coefficients, a second
decomposition transforms unit-coefficient operators and accumulates scaled raw
coefficient maps before repeating the same reconstruction audit.

This implements the documented nesting construction and adjacent inverse-Clifford
cancellation in Jordan–Wigner parity circuits. It does not claim to reproduce
the complete proprietary LIQUi|> optimizer or the paper's PAR architecture.
Nesting primarily targets depth: the evaluator reports logical gate counts and
T depth separately from total T95. Depth assumes all-to-all logical connectivity
and unit gate duration; it is not an architecture runtime estimate.

Changing grouping changes the ideal product formula. **Do not reuse old-order
beta as the calibration of the nested baseline.** Prepare a new external profile,
reusing the matching cached exact reference:

```bash
uv sync --extra chemistry --extra gpu-lanczos
uv run hsopt-elote prepare --problem /work/data/molecule.json \
  --exact-reference /work/data/exact_reference.json \
  --baseline tight --output-dir /work/data/nested-tight/molecule
```

Tight beta is measured for this grouped splitting at 0.1 mHa; loose beta uses
Eq. (19) on its ordered Pauli expansion. Eq. (34) allocation uses that beta and
the actual rotation count for uniform epsilon. The allocated ideal step and the
unchanged compiled iteration 0 must both pass before a calibration is sealed.
Preparation uses GPU Lanczos and never a dense fallback. A small-system dense
oracle exists only for tests. A failed/uncertified start is not licensed for
mutations and must be investigated separately.

Ground-input certification checks literal-U residuals, majority input overlap,
independent-start agreement and absence of repetition wrapping for the selected
peak. It does not require a global triangle bound on all expanded rotations:
that bound can overcount cancelling fermionic contributions. The lowest-HF-peak
solver still requires its global phase-order certificate. No residual, overlap,
energy-accuracy or selected-branch wrapping threshold is relaxed.
An inconclusive ground-input solve first refines its anchor and can then retry
with the Hermitian action `Im(exp(-i theta) U)`, where `theta` comes from the
fixed input's compiled-unitary Rayleigh quotient. It uses the same literal
operator and input, without consulting the exact energy. Independent-start
agreement and literal-U residual checks still apply; prior qualified energies
and integer QPE repetition costs must also agree across projections. Reported
overlap spreads are numerical diagnostics, not rigorous absolute error bounds.
After a verifier upgrade, `hsopt-elote refresh` can recheck an unchanged sealed
preparation with `--problem`, `--exact-reference` and `--output-dir`, preserving
its beta, repetitions, precision and reference. New campaigns require validation
with the current verifier; historical campaign receipts remain immutable.

Use these new `calibration.json` paths in a shared experiment selection. Pass
`--policy-profile fermionic_nested` to either `abs_protocol/mutate.py prepare`
or `diff_protocol/mutate.py prepare`. All species of a reaction need matching
new calibrations under its selected `calibration_root`. Existing Pauli profiles,
cached inputs and pinned campaigns keep their original definitions. A new
experiment receives its own run directory and iteration 0.

Inspect any bound baseline or learned candidate without compilation:

```bash
uv run hsopt-elote describe --problem /work/data/molecule.json \
  --policy-file /work/runs/seed-01/iterations/0020/bound_policies/molecule_01.py \
  --output /work/figures/elote_recipe.json
```

The shared candidate has an empty sealed parameter placeholder. Use the actual
per-case bound `policy.py` in the evaluation request when inspecting a shared run.
Acceptance remains all-cases accuracy plus strict improvement of the geometric
mean of baseline-relative T95; no grouping or lower-depth claim bypasses it.

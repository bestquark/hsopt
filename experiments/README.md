# Experiments

The individual optimization experiments are:

- `abs_energy/`: the canonical 82-instance HamLib selection through 12
  qubits. Each selected Hamiltonian has a visible directory containing an
  `instance.json` definition plus generated `initial_pf_loose.py` and
  `initial_pf_tight.py`. The family-level
  `abs_energy/evaluate.py` contains its explicit fixed evaluator entry point,
  alongside the source/checksum manifests used to prepare external inputs.
- `diff_energy/`: ten visible reaction directories. Each contains a
  `definition.json` plus the same two baseline variants. Its explicit fixed
  evaluator entry point is `diff_energy/evaluate.py`.

Two separate experiments optimize one shared protocol per trajectory:

- [`abs_protocol/`](abs_protocol/README.md): one mutation evaluated on all selected molecules.
- [`diff_protocol/`](diff_protocol/README.md): one mutation evaluated on all selected reactions,
  separately for each reaction-error contract.

Both use calibrated baseline parameters, explicit mixed-precision mutation prompts,
and geometric-mean baseline-relative T95 with case-wise accuracy constraints.

`generate_initial_pf.py` exposes the compact `ProductFormula`, returns an object
via `product_formula(problem, parameters)`, and generates both iteration-0 policies.
Calibration math and provenance live in `cli/src/hsopt/campaign/calibration.py`.
The generator verifies sealed per-Hamiltonian calibration records and
embeds two controlled forms of the Reiher--Wiebe--Svore--Wecker--Troyer PF2
baseline. `loose` uses the rigorous Eq. (19) double-commutator upper bound for
$\beta(H)$ without the exact energy. `tight` numerically finds the first PF2
step count passing the same 0.1 mHa reference error against the cached exact
ground energy. Both use lexicographic Strang ordering, the same Eq. (34) error
allocation, normalized physical time $t=1/\sum_j|h_j|$, and 1 kcal/mol total
target. The individual policies stay beside the same definition; shared-protocol
code lives in its own experiment folders and consumes an explicit external suite.

The agent edits the emitted class, not the calibration machinery:

| Hook | Agent can change |
| --- | --- |
| `repetitions()` | Step count $r$; the evaluator sets $\Delta t=t/r$. |
| `build_step(terms, step_time)` | Ordering, stages, repetitions of terms, and rotation angles. |
| `synthesis_epsilon(term, angle)` | The tolerance for each rotation, using its term and angle. |

Uniform precision belongs to the starting PF2 baseline. A mutation may return
different tolerances for different rotations; feasibility still comes from the
fixed compiled-evolution energy test. Total physical time remains fixed.

Run `uv run hsopt-layout --check` to prove that the 82 + 10 visible directories
match their compact registries and source checksums exactly. Run
`uv run hsopt-layout --write` after an intentional registry update to
regenerate the deterministic definition files.

The coefficient-heavy production Hamiltonians, sealed per-instance
calibrations, exact references, mutation lanes, scheduler logs, agent
transcripts, exports, and reports live below an external `HSOPT_DATA_ROOT`.
Only the compact generated loose and tight iteration-0 policies and the
requested folder-level comparison scalar are duplicated in Git.

Once matching same-placement loose and tight iteration-0 censuses have both
completed, `hsopt-baseline-drop` writes
`baseline_t_gate_drop_percent.txt` beside those two policies. That file is
deliberately one signed decimal number: the percentage reduction in the
confidence-adjusted QPE T count from the loose baseline to the tight baseline.
Its detailed counts, hashes, evaluator contract, backend, and accelerator
remain in the external audit specified with `--audit-output`.

The external campaign layout mirrors the two families:

```text
$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy/
$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy/
$HSOPT_DATA_ROOT/exact/production-cpu82-v1/lanes/{abs_energy,diff_energy}/
$HSOPT_DATA_ROOT/exact/production-gpu82-v2/lanes/{abs_energy,diff_energy}/
```

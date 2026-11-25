# Selected molecular ELOTE

[`best_policy.py`](best_policy.py) is the exact best feasible shared molecular
policy from five independent searches, each completed through 100 mutations.
Its SHA-256 is
`d199f6a4efef965aa015dbd73f69d14f628fd1b9eaa3653391dad8e3a75e472e`.
This is the policy used for the paper's final molecular Figures 2 and 3.

| Dataset | Hamiltonians | ELOTE passes | Geometric-mean $T_{95}$ reduction |
| --- | ---: | ---: | ---: |
| Training, 4–16 qubits | 115 | 115/115 | 84.82% |
| Testing, 18–22 qubits | 76 evaluated pairs | 76/76 | 86.81% over 75 pairs where both policies pass |

The testing census contains 78 Hamiltonians. NaLi-18 and NaLi-20 are baseline
construction exclusions. The extrapolated tight baseline fails chemical
accuracy for HF-18 (1.833 mHa); ELOTE passes it. All 78 census rows and their
statuses are preserved in [`testing_results.csv`](testing_results.csv).
Testing uses the baseline fit frozen from training, with no fit to testing
energies. Testing results did not enter policy selection.

## Files

| File | Purpose |
| --- | --- |
| [`best_policy.py`](best_policy.py) | Byte-identical archived shared policy; `baseline_parameters = {}` is filled by the host. |
| [`training_parameters.json`](training_parameters.json) | The original per-Hamiltonian `n_steps`, `uniform_epsilon` and `reference_trotter_number_beta`, keyed by exact problem name. |
| [`training_results.csv`](training_results.csv) | All 115 recorded costs, energy errors, problem/reference/calibration hashes and resolved schedule hashes. |
| [`testing_results.csv`](testing_results.csv) | Recorded testing costs and errors, including the failed baseline and two construction exclusions. |
| [`manifest.json`](manifest.json) | Accepted decision, original search/round IDs, five-search selection, suite/result hashes, validation and source references. |
| [`SHA256SUMS`](SHA256SUMS) | Checksums of this snapshot's files. |

This folder is a frozen release artifact. Live mutation workspaces, full
Hamiltonians, reference vectors, compiler caches and raw traces stay in the
external campaign archive. Future selected protocols should get a new snapshot
directory rather than overwrite this one.

## What the policy does

- Adds contributions with the same Pauli string and preserves the signed
  aggregate coefficient in every rotation angle.
- Uses flat first-order Lie–Trotter splitting in lexicographic Pauli order,
  with one rotation per supplied string per step. Exactly zero aggregate
  coefficients give identity rotations.
- Retains each Hamiltonian's baseline repetition count and uses no Clifford
  change of frame.
- Allocates mixed synthesis tolerances using coefficient magnitude, Pauli
  support, diagonal/off-diagonal character and Hamiltonian feature scales.
  Rotation angle is not an independent feature of this selected rule.

For the detailed interface and baseline construction, see
[`skills/hsopt/references/elote.md`](../../skills/hsopt/references/elote.md). The feature scales are calculated
from the host-verified fermionic block contributions **before** coefficient
aggregation. The host still needs this decomposition, although the executable
schedule is flat.

**The source is preserved verbatim.** Its opening docstring and unused
`groups()` method describe the original nested initializer. The learned
`build_step()` defines the flat first-order schedule; the archived comments do
not override that executable behavior. See the policy explanation in
[the paper's appendix](https://github.com/bestquark/eqd-paper/blob/175311350f27540bca1a5510692af1e40c97476e/tex/appendix/understanding_elote.tex).

## Inspect or replay

From the repository root, verify the snapshot:

```bash
(cd results/elote && shasum -a 256 -c SHA256SUMS)
```

The shared file cannot be instantiated with an empty parameter table. Bind
the saved training rows using the ordinary host function, writing the derived
policy outside the checkout:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from hsopt.campaign.protocol import bind_policy

snapshot = Path("results/elote")
parameters = json.loads((snapshot / "training_parameters.json").read_text())
bind_policy(
    snapshot / "best_policy.py",
    parameters,
    Path("../elote-replay/bound_policy.py"),
)
PY
```

With a matching prepared training Hamiltonian, inspect the resolved schedule
and tolerances without synthesis or a GPU solve:

```bash
uv run hsopt-elote describe --problem /path/to/prepared/problem.json \
  --policy-file ../elote-replay/bound_policy.py \
  --output ../elote-replay/recipe.json
```

Match the problem's exact `name` to a parameter key and its SHA-256 to
`training_results.csv`. Binding changes the derived policy's file hash; the
unbound `best_policy.py` hash above remains the shared algorithm's identity.
Use the archived per-case requests and pinned scientific environment for an
exact numerical replay. Preparing inputs or running a new campaign is described
in [the reproduction guide](../../skills/hsopt/references/reproduce.md); a new stochastic
search need not rediscover this exact code.

These training rows are specific to the recorded Hamiltonians. Applying the
policy to another Hamiltonian requires matching baseline parameters and the
verified host representation; the tolerance equation alone is not the complete
protocol. The testing results here record the separately frozen training-fit
extrapolation, whose full allocation provenance remains in
[the paper's testing manifest](https://github.com/bestquark/eqd-paper/blob/175311350f27540bca1a5510692af1e40c97476e/data/elote_testing_manifest.json).

## Cost and accuracy scope

`baseline_t95` and `best_t95`/`elote_t95` include QPE precision queries and enough
independent repetitions for 95% success. Columns ending in `t_count` contain
literal compiled T gates for one full evolution, including all Trotter steps,
and exclude those QPE factors. The selection score is the geometric mean of
per-Hamiltonian baseline-relative $T_{95}$, with every training Hamiltonian
required to pass its own chemical-accuracy constraint.

The recorded evaluator uses CUDA-Q synthesis and matrix-free GPU Lanczos on
the ground-state input branch. Ideal ground-state preparation cost is excluded.
The maximum ELOTE energy errors are 0.356 mHa in training and 0.680 mHa in
testing; the chemical-accuracy threshold is 1.594 mHa. These are observed
compiled-energy results on the recorded cases, not a general operator-error
guarantee for arbitrary Hamiltonians or input states.

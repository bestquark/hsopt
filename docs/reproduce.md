# Reproduce shared ELOTE and δ-ELOTE workflows

Use an external workspace next to the source checkout:

```text
hsopt_workdir/
    hsopt/       Source, two baseline registries, generic starting policies
    data/        Acquired/generated Hamiltonians, exact references, calibrations
    runs/        Shared protocol histories and immutable per-round receipts
    figures/     Generated plots
    runtime/     Deployment environments, dispatchers and machine configuration
```

The default molecular registry is the 115-Hamiltonian training selection through
16 qubits. The reaction registry defines ten reactions over 21 species. The
[baseline guide](../baseline/README.md) distinguishes new preparation from exact
historical replay. The [`single_molecule` branch](https://github.com/bestquark/hsopt/tree/single_molecule)
contains the earlier per-instance experiment workflows.

## Install and check the source registries

```bash
uv sync --locked --extra chemistry --extra gpu-lanczos
uv run hsopt-layout --check
uv run hsopt-data export --output-root ../data/registry
```

On Apple silicon, follow [the CUDA-Q CPU setup](synthesis.md#apple-silicon) for
small local examples. Production shared-protocol evaluation uses GPU Lanczos.
The two checked-in input sources are `baseline/molecules.json` and
`baseline/reactions.json`. Exported CSV/JSON preparation views include source
hashes and remain outside Git.

## Molecular inputs

```bash
uv run hsopt-data hamlib --output-root ../data/hamlib
uv run python experiments/abs_protocol/prepare_inputs.py \
  --archive-root ../data/hamlib --output ../data/molecules115
```

Downloads verify the pinned archive checksums and refuse corrupt existing data.
Preparation uses the original HamLib parser and norm-based phase encoding,
with the fixed electron count and signed Pauli coefficients from the registry.
The public `hsopt-campaign prepare-hamlib` command can also consume
`--active-manifest baseline/molecules.json --checksum-manifest baseline/molecules.json`
with `--max-qubits 16`. For an explicit historical selection, first export it
with `hsopt-data export --selection generalization100` or `single_molecule82`.

## Reaction inputs

```bash
uv run hsopt-data reactions --output-root ../data/reactions-physical
uv run hsopt-campaign prepare-reactions \
  --reaction-manifest baseline/reactions.json \
  --cache-checksums ../data/reactions-physical/problem_cache.sha256 \
  --cache-root ../data/reactions-physical/problem_cache \
  --output-root ../data/reactions-prepared
```

Generation uses the fixed geometries, charges, basis, active spaces, RHF,
occupied/virtual localization and Jordan–Wigner mapping recorded in the registry.
It records dependency versions, the chemistry convention and fresh hashes.
`--species h2` provides a bounded generation check, but that partial output
cannot prepare all ten reactions. Use a fresh output directory for each run.

The default chemistry conversion is corrected physical generation. The
registry's historical cache hashes describe the archived conversion only.
`--convention historical-replay` explicitly requests that old convention; it
does not promise byte-identical regeneration of the old cache. Keep archived
inputs for historical replay, and calibrate/evaluate fresh inputs separately.

## References and grouped starting policies

For each prepared molecule/species, cache the exact ground reference once on
the selected GPU backend. The existing Python reference API is:

```python
import json
from pathlib import Path
from hsopt.evaluation.compiled_unitary import evaluate_exact_reference

problem = Path("/work/data/molecule.json")
reference = evaluate_exact_reference(problem, backend="cupy-lanczos")
Path("/work/data/reference.json").write_text(json.dumps(reference, indent=2) + "\n")
```

Then calibrate and generate the shared grouped PF2 baseline:

```bash
uv run hsopt-elote prepare --problem /work/data/molecule.json \
  --exact-reference /work/data/reference.json --baseline tight \
  --output-dir /work/data/baselines/molecule
```

This uses `baseline/initial_nested_pf.py`, verifies its fermionic decomposition,
measures tight beta at the original 0.1 mHa reference error, applies Eq. (34),
and checks the allocated ideal step and unchanged compiled policy before sealing
`calibration.json`. Reuse the matching exact reference rather than recompute it
during mutations. Loose grouped preparation is selected explicitly with
`--baseline loose`. The separate compact Pauli-profile generator is
`baseline/generate_initial_pf.py`; do not apply its calibration to a different
nested splitting.

For reactions, assemble exact difference references from the species caches
using `assemble_exact_difference_reference(..., energy_target="ground_branch")`.
The helper in `experiments/diff_protocol/prepare_ground_references.py` converts
an existing species-referenced selection while preserving its calibration
paths and recording source hashes.

## Shared protocol selections and search

Create an external molecular selection listing every selected case:

```json
{"cases": [{"id": "molecule_01", "source": "molecule.json",
  "reference": "reference.json", "calibration": "baselines/molecule/calibration.json"}]}
```

Paths are relative to that selection. Use all 115 cases for the published
training suite; the one-case object above only illustrates the schema.

```bash
uv run python experiments/abs_protocol/mutate.py prepare \
  --selection /work/data/molecular-selection.json \
  --run-dir /work/runs/elote/search-01 --baseline tight \
  --policy-profile fermionic_nested --energy-target ground_branch \
  --iterations 100 --replicate search-01
uv run python experiments/abs_protocol/evaluate.py \
  --run-dir /work/runs/elote/search-01
uv run python experiments/abs_protocol/mutate.py advance \
  --run-dir /work/runs/elote/search-01
```

Evaluate unchanged iteration 0 first. Alternate evaluation and `advance`, or
use an external dispatcher to deliver the ordinary portable requests. No
mutation can proceed before the starting suite passes. Every proposed shared
policy is scored on the whole sealed suite. Keep each independent search in a
separate directory; never edit an existing suite, baseline or trajectory.

Reaction selections point to reaction JSON, assembled references, and the
species calibration root. Use the `diff_protocol` workflow with an explicit
`individual_errors` or `error_cancellation` contract and ground-branch target.
To warm-start δ-ELOTE from molecular ELOTE, call
`hsopt.campaign.protocol.prepare(..., family="diff_protocol", initial_policy=...,
policy_profile="fermionic_nested", energy_target="ground_branch", ...)`.
The transferred policy is measured and sealed as iteration 0; reaction-specific
mutations follow it. The tight PF2 comparison remains a separate validation.
See [the reaction workflow](../experiments/diff_protocol/README.md).

## Replaying and comparing

The saved [best molecular ELOTE](../experiments/elote_search/results/elote115/README.md)
includes its exact shared code, 115 bound-parameter rows, compact recorded
results and original hashes. Bind those rows for the matching Hamiltonians;
use archived requests and the pinned scientific environment for exact numerical
replay. A fresh agent search need not rediscover the same code.

Testing freezes the selected training policy and parameter-fit protocol before
reading testing energies. Record extrapolation failures and construction
exclusions. Never relabel old-policy energies as results for a new checkpoint.
For the [three-strategy comparison](../experiments/elote_search/README.md), all
hybrid PySR probes count toward the evaluation budget.

[Runtime dispatch](runtime.md), [the policy interface](elote.md) and
[numerical certification](numerical-certification.md) document the remaining
execution and scientific contracts. Source relocation changes no energy,
compiler, cost or acceptance equations in existing pinned campaigns.

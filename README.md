<p align="center">
  <img src="docs/assets/hsopt-logo.png" alt="HSOpt logo" width="760">
</p>

# HSOpt

HSOpt searches for shared product-formula protocols that reduce fault-tolerant
energy-estimation costs while preserving a fixed accuracy target. **ELOTE**
(Error-Limited Optimized Trotter Evolution) optimizes molecular energies;
**δ-ELOTE** optimizes signed reaction-energy differences. The fixed evaluator
uses literal CUDA-Q Clifford+T synthesis and matrix-free GPU Lanczos.

The best released molecular ELOTE is in
[`experiments/elote_search/results/elote115/`](experiments/elote_search/results/elote115/README.md),
with its exact policy, all 115 training parameter rows, validation results and
checksummed provenance.

## Repository layout

```text
baseline/
    molecules.json          Molecular selections and pinned HamLib archive hashes
    reactions.json          Reaction/species definitions and historical cache audit
    initial_nested_pf.py    Shared grouped PF2 starting-policy template
    generate_initial_pf.py  Compact Pauli PF2 generator for sealed calibrations
experiments/
    abs_protocol/           Shared molecular ELOTE optimization
    diff_protocol/          Shared reaction δ-ELOTE optimization
    elote_search/           Agent/PySR comparisons and released protocols
cli/src/hsopt/              Preparation, evaluator, search and sandbox implementation
cli/tests/                 Scientific checks and small executable fixtures
plots/                     Publication-plot source code
skills/hsopt/              Mutation-agent instructions
```

Single-molecule and single-reaction experiment folders, their generated
initializers, and their original reproduction instructions are preserved on
[the `single_molecule` branch](https://github.com/bestquark/hsopt/tree/single_molecule).
`main` uses the compact [baseline registries](baseline/README.md). Generated
Hamiltonians, calibrated policies, references, campaigns, caches and raw traces
belong in an external `hsopt_workdir/`.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync --locked --python 3.12
uv run hsopt-layout --check
```

Linux installs the pinned CUDA-Q synthesis package. Apple silicon requires
[the CUDA-Q CPU installation](docs/synthesis.md#apple-silicon). The included
small fixtures can then be evaluated without an agent:

```bash
uv run python cli/tests/fixtures/abs_energy_h2/evaluate.py --backend numpy
uv run python cli/tests/fixtures/diff_energy_h2_bond_stretch/evaluate.py \
  --error-contract error_cancellation --backend numpy
```

These are bounded executable tests. Shared production searches use GPU Lanczos;
they do not fall back to dense diagonalization.

## Prepare and optimize a shared protocol

Follow [the reproduction guide](docs/reproduce.md) to acquire checksummed
HamLib inputs, generate reaction species, prepare exact references and grouped
baselines, and construct an external selection. The default molecular registry
contains the 115 training Hamiltonians through 16 qubits. Historical 100- and
82-case selections remain explicit metadata choices, not runtime defaults.

```bash
uv run hsopt-data export --output-root ../data/registry
uv run hsopt-data hamlib --output-root ../data/hamlib
uv run python experiments/abs_protocol/prepare_inputs.py \
  --archive-root ../data/hamlib --output ../data/molecules115
```

One shared candidate is evaluated on every selected Hamiltonian or reaction.
All cases must pass accuracy before a strict geometric-mean baseline-relative
$T_{95}$ improvement can be retained. $T_{95}$ includes full evolution, QPE
precision queries and repetitions for 95% success; ideal state-preparation cost
is excluded. Reactions select either a componentwise error constraint or a
signed-difference constraint, with isolated histories for each.

The agent may change repetitions, grouping, splitting, ordering, rotation
angles, mixed synthesis precision and supported Clifford frames. The executable
`ProductFormula` defines the algorithm; [the ELOTE interface](docs/elote.md)
describes its validated boundaries. Calibration values, reference states,
Hamiltonians, physical times and acceptance thresholds remain fixed.

- [Molecular protocol workflow](experiments/abs_protocol/README.md)
- [Reaction protocol workflow](experiments/diff_protocol/README.md)
- [Agent-only, PySR-only and hybrid comparison](experiments/elote_search/README.md)
- [Portable worker/runtime contract](docs/runtime.md)

## Installation

HSOpt requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for its environment:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync --locked --python 3.12
```

Install the optional cuPyNumeric/Legate dependencies on a supported Linux
system:

```bash
uv sync --extra cupynumeric
```

## Agent skill

[`skills/hsopt/SKILL.md`](skills/hsopt/SKILL.md) defines the mutation contract;
the runtime lives separately in `cli/src/hsopt/`. The agent reads its policy,
research context and required benchmark inputs before editing. It can explore
new product formulas and mixed precision while the host preserves sealed
inputs, calibration and evaluation constraints.

## Development

```bash
uv sync --extra chemistry
uv run hsopt-layout --check
uv run ruff check cli/src cli/tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest -q
uv build
```

The [CLI guide](cli/README.md) describes the runtime and mutation audit. Plot
sources and reproduction notes live in [plots/](plots/README.md).

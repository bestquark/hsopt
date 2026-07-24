<p align="center">
  <img src="docs/assets/hsopt-logo.png" alt="HSOpt logo" width="760">
</p>

# HSOpt

HSOpt uses an AI mutation loop to reduce the logical $T$-gate cost of
product-formula quantum phase estimation while preserving a fixed energy
accuracy target. Every candidate is evaluated by constructing and
diagonalizing the compiled Clifford+$T$ unitary; analytical product-formula
bounds are not used as the acceptance test.

The repository contains one benchmark tree, [`experiments/`](experiments/),
and one plotting-source tree, [`plots/`](plots/). Raw campaigns, databases,
Hamiltonian collections, caches, scheduler logs, and agent transcripts stay
outside the Git checkout.

## Objective

For a fixed Hamiltonian $H$, evolution time $t$, electron number, and
Hartree--Fock (HF) input, HSOpt caches the exact ground energy
$E^{\mathrm{exact}}$ once. A candidate policy may change only the
product-formula and rotation-synthesis parameters used to compile
$U_{\mathrm{PF}}$.

The evaluator diagonalizes $U_{\mathrm{PF}}$, unwraps its eigenphases into
energies, and selects the lowest-energy peak with numerically resolved HF
weight $p_{\mathrm{HF}}$. If one QPE attempt costs $T_{\mathrm{raw}}$, the number of
independent attempts required to observe that peak with 95% confidence is

$$
R_{95} =
\begin{cases}
1, & p_{\mathrm{HF}}=1, \\
\left\lceil \dfrac{\log(0.05)}{\log\left(1-p_{\mathrm{HF}}\right)} \right\rceil,
& 0<p_{\mathrm{HF}}<1,
\end{cases}
\qquad T_{95} = R_{95}T_{\mathrm{raw}}.
$$

A molecular candidate is feasible when

$$
\left|E^{\mathrm{PF}}-E^{\mathrm{exact}}\right|
\leq 1\ \mathrm{kcal}\,\mathrm{mol}^{-1}.
$$

Among feasible candidates, HSOpt minimizes $T_{95}$.
Reports display this objective as **T95 (raw x R)**.

For a reaction, each unique species is evaluated independently and the signed
reaction energy is formed classically. The five-percent failure budget is
shared equally across the unique species, giving at least 95% joint confidence
by the union bound. Stoichiometric coefficients affect the energy difference
but do not duplicate circuits. The reaction objective is the sum of the
confidence-adjusted species costs and is reported as **T95 (raw total)**.

## Repository layout

```text
hsopt/
├── experiments/
│   ├── abs_energy/             91 HamLib folders: definition + initial_pf.py
│   ├── diff_energy/            10 reaction folders: definition + initial_pf.py
│   └── generate_initial_pf.py  Immutable initializer for both families
├── plots/                      Publication-plot source code
├── skills/hsopt/
│   ├── SKILL.md                Optimization-agent instructions
│   ├── references/             On-demand workflow references
│   └── scripts/
│       └── hsopt/              Evaluator, queue, supervisor, export, and reports
└── tests/                      Scientific-contract and recovery tests
```

## Installation

HSOpt requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for its environment:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync
```

Install the optional cuPyNumeric/Legate dependencies on a supported Linux
system:

```bash
uv sync --extra cupynumeric
```

## Run the compact checks

Evaluate the checked-in HamLib fixture on NumPy:

```bash
uv run python tests/fixtures/abs_energy_h2/evaluate.py \
  --backend numpy
```

Evaluate the checked-in bond-stretch energy difference:

```bash
uv run python tests/fixtures/diff_energy_h2_bond_stretch/evaluate.py \
  --backend numpy
```

Verify that an immutable iteration-0 policy still matches the embedded paper
contract:

```bash
uv run hsopt-paper-baseline \
  tests/fixtures/abs_energy_h2/hamiltonian.json \
  --backend numpy \
  --output /tmp/h2-paper-calibration.json

uv run python experiments/generate_initial_pf.py \
  tests/fixtures/abs_energy_h2/hamiltonian.json \
  --output tests/fixtures/abs_energy_h2/initial_pf.py \
  --calibration /tmp/h2-paper-calibration.json \
  --check
```

These fixtures are development checks and therefore live under `tests/`.
Under `experiments/abs_energy/`, every production Hamiltonian has its own
folder containing `instance.json` and its generated `initial_pf.py`. Under
`experiments/diff_energy/`, every production reaction has its own folder
containing `definition.json` and its generated `initial_pf.py`. Verify the
definitions against the compact source registries with:

```bash
uv run hsopt-layout --check
```

The large coefficient archives/caches and their materialized evaluator inputs
remain external datasets supplied to `hsopt-campaign`.

## Campaign lifecycle

Every optimization lane follows the same order:

1. Prepare the paper baseline once per unique Hamiltonian. Compute the
   supplement's Eq. (19) Pauli double-commutator coefficient bound exactly,
   derive $\beta(H)$ at the paper's $0.1$ mHa reference error without using
   $E_{\mathrm{exact}}$, and directly allocate a $1$ kcal/mol Trotter/QPE/
   synthesis budget using PyGridSynth T counts. Separately compute and cache
   $U_{\mathrm{exact}}$ once for evaluator reuse.
2. Evaluate the generated paper baseline unchanged as iteration 0.
3. Run iterations 1--50 sequentially, retaining every proposal, evaluator
   result, and accept/reject decision.
4. Accept only feasible strict improvements in confidence-adjusted $T$ cost.
5. Export and report only after every expected lane is complete.

The current dense production selection contains exactly 91 HamLib
Hamiltonians through 14 qubits, with five mutation lanes per Hamiltonian. It
also contains ten reactions with two policy contracts each.
This gives 475 lanes and 23,750 mutation evaluations, plus iteration 0 for
every lane.

Set an absolute data root before initializing a campaign:

```bash
export HSOPT_DATA_ROOT=/absolute/path/to/hsopt-data
```

Build the unique-Hamiltonian calibration manifest, run four independent CPU
shards, and require a complete sealed result set:

```bash
uv run hsopt-paper-baseline-batch build-manifest \
  --abs-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --output "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json"

for shard in 0 1 2 3; do
  uv run hsopt-paper-baseline-batch run \
    "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
    --output-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
    --backend numpy --mode hamiltonian_bound \
    --shard-index "$shard" --shard-count 4
done

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

On NVPark, run each shard as a separate job on the `cpu` partition with no
GPU request. The command is resumable: a sealed source-matching result is
verified and skipped. `hamiltonian_bound` is the canonical baseline: it
computes the complete Eq. (19) Pauli coefficient sum, not a Monte Carlo
estimate. The same pass caches one exact reference solely for evaluator reuse;
that energy does not enter $\beta(H)$.
The final command refuses incomplete calibration sets, writes a checksummed
external index, and materializes byte-identical `initial_pf.py` files in all
101 checked-in experiment directories.

```bash
uv run hsopt-campaign initialize \
  --hamlib-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --calibration-root "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --generator experiments/generate_initial_pf.py \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --hamlib-replicates 5 \
  --iterations 50
```

The canonical CPU campaign owns the stochastic OpenCode/Nemotron mutation
trace. The A100 campaign must use the archived CPU candidates in the same order
with `--no-mutations` and `--replay-campaign-root`; it must not run a second
independent mutation pass.

Mutable campaign material belongs below the external data root:

```text
$HSOPT_DATA_ROOT/
├── exact/                       Production inputs and campaigns
├── development/                 Development campaigns
├── old_experiments/             Preserved superseded campaigns
└── scratch/                     Preflight and disposable runs
```

Campaign IDs are immutable after work begins. Keep candidate policies, exact
references, SQLite databases, OpenCode/OpenShell logs, Slurm output, telemetry,
and virtual environments outside the repository.

## Main commands

| Command | Purpose |
|---|---|
| `hsopt` / `hsopt-loop` | Initialize, reconcile, inspect, propose, or replay one sequential lane |
| `hsopt-exact` | Enqueue exact references and candidates, prepare batches, run them, and merge results |
| `hsopt-layout` | Verify or regenerate the checked-in 91-Hamiltonian/10-reaction directory layout |
| `hsopt-paper-baseline` | Prepare one Eq. (19) bound-derived PF2 baseline and exact reference |
| `hsopt-paper-baseline-batch` | Build, run, and audit resumable calibration shards |
| `hsopt-campaign` | Prepare external inputs and initialize the complete lane grid |
| `hsopt-supervisor` | Persistently coordinate mutation, batching, recovery, export, and reports |
| `hsopt-openshell` | Run the isolated OpenShell/OpenCode mutation adapter |
| `hsopt-export-exact` | Produce a strict checksummed export from a completed campaign |
| `hsopt-reports` | Compile summary and extended PDFs from a strict export |
| `hsopt-exact-parity` | Certify the CPU campaign against its A100 replay |

Run any command with `--help` for its complete interface.

## Reports

First produce a normalized, checksummed export:

```bash
uv run hsopt-export-exact \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --family abs_energy \
  --output-dir "$HSOPT_DATA_ROOT/exact/final-abs-energy-export"
```

Then compile the paper reports:

```bash
uv run hsopt-reports \
  --family abs_energy \
  --export-dir "$HSOPT_DATA_ROOT/exact/final-abs-energy-export" \
  --output-dir "$HSOPT_DATA_ROOT/exact/final-abs-energy-reports"
```

The reporting gate emits two PDFs per Hamiltonian or reaction:

- a summary comparing the initial and final accepted policies;
- an extended table containing the initial policy and every mutation,
  acceptance decision, exact-reference error, product-formula settings, raw
  $T$ count, repetition factor, and confidence-adjusted $T_{95}$ cost.

LaTeX is used when available; ReportLab provides the portable renderer.

After the CPU and A100 campaigns finish, run the cross-backend audit:

```bash
uv run hsopt-exact-parity \
  --cpu-campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-v1" \
  --gpu-campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu91-v2" \
  --output "$HSOPT_DATA_ROOT/exact/production-gpu91-v2/cross_backend_audit.json" \
  --wait \
  --interval-seconds 60
```

The audit checks the complete lane grid, source and policy hashes, compiled
synthesis records, confidence-adjusted $T$ counts, policy settings,
acceptance trajectories, and energies. CPU and A100 energies are compared
within the configured numerical tolerance rather than requiring bitwise
floating-point identity.

## Plotting

Reusable code for publication figures belongs in [`plots/`](plots/). Plot
scripts should consume checksummed final exports or compact derived tables,
write generated figures to an explicit output directory, and keep raw campaign
data outside Git. Operational telemetry dashboards remain with their campaign
runners because they are part of execution rather than paper plotting.

## Agent skill

The repository contains `$hsopt` as a single source bundle: its instructions,
references, and complete Python runtime live together under `skills/hsopt/`.
The package is installed from `skills/hsopt/scripts/hsopt/` by `uv sync`.

Cloning the repository does not automatically activate the skill in an agent
environment. Install that agent-facing entry separately with:

```bash
npx skills add bestquark/hsopt --skill hsopt -y
```

The skill defines the immutable evaluator contract, permitted mutation
boundary, production ordering, and CPU/A100 replay rules.

### OpenShell + OpenCode mutation agent

[OpenShell](https://docs.nvidia.com/openshell/latest/about/installation) base
sandboxes include [OpenCode](https://opencode.ai/docs/), so OpenCode does not
need a second installation inside the sandbox. Install OpenShell on the host,
verify its gateway, and configure the protected NVIDIA inference route once:

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
openshell status
export NVIDIA_API_KEY="your-key"
uv run hsopt-openshell configure
```

The supervisor then creates one disposable sandbox per proposal, installs
`skills/hsopt/SKILL.md` as the sandbox's `hsopt` skill, exposes only the
isolated `candidate.py` mutation target plus bounded research tools, records
the OpenCode/OpenShell action logs, and deletes the sandbox. OpenCode's
[standalone installer](https://opencode.ai/docs/) is needed only when using a
custom image that does not already include it.

## Development

Run formatting, lint, and the complete test suite before publishing changes:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

See [`experiments/README.md`](experiments/README.md) for experiment-storage
rules and [`BRANCHES.md`](BRANCHES.md) for preserved historical branches.

## License

MIT License. See [`LICENSE`](LICENSE).

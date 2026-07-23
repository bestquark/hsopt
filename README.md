<p align="center">
  <img src="docs/assets/hsopt-logo.png" alt="HSOpt logo" width="760">
</p>

# HSOpt

HSOpt uses an AI mutation loop to reduce the logical \(T\)-gate cost of
product-formula quantum phase estimation while preserving a fixed energy
accuracy target. Every candidate is evaluated by constructing and
diagonalizing the compiled Clifford+\(T\) unitary; analytical product-formula
bounds are not used as the acceptance test.

The clean branch contains one benchmark tree, [`experiments/`](experiments/),
and one plotting-source tree, [`plots/`](plots/). Raw campaigns, databases,
Hamiltonian collections, caches, scheduler logs, and agent transcripts stay
outside the Git checkout.

## Objective

For a fixed Hamiltonian \(H\), evolution time \(t\), electron number, and
Hartree--Fock (HF) input, HSOpt caches the exact ground energy \(E^{\rm exact}\)
once. A candidate policy may change only the product-formula and
rotation-synthesis parameters used to compile \(U_{\rm PF}\).

The evaluator diagonalizes \(U_{\rm PF}\), unwraps its eigenphases into
energies, and selects the lowest-energy peak with numerically resolved HF
weight \(p_{\rm HF}\). If one QPE attempt costs \(T_{\rm raw}\), the number of
independent attempts required to observe that peak with 95% confidence is

```text
R95 = ceil(log(0.05) / log(1 - p_HF))
T95 = R95 * T_raw
```

A molecular candidate is feasible when
\(\lvert E^{\rm PF}-E^{\rm exact}\rvert\) is no larger than
1 kcal mol\(^{-1}\). Among feasible candidates, HSOpt minimizes \(T_{95}\).
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
│   ├── abs_energy/             HamLib91 definition and compact fixture
│   ├── diff_energy/            Ten-reaction definition and compact fixture
│   └── generate_initial_pf.py  Immutable initializer for both families
├── plots/                      Publication-plot source code
├── skills/hsopt/               Optimization-agent instructions
├── src/hsopt/                  Evaluator, queue, supervisor, export, and reports
└── tests/                      Scientific-contract and recovery tests
```

## Installation

HSOpt requires Python 3.12 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for its environment:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
git switch clean
uv sync
```

Install the optional cuPyNumeric/Legate dependencies on a supported Linux
system:

```bash
uv sync --extra cupynumeric
```

The optional GEPA prompt-optimization dependencies are installed with:

```bash
uv sync --extra gepa
```

## Run the compact checks

Evaluate the checked-in HamLib fixture on NumPy:

```bash
uv run python experiments/abs_energy/hamlib_h2_r0_7/evaluate.py \
  --backend numpy
```

Evaluate the checked-in bond-stretch energy difference:

```bash
uv run python experiments/diff_energy/h2_bond_stretch/evaluate.py \
  --backend numpy
```

Verify that an immutable iteration-0 policy still matches the embedded paper
contract:

```bash
uv run python experiments/generate_initial_pf.py \
  experiments/abs_energy/hamlib_h2_r0_7/hamiltonian.json \
  --output experiments/abs_energy/hamlib_h2_r0_7/initial_pf.py \
  --check
```

These fixtures are development checks. Production absolute- and
difference-energy inputs are external datasets supplied to `hsopt-campaign`.

## Campaign lifecycle

Every optimization lane follows the same order:

1. Build and cache the exact reference once.
2. Evaluate the generated paper baseline unchanged as iteration 0.
3. Run iterations 1--50 sequentially, retaining every proposal, evaluator
   result, and accept/reject decision.
4. Accept only feasible strict improvements in confidence-adjusted \(T\) cost.
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

```bash
uv run hsopt-campaign initialize \
  --hamlib-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --seed-validation experiments/diff_energy/inputs/seed_validation.json \
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
  \(T\) count, repetition factor, and confidence-adjusted \(T_{95}\) cost.

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
synthesis records, confidence-adjusted \(T\) counts, policy settings,
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

The repository contains the `$hsopt` agent skill, but cloning the Python
runtime does not automatically install it into an agent environment. Install
the skill separately with:

```bash
npx skills add bestquark/hsopt --skill hsopt -y
```

The skill defines the immutable evaluator contract, permitted mutation
boundary, production ordering, and CPU/A100 replay rules.

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

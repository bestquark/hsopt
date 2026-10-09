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

Follow [Reproduce the paper workflow](docs/reproduce.md) from a fresh clone:
acquire and verify inputs, inspect Hamiltonians, generate loose/tight policies,
set up OpenShell, run mutations, and reproduce plots. Keep generated files in
one external `hsopt_workdir/`.

## Quick start on your own computer

The checked-in H2 examples run on a CPU with NumPy on macOS or Linux. They
include their Hamiltonians and policies; no NVIDIA account, GPU, SSH access,
API key, or external campaign data is needed. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync --locked --python 3.12
uv run python cli/tests/fixtures/abs_energy_h2/evaluate.py --backend numpy
uv run python cli/tests/fixtures/diff_energy_h2_bond_stretch/evaluate.py \
  --error-contract error_cancellation --backend numpy
```

Each command prints a JSON result. Look for `passes_chemical_accuracy: true`,
the energy error, and `confidence_adjusted_t_count` (the optimization score).
These commands evaluate the supplied policies; they do not invoke an AI agent.

To try a policy change, copy the fixture policy to an external directory and
pass it with `--policy-file`:

```bash
mkdir -p ../hsopt-demo
cp cli/tests/fixtures/abs_energy_h2/initial_pf.py ../hsopt-demo/candidate.py
# Edit ../hsopt-demo/candidate.py, then evaluate it:
uv run python cli/tests/fixtures/abs_energy_h2/evaluate.py \
  --backend numpy --policy-file ../hsopt-demo/candidate.py
```

The [product-formula interface](#product-formula-policy) below describes the
editable policy. Automated mutations additionally require the
[OpenShell agent setup](#openshell--opencode-mutation-agent). Full paper
campaigns require separately prepared inputs and suitable compute resources;
large HamLib archives are downloaded explicitly, and the exact reaction inputs
are supplied as a compact checksummed bundle. Historical run outputs and
site deployment state remain external.

GitHub Actions checks installation, every command's `--help`, the two CPU
examples, and the test suite on macOS and Linux with Python 3.12.

## Objective

For each physical Hamiltonian,

```math
H=c_0I+\sum_j h_jP_j,
\qquad
\lambda=\sum_j |h_j|,
```

HSOpt uses the explicit QPE encoding

```math
U=\exp\left(+i\frac{H-c_0I}{\lambda}\right).
```

Equivalently, the product formula simulates the physical Hamiltonian for
$t=1/\lambda$ Hartree$^{-1}$. Since
$\lVert H-c_0I\rVert\leq\lambda$, every exact encoded eigenphase lies in
$[-1,1]$ radians and cannot wrap across $\pm\pi$. The principal phase is
decoded without consulting the exact answer:

```math
E_{\mathrm{Ha}}=c_0+\lambda\phi.
```

The exact ground energy is cached once only to score candidates. A candidate
may change the product formula and rotation-synthesis parameters, but not this
encoding, the Hamiltonian, or the electron sector.

The evaluator diagonalizes the compiled $U_{\mathrm{PF}}$ and selects the
lowest decoded energy with numerically resolved HF weight $p_{\mathrm{HF}}$.
After measuring its compiled-unitary systematic error
$\epsilon_{\mathrm{sys}}$, the remaining chemical-accuracy budget is assigned
to QPE:

```math
\epsilon_{\mathrm{QPE}}
=
\epsilon_{\mathrm{chem}}-\epsilon_{\mathrm{sys}}.
```

Using the rigorous phase-estimation row of the Reiher *et al.* supplement, one
QPE attempt uses

```math
M_{\mathrm{QPE}}
=
\left\lceil
\frac{8\pi}{t\epsilon_{\mathrm{QPE}}}
\right\rceil
```

full-evolution queries and succeeds with probability at least
$p_{\mathrm{HF}}/2$. The number of independent attempts required for 95%
confidence is therefore

```math
R_{95}
=
\left\lceil
\frac{\log(0.05)}{\log\left(1-p_{\mathrm{HF}}/2\right)}
\right\rceil,
\qquad
T_{95}=R_{95}M_{\mathrm{QPE}}T_U,
```

where $T_U$ is the directly synthesized PyGridSynth $T$ count of one compiled
product-formula evolution. A molecular candidate is feasible when

$$
\left|E^{\mathrm{PF}}-E^{\mathrm{exact}}\right|
< 1\ \mathrm{kcal}\,\mathrm{mol}^{-1}.
$$

Among feasible candidates, HSOpt minimizes $T_{95}$.
Reports retain $T_U$, $M_{\mathrm{QPE}}$, $R_{95}$, and their full product.

For a reaction, each unique species is evaluated independently and the signed
reaction energy is formed classically. The remaining QPE energy budget is
allocated so every species contributes an equal worst-case absolute error
after its stoichiometric coefficient. The five-percent failure budget is shared
equally across unique species, giving at least 95% joint confidence by the
union bound. Stoichiometric coefficients affect the energy difference but do
not duplicate circuits. The reaction objective is the sum of the full
confidence-adjusted species QPE costs.

Reaction optimization is deliberately split into two independent contracts.
The conservative `individual_errors` lane requires

$$
\sum_s \left|\nu_s\left(\widetilde E_s-E_s\right)\right|
< \varepsilon_{\mathrm{chem}},
$$

while the `error_cancellation` lane requires

$$
\left|\sum_s \nu_s\left(\widetilde E_s-E_s\right)\right|
< \varepsilon_{\mathrm{chem}}.
$$

Every evaluation records both quantities. The selected contract is sealed in
the lane manifest and queue task ID and alone controls feasibility and
acceptance; mutations, incumbents, exports, and plots are never pooled across
the two contracts.

## Product-formula policy

The agent edits a small executable `ProductFormula` class, not the baseline's
derivation record. Its three scientific hooks are:

```python
class ProductFormula:
    def repetitions(self): ...
    def build_step(self, terms, step_time): ...
    def synthesis_epsilon(self, term, angle): ...
```

`build_step()` returns the ordered `(Pauli term, angle)` rotations for one
stationary step. That returned schedule is authoritative: the evaluator can
apply it with ideal rotations to construct the exact mathematical PF step or
compile the same rotations with PyGridSynth. No `formula_order = 2` constraint
is imposed. The generated baseline implements PF2/Strang, while mutations may
replace its recurrence with PF1, a higher-order composition, grouping, or
another deterministic stationary product formula.

The evaluator infers standard PF1/PF2 and common ordering labels from the
actual schedule. Arbitrary recurrences are recorded as custom together with a
schedule checksum, so a stale human-written label cannot affect evaluation.
The full beta/Eq. (34) derivation, checksums, error allocation, and cached exact
energy remain in sealed campaign metadata outside the agent-editable file.
The `loose` initializer uses the rigorous Eq. (19) beta upper bound; the
`tight` initializer measures beta numerically against the cached exact energy
at the same 0.1 mHa reference error.

Each experiment family has one explicit evaluator entry point. Because
coefficient-heavy Hamiltonians and exact references remain external, invoke it
with the selected experiment directory and those sealed paths explicitly:

```bash
uv run python experiments/abs_energy/evaluate.py \
  experiments/abs_energy/<instance> \
  --problem /path/to/hamiltonian.json \
  --exact-reference /path/to/exact-reference.json \
  --baseline loose \
  --backend numpy --output /path/to/result.json
```

## Repository layout

```text
hsopt/
├── experiments/
│   ├── abs_energy/             82 definitions × two baseline policies
│   ├── diff_energy/            10 definitions × two baseline policies
│   └── generate_initial_pf.py  Loose/tight initializer for both families
├── plots/                      Publication-plot source code
├── skills/hsopt/               Compact agent instructions and references
└── cli/
    ├── src/hsopt/
    │   ├── campaign/           Lane lifecycle, queues, supervisor, and exports
    │   ├── cluster/            NVPark/Slurm execution adapter
    │   ├── evaluation/         Fixed compiled-unitary judge and diagnostics
    │   ├── sandbox/            OpenShell/OpenCode mutation boundary
    │   ├── view/               Mutation ledger and final-report views
    │   └── tgate_opt/          Generic mutation execution helpers
    └── tests/                  Scientific-contract tests and compact fixtures
```

The complete Reiher baseline derivation and policy renderer share the single
`experiments/generate_initial_pf.py` source. The installable campaign package
only contains a thin loader for invoking it.

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

## Run the compact checks

Evaluate the checked-in HamLib fixture on NumPy:

```bash
uv run python cli/tests/fixtures/abs_energy_h2/evaluate.py \
  --backend numpy
```

Evaluate the checked-in bond-stretch energy difference:

```bash
uv run python cli/tests/fixtures/diff_energy_h2_bond_stretch/evaluate.py \
  --error-contract error_cancellation \
  --backend numpy
```

Verify that an immutable iteration-0 policy still matches its sealed paper
calibration:

```bash
uv run hsopt-paper-baseline \
  cli/tests/fixtures/abs_energy_h2/hamiltonian.json \
  --backend numpy \
  --output /tmp/h2-paper-calibration.json

uv run python experiments/generate_initial_pf.py \
  cli/tests/fixtures/abs_energy_h2/hamiltonian.json \
  --output cli/tests/fixtures/abs_energy_h2/initial_pf.py \
  --calibration /tmp/h2-paper-calibration.json \
  --baseline loose \
  --check
```

These fixtures are development checks and therefore live under `cli/tests/`.
Under `experiments/abs_energy/`, every production Hamiltonian has its own
folder containing `instance.json`, `initial_pf_loose.py`, and
`initial_pf_tight.py`. Under
`experiments/diff_energy/`, every production reaction has its own folder
containing `definition.json` and the same two generated policy variants. Verify the
definitions against the compact source registries with:

```bash
uv run hsopt-layout --check
```

The large coefficient archives/caches and their materialized evaluator inputs
remain external datasets supplied to `hsopt-campaign`.

## Campaign lifecycle

Every optimization lane follows the same order:

1. Compute and cache $U_{\mathrm{exact}}$ once per unique Hamiltonian. Prepare
   the selected paper baseline: `loose` derives $\beta(H)$ from the rigorous
   Eq. (19) Pauli double-commutator upper bound without
   $E_{\mathrm{exact}}$; `tight` finds the first PF2 step count passing 0.1 mHa
   against that cached exact energy. Both allocate the 1 kcal/mol
   Trotter/QPE/synthesis budget using the same rigorous Eq. (32)/(34) calculus
   with $\alpha=8\pi$, $\gamma=4$, and $\delta_0=11$.
2. Evaluate the generated paper baseline unchanged as iteration 0. This is
   the first stage that invokes PyGridSynth and records the realized T count.
3. Run iterations 1--50 sequentially, retaining every proposal, evaluator
   result, and accept/reject decision.
4. Accept only feasible strict improvements in confidence-adjusted $T$ cost.
5. Export and report only after every expected lane is complete.

The current dense production selection contains exactly 82 HamLib
Hamiltonians through 12 qubits, with three mutation lanes per Hamiltonian. It
also contains ten reactions with the isolated `individual_errors` and
`error_cancellation` contracts.
This gives 266 lanes and 13,300 mutation evaluations, plus iteration 0 for
every lane.

Set an absolute data root before initializing a campaign:

```bash
export HSOPT_DATA_ROOT=/absolute/path/to/hsopt-data
```

When an existing materialization contains the physical Pauli coefficients but
predates the norm-based QPE contract, create a fresh input-only tree. This
copies no exact references, policies, queues, or campaign state:

```bash
uv run hsopt-campaign normalize-inputs \
  --abs-inputs "$HSOPT_DATA_ROOT/exact/physical-inputs/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/physical-inputs/diff_energy" \
  --output-root "$HSOPT_DATA_ROOT/exact/normalized-inputs-v1"
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
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --baseline loose

uv run hsopt-paper-baseline-batch generate \
  "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
  --calibration-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/initial-policies" \
  --generator experiments/generate_initial_pf.py \
  --baseline loose \
  --repository-root "$PWD"
```

On NVPark, run each shard as a separate job on the `cpu` partition with no
GPU request. The command is resumable: a sealed source-matching result is
verified and skipped. `hamiltonian_bound` is the canonical baseline: it
computes the complete Eq. (19) Pauli coefficient sum, not a Monte Carlo
estimate. The same pass caches one exact reference solely for evaluator reuse;
that energy does not enter $\beta(H)$.
The final command refuses incomplete calibration sets, writes a checksummed
external index, and materializes byte-identical `initial_pf_loose.py` files in
all 92 checked-in experiment directories. Prepare the tight comparison in a
separate result tree while reusing those exact references:

```bash
uv run hsopt-paper-baseline-batch run \
  "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-tight-v1/results" \
  --backend cupynumeric --mode empirical_exact \
  --exact-reference-root "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results"

uv run hsopt-paper-baseline-batch generate \
  "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/manifest.json" \
  --calibration-dir "$HSOPT_DATA_ROOT/exact/baseline-tight-v1/results" \
  --output-dir "$HSOPT_DATA_ROOT/exact/baseline-tight-v1/initial-policies" \
  --generator experiments/generate_initial_pf.py \
  --baseline tight \
  --repository-root "$PWD"
```

`--exact-reference-root` prevents a second exact-Hamiltonian diagonalization;
only the ideal PF2 step search is new. Keep the calibration trees separate so
one problem hash can have one sealed result per beta definition.

After the independent iteration-0 T95 census has completed for the loose and
tight baselines on the same backend/placement, materialize the baseline-only
reduction used for comparison with autoresearch:

```bash
uv run hsopt-baseline-drop \
  --loose-census-root "$HSOPT_DATA_ROOT/exact/iteration0-t95-loose-v1" \
  --tight-census-root "$HSOPT_DATA_ROOT/exact/iteration0-t95-tight-v1" \
  --loose-placement omni-l4 \
  --tight-placement omni-l4 \
  --repository-root "$PWD" \
  --audit-output "$HSOPT_DATA_ROOT/exact/baseline-t-gate-drop-audit.json"
```

The census deduplicates absolute-energy replicate lanes only after their
immutable inputs agree. It never deduplicates the two reaction contracts, so a
complete census contains 82 absolute tasks plus 20 reaction-contract tasks.
The comparison command then requires matching complete grids, source hashes,
exact-reference hashes, evaluator contracts, execution
backends/accelerators, placement labels, and chemical-accuracy passes. Only
then does it write `baseline_t_gate_drop_percent.txt` in each absolute-energy
folder and one
`baseline_t_gate_drop_<contract>_percent.txt` file for each reaction contract.
The signed value is
$100(T_{\mathrm{loose}}-T_{\mathrm{tight}})/T_{\mathrm{loose}}$, where $T$ is
the confidence-adjusted QPE T count (`T95`) optimized by autoresearch. Thus a
positive value is a reduction from the loose bound to the tight baseline. For
reactions, $T$ is the evaluator's summed reaction cost under the named,
independently evaluated error contract.

```bash
uv run hsopt-campaign initialize \
  --hamlib-inputs "$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy" \
  --calibration-root "$HSOPT_DATA_ROOT/exact/baseline-bound-v1/results" \
  --generator experiments/generate_initial_pf.py \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu82-v1" \
  --baseline loose \
  --hamlib-replicates 3 \
  --iterations 50
```

Use `--reaction-only` for a replacement or validation campaign that must start
both reaction contracts from iteration 0 without duplicating valid
absolute-energy lanes.

To narrow an older five-replicate campaign without deleting its existing
histories, first stop its supervisor and local evaluator workers, then run:

```bash
uv run hsopt-campaign-scope \
  --campaign-root /path/to/existing-campaign \
  --hamlib-replicates 3
```

The command retains the predetermined `run_01`--`run_03` lanes, leaves all
`run_04`/`run_05` histories in place, archives their uncollected queue work,
and writes a checksummed scope-change record. Reaction contracts are unchanged.

The canonical campaign runs compiled-unitary construction and dense
eigendecomposition through cuPyNumeric on one A100 per Slurm job. PyGridSynth
uses the job's CPU cores before the compiled rotations are applied on the GPU.
This A100 campaign owns the single stochastic OpenCode/Nemotron mutation trace.
An optional CPU validation campaign must use the archived A100 candidates in
the same order with `--no-mutations` and `--replay-campaign-root`; it must not
run a second independent mutation pass.

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
| `hsopt-baseline-drop` | Write the loose-to-tight iteration-0 T95 reduction per experiment |
| `hsopt-exact` | Enqueue exact references and candidates, prepare batches, run them, and merge results |
| `hsopt-layout` | Verify or regenerate the checked-in 82-Hamiltonian/10-reaction directory layout |
| `hsopt-paper-baseline` | Prepare one loose or tight PF2 calibration and exact reference |
| `hsopt-paper-baseline-batch` | Build, run, and audit resumable calibration shards |
| `hsopt-campaign` | Prepare external inputs and initialize the complete lane grid |
| `hsopt-supervisor` | Persistently coordinate mutation, batching, recovery, export, and reports |
| `hsopt-openshell` | Run the isolated OpenShell/OpenCode mutation adapter |
| `hsopt-export-exact` | Produce a strict checksummed export from a completed campaign |
| `hsopt-reports` | Compile summary and extended PDFs from a strict export |
| `hsopt-exact-parity` | Certify cross-backend CPU/A100 replay parity |
| `hsopt-view` | Export or serve the read-only mutation-history grid from a campaign directory |

Run any command with `--help` for its complete interface.

## Reports

First produce a normalized, checksummed export:

```bash
uv run hsopt-export-exact \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu82-v1" \
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

After the A100 campaign and optional CPU replay finish, run the cross-backend
audit:

```bash
uv run hsopt-exact-parity \
  --cpu-campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu82-replay-v1" \
  --gpu-campaign-root "$HSOPT_DATA_ROOT/exact/production-gpu82-v1" \
  --output "$HSOPT_DATA_ROOT/exact/production-gpu82-v1/cross_backend_audit.json" \
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

The repository keeps the `$hsopt` agent contract deliberately small under
`skills/hsopt/`. The installable evaluator and campaign runtime live separately
under `cli/src/hsopt/` and are installed by `uv sync`. The mutation sandbox receives
the compact skill, the fixed evaluator source, and only the selected lane's
read-only benchmark inputs; it does not receive the full campaign runtime.

Cloning the repository does not automatically activate the skill in an agent
environment. Install that agent-facing entry separately with:

```bash
npx skills add bestquark/hsopt --skill hsopt -y
```

The skill defines the immutable evaluator contract, permitted mutation
boundary, production ordering, and A100/CPU replay rules.

### OpenShell + OpenCode mutation agent

The implementation and its tests have one explicit home:
[`cli/src/hsopt/sandbox/openshell.py`](cli/src/hsopt/sandbox/openshell.py) owns
host setup, disposable sandbox creation, OpenCode permissions, cleanup, and
tool/security telemetry; [`cli/README.md`](cli/README.md) documents that
boundary and every emitted audit file.

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

Each mutation writes the raw `opencode.jsonl` and `openshell.log` alongside
redacted action and metric summaries. Rebuild those summaries without rerunning
the mutation with:

```bash
uv run hsopt-openshell audit --run-dir /path/to/mutation-run
```

The supervisor then creates one disposable sandbox per proposal, installs
`skills/hsopt/SKILL.md` as the sandbox's `hsopt` skill, exposes only the
isolated `candidate.py` mutation target plus bounded research tools, records
the OpenCode/OpenShell action logs, and deletes the sandbox. OpenCode's
[standalone installer](https://opencode.ai/docs/) is needed only when using a
custom image that does not already include it.

Run loose and tight campaigns with distinct supervisors, mutation pools,
queues, evaluator workers, and output roots. An evaluator worker must accept
exactly one baseline-tagged campaign root; deployments may coordinate access
to a shared accelerator with hardware locks, but must never round-robin tasks
from loose and tight queues through the same worker.

## Development

Run formatting, lint, and the complete test suite before publishing changes:

```bash
uv run ruff format --check cli/src cli/tests
uv run ruff check .
uv run pytest -q
```

See [`experiments/README.md`](experiments/README.md) for experiment-storage
rules and [`BRANCHES.md`](BRANCHES.md) for preserved historical branches.

## License

MIT License. See [`LICENSE`](LICENSE).

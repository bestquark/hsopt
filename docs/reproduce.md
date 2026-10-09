# Reproduce the paper workflow

Use one workspace. The Git checkout contains reusable code and benchmark
definitions; generated files live beside it:

```text
hsopt_workdir/
    hsopt/          Git checkout
    data/           Raw archives, prepared Hamiltonians, sealed calibrations
    runs/           Separate loose/tight campaigns and their exports
    figures/        Generated paper plots
    runtime/        Deployment environments and operational configuration
```

A fresh run uses the current selection: 82 HamLib instances through 12 qubits
and ten reactions. Historical runs retain their own pinned evaluator, inputs,
candidate files, and manifests. Replaying a saved mutation trace requires those
archived candidates; a fresh stochastic agent run will propose different code.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
mkdir hsopt_workdir
cd hsopt_workdir
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync --locked
export HSOPT_WORKDIR="$(cd .. && pwd)"
export HSOPT_DATA_ROOT="$HSOPT_WORKDIR/data"
uv run hsopt-layout --check
uv run python cli/tests/fixtures/abs_energy_h2/evaluate.py --backend numpy
```

The compact example evaluates supplied code and never invokes an agent.

## Acquire and prepare inputs

The HamLib command selects only the public NERSC archives required by the
checked-in instance registry and verifies every SHA-256 checksum. Repeating
it skips verified archives; changed data raises an error.

The 21 reaction species are included in `reaction_cache.tar.gz`: the original
42 JSON/NPZ files, with their geometries, active spaces, localization, and
coefficients. Extraction verifies the existing `problem_cache.sha256` manifest.
It needs neither a private storage path nor a new chemistry calculation.

```bash
uv run hsopt-data hamlib --output-root "$HSOPT_DATA_ROOT/raw/hamlib"
uv run hsopt-data reactions --output-root "$HSOPT_DATA_ROOT/raw/reactions"
uv run hsopt-campaign prepare-hamlib \
  --active-manifest experiments/abs_energy/inputs/instances.csv \
  --checksum-manifest experiments/abs_energy/inputs/hamlib_zip_sha256.txt \
  --data-root "$HSOPT_DATA_ROOT/raw/hamlib" \
  --output-root "$HSOPT_DATA_ROOT/prepared/abs_energy"
uv run hsopt-campaign prepare-reactions \
  --reaction-manifest experiments/diff_energy/inputs/reactions.json \
  --cache-checksums experiments/diff_energy/inputs/problem_cache.sha256 \
  --cache-root "$HSOPT_DATA_ROOT/raw/reactions" \
  --output-root "$HSOPT_DATA_ROOT/prepared/diff_energy"
```

Preparation requires a fresh output directory and does not replace campaign inputs.

## Inspect Hamiltonians and coefficients

`plots/_paper_plot.py` provides the common NVIDIA green/graphite palette and
LaTeX typography. Install a TeX distribution for the paper figures.

```bash
uv run python plots/plot_hamlib_active_spaces.py \
  --output-dir "$HSOPT_WORKDIR/figures/dataset"
uv run python plots/plot_hamlib_coefficients.py \
  --hamlib-input-dir "$HSOPT_DATA_ROOT/prepared/abs_energy" \
  --output-dir "$HSOPT_WORKDIR/figures/dataset"
```

## Prepare both initial policies

`experiments/generate_initial_pf.py` is the sole baseline generator. The public
filenames are `initial_pf_loose.py` and `initial_pf_tight.py`. Each campaign
selects one variant as its immutable `initial_pf.py`; mutations edit only the
external lane's `candidate.py`.

Build one manifest of unique Hamiltonians. Loose calibration constructs the
exact references once; tight calibration reuses them. These are substantive
benchmark calculations, rather than installation checks.

```bash
uv run hsopt-paper-baseline-batch build-manifest \
  --abs-inputs "$HSOPT_DATA_ROOT/prepared/abs_energy" \
  --reaction-inputs "$HSOPT_DATA_ROOT/prepared/diff_energy" \
  --output "$HSOPT_DATA_ROOT/calibrations/manifest.json"
uv run hsopt-paper-baseline-batch run \
  "$HSOPT_DATA_ROOT/calibrations/manifest.json" \
  --output-dir "$HSOPT_DATA_ROOT/calibrations/loose" \
  --backend numpy --mode hamiltonian_bound
uv run hsopt-paper-baseline-batch run \
  "$HSOPT_DATA_ROOT/calibrations/manifest.json" \
  --output-dir "$HSOPT_DATA_ROOT/calibrations/tight" \
  --backend numpy --mode empirical_exact \
  --exact-reference-root "$HSOPT_DATA_ROOT/calibrations/loose"
for baseline in loose tight; do
  uv run hsopt-paper-baseline-batch audit \
    "$HSOPT_DATA_ROOT/calibrations/manifest.json" \
    --output-dir "$HSOPT_DATA_ROOT/calibrations/$baseline" --baseline "$baseline"
  uv run hsopt-paper-baseline-batch generate \
    "$HSOPT_DATA_ROOT/calibrations/manifest.json" \
    --calibration-dir "$HSOPT_DATA_ROOT/calibrations/$baseline" \
    --output-dir "$HSOPT_DATA_ROOT/policies/$baseline" \
    --generator "$PWD/experiments/generate_initial_pf.py" --baseline "$baseline"
done
```

## Install OpenShell and initialize mutation campaigns

OpenShell runs on the agent host. The historical live campaigns used 0.0.89;
its [tagged installer](https://github.com/NVIDIA/OpenShell/tree/v0.0.89) and
[release](https://github.com/NVIDIA/OpenShell/releases/tag/v0.0.89) provide a
versioned setup. The [installation guide](https://docs.nvidia.com/openshell/latest/about/installation)
describes container-runtime requirements and newer versions.

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/v0.0.89/install.sh | \
  OPENSHELL_VERSION=v0.0.89 sh
openshell --version
openshell status
export NVIDIA_API_KEY="your-key"
uv run hsopt-openshell configure
for baseline in loose tight; do
  uv run hsopt-campaign initialize \
    --hamlib-inputs "$HSOPT_DATA_ROOT/prepared/abs_energy" \
    --reaction-inputs "$HSOPT_DATA_ROOT/prepared/diff_energy" \
    --calibration-root "$HSOPT_DATA_ROOT/calibrations/$baseline" \
    --generator "$PWD/experiments/generate_initial_pf.py" \
    --campaign-root "$HSOPT_WORKDIR/runs/$baseline" \
    --baseline "$baseline" --hamlib-replicates 3 --iterations 50
done
```

The base sandbox includes OpenCode. Credentials remain with the host inference
provider. Each proposal receives the repo's skill, one editable candidate, and
read-only benchmark inputs in a disposable sandbox. Record the OpenShell
version, sandbox image, inference model, and HSOpt commit with the run.
Keep raw logs and redacted action summaries. Existing gateways retain their
campaign's version.

The [campaign lifecycle](../README.md#campaign-lifecycle) provides supervisor
and evaluator-worker commands. The [runtime guide](../cli/README.md) explains
the mutation boundary and audit files. Loose and tight campaigns use distinct
roots; reaction contracts remain isolated within each baseline.

Each task freezes the candidate, Hamiltonian inputs, references, and scoring
contract. Explicitly stage those files and a versioned evaluator on the compute
system; submitting from Omni does not make Omni's disk visible there. NVIDIA
Slurm operations use the site's governed broker. The existing `hsopt-nvpark`
adapter uses SSH/rsync and native Slurm; it has not been converted to the broker
and requires a site-permitted native submission path.

For that adapter, set `HSOPT_CLUSTER_USER` if your cluster username differs
from your local username. Its default remote directory is
`/lustre/fsw/portfolios/nvr/users/<user>/autoresearch/hsopt_workdir/runs/evaluation`.
Set `HSOPT_REMOTE_ROOT` before starting the CLI, or pass `--remote-root`, to
select your site's filesystem and a distinct campaign directory. Existing
campaign services retain their explicit per-campaign remote roots.

## Export and plot results

Wait for every expected lane to finish. Run `hsopt-export-exact` and
`hsopt-reports` for each baseline and experiment family separately, using the
options in the [campaign lifecycle](../README.md#campaign-lifecycle).
The completion gate rejects partial runs.

The [paper-figure guide](../plots/README.md) maps each script to its PDF and
input export. It includes precision/energy trajectories, T-gate reduction,
reaction comparisons, baseline tolerance, and policy angle violins. Result
plots verify final-export checksums; live dashboard plots retain partial labels.

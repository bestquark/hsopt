# hsopt

Hamiltonian-simulation T-gate optimization with fixed fault-tolerant chemistry
benchmarks and agent-friendly mutation loops.

`main` is intentionally narrow. It keeps the product-formula benchmark
definition in `examples/pfs/` and the two repo skills that operate
on those lanes:

- `$hsopt`: sequential `simple_pf.py` mutation runs scored by a fixed
  evaluator.
- `$hsgepa`: GEPA prompt optimization for the agent strategy that drives those
  mutation runs.

Raw campaign workspaces stay outside the benchmark tree. Only compact,
completion-gated results for the five 121-Hamiltonian campaigns and the ten
reaction comparisons belong on `main`.

The portable paper runners and final-artifact gates are documented in
[`experiments/`](experiments/README.md).

The older CPU/GPU/QPU constraint framework, GPU-memory experiments, generic
telemetry APIs, broad protocol examples, and large run artifacts are preserved
off `main` on branches listed in `BRANCHES.md`.

## Experiment storage

Set one absolute external data root before creating a campaign:

```bash
export HSOPT_DATA_ROOT=/absolute/path/to/hsopt-data
```

The standard layout is:

```text
$HSOPT_DATA_ROOT/
  paper/hamlib121/hamlib121_paper_YYYYMMDD_<commit>/run_01 ... run_05
  paper/reactions10/reactions10_paper_YYYYMMDD_<commit>/
  development/<benchmark>/<campaign-id>/
  old_experiments/<benchmark>/<campaign-id>/
  scratch/preflight/<campaign-id>/
```

Campaign IDs and their contents are immutable once work starts. Move whole
inactive campaigns between `development`, `paper`, and `old_experiments`. Do
not scatter individual iterations or logs across the home directory. Raw mutation traces,
scheduler logs, caches, databases, and environments never belong in Git. A
small validated final export or summary PDF may be copied to the corresponding
`experiments/*/results/` directory after its completion gate passes.

## Final paper reports

The strict exporters retain normalized product-formula settings, exact and
estimated energies, accept/reject decisions, and a bounded mutation summary.
After an exporter writes `FINAL_EXPORT_COMPLETE` and `SHA256SUMS`, build the
two reports for every scientific instance with:

```bash
uv run hsopt-reports --family hamlib121 \
  --export-dir /external/final/hamlib121-export \
  --output-dir experiments/hamlib121/results/final

uv run hsopt-reports --family reactions10 \
  --export-dir /external/final/reactions10-export \
  --output-dir experiments/reactions10/results/final
```

The renderer verifies the complete checksum set and expected iteration shape,
compiles in a temporary directory, and atomically publishes exactly two PDFs
per Hamiltonian or reaction. It never reads a live database or raw transcript.
The extended PDF records every mutation and evaluator outcome; the summary PDF
compares the initial policy with the final incumbent in the compact paired-row
style used by the paper appendices.

## Install the Agent Skills

The agent skills and the benchmark runtime are separate prerequisites.
Installing the skills teaches an agent the HSOpt workflow, but it does not
clone `examples/pfs/` or install the Python dependencies. Conversely, cloning
this repository does not register the skills with your agent. Complete both
steps when you want the agent to run the local benchmarks.

Install the hsopt agent skills directly from GitHub:

```bash
npx skills add bestquark/hsopt
```

For non-interactive setup that installs both bundled skills:

```bash
npx skills add bestquark/hsopt --skill '*' -y
```

For Claude Code specifically:

```bash
npx skills add bestquark/hsopt --agent claude-code --skill '*' -y
```

To inspect the available skills before installing:

```bash
npx skills add bestquark/hsopt --list
```

To update installed skills later:

```bash
npx skills update hsopt hsgepa -y
```

For local testing from inside this repo:

```bash
npx skills add . --list
```

## Clone the Benchmark Runtime

Separately clone and sync the repository to run the Python CLIs and fixed
benchmark evaluators:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync
```

For GEPA prompt optimization:

```bash
uv sync --extra gepa
```

For optional cuPyNumeric execution of the PFS evaluators on a Legate-capable
local NVIDIA GPU or cluster:

```bash
uv sync --extra cupynumeric
```

## T-Gate Lanes

The canonical benchmark tree is `examples/pfs/`. It contains fixed
chemistry instances and editable lane-local policies:

- `h2/`, `bh/`, `lih/`, `dmds/`: small runnable baselines.
- `n2/`, `co/`, `nh3/`, `hcn/`: larger CAS(6e,5o) lanes seeded for campaigns.
- `evaluate_energy.py`: T count with chemical accuracy on the lowest resolved
  HF-reference spectral peak.
- `evaluate_fidelity.py`: T count with final-state infidelity below the selected
  epsilon budget.

During an optimization run, edit only the active lane's `simple_pf.py` unless
you are deliberately changing the benchmark framework.

## Quick Commands

List lanes:

```bash
uv run hsopt list
```

Inspect a lane:

```bash
uv run hsopt status examples/pfs/h2
```

Print the direct evaluator command:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy --print-only
uv run hsopt evaluate examples/pfs/h2 --evaluator fidelity --epsilon 1e-3 --print-only
```

Render the prompt for an agent run:

```bash
uv run hsopt prompt examples/pfs/h2 --evaluator energy
```

Run the evaluator locally on CPU, or on one local NVIDIA GPU through
Legate/cuPyNumeric:

```bash
uv run hsopt evaluate examples/pfs/h2 --transport local --backend numpy
uv run hsopt evaluate examples/pfs/h2 --transport local --backend cupynumeric
```

Create one isolated proposal with Codex or OpenCode. Mutation histories are
required to live outside the checkout:

```bash
uv run hsopt mutate examples/pfs/h2 --agent codex \
  --run-dir "$HSOPT_DATA_ROOT/development/pfs/codex-001"

NVIDIA_API_KEY=... uv run hsopt mutate examples/pfs/h2 --agent opencode \
  --run-dir "$HSOPT_DATA_ROOT/development/pfs/nemotron-001"
```

OpenCode is pinned to `opencode-ai@1.4.11`; its default model is
`nvidia/nvidia/nemotron-3-super-v3` through NVIDIA's OpenAI-compatible API.
The key is read only from `NVIDIA_API_KEY`, and the agent is denied every tool
except read/edit access to the isolated `simple_pf.py`.

For a shared cluster checkout, `--transport slurm` submits the same evaluator
as one blocking `sbatch --wait` job with one GPU and no scheduler polling. Use
`--print-only` to inspect the exact SSH, Slurm, and Legate handoff before
running it:

```bash
uv run hsopt evaluate examples/pfs/h2 --transport slurm --print-only \
  --submit-host nvlogin --remote-repo-root /shared/hsopt \
  --account ACCOUNT --partition PARTITION \
  --remote-python /shared/hsopt/.venv/bin/python \
  --remote-legate /shared/hsopt/.venv/bin/legate
```

Seed a GEPA prompt-optimization run:

```bash
HSOPT_RUNS=/absolute/path/outside/hsopt
mkdir -p "$HSOPT_RUNS/hsgepa"
uv run hsgepa dataset --smoke --output "$HSOPT_RUNS/hsgepa/dataset_smoke.json"
uv run hsgepa optimize --smoke --run-root "$HSOPT_RUNS/hsgepa/smoke" \
  --max-metric-calls 6 --no-parallel
```

`hsgepa dataset`, `evaluate-prompt`, and `optimize` require explicit output or
run paths. Keep those mutable workspaces outside the repository checkout.

## Compatibility

Some PFS evaluator environment variables still use the `TROVOLUT_` prefix so
existing campaign scripts and lane wrappers keep working. Lane discovery and
the `hsopt` CLI do not require per-lane manifest files.

The old Python CLI aliases `t-autoresearch` and `t-gepa` are still registered
for existing scripts, but the primary commands and skills use `hsopt` and
`hsgepa`.

## License

MIT License - see `LICENSE` for details.

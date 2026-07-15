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

The older CPU/GPU/QPU constraint framework, GPU-memory experiments, generic
telemetry APIs, broad protocol examples, and large run artifacts are preserved
off `main` on branches listed in `BRANCHES.md`.

## Install the Skills

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

These commands install the agent-facing skills. To run the local Python CLIs and
benchmarks, set up the repo environment:

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
cluster:

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

Seed a GEPA prompt-optimization run:

```bash
uv run hsgepa dataset --smoke --output runs/hsgepa/dataset_smoke.json
uv run hsgepa optimize --smoke --run-root runs/hsgepa/smoke --max-metric-calls 6 --no-parallel
```

## Compatibility

Some PFS evaluator environment variables still use the `TROVOLUT_` prefix so
existing campaign scripts and lane wrappers keep working. Lane discovery and
the `hsopt` CLI do not require per-lane manifest files.

The old Python CLI aliases `t-autoresearch` and `t-gepa` are still registered
for existing scripts, but the primary commands and skills use `hsopt` and
`hsgepa`.

## License

MIT License - see `LICENSE` for details.

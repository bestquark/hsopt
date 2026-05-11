# hsopt

Hamiltonian-simulation T-gate optimization with fixed fault-tolerant chemistry
benchmarks and agent-friendly mutation loops.

`main` is intentionally narrow. It keeps the product-formula benchmark
definition in `examples/ft_protocols/pfs/` and the two repo skills that operate
on those lanes:

- `$t-autoresearch`: sequential `simple_pf.py` mutation runs scored by a fixed
  evaluator.
- `$t-gepa`: GEPA prompt optimization for the agent strategy that drives those
  mutation runs.

The older CPU/GPU/QPU constraint framework, GPU-memory experiments, generic
telemetry APIs, broad protocol examples, and large run artifacts are preserved
off `main` on branches listed in `BRANCHES.md`.

## Installation

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

The canonical benchmark tree is `examples/ft_protocols/pfs/`. It contains fixed
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
uv run t-autoresearch list
```

Inspect a lane:

```bash
uv run t-autoresearch status examples/ft_protocols/pfs/h2
```

Print the direct evaluator command:

```bash
uv run t-autoresearch evaluate examples/ft_protocols/pfs/h2 --evaluator energy --print-only
uv run t-autoresearch evaluate examples/ft_protocols/pfs/h2 --evaluator fidelity --epsilon 1e-3 --print-only
```

Render the prompt for an agent run:

```bash
uv run t-autoresearch prompt examples/ft_protocols/pfs/h2 --evaluator energy
```

Seed a GEPA prompt-optimization run:

```bash
uv run t-gepa dataset --smoke --output runs/t_gepa/dataset_smoke.json
uv run t-gepa optimize --smoke --run-root runs/t_gepa/smoke --max-metric-calls 6 --no-parallel
```

## Compatibility

Some PFS evaluator environment variables and molecule manifests still use the
`TROVOLUT_` prefix and `.trovolut.json` names. They are retained so old campaign
scripts and lane wrappers keep working while the repo/package identity moves to
`hsopt`.

## License

MIT License - see `LICENSE` for details.

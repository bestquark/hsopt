# Product-Formula Benchmarks

`examples/ft_protocols/pfs/` is the canonical home for the product-formula
Hamiltonian-simulation benchmarks in trovolut.

- `h2/`, `bh/`, `lih/`, `dmds/`: runnable baseline molecule folders.
- `n2/`, `co/`, `nh3/`, `hcn/`: histogram-derived larger molecule folders
  seeded for autoresearch with CAS(6e,5o) active spaces.
- `evaluate_energy.py`: T count with a chemical-accuracy constraint on the
  lowest resolved HF-reference spectral peak in the fixed-electron sector.
- `evaluate_fidelity.py`: T count with a final-state infidelity constraint
  against exact `exp(+i H t)`.
- `chemistry.py`: fixed molecule registry, cached Hamiltonian data, and shared
  product-formula compilation helpers.

Run either evaluator directly:

```bash
uv run python examples/ft_protocols/pfs/evaluate_energy.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py
uv run python examples/ft_protocols/pfs/evaluate_fidelity.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py --epsilon 1e-3
```

Use molecule folders directly with the helper CLI:

```bash
uv run t-autoresearch status examples/ft_protocols/pfs/h2
uv run t-autoresearch prompt examples/ft_protocols/pfs/h2 --evaluator fidelity --epsilon 1e-3
```

Campaign histories should be created by the external run harness. This folder
only stores the benchmark definitions, the molecule-local `simple_pf.py` files,
and fixed evaluator infrastructure.

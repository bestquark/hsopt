# Fault-Tolerant Protocol Examples

`main` currently keeps one canonical protocol family: product-formula/Trotter
dynamics in `pfs/`.

The removed Taylor/LCU, QSVT, qubitization, factorization, GPU-memory, and
broader resource-estimation examples are preserved on the
`resource/hamiltonian-simulation-resources` branch.

Run the PFS smoke checks from the repo root:

```bash
uv run t-autoresearch list
uv run t-autoresearch status examples/ft_protocols/pfs/h2
uv run python examples/ft_protocols/pfs/evaluate_energy.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py
uv run python examples/ft_protocols/pfs/evaluate_fidelity.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py --epsilon 1e-3
```

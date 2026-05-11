# Fault-Tolerant Protocol Examples

`main` currently keeps one canonical protocol family: product-formula/Trotter
dynamics in `pfs/`.

The removed Taylor/LCU, QSVT, qubitization, factorization, GPU-memory, and
broader resource-estimation examples are preserved on the
`resource/classical-compute` branch.

Run the PFS smoke checks from the repo root:

```bash
uv run t-autoresearch list
uv run t-autoresearch status examples/pfs/h2
uv run python examples/pfs/evaluate_energy.py --molecule h2 --script examples/pfs/h2/simple_pf.py
uv run python examples/pfs/evaluate_fidelity.py --molecule h2 --script examples/pfs/h2/simple_pf.py --epsilon 1e-3
```

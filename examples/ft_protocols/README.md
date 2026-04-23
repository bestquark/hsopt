# Fault-Tolerant Protocol Examples

This folder contains small, self-contained H2 resource-estimation examples for
fault-tolerant chemistry protocols. They are meant to expose editable policy
knobs for T-count optimization while keeping the chemistry instance and
chemical-accuracy threshold fixed.

Direct dynamics methods for implementing exp(-i H t):

- `pfs/`: product-formula/Trotter dynamics with dense H2 error checks and
  pygridsynth rotation T counts
- `taylor_lcu/`: segmented truncated Taylor/LCU dynamics using Qualtran
  chemistry SELECT/PREPARE counts
- `qsvt/`: qubitized QSVT/QSP dynamics using a sparse chemistry walk

Energy-estimation method:

- `qubitization/`: qubitization-QPE over the same sparse chemistry walk

Install the optional Qualtran dependency with:

```bash
uv sync --extra ft-protocols
```

Run the smoke checks from the repo root:

```bash
uv run --extra ft-protocols python examples/ft_protocols/pfs/evaluate.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py
uv run --extra ft-protocols python examples/ft_protocols/taylor_lcu/evaluate.py --molecule h2 --script examples/ft_protocols/taylor_lcu/h2/simple_taylor_lcu.py
uv run --extra ft-protocols python examples/ft_protocols/qsvt/evaluate.py --molecule h2 --script examples/ft_protocols/qsvt/h2/simple_qsvt.py
uv run --extra ft-protocols python examples/ft_protocols/qubitization/evaluate.py --molecule h2 --script examples/ft_protocols/qubitization/h2/simple_qubitization.py
```

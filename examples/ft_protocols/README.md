# Fault-Tolerant Protocol Examples

This folder contains fault-tolerant chemistry protocol examples with editable
policy knobs for T-count optimization while keeping the chemistry instance and
chemical-accuracy threshold fixed.

Direct dynamics methods for implementing exp(-i H t):

- `pfs/`: product-formula/Trotter dynamics for `h2`, `bh`, `lih`, and `dmds`,
  using pygridsynth rotation T counts and a phase-to-energy chemical-accuracy
  constraint
- `taylor_lcu/`: segmented truncated Taylor/LCU dynamics using Qualtran
  chemistry SELECT/PREPARE counts
- `qsvt/`: qubitized QSVT/QSP dynamics using a sparse chemistry walk

Energy-estimation method:

- `qubitization/`: qubitization-QPE over the same sparse chemistry walk

Coefficient-distribution diagnostic:

- `hamiltonian_factorization_histograms.py`: builds four similar-size localized
  CAS(6e,6o) molecular Hamiltonians and compares `|alpha_i|` coefficient/data-entry
  histograms for the unfactorized, SF, DF, and THC representations.

Install the optional Qualtran dependency with:

```bash
uv sync --extra ft-protocols
```

Run the smoke checks from the repo root:

```bash
uv run t-autoresearch list
uv run t-autoresearch status examples/ft_protocols/pfs/h2
uv run --extra ft-protocols python examples/ft_protocols/pfs/evaluate.py --molecule h2 --script examples/ft_protocols/pfs/h2/simple_pf.py
uv run --extra ft-protocols python examples/ft_protocols/taylor_lcu/evaluate.py --molecule h2 --script examples/ft_protocols/taylor_lcu/h2/simple_taylor_lcu.py
uv run --extra ft-protocols python examples/ft_protocols/qsvt/evaluate.py --molecule h2 --script examples/ft_protocols/qsvt/h2/simple_qsvt.py
uv run --extra ft-protocols python examples/ft_protocols/qubitization/evaluate.py --molecule h2 --script examples/ft_protocols/qubitization/h2/simple_qubitization.py
uv run python examples/ft_protocols/hamiltonian_factorization_histograms.py
```

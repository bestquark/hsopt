# Trovolut

> [![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bestquark.github.io/trovolut/) &nbsp;&nbsp;&nbsp; 🔑 **Password**: `gordonbell2026`

Evolutionary Trotterization for quantum simulation with modular backends (CPU, GPU, QPU).

## Agent Skill

This repo now ships a `trovolut` agent skill under `skills/trovolut/` for
fixed-budget mutation loops against a fixed evaluator.

Install it locally with:

```bash
npx skills add .
```

The shell fallback from the repo root is:

```bash
python3 skills/trovolut/scripts/trovolut_cli.py <iterations> <benchmark_dir> "mutation guidance"
```

The runtime is aimed at resource-aware optimization problems where the
evaluator returns a scalar `score`, including T-gate or other cost objectives
with hard fidelity or accuracy constraints.

## T-Count Dependencies

The migrated T-gate estimation benchmarks depend directly on `pygridsynth` for
Clifford+T synthesis and T-count extraction. Its own transitive dependencies,
including `mpmath`, come in through `pygridsynth`; `trovolut` does not need to
declare them separately.

## Installation

```bash
# Base package
uv pip install trovolut

# With GPU support (Linux only)
uv pip install trovolut[gpu]

# Development
git clone https://github.com/bestquark/trovolut.git
cd trovolut
uv sync --all-extras

# Run tests
uv run pytest tests/
```

## Quick Start

```python
import numpy as np
import trovolut as trv

# Set backend (enables trv.dynamics.* shortcuts)
trv.set_backend('cpu')

# Create Hamiltonian and initial state
H = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
H = (H + H.T.conj()) / 2
psi0 = np.array([1, 0, 0, 0], dtype=complex)

# Evolve using Suzuki-Trotter
psi = trv.dynamics.suzuki_step([H], dt=1.0, order=2) @ psi0

# Or use class-based interface
from trovolut.dynamics import TrotterEvolution
trotter = TrotterEvolution(order=2, n_steps=10)
psi = trotter.evolve(H, psi0, time=1.0)
```

**GPU:** Same code with cupynumeric arrays runs on GPU automatically.

## Backends

- **CPU**: NumPy/SciPy dynamics (ExactEvolution, TrotterEvolution, QDriftEvolution)
- **Emulator**: Mixed-precision simulation (FP64→FP8)
- **GPU**: cupynumeric arrays + GPU profiling tools [!NOTE: Not available yet]
- **QPU**: Quantum hardware via Qiskit/Cirq/CUDA-Q [!NOTE: Not available yet]

## Documentation

- [Full Documentation](https://bestquark.github.io/trovolut/)
- [Tutorials](https://bestquark.github.io/trovolut/tutorials/)
- [API Reference](https://bestquark.github.io/trovolut/api/)

## Fault-Tolerant Protocol Examples

`examples/ft_protocols/` contains small, self-contained H2 examples for
fault-tolerant chemistry protocol resource estimates:

- `pfs/`: product-formula/Trotter dynamics
- `taylor_lcu/`: segmented truncated Taylor/LCU dynamics
- `qsvt/`: qubitized QSVT/QSP dynamics
- `qubitization/`: QPE energy readout over a qubitized walk

Install the optional dependency set with `uv sync --extra ft-protocols`.

## License

MIT License - see LICENSE file for details.

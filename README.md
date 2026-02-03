# Trovolut

> [![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bestquark.github.io/trovolut/) &nbsp;&nbsp;&nbsp; 🔑 **Password**: `gordonbell2026`

Evolutionary Trotterization for quantum simulation with modular backends (CPU, GPU, QPU).

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
```

## Quick Start

```python
import numpy as np
from trovolut.backends.cpu.dynamics import TrotterEvolution

# Create Hamiltonian and initial state
H = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
H = (H + H.T.conj()) / 2
psi0 = np.array([1, 0, 0, 0], dtype=complex)

# Evolve
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

## License

MIT License - see LICENSE file for details.

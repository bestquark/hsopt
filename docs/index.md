# Trovolut

Evolutionary Trotterization for quantum simulation with modular computational backends.

## Installation

```bash
# Base package (CPU backend)
uv pip install trovolut

# With optional backends
uv pip install trovolut[gpu]         # NVIDIA GPU (Linux only)
uv pip install trovolut[qpu]         # Quantum hardware (Qiskit, Cirq, CUDA-Q)
uv pip install trovolut[gpu-emulate] # Mixed-precision emulation
uv pip install trovolut[all]         # All backends
```

## Quick Start

```python
import trovolut as trv
from trovolut import qm

# Check available backends
print(trv.backends.list_available())

# Create Hamiltonian
H = qm.models.hubbard_2d(Lx=3, Ly=3, t_hop=1.0, U_onsite=4.0)

# Get backend dynamics
dynamics = trv.backends.get_dynamics('cpu')
```

## Backends

| Backend | Platform | Dependencies | Use Case |
|---------|----------|--------------|----------|
| `cpu` | All | NumPy/SciPy | Standard dynamics (always available) |
| `gpu` | Linux | CuPy, cuQuantum | NVIDIA GPU acceleration |
| `qpu` | All | Qiskit, Cirq | Quantum hardware/simulators |
| `gpu-emulate` | All | None | Mixed-precision studies (FP64→FP8) |

## Development

```bash
git clone https://github.com/bestquark/trovolut.git
cd trovolut
uv sync --all-extras

# Serve documentation locally
uv run --group docs mkdocs serve
```

## Documentation

- [API Reference](api/index.md)
- [Tutorials](tutorials/index.md)

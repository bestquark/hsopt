# Trovolut Backends

This module provides a plugin-style system for different computational backends in Trovolut.

## Available Backends

### CPU (always available)
Standard NumPy/SciPy implementation. No additional dependencies required.

```python
import trovolut as trv

# CPU backend is always available
cpu_dynamics = trv.backends.get_dynamics('cpu')
```

### GPU (Linux only)
NVIDIA GPU acceleration using CuPy and cuQuantum.

**Requirements:**
- Linux operating system
- NVIDIA GPU with CUDA support
- Install with: `uv pip install trovolut[gpu]`

```python
import trovolut as trv

# Check if GPU backend is available
if trv.backends.is_available('gpu'):
    gpu_dynamics = trv.backends.get_dynamics('gpu')
    gpu_evaluator = trv.backends.get_evaluator('gpu')
    gpu_metrics = trv.backends.get_metrics('gpu')
else:
    print("GPU backend not available")
```

### QPU (cross-platform)
Quantum hardware access via Qiskit, Cirq, and CUDA Quantum.

Works on all platforms with Qiskit and Cirq. CUDA Quantum (cudaq) provides additional
GPU-accelerated simulation capabilities on Linux systems.

**Requirements:**
- Install with: `uv pip install trovolut[qpu]`
- Core: Qiskit, Cirq (cross-platform)
- Optional: CUDA Quantum (Linux-only, for GPU acceleration)

```python
import trovolut as trv

if trv.backends.is_available('qpu'):
    qpu_dynamics = trv.backends.get_dynamics('qpu')
```

### GPU Emulator
CPU-based emulation for studying quantized exponentials and mixed-precision Trotter methods.

This backend simulates GPU low-precision arithmetic without requiring actual GPU hardware.
It's useful for:
- Studying numerical precision effects on quantum dynamics
- Testing mixed-precision algorithms (FP64, FP32, FP16, BF16, FP8, INT8)
- Developing quantization-aware algorithms

**Requirements:**
- Install with: `uv pip install trovolut[gpu-emulate]`
- No additional dependencies (uses NumPy)

```python
import trovolut as trv
from trovolut.backends.gpu_emulator.mixed_fps.mp_trotter import Precision

# Get emulator dynamics
dynamics = trv.backends.get_dynamics('gpu-emulate')

# Standard Suzuki-Trotter
psi_final = dynamics.evolve(H_terms, psi0, time=1.0, order=2, steps=100)

# Mixed-precision Trotter
from trovolut.backends.emulator.dynamics import create_exp_terms_from_hamiltonian
exp_terms = create_exp_terms_from_hamiltonian(
    H_terms,
    precisions=[Precision.FP32, Precision.FP16, Precision.FP8_E4M3],
)
psi_final, metrics = dynamics.evolve_mixed_precision(
    exp_terms, psi0, time=1.0, steps=100
)
```

## Usage Examples

### Listing Available Backends

```python
import trovolut as trv

# List only available backends
print("Available backends:", trv.backends.list_available())
# Example output: ['cpu', 'gpu']

# List all backends with status
all_backends = trv.backends.list_all()
for name, info in all_backends.items():
    status = "✓" if info["available"] else "✗"
    print(f"{status} {name}: {info['description']}")
```

### Using a Backend

```python
import trovolut as trv
import numpy as np

# Get backend components
dynamics = trv.backends.get_dynamics('cpu')
evaluator = trv.backends.get_evaluator('cpu')
metrics = trv.backends.get_metrics('cpu')

# Use dynamics for quantum evolution
H = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
H = H + H.T.conj()  # Make Hermitian
psi0 = np.array([1, 0, 0, 0], dtype=complex)

# psi_final = dynamics.evolve(H, psi0, time=1.0)
```

### Error Handling

The backend system provides helpful error messages when dependencies are missing:

```python
import trovolut as trv

try:
    gpu_dynamics = trv.backends.get_dynamics('gpu')
except trv.backends.BackendNotAvailableError as e:
    print(e)
    # Output: Backend 'gpu' is not available.
    #         Missing packages: cupy, cuquantum, pynvml
    #         Install with: uv pip install trovolut[gpu]
```

## Installation Options

```bash
# Install with all backends
uv pip install trovolut[all]

# Install with specific backends
uv pip install trovolut[gpu]
uv pip install trovolut[qpu]
uv pip install trovolut[gpu-emulate]

# Install multiple backends
uv pip install "trovolut[gpu,gpu-emulate]"

# CPU backend is included by default
uv pip install trovolut
```

## Backend Architecture

Each backend implements three main components:

1. **Dynamics** (`get_dynamics()`): Time evolution of quantum systems
   - Implements various matrix exponential methods
   - Backend-specific optimizations

2. **Evaluator** (`get_evaluator()`): Performance evaluation
   - Measures execution time, memory usage, fidelity
   - Used by OpenEvolve for optimization

3. **Metrics** (`get_metrics()`): Hardware monitoring
   - Collects runtime performance data
   - Backend-specific metrics (GPU utilization, circuit depth, etc.)

## Platform Requirements

| Backend       | Linux | macOS | Windows | Hardware | Notes |
|---------------|-------|-------|---------|----------|-------|
| CPU           | ✓     | ✓     | ✓       | Any      | Always available |
| GPU           | ✓     | ✗     | ✗       | NVIDIA   | CUDA required |
| QPU           | ✓     | ✓     | ✓       | Optional | cudaq Linux-only |
| GPU Emulator  | ✓     | ✓     | ✓       | Any      | NumPy-based |

## Adding Custom Backends

To add a new backend, implement the following structure:

```
src/trovolut/backends/mybackend/
├── __init__.py          # is_available(), get_dynamics(), get_evaluator(), get_metrics()
├── dynamics.py          # MyBackendDynamics(BaseDynamics)
├── evaluator.py         # MyBackendEvaluator(BaseEvaluator)
└── metrics.py           # MyBackendMetrics(BaseMetrics)
```

Register it in `backends/__init__.py`:

```python
_backends = {
    # ...
    "mybackend": {
        "module": "trovolut.backends.mybackend",
        "required_packages": ["mypackage"],
        "description": "My custom backend",
        "platform": "all",  # or "linux", "darwin", etc.
    },
}
```

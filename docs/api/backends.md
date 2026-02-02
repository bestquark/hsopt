# `trovolut.backends`

Backend registry and implementations.

## API

::: trovolut.backends

## Usage

```python
import trovolut as trv

# List available backends
trv.backends.list_available()

# Check availability
trv.backends.is_available('gpu')

# Import dynamics methods directly
from trovolut.backends.cpu.dynamics import ExactEvolution, TrotterEvolution

# Import hardware-specific tools
from trovolut.backends.gpu.evaluator import GPUEvaluator
```

## Backends

### CPU
::: trovolut.backends.cpu

Provides dynamics methods (ExactEvolution, TrotterEvolution, QDriftEvolution) that work with any array backend.

### GPU  
::: trovolut.backends.gpu

Install: `uv pip install trovolut[gpu]` (Linux + NVIDIA only)

Provides GPU-specific profiling tools (GPUEvaluator). For dynamics on GPU, use CPU backend methods with CuPy arrays.

### QPU
::: trovolut.backends.qpu

Install: `uv pip install trovolut[qpu]`

Provides QPU-specific tools for circuit compilation and hardware execution.

### GPU Mixed-Precision Emulator
::: trovolut.backends.gpu_mp_emulator

Install: `uv pip install trovolut[gpu-emulate]`

Provides mixed-precision emulation tools for studying quantization effects (FP64 → FP32 → FP16 → FP8).

## Platform Support

| Backend | Linux | macOS | Windows |
|---------|-------|-------|---------|
| CPU | ✓ | ✓ | ✓ |
| GPU | ✓ | ✗ | ✗ |
| QPU | ✓ | ✓ | ✓ |
| GPU MP Emulator | ✓ | ✓ | ✓ |

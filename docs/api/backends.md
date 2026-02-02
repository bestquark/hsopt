# `trovolut.backends`

Backend registry and implementations.

## API

::: trovolut.backends

## Usage

```python
import trovolut as trv

# List available
trv.backends.list_available()

# Check availability
trv.backends.is_available('gpu')

# Get backend
dynamics = trv.backends.get_dynamics('cpu')
evaluator = trv.backends.get_evaluator('cpu')
metrics = trv.backends.get_metrics('cpu')
```

## Backends

### CPU
::: trovolut.backends.cpu

### GPU  
::: trovolut.backends.gpu

Install: `uv pip install trovolut[gpu]` (Linux + NVIDIA only)

### QPU
::: trovolut.backends.qpu

Install: `uv pip install trovolut[qpu]`

### GPU Emulator
::: trovolut.backends.emulator

Install: `uv pip install trovolut[gpu-emulate]`

## Platform Support

| Backend | Linux | macOS | Windows |
|---------|-------|-------|---------|
| CPU | ✓ | ✓ | ✓ |
| GPU | ✓ | ✗ | ✗ |
| QPU | ✓ | ✓ | ✓ |
| GPU Emulator | ✓ | ✓ | ✓ |

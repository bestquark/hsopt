"""
GPU Emulator backend modules.

This module contains:
- dynamics: Suzuki-Trotter methods
- evaluator: OpenEvolve evaluator builder
- hamsim: Hamiltonian simulation utilities
- mp_evals: Mixed-precision evaluation
- constraints: Constraint definitions for OpenEvolve
- mixed_fps: Mixed-precision Trotter framework

These modules are used by the gpu-emulate backend for CPU-based
emulation of quantized exponentials and mixed-precision dynamics.
"""

# Make key modules available
from . import dynamics
from . import evaluator
from . import hamsim
from . import mp_evals
from . import constraints
from . import mixed_fps

__all__ = [
    "dynamics",
    "evaluator",
    "hamsim",
    "mp_evals",
    "constraints",
    "mixed_fps",
]

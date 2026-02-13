# Tutorials

## Overview

- [Creating Hamiltonians](dmds_hamiltonian.ipynb) - Built-in models and electronic structure
- [Trotter Dynamics](trotter_dynamics.ipynb) - Suzuki-Trotter algorithms
- [Filemaker](filemaker.ipynb) - Generate evaluator files for OpenEvolve
- [Evaluations](evaluations.ipynb) - Performance metrics and optimization
- [GPU Optimization](mp_optimization.ipynb) - Mixed-precision optimization on GPU hardware
- [Roofline, Counters & Calibration](roofline_calibration.ipynb) - Roofline cost model, NVML hardware counters, and emulator calibration

## Backend Requirements

| Tutorial | Backend | Install Command |
|----------|---------|-----------------|
| Hamiltonians, Filemaker, Evaluations | CPU | `uv pip install trovolut` |
| Trotter Dynamics | CPU | `uv pip install trovolut` |
| GPU Optimization | GPU Emulator | `uv pip install trovolut[gpu-emulate]` |
| Roofline, Counters & Calibration | GPU Emulator (+ optional pynvml) | `uv pip install trovolut[gpu-emulate]` |

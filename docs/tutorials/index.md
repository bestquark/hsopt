# Tutorials

## Overview

- [Creating Hamiltonians](dmds_hamiltonian.ipynb) - Built-in models and electronic structure
- [Trotter Dynamics](trotter_dynamics.ipynb) - Suzuki-Trotter algorithms
- [Filemaker](filemaker.ipynb) - Generate evaluator files for OpenEvolve
- [Evaluations](evaluations.ipynb) - Performance metrics and optimization
- [Mixed Precision](mp_trotter.ipynb) - Quantized exponentials (FP64→FP8)
- [Precision Optimization](mp_evolve.ipynb) - Evolutionary precision allocation

## Backend Requirements

| Tutorial | Backend | Install Command |
|----------|---------|-----------------|
| Hamiltonians, Filemaker, Evaluations | CPU | `uv pip install trovolut` |
| Trotter Dynamics | CPU | `uv pip install trovolut` |
| Mixed Precision, Optimization | GPU Emulator | `uv pip install trovolut[gpu-emulate]` |

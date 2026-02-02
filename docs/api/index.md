# API Overview

Welcome to the Trovolut API reference. Use the navigation tree to explore each module.

## Core Modules

- [`trovolut.core`](core.md) - Base classes and types for all backends
- [`trovolut.backends`](backends.md) - Backend registry and implementations
- [`trovolut.qm`](qm.md) - Quantum mechanics (Hamiltonians, operators)

## Workflow Modules

- [`trovolut.filemaker`](filemaker.md) - Workflow and file generation
- [`trovolut.utils`](utils.md) - Utility functions

## Legacy Modules (deprecated paths)

- [`trovolut.eval`](eval.md) - Legacy evaluation (moved to `backends.gpu_emulator`)
- [`trovolut.mixed_fps`](mixed_fps.md) - Legacy mixed-precision (moved to `backends.gpu_emulator.mixed_fps`)

!!! note "Module Organization"
    The codebase has been reorganized into a backend-based architecture:
    
    - **`trovolut.qm`**: Backend-agnostic Hamiltonian construction
    - **`trovolut.backends`**: Backend-specific dynamics implementations
    - **`trovolut.core`**: Shared base classes and types
    
    See the [Backend Guide](backends.md) for details.

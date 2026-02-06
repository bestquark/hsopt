# `trovolut.core`

Base classes and types for backend implementations.

## Module Dependency Graph

```mermaid
flowchart LR
    subgraph core [trovolut.core]
        BD[base_dynamics]
        BM[base_metrics]
        BH[base_hardware]
        BC[base_constraints]
        T[types]
    end

    subgraph backends [Backends]
        CPU[cpu]
        GPU[gpu]
        MPE[gpu_mp_emulator]
        QPU[qpu]
    end

    BD --> CPU & GPU & MPE & QPU
    BM --> GPU & MPE
    BH --> MPE
    BC --> CPU & GPU & MPE
```

## Base Classes

::: trovolut.core.BaseDynamics

::: trovolut.core.BaseMetrics

## Product Formulas

::: trovolut.core.ProductFormula

::: trovolut.core.LieTrotter

::: trovolut.core.SuzukiTrotter

::: trovolut.core.SuzukiHighOrder

## Hardware

::: trovolut.core.BaseHardware

::: trovolut.core.HardwareCluster

## Constraint System

::: trovolut.core.Constraint

::: trovolut.core.Group

## Types

::: trovolut.core.types

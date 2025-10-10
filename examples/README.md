# GPU4PySCF Examples

This directory contains examples demonstrating the usage of GPU-accelerated quantum chemistry calculations using `gpu4pyscf`.

## Overview

The `gpu4pyscf_example.py` script demonstrates:

1. **Basic HF (Hartree-Fock) calculations** - Ground state energy for H2
2. **DFT calculations with different functionals** - H2O with LDA, PBE, and B3LYP
3. **MP2 calculations** - Møller-Plesset 2nd order perturbation theory
4. **QMMM calculations with point charges** - QM/MM with external electrostatic potential
5. **Unified interface** - Single function to access all methods

## Requirements

To run these examples, you need to have `gpu4pyscf` installed:

```bash
pip install gpu4pyscf
```

Note: `gpu4pyscf` requires a CUDA-capable GPU and appropriate CUDA drivers.

## Running the Examples

From the repository root:

```bash
python examples/gpu4pyscf_example.py
```

The script will check if `gpu4pyscf` is available and run all examples if it is installed. If not, it will display an informative message.

## API Usage

### Basic Usage

```python
from src.qm import create_molecule, compute_hf_ground_state

# Create a molecule
mol = create_molecule(
    atoms="H 0 0 0; H 0 0 0.74",
    basis="6-31g"
)

# Compute ground state
results = compute_hf_ground_state(mol)
print(f"Energy: {results['energy']:.6f} Hartree")
```

### QMMM with External Charges

```python
import numpy as np
from src.qm import create_molecule, compute_hf_ground_state

mol = create_molecule(atoms="H 0 0 0; H 0 0 0.74", basis="6-31g")

# Define MM point charges: [x, y, z, charge]
mm_charges = np.array([
    [2.0, 0.0, 0.0, -0.5],
    [-2.0, 0.0, 0.0, -0.5],
])

# Compute with QMMM
results = compute_hf_ground_state(mol, external_potential=mm_charges)
```

### DFT Calculations

```python
from src.qm import create_molecule, compute_dft_ground_state

mol = create_molecule(atoms="O 0 0 0; H 0 1 0; H 0 0 1", basis="6-31g")

# Use different functionals
results = compute_dft_ground_state(mol, xc="B3LYP")
```

### Unified Interface

```python
from src.qm import create_molecule, compute_ground_state

mol = create_molecule(atoms="H 0 0 0; H 0 0 0.74", basis="6-31g")

# All methods through one interface
hf_results = compute_ground_state(mol, method="HF")
dft_results = compute_ground_state(mol, method="DFT", xc="B3LYP")
mp2_results = compute_ground_state(mol, method="MP2")
```

## Available Methods

- **HF** (`compute_hf_ground_state`): Hartree-Fock method
- **DFT** (`compute_dft_ground_state`): Density Functional Theory with various functionals
- **MP2** (`compute_mp2_ground_state`): Møller-Plesset 2nd order perturbation theory

All methods support QMMM external potentials through the `external_potential` parameter.

## QMMM External Potentials

Two types of external potentials are supported:

1. **Point charges** (most common for QMMM):
   ```python
   mm_charges = np.array([[x, y, z, charge], ...])
   results = compute_hf_ground_state(mol, external_potential=mm_charges)
   ```

2. **Custom potential functions**:
   ```python
   def custom_potential(coords):
       # coords shape: (n_points, 3)
       return potential_values  # shape: (n_points,)
   
   results = compute_hf_ground_state(mol, external_potential=custom_potential)
   ```

## Notes

- All coordinates are in Angstroms by default
- Energies are returned in Hartree atomic units
- The module gracefully handles cases where `gpu4pyscf` is not installed

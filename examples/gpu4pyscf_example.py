#!/usr/bin/env python
"""
Example demonstrating gpu4pyscf usage for ground state calculations.

This script shows how to compute ground states using HF, DFT, and MP2 methods
with optional QMMM external potentials.
"""

import numpy as np

# Import the new gpu4pyscf functionality
from src.qm import (
    create_molecule,
    compute_ground_state,
    compute_hf_ground_state,
    compute_dft_ground_state,
    compute_mp2_ground_state,
    GPU4PYSCF_AVAILABLE,
)


def example_basic_hf():
    """Example: Basic Hartree-Fock calculation for H2."""
    print("\n" + "=" * 60)
    print("Example 1: Basic HF calculation for H2")
    print("=" * 60)
    
    if not GPU4PYSCF_AVAILABLE:
        print("gpu4pyscf not available. Skipping example.")
        return
    
    # Create H2 molecule
    mol = create_molecule(
        atoms="H 0 0 0; H 0 0 0.74",
        basis="6-31g",
    )
    
    # Compute HF ground state
    results = compute_hf_ground_state(mol)
    
    print(f"HF Energy: {results['energy']:.6f} Hartree")
    print(f"Converged: {results['converged']}")
    print(f"Number of MOs: {results['mo_coeff'].shape}")


def example_dft_with_different_functionals():
    """Example: DFT calculations with different functionals."""
    print("\n" + "=" * 60)
    print("Example 2: DFT with different functionals for H2O")
    print("=" * 60)
    
    if not GPU4PYSCF_AVAILABLE:
        print("gpu4pyscf not available. Skipping example.")
        return
    
    # Create H2O molecule
    mol = create_molecule(
        atoms="""
        O 0.0000 0.0000 0.1173
        H 0.0000 0.7572 -0.4692
        H 0.0000 -0.7572 -0.4692
        """,
        basis="6-31g",
    )
    
    functionals = ["LDA", "PBE", "B3LYP"]
    
    for xc in functionals:
        results = compute_dft_ground_state(mol, xc=xc)
        print(f"{xc:8s} Energy: {results['energy']:.6f} Hartree")


def example_mp2():
    """Example: MP2 calculation."""
    print("\n" + "=" * 60)
    print("Example 3: MP2 calculation for H2")
    print("=" * 60)
    
    if not GPU4PYSCF_AVAILABLE:
        print("gpu4pyscf not available. Skipping example.")
        return
    
    # Create H2 molecule
    mol = create_molecule(
        atoms="H 0 0 0; H 0 0 0.74",
        basis="6-31g",
    )
    
    # Compute MP2 ground state
    results = compute_mp2_ground_state(mol)
    
    print(f"HF Energy:          {results['hf_energy']:.6f} Hartree")
    print(f"Correlation Energy: {results['correlation_energy']:.6f} Hartree")
    print(f"MP2 Total Energy:   {results['mp2_energy']:.6f} Hartree")


def example_qmmm_with_point_charges():
    """Example: QMMM calculation with point charges."""
    print("\n" + "=" * 60)
    print("Example 4: QMMM HF calculation with external point charges")
    print("=" * 60)
    
    if not GPU4PYSCF_AVAILABLE:
        print("gpu4pyscf not available. Skipping example.")
        return
    
    # Create H2 molecule (QM region)
    mol = create_molecule(
        atoms="H 0 0 0; H 0 0 0.74",
        basis="6-31g",
    )
    
    # Define MM point charges: [x, y, z, charge]
    # Place two negative charges near the molecule
    mm_charges = np.array([
        [2.0, 0.0, 0.0, -0.5],  # -0.5e at (2.0, 0, 0) Angstrom
        [-2.0, 0.0, 0.0, -0.5],  # -0.5e at (-2.0, 0, 0) Angstrom
    ])
    
    # Compute with and without external charges
    results_no_mm = compute_hf_ground_state(mol)
    results_with_mm = compute_hf_ground_state(mol, external_potential=mm_charges)
    
    print("Without MM charges:")
    print(f"  Energy: {results_no_mm['energy']:.6f} Hartree")
    
    print("\nWith MM charges:")
    print(f"  Energy: {results_with_mm['energy']:.6f} Hartree")
    
    energy_diff = results_with_mm['energy'] - results_no_mm['energy']
    print(f"\nEnergy difference: {energy_diff:.6f} Hartree")
    print(f"                   {energy_diff * 627.509:.3f} kcal/mol")


def example_unified_interface():
    """Example: Using the unified compute_ground_state interface."""
    print("\n" + "=" * 60)
    print("Example 5: Unified interface for different methods")
    print("=" * 60)
    
    if not GPU4PYSCF_AVAILABLE:
        print("gpu4pyscf not available. Skipping example.")
        return
    
    mol = create_molecule(
        atoms="H 0 0 0; H 0 0 0.74",
        basis="6-31g",
    )
    
    methods = ["HF", "DFT", "MP2"]
    
    for method in methods:
        if method == "DFT":
            results = compute_ground_state(mol, method=method, xc="B3LYP")
        else:
            results = compute_ground_state(mol, method=method)
        
        energy_key = "mp2_energy" if method == "MP2" else "energy"
        print(f"{method:4s} Energy: {results[energy_key]:.6f} Hartree")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# GPU4PySCF Ground State Calculation Examples")
    print("#" * 60)
    
    if not GPU4PYSCF_AVAILABLE:
        print("\nWARNING: gpu4pyscf is not available!")
        print("These examples require gpu4pyscf to be installed.")
        print("Install with: pip install gpu4pyscf")
        print("\nThe module will still import successfully and raise")
        print("appropriate errors when methods are called.")
        return
    
    # Run examples
    example_basic_hf()
    example_dft_with_different_functionals()
    example_mp2()
    example_qmmm_with_point_charges()
    example_unified_interface()
    
    print("\n" + "#" * 60)
    print("# All examples completed successfully!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()

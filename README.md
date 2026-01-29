# Welcome to the `trovolut` repository

> [![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bestquark.github.io/trovolut/) &nbsp;&nbsp;&nbsp; 🔑 **Password**: `gordonbell2026`

The `trovolut` package is a software for evolutionary Trotterization in quantum mechanics. 
Here, you will find code for automatically evolving simulations of quantum systems using mixed-precision arithmetic, embedded in a classical environment. 

## Installation

Make sure you have an API key for OpenEvolve and have it set in your environment:
```bash
export OPENAI_API_KEY="your_key_here"  
```

If you are only running only the QM simulations, clone the repository and sync the environment:

```bash
git clone https://github.com/bestquark/trovolut.git
cd trovolut
uv sync
uv run pre-commit install
```

Configure the number of GPUs to use with:
```bash
source ./config_gpus.sh
echo "$CUDA_VISIBLE_DEVICES" # Check the GPUs assigned
```

## Usage

### For protein-ligand binding free energy calculations:

An example is provided in the [README](examples/qmmm/biocomplex/README.md) of the `examples/qmmm/biocomplex/` directory.

### GPU-Accelerated Quantum Chemistry

The repository now includes GPU-accelerated ground state calculations using `gpu4pyscf`. This enables fast QM calculations with optional QMMM external potentials:

```python
from src.qm import create_molecule, compute_ground_state

# Create molecule
mol = create_molecule(atoms="H 0 0 0; H 0 0 0.74", basis="6-31g")

# Compute ground state with HF, DFT, or MP2
results = compute_ground_state(mol, method="HF")
print(f"Energy: {results['energy']:.6f} Hartree")
```


### Code Evolution

Code evolution takes place in the `code_evolution/` directory.

To evolve a program, run:
```bash
uv run python -m code_evolution.openevolve-run code_evolution/initial_program.py code_evolution/evaluator.py --config code_evolution/config.yaml --iterations 5
```
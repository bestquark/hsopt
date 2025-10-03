# Evolutionary low-precision QM/MM
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://bestquark.github.io/evolutionary-qmmm/)  
<sub>🔑 Password: `gordonbell2026`</sub>


This repository contains code for simulating quantum systems using mixed-precision arithmetic, embedded in a classical environment.
The main idea is to use this technique for speeding up free-energy calculations (e.g. FEP) while maintaining accuracy.

<p style="text-align:center;">
<img src="docs/assets/traj.gif" alt="QM/MM Trajectory" width="40%">
</p>


## Usage

Make sure you have an API key for OpenEvolve and have it set in your environment:
```bash
export OPENAI_API_KEY="your_key_here"  
```

If you are only running only the QM simulations, clone the repository and sync the environment:

```bash
git clone https://github.com/bestquark/evolutionary-qmmm.git
cd evolutionary-qmmm
uv sync
```

If you are running MD simulations, you need conda to install 
```bash
conda install openmmforcefields
```


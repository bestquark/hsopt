# Mixed-precision QM/MM

This repository contains code for simulating quantum systems using mixed-precision arithmetic, embedded in a classical environment.
The main idea is to use this technique for speeding up free-energy calculations (e.g. FEP) while maintaining accuracy.

## Usage

Make sure you have an API key for OpenEvolve and have it set in your environment:
```bash
export OPENAI_API_KEY="your_key_here"  
```

Then, clone the repository and sync the environment:

```bash
git clone https://github.com/bestquark/gordonEPR.git
cd gordonEPR
uv sync
```


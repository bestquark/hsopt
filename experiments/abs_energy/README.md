# Absolute-energy benchmark

This family is the 82-instance HamLib electronic-structure selection that fits
the canonical dense evaluator through 12 qubits:

| Qubits | Instances |
|---:|---:|
| 4 | 20 |
| 6 | 2 |
| 8 | 24 |
| 10 | 11 |
| 12 | 25 |
| **Total** | **82** |

`inputs/instances.csv` records exactly those 82 selected instances and
`inputs/hamlib_zip_sha256.txt` pins the upstream HamLib archives. Running
`hsopt-campaign prepare-hamlib --max-qubits 12` materializes the 82 compact
Hamiltonian JSON files under an external input directory. Campaign
baseline preparation computes one exact reference per Hamiltonian and two
sealed calibrations: the rigorous Eq. (19) upper-bound `loose` beta and the
exact-energy-calibrated `tight` beta. `experiments/generate_initial_pf.py`
writes `initial_pf_loose.py` or `initial_pf_tight.py`. Campaign initialization
selects exactly one, copies only that policy into each mutation lane, and
reuses the same exact reference.

The 82 directories beside `inputs/` are the visible production selection. Each
contains one deterministic `instance.json` and both generated initial policies.
The definition records its HamLib archive and dataset, archive checksum,
mapped qubit/electron/term counts, one-norm, phase-encoding policy, and dense-exact fit
metadata. The large coefficient payload is not duplicated in Git.

Verify or intentionally regenerate this directory view with:

```bash
uv run hsopt-layout --check
uv run hsopt-layout --write
```

Compact executable evaluator fixtures live under `cli/tests/fixtures/`.

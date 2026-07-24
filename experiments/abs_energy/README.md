# Absolute-energy benchmark

This family is the 91-instance HamLib electronic-structure selection that fits
the canonical dense evaluator through 14 qubits:

| Qubits | Instances |
|---:|---:|
| 4 | 20 |
| 6 | 2 |
| 8 | 24 |
| 10 | 11 |
| 12 | 25 |
| 14 | 9 |
| **Total** | **91** |

`inputs/instances.csv` records exactly those 91 selected instances and
`inputs/hamlib_zip_sha256.txt` pins the upstream HamLib archives. Running
`hsopt-campaign prepare-hamlib --max-qubits 14` materializes the 91 compact
Hamiltonian JSON files under an external input directory. Campaign
baseline preparation first computes one sealed Eq. (19) bound $\beta(H)$ and
exact reference per Hamiltonian on CPU nodes. It then runs
`experiments/generate_initial_pf.py` once per Hamiltonian, writing the
immutable policy both beside its `instance.json` and into the external
campaign data. Campaign initialization creates five mutation lanes that reuse
that same exact reference.

The 91 directories beside `inputs/` are the visible production selection. Each
contains one deterministic `instance.json` and one generated `initial_pf.py`.
The definition records its HamLib archive and dataset, archive checksum,
mapped qubit/electron/term counts, one-norm, target time, and dense-exact fit
metadata. The large coefficient payload is not duplicated in Git.

Verify or intentionally regenerate this directory view with:

```bash
uv run hsopt-layout --check
uv run hsopt-layout --write
```

Compact executable evaluator fixtures live under `tests/fixtures/`.

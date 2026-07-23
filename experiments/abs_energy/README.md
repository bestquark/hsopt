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
initialization then runs `experiments/generate_initial_pf.py` once for each
Hamiltonian and creates five mutation lanes that share its exact reference.

`hamlib_h2_r0_7/` is only a compact local evaluator check; it is not an extra
production family.

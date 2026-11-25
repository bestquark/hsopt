# Current manuscript HamLib diagnostics

- hamlib_molecular_dataset_active_spaces.json supplies Figure 5's 663-instance
  census and the 115 training/78 testing census memberships.
- baseline_initialization_vs_qubits.csv and its metadata preserve the historical
  121-Hamiltonian bound-based diagnostic at fixed physical time. They no longer
  supply Figure 6. The current figure reads the actual 115 training baselines
  from the frozen elote115-paper iteration-zero trajectories.
- schematic_coefficients.npz contains the four exact nonidentity coefficient
  samples for Figure 1's illustration inset. Archives and arrays are SHA256
  pinned; coefficients at or below 1e-15 are omitted, matching the original
  inset recipe. These illustrative 18–30-qubit examples are not training cases.

All current figure processing is under figures/. The original one-time coefficient
extractor is preserved at the export commit and hash in experiments/manuscript.json.
Rebuilding Figure 1 uses the frozen coefficient arrays without another download.

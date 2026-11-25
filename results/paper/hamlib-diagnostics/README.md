# Current manuscript HamLib diagnostics

- hamlib_molecular_dataset_active_spaces.json supplies Figure 5's 663-instance
  census and the 115 training/78 testing census memberships.
- baseline_initialization_vs_qubits.csv and its metadata supply Figure 6's
  separate 121-Hamiltonian bound-based initialization diagnostic. Its fixed
  physical-time construction must not be relabeled as the normalized-time
  grouped tight baseline used to train ELOTE.
- schematic_coefficients.npz contains the four exact nonidentity coefficient
  samples for Figure 1's illustration inset. Archives and arrays are SHA256
  pinned; coefficients at or below 1e-15 are omitted, matching the original
  inset recipe. These illustrative 18–30-qubit examples are not training cases.

All current figure processing is under figures/. The original one-time coefficient
extractor is preserved at the export commit and hash in experiments/manuscript.json.
Rebuilding Figure 1 uses the frozen coefficient arrays without another download.

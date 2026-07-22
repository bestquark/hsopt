# Reaction paper results

This directory intentionally contains no checkpoint or partial campaign data.
The archived Omni iteration-13/14 checkpoint is provenance evidence, not a
finished paper result.

After the cleaned fixed campaign is rerun through baseline iteration 0 and all
50 proposals for each of the 20 lanes, use
`../runner/export_results.py` to write the small, normalized, checksummed paper
tables here. Keep the full mutable campaign workspace and the 42 Hamiltonian
cache files external to Git.

The normalized candidate table includes the PF parameters and per-species
exact/PF energies required for report arithmetic. After strict export, run the
shared renderer to create:

```text
final/
  reactions/<reaction_key>/
    extended_log.pdf
    summary.pdf
  report_manifest.json
  SHA256SUMS
```

The renderer requires all 1,020 candidates and both contracts' contiguous
iterations. `extended_log.pdf` records every mutation, acceptance decision,
and exact reaction/species error; `summary.pdf` compares each lane's initial
and final incumbent and labels the pooled paper selections separately.

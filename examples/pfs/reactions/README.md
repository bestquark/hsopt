# Ten-reaction product-formula benchmark

This directory contains the fixed scientific evaluator and the small editable
policy used by the reaction autoresearch campaign.

- `evaluate.py` loads a cache for an explicit `MoleculeSpec`, constructs the
  compiled one-step product-formula unitary in the full Hilbert space, matches
  its ground phase by eigenvector overlap, and counts the selected PyGridSynth
  `T` gates exactly.
- `simple_pf.py` is the mutation surface. Campaign lanes copy it before making
  changes.

The ordered species definitions and stoichiometry for all ten reactions live
only in `experiments/reactions10/benchmark.py`. The fixed reaction target is
exactly `1 / 627.5094740631` Hartree (one kcal/mol). Problem caches and campaign
histories are external inputs; they are not written into this example folder.

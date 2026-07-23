# Reaction-energy benchmark

This family contains the ten fixed reaction definitions used by the production
campaign:

1. HCN → HNC
2. HCO⁺ → HOC⁺
3. HNO → NOH
4. H₂CO → HCOH
5. acetylene → vinylidene
6. H₂ + Cl₂ → 2 HCl
7. N₂ + 3 H₂ → 2 NH₃
8. CO + H₂O → CO₂ + H₂
9. H₂ + F₂ → 2 HF
10. CO + 3 H₂ → CH₄ + H₂O

`inputs/seed_manifest.json` is the authoritative reaction ordering and
stoichiometry registry. `inputs/` also contains the validated per-contract
seed budgets and checksums for the 21 external species-cache pairs.
`hsopt-campaign prepare-reactions` converts those checksummed caches into ten
external reaction directories consumed by the fixed evaluator in
`skills/hsopt/scripts/hsopt/compiled_unitary.py`.

The ten directories beside `inputs/` make that production set visible on
GitHub. Each `definition.json` records the signed reaction terms and the
metadata/coefficients SHA-256 checksums for every external species
Hamiltonian. Run `uv run hsopt-layout --check` to verify that all ten match the
compact registry, or `uv run hsopt-layout --write` after an intentional
registry change.

Each reaction has two policy lanes, `equal_share` and `reaction_aware`, but
both use the same exact signed reaction reference. During campaign
initialization, `experiments/generate_initial_pf.py` generates the immutable
iteration-0 `initial_pf.py` for every reaction/contract pair.

Compact executable evaluator fixtures live under `tests/fixtures/`. All
generated ten-reaction inputs, mutation history, exports, and reports remain
under the external `HSOPT_DATA_ROOT`.

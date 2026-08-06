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

`inputs/reactions.json` is the authoritative reaction ordering and
stoichiometry registry. `inputs/problem_metadata.json` records compact
scientific metadata for the 21 external species, while
`inputs/problem_cache.sha256` pins their external JSON/NPZ Hamiltonian caches.
`hsopt-campaign prepare-reactions` converts those checksummed caches into ten
external reaction directories consumed by the fixed evaluator in
`cli/src/hsopt/evaluation/compiled_unitary.py`.

The ten directories beside `inputs/` make that production set visible on
GitHub. Each `definition.json` records the signed reaction terms and the
metadata/coefficients SHA-256 checksums for every external species
Hamiltonian; its `initial_pf_loose.py` and `initial_pf_tight.py` contain the
two PF2 variants for every species in the reaction. Run `uv run hsopt-layout --check` to verify that
all ten definitions match the compact registry, or `uv run hsopt-layout
--write` after an intentional registry change.

Each reaction is optimized in two scientifically distinct, isolated lanes:

- `individual_errors` enforces the conservative componentwise condition
  $\sum_s |\nu_s(\widetilde E_s-E_s)| < \varepsilon_{\mathrm{chem}}$.
- `error_cancellation` enforces the signed aggregate condition
  $|\sum_s \nu_s(\widetilde E_s-E_s)| < \varepsilon_{\mathrm{chem}}$.

The evaluator records both metrics on every proposal, but feasibility,
acceptance, queue identity, and optimization history use only the lane's sealed
contract. This lets the final plots measure cancellation without contaminating
the conservative and cancellation-aware mutation trajectories. Both contracts
start from the same explicitly selected baseline variant and may share the same
immutable exact signed reaction reference; candidate evaluations and mutations
are never shared. Species exact references are computed once, and the signed
reaction reference is assembled from those cached values without another
eigensolve. The agent sees only the selected initializer after it is copied to
its isolated lane.

Compact executable evaluator fixtures live under `cli/tests/fixtures/`. All
generated ten-reaction inputs, mutation history, exports, and reports remain
under the external `HSOPT_DATA_ROOT`.

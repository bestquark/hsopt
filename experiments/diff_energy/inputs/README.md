# Reaction inputs from source

`species.json` fixes all 21 geometries (Angstrom), charges, singlet
multiplicities, STO-3G basis sets, active electron/orbital counts, and the
localization recipe. `reactions.json` fixes the ten balanced reactions,
stoichiometric coefficients, paper ordering, and 1 kcal/mol target. Their
source provenance is recorded in the species registry.

The generation code is `cli/src/hsopt/data/reactions.py`, invoked by
`hsopt-data reactions`. A fresh clone can generate the Hamiltonians and their
metadata without any historical checkout, private storage, or bundled tarball:

```bash
uv sync --locked --extra chemistry
uv run --extra chemistry hsopt-data reactions \
  --output-root ../data/raw/reactions-physical
uv run hsopt-campaign prepare-reactions \
  --reaction-manifest experiments/diff_energy/inputs/reactions.json \
  --cache-checksums ../data/raw/reactions-physical/problem_cache.sha256 \
  --cache-root ../data/raw/reactions-physical/problem_cache \
  --output-root ../data/prepared/diff_energy-physical
```

Outputs are published only when generation finishes and only to a fresh
external directory. `--species h2` restricts generation to a bounded smoke
check; such an output is labeled incomplete. `--threads` defaults to one.

## Chemistry recipe

1. Run converged closed-shell RHF with the fixed geometry, basis, and charge.
2. Freeze the lowest `(total_electrons - active_electrons) / 2` RHF orbitals.
   Retain the next `active_electrons / 2` occupied orbitals and enough lowest
   virtual orbitals to reach `active_orbitals`.
3. Apply Pipek--Mezey localization separately to occupied and virtual active
   blocks, using PySCF's meta-Lowdin populations. Single-orbital blocks need no
   localization. The registry records SCF/localization tolerances and limits.
4. Obtain the effective one-electron integrals, chemist-order two-electron
   integrals `(pq|rs)`, and nuclear/frozen-core energy from PySCF CASCI. This
   uses its integral routines, without a CASCI eigensolver calculation.
5. Interleave alpha/beta spin orbitals, perform the Jordan--Wigner transform,
   remove the identity coefficient, and lexicographically sort nonidentity
   Pauli strings with qubit zero leftmost. Integral and Pauli cutoffs are
   respectively `1e-14` and `1e-12`.

The generated directory contains:

```text
problem_cache/           21 JSON metadata files + 21 NPZ integral/Pauli files
problem_cache.sha256     Fresh hashes of those 42 files
problem_metadata.json   Computed basis/CAS/qubit/term counts and metadata hashes
provenance.json         Convention, recipe/source hashes, dependency versions
```

Dense matrices, exact eigenstates, target-time evolution, and calibrated
policies are deliberately computed later by the existing calibration workflow.
Both baseline variants use norm-derived physical target times. See
[the reproduction guide](../../../docs/reproduce.md) for calibration, initial
policies, OpenShell setup, mutations, and plots.

## Corrected conversion and historical replay

The default `--convention physical` uses the physical interaction

```math
H_2 = \frac{1}{2}\sum_{pqrs,\sigma\tau}
(pr|qs)\,a^\dagger_{p\sigma}a^\dagger_{q\tau}
a_{s\tau}a_{r\sigma}.
```

[OpenFermion's InteractionOperator](https://quantumai.google/reference/python/openfermion/ops/InteractionOperator)
instead stores the coefficient of `a†_p a†_q a_r a_s`, with no implicit factor
of one half. Exchanging annihilators contributes a minus sign: its tensor must
therefore contain `-0.5 * (pr|qs)` in the corresponding spin blocks.

The recovered historical generator used `+(pr|qs)` in those positions, giving
**minus twice the physical two-electron interaction**. With the same archived
H2 integrals, the two-electron-sector ground energies are -3.223473789760 Ha
historically and -1.137283834489 Ha after correction. The tests check the
mapping against an independent occupation-basis construction and PySCF FCI.

For auditing the archived conversion explicitly:

```bash
uv run --extra chemistry hsopt-data reactions \
  --convention historical-replay \
  --output-root ../data/raw/reactions-historical-replay
```

Every output records its convention. This mode recreates the historical
Hamiltonian construction, not the bytes of the old cache: dependency versions,
SCF/localization results, and generated metadata can differ. The original
chemistry environment was not fully recorded. Preserve original sealed inputs
and candidates when replaying a saved campaign; do not substitute regenerated
inputs into it. Original caches remain outside the source checkout and in
historical Git revisions.

The checked-in `problem_metadata.json`, `problem_cache.sha256`, and sibling
`definition.json` files describe the **historical** cache. They are audit
records, not checksums for new physical generation. `hsopt-layout --check`
checks that historical inventory. Fresh generation computes its own metadata
and checksums, which must be passed to preparation. Corrected chemistry needs
new exact references, calibrated policies, and results before paper claims can
be made about it. Existing campaigns remain pinned to their original inputs.

# Baseline inputs and starting policies

The source metadata for shared ELOTE and δ-ELOTE preparation is kept in two
registries. Large Hamiltonians and fitted numerical calibrations are generated
outside the checkout.

| File | Contents |
| --- | --- |
| [`molecules.json`](molecules.json) | Default 115-Hamiltonian ELOTE selection, explicit historical 100/82-case selections, and pinned HamLib archive SHA-256 hashes. |
| [`reactions.json`](reactions.json) | Ten reactions, 21 species' geometries/basis/active spaces, the generation recipe, and historical cache hashes/metadata. |
| [`initial_nested_pf.py`](initial_nested_pf.py) | The common orbital-disjoint grouped PF2 template used to initialize shared molecular and reaction protocols. |
| [`generate_initial_pf.py`](generate_initial_pf.py) | The compact lexicographic Pauli PF2 generator from sealed loose/tight calibrations, retained for explicit Pauli-profile preparation and replay. |

The moved generator and grouped template are byte-identical to their previous
versions. `hsopt-elote prepare` owns grouped-baseline calibration and policy
binding; calibration math stays in `hsopt.campaign.calibration`. Metadata does
not supply invented numerical beta, repetition or synthesis values.

## Export input views

From the repository root:

```bash
uv run hsopt-layout --check
uv run hsopt-data export --output-root ../data/registry
```

The external directory contains `molecules.csv`, archive checksums,
`species.json`, `reactions.json`, explicitly named historical cache-audit files,
and a provenance manifest. Existing output directories are never overwritten.
The CSV and split JSON files are views of these two registries, not additional
checked-in sources. To export a historical selection explicitly:

```bash
uv run hsopt-data export --selection generalization100 \
  --output-root ../data/registry100
```

The `single_molecule82` metadata choice preserves its original ordered rows.
The complete old workflows and generated policies are on the
[`single_molecule` branch](https://github.com/bestquark/hsopt/tree/single_molecule).

## Prepare inputs and baselines

[The reproduction guide](../docs/reproduce.md) gives acquisition, preparation,
reference and calibration steps. HamLib input preparation accepts
`baseline/molecules.json` directly or an exported CSV. Reaction generation and
preparation accept `baseline/reactions.json` directly or the exported split
JSON files. Their Hamiltonian parsing and chemistry equations are unchanged.

A new grouped tight baseline is prepared once per molecule/species, with a
matching cached exact-ground reference:

```bash
uv run hsopt-elote prepare --problem /work/data/molecule.json \
  --exact-reference /work/data/reference.json \
  --baseline tight --output-dir /work/data/baselines/molecule
```

It measures beta for this grouped PF2, applies the existing Eq. (34) allocation,
binds `initial_nested_pf.py`, and verifies its unchanged compiled iteration 0.
The resulting `calibration.json`, policy and checksummed references remain
external. Both shared protocol families consume these sealed records. Reaction
preparation retains the existing stoichiometric allocation and chosen error
contract. A warm-started δ-ELOTE run separately seals its transferred ELOTE code.

## Historical reaction inputs

The registry distinguishes the fixed species recipe from the audit of the
historical coefficient cache. An earlier generator used an incorrect sign and
factor in its two-electron conversion. New `hsopt-data reactions` generation
uses the corrected physical convention; `--convention historical-replay` is an
explicit audit mode for the archived conversion. Both record their convention.

The historical metadata and cache checksums in `reactions.json` do not certify
new physical generation. New generation writes its own metadata, hashes and
dependency provenance, which must be used in preparation. Fresh chemistry needs
fresh exact references, calibration and evaluations. A directory cleanup does
not change the scientific identity or validity of archived results.

Replaying a saved campaign requires its original sealed Hamiltonians,
references, candidates and pinned environment. Regenerating from the recipe can
change numerical bytes across dependency/SCF versions, even in historical mode;
never substitute newly generated inputs into an existing trajectory.

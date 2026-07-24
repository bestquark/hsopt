# Experiments

The checked-in benchmark surface has exactly two families:

- `abs_energy/`: the canonical 91-instance HamLib selection through 14
  qubits. Each selected Hamiltonian has a visible directory containing an
  `instance.json` definition, alongside the source/checksum manifests used to
  prepare external inputs.
- `diff_energy/`: ten visible reaction directories. Each contains a
  `definition.json` with its signed stoichiometry and the checksums of every
  external Hamiltonian cache it consumes.

`generate_initial_pf.py` is the single immutable iteration-0 policy generator
for both families. It verifies sealed per-Hamiltonian calibration records and
embeds the Reiher--Wiebe--Svore--Wecker--Troyer PF2 baseline: lexicographic
Strang ordering, empirically calibrated $\beta(H)$, $t=1$, the resulting
step count and $\Delta t$, and a directly synthesized uniform rotation
tolerance. There is no second baseline-definition folder to keep in sync.

Run `uv run hsopt-layout --check` to prove that the 91 + 10 visible directories
match their compact registries and source checksums exactly. Run
`uv run hsopt-layout --write` after an intentional registry update to
regenerate the deterministic definition files.

The coefficient-heavy production Hamiltonians, generated per-instance
calibrations, `initial_pf.py` files, exact references, mutation lanes, scheduler logs, agent
transcripts, exports, and reports live below an external `HSOPT_DATA_ROOT`,
never in this checkout. Compact executable development fixtures live under
`tests/fixtures/`, not inside either production family.

The external campaign layout mirrors the two families:

```text
$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy/
$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy/
$HSOPT_DATA_ROOT/exact/production-cpu91-v1/lanes/{abs_energy,diff_energy}/
$HSOPT_DATA_ROOT/exact/production-gpu91-v2/lanes/{abs_energy,diff_energy}/
```

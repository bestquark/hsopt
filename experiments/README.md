# Experiments

The checked-in benchmark surface has exactly two families:

- `abs_energy/`: the canonical 91-instance HamLib selection through 14
  qubits, plus one compact runnable fixture and the source/checksum manifests
  used to prepare external inputs.
- `diff_energy/`: ten reaction-energy definitions, their validated seed
  inputs, the portable reaction runtime, and one compact bond-stretch fixture.

`generate_initial_pf.py` is the single immutable iteration-0 policy generator
for both families. It embeds the Reiher--Wiebe--Svore--Wecker--Troyer baseline
contract, so there is no second baseline-definition folder to keep in sync.

Production Hamiltonians, generated per-instance `initial_pf.py` files, exact
references, mutation lanes, scheduler logs, agent transcripts, exports, and
reports live below an external `HSOPT_DATA_ROOT`, never in this checkout.

The external campaign layout mirrors the two families:

```text
$HSOPT_DATA_ROOT/exact/inputs-v2/abs_energy/
$HSOPT_DATA_ROOT/exact/inputs-v1/diff_energy/
$HSOPT_DATA_ROOT/exact/production-cpu91-v1/lanes/{abs_energy,diff_energy}/
$HSOPT_DATA_ROOT/exact/production-gpu91-v2/lanes/{abs_energy,diff_energy}/
```

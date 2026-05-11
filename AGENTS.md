# hsopt Fault-Tolerant Protocol Agent Notes

The canonical product-formula T-gate optimization code lives in
`examples/pfs/`. Do not use the old standalone checkout as a runtime
dependency or source of truth.

Use the repo skills first:

- `$hsopt` for sequential code-mutation runs that edit only a lane's
  `simple_pf.py` and score proposals with one of the fixed evaluators.
- `$hsgepa` for optimizing the prompt that drives those T-gate mutation runs
  through GEPA's `optimize_anything` API.

Treat these as fixed infrastructure unless the user explicitly asks to change
the benchmark framework:

- `examples/pfs/*/evaluate.py`
- `examples/pfs/evaluate_energy.py`
- `examples/pfs/evaluate_fidelity.py`
- `examples/pfs/chemistry.py`

The usual mutation target is the active lane's `simple_pf.py`. Do not change
molecules, geometries, basis sets, active spaces, target times, localization, or
evaluator scoring semantics during an optimization run.

Canonical benchmark folders:

- `examples/pfs/h2`
- `examples/pfs/bh`
- `examples/pfs/lih`
- `examples/pfs/dmds`
- `examples/pfs/n2`
- `examples/pfs/co`
- `examples/pfs/nh3`
- `examples/pfs/hcn`

The PF examples expose two fixed scoring contracts:

- `evaluate_energy.py`: compiled T count subject to chemical accuracy on the
  lowest HF-reference spectral peak in the fixed-electron sector.
- `evaluate_fidelity.py`: compiled T count subject to final-state infidelity
  against exact `exp(+i H t)` being below the selected epsilon budget.

Campaign histories should live in external run directories. The `pfs/` tree is
kept as benchmark definition plus fixed evaluator infrastructure.

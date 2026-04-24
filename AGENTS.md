# trovolut Fault-Tolerant Protocol Agent Notes

The canonical product-formula T-gate optimization code lives in
`examples/ft_protocols/pfs/`. Do not use the old standalone checkout as a runtime
dependency or source of truth.

Use the repo skills first:

- `$t-autoresearch` for sequential code-mutation runs that edit only a lane's
  `simple_pf.py` and score proposals with one of the fixed evaluators.
- `$t-gepa` for optimizing the prompt that drives those T-gate mutation runs
  through GEPA's `optimize_anything` API.

Treat these as fixed infrastructure unless the user explicitly asks to change
the benchmark framework:

- `examples/ft_protocols/pfs/*/evaluate.py`
- `examples/ft_protocols/pfs/evaluate_energy.py`
- `examples/ft_protocols/pfs/evaluate_fidelity.py`
- `examples/ft_protocols/pfs/chemistry.py`

The usual mutation target is the active lane's `simple_pf.py`. Do not change
molecules, geometries, basis sets, active spaces, target times, localization, or
evaluator scoring semantics during an optimization run.

Canonical benchmark folders:

- `examples/ft_protocols/pfs/h2`
- `examples/ft_protocols/pfs/bh`
- `examples/ft_protocols/pfs/lih`
- `examples/ft_protocols/pfs/dmds`
- `examples/ft_protocols/pfs/n2`
- `examples/ft_protocols/pfs/co`
- `examples/ft_protocols/pfs/nh3`
- `examples/ft_protocols/pfs/hcn`

The PF examples expose two fixed scoring contracts:

- `evaluate_energy.py`: compiled T count subject to chemical accuracy on the
  lowest HF-reference spectral peak in the fixed-electron sector.
- `evaluate_fidelity.py`: compiled T count subject to final-state infidelity
  against exact `exp(+i H t)` being below the selected epsilon budget.

Campaign histories should live in external run directories. The `pfs/` tree is
kept as benchmark definition plus fixed evaluator infrastructure.

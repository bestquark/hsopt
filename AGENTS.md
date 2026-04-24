# trovolut Fault-Tolerant Protocol Agent Notes

The canonical product-formula T-gate optimization code lives in
`examples/ft_protocols/pfs/`. Do not use the old standalone checkout as a runtime
dependency or source of truth.

Use the repo skills first:

- `$t-autoresearch` for sequential code-mutation runs that edit only a lane's
  `simple_pf.py` and archive every scored proposal.
- `$t-gepa` for optimizing the prompt that drives those T-gate mutation runs
  through GEPA's `optimize_anything` API.

Treat these as fixed infrastructure unless the user explicitly asks to change
the benchmark framework:

- `examples/ft_protocols/pfs/evaluate.py`
- `examples/ft_protocols/pfs/track_iteration.py`
- `examples/ft_protocols/pfs/queued_track_iteration.py`
- `examples/ft_protocols/pfs/restore_best_iteration.py`
- `examples/ft_protocols/pfs/chemistry.py`
- `examples/ft_protocols/pfs/model_registry.py`

The usual mutation target is the active lane's `simple_pf.py`. Do not change
molecules, geometries, basis sets, active spaces, target times, localization, or
evaluator scoring semantics during an optimization run.

Canonical benchmark folders:

- `examples/ft_protocols/pfs/h2`
- `examples/ft_protocols/pfs/bh`
- `examples/ft_protocols/pfs/lih`
- `examples/ft_protocols/pfs/dmds`

The active PF evaluator optimizes compiled T count subject to chemical accuracy
on the ground-state energy extracted from the phase of
`<psi_0|exp(+i H t)|psi_0>`. Older fidelity-only histories are retained only as
legacy records and should not define the current best phase-constrained policy.

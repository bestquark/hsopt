# hsopt T-Gate Layout

Current organization:

- `skills/hsopt/`: instructions for sequential code-mutation T-gate optimization.
- `skills/hsgepa/`: instructions for GEPA prompt optimization.
- `src/hsopt/tgate_opt/`: shared CLI/runtime used by both skills.
- `src/hsopt/tgate_opt/mutation_agents.py`: Codex/OpenCode mutation adapters.
- `src/hsopt/tgate_opt/execution.py`: local/Slurm evaluator transports.
- `examples/pfs/<molecule>/`: primary runnable PF1 molecule folders.
- caller-supplied external `HSOPT_DATA_ROOT`: all campaign workspaces and logs.

Keep active product-formula families under `examples/pfs/`. Each
runnable molecule should be targetable directly by path, for example
`examples/pfs/h2`. The CLI also accepts legacy shorthand such as
`pfs/h2`.

Keep mutable campaign histories outside the repository checkout under
`$HSOPT_DATA_ROOT/{paper,development,old_experiments,scratch}`. Paper HamLib-121 runs
live below `paper/hamlib121/<campaign-id>/run_01` through `run_05`; reaction
runs live below `paper/reactions10/<campaign-id>`. The `hsgepa` write commands
require explicit output and run paths so a live trial does not silently create
a repo-local `runs/` tree.

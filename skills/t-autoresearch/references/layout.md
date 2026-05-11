# hsopt T-Gate Layout

Current organization:

- `skills/t-autoresearch/`: instructions for sequential code-mutation T-gate optimization.
- `skills/t-gepa/`: instructions for GEPA prompt optimization.
- `src/hsopt/tgate_opt/`: shared CLI/runtime used by both skills.
- `examples/ft_protocols/pfs/<molecule>/`: primary runnable PF1 molecule folders.
- `runs/t_gepa/`: local GEPA trial workspaces and logs.

Keep active product-formula families under `examples/ft_protocols/pfs/`. Each
runnable molecule should be targetable directly by path, for example
`examples/ft_protocols/pfs/h2`. The CLI also accepts legacy shorthand such as
`pfs/h2`.

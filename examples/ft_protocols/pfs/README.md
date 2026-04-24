# Product-Formula Benchmarks

`examples/ft_protocols/pfs/` is the canonical home for the default PF1
product-formula benchmarks in trovolut.

- `h2/`, `bh/`, `lih/`, `dmds/`: runnable molecule folders.
- `evaluate.py`, `track_iteration.py`, `queued_track_iteration.py`,
  `restore_best_iteration.py`: shared PF1 runner infrastructure.
- `snapshots/results.tsv`: phase-energy objective history. Older fidelity-only
  tables are retained as `results_legacy_fidelity.tsv` when a lane is first
  tracked under the phase objective.

Use molecule folders directly with the skills:

```bash
uv run t-autoresearch status examples/ft_protocols/pfs/h2
uv run t-autoresearch prompt examples/ft_protocols/pfs/h2
```

For compatibility with older notes, the CLI also accepts shorthand targets such
as `pfs/h2`.

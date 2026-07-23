# Experiments

This is the repository's only benchmark tree. It contains compact runnable
fixtures, portable paper experiment definitions, and completion-gated paper
artifacts. It does not contain live campaign workspaces.

- `abs_energy/` contains compact HamLib absolute-energy fixtures.
- `diff_energy/` contains compact molecular and reaction energy-difference
  fixtures plus the portable reactions10 runtime.
- `hamlib121/` defines five independent autoresearch runs over the fixed
  121-Hamiltonian benchmark.
- `reactions10/` defines the ten-reaction equal-share versus reaction-aware
  error-allocation comparison.
- `generate_initial_pf.py` and `paper_baselines/` define the immutable
  iteration-0 initializer.

Campaign databases, mutation directories, scheduler logs, telemetry, agent
transcripts, virtual environments, and intermediate checkpoints stay in
external storage. Only a reusable runner and compact, validated final results
belong here.

Final report trees contain two vector PDFs per scientific instance, plus a
machine-readable report manifest and checksums. These PDFs are derived only
from the strict portable export: bounded mutation summaries are included, but
prompts, standard output/error, database files, and absolute paths are not.

## External workspace convention

Every runner receives an explicit campaign root below one absolute
`HSOPT_DATA_ROOT`; runners must refuse roots inside the Git checkout. Use these
categories consistently:

```text
$HSOPT_DATA_ROOT/paper/hamlib121/<campaign-id>/run_01 ... run_05
$HSOPT_DATA_ROOT/paper/reactions10/<campaign-id>
$HSOPT_DATA_ROOT/development/<benchmark>/<campaign-id>
$HSOPT_DATA_ROOT/old_experiments/<benchmark>/<campaign-id>
$HSOPT_DATA_ROOT/scratch/preflight/<campaign-id>
```

Use `<family>_YYYYMMDD_<source-commit>` for a paper campaign ID. Keep failed and
superseded campaigns intact, then move the whole directory to `old_experiments`; never
reuse its ID for a fresh run. Git ignores emergency checkout-local roots such
as `hsopt-data/`, `.hsopt-runs/`, and `experiments/_runs/`, but external storage
is the normal location.

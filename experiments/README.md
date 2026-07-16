# Paper Experiments

This tree contains portable experiment definitions and completion-gated paper
artifacts. It does not contain live campaign workspaces.

- `hamlib121/` defines five independent autoresearch runs over the fixed
  121-Hamiltonian benchmark.
- `reactions10/` defines the ten-reaction equal-share versus reaction-aware
  error-allocation comparison.

Campaign databases, mutation directories, scheduler logs, telemetry, agent
transcripts, virtual environments, and intermediate checkpoints stay in
external storage. Only a reusable runner and compact, validated final results
belong here.

## External workspace convention

Every runner receives an explicit campaign root below one absolute
`HSOPT_DATA_ROOT`; runners must refuse roots inside the Git checkout. Use these
categories consistently:

```text
$HSOPT_DATA_ROOT/paper/hamlib121/<campaign-id>/run_01 ... run_05
$HSOPT_DATA_ROOT/paper/reactions10/<campaign-id>
$HSOPT_DATA_ROOT/development/<benchmark>/<campaign-id>
$HSOPT_DATA_ROOT/archive/<benchmark>/<campaign-id>
$HSOPT_DATA_ROOT/scratch/preflight/<campaign-id>
```

Use `<family>_YYYYMMDD_<source-commit>` for a paper campaign ID. Keep failed and
superseded campaigns intact, then move the whole directory to `archive`; never
reuse its ID for a fresh run. Git ignores emergency checkout-local roots such
as `hsopt-data/`, `.hsopt-runs/`, and `experiments/_runs/`, but external storage
is the normal location.

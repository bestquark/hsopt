# HamLib 121-Hamiltonian Experiment

This experiment measures autoresearch T-gate improvements across five
independent runs of a fixed 121-Hamiltonian benchmark.

The benchmark identity is guarded by two manifests in `runner/manifests/`.
`prepare_campaign.py` must read exactly 401 source rows, apply the fixed
64 GiB/workspace-factor-4 selection, and reproduce all fields of the canonical
121-row active manifest. Any count, lane-ID, ordering, or metadata drift stops
campaign preparation.

Each external workspace is named `run_01` through `run_05`. Every run first
records an accepted iteration-0 baseline for all 121 lanes, then evaluates
exactly 50 mutation proposals per lane. The final gate is therefore:

```text
5 runs x 121 lanes x 50 proposals = 30,250 proposals
```

Iteration 0 is required provenance but is not included in that proposal count.
The strict exporter also verifies a common lane set, identical baselines across
runs, complete iteration coverage, acceptance decisions, policy hashes, and
identical model/reasoning/Codex CLI/runner/lockfile provenance.

- `runner/` is copied into external workspaces to launch or resume campaigns.
- `results/` accepts only portable output from a successful strict final export.

The incomplete July checkpoint remains in external archival storage for audit
and recovery. It is not a final result and must not be copied into `results/`.

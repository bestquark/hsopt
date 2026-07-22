# HamLib121 Final Results

This directory is final-only. An export may be added here only when the strict
runner exporter succeeds without `--allow-incomplete` and proves all of the
following:

- five workspaces named `run_01` through `run_05`;
- the same 121 active lanes in every run;
- one accepted iteration-0 baseline per lane, identical across runs;
- iterations 1 through 50 present for every lane, totaling 30,250 proposals;
- valid feasibility and accept/reject lineage for every proposal;
- exact lane identity and metadata matching the canonical 121-row manifest;
- verified initial and best-policy hashes; and
- identical model, reasoning effort, Codex CLI, runner, and lockfile provenance;
- a portable, checksummed `provenance.json`; and
- portable output with no machine-specific absolute paths.

Accepted artifacts are compact normalized CSV/JSON summaries, selected policy
snapshots, `provenance.json`, a checksummed `FINAL_EXPORT_COMPLETE` marker,
`SHA256SUMS`, and the final report tree:

```text
final/
  hamiltonians/<lane_id>/
    extended_log.pdf
    summary.pdf
  report_manifest.json
  SHA256SUMS
```

The report renderer requires 30,250 normalized mutation rows, 605 run/lane
summaries, contiguous iterations, and valid export checksums before it writes
anything. The extended PDF includes the validated, whitespace-normalized
mutation summary and decision--not raw prompts or agent transcripts.
Diagnostic-shape and incomplete exports are always marked incomplete and must
stay external. Do not add campaign SQLite files, WAL/SHM files, mutation
workspaces, raw scheduler logs, telemetry, agent transcripts, caches, or virtual
environments.

The incomplete July checkpoint is retained externally for recovery and audit.
It does not satisfy this gate and is not a final paper artifact.

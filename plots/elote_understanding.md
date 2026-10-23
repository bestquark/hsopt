# Understanding ELOTE

These read-only analyses support Appendix B of the paper. Plotting requires a
working LaTeX installation for the manuscript fonts. They do not mutate a
candidate, change the evaluator, recalibrate a baseline, or dispatch GPU jobs.

## Policy diagnostics

Freeze/export the Figure 2 policy first, then verify its receipts and plot:

```bash
uv run --extra plots python plots/plot_elote_understanding.py \
  --snapshot /path/to/figure2/snapshot --output /path/to/policy-diagnostics
```

The source snapshot is the seed-05, mutation-78 policy across 115 Hamiltonians.
The script pins its SHA-256, verifies the recorded mixed-precision factors against
the archived policy, and checks signed-angle aggregation and schedule counts.
The coefficient reference is the maximum original fermionic-block contribution,
not necessarily the maximum combined Pauli coefficient. The active schedule is
flat first order with lexicographic labels, unchanged repetitions, and no
Clifford-frame change. Its tolerance has no independent angle dependence.

The output includes a compact `elote_policy_plot_data.npz` replay input containing
only plotted arrays and diagnostic metadata. Rebuild the figure without raw
rotation exports or a live campaign:

```bash
uv run --extra plots python plots/plot_elote_understanding.py \
  --replay /path/to/elote_policy_plot_data.npz --output /path/to/rebuilt-figure
```

Replay preserves the original scientific-verification receipt; it does not
claim to have repeated checks that require the original source snapshot.
Panels showing conditional coefficient, support and angle responses are exact
slices of the learned rule, not additional evaluation measurements.

## Seed decisions and inference effort

Collect a campaign locally or read it remotely through SSH:

```bash
python3 plots/export_elote_seed_effort.py \
  --campaign /path/to/campaign --output /path/to/seed-effort
python3 plots/export_elote_seed_effort.py --ssh omni \
  --campaign /remote/path/to/campaign --output /path/to/seed-effort
```

Only the destination is written. Completed mutation prefixes exclude iteration
0 and pending-round work. Identified archived mutation retries associated with
completed rounds are included in inference accounting. Stable receipt IDs
prevent double counting; missing or interrupted usage is never imputed as zero.
Input tokens include repeated context across calls. Preparation and editing
already present in the concatenated event stream are counted once.

Regenerate the table and CSVs from an existing JSON export:

```bash
python3 plots/export_elote_seed_effort.py \
  --snapshot /path/to/seed_effort.json --output /path/to/rebuilt-table
```

The table uses observed lower bounds on mean input/output tokens over **all**
completed rounds, rounded downward. Separate JSON fields give means over fully
logged rounds. Numerical uncertainty, policy-execution errors, physical accuracy
rejections and nonimproving valid proposals have distinct classifications.
“Not retained” is not a physical failure rate.

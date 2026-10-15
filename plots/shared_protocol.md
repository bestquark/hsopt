The shared-protocol figure shows accepted incumbent costs relative to iteration 0.
Each gray curve averages one molecule over five independent seeds. The green curve
averages the five seed-level molecule means, with a band of ±1 sample standard
deviation across those five means (`ddof=1`). This is a standard-deviation band.
The per-molecule bars use the same five seeds, with per-molecule standard-deviation
whiskers. The energy-error plot averages absolute compiled-unitary molecular
errors, in mHa, using the same five-seed means and standard-deviation band.
Its red horizontal line is the mean baseline simulation budget:
`epsilon_chem - epsilon_QPE_baseline`, from the sealed calibration's
`phase_estimation_error_hartree`. The narrow red band spans the per-molecule
budget range; it is not statistical uncertainty. These fixed budgets are
calibration references, not candidate acceptance limits: the current T95
score adapts QPE precision to each candidate's measured simulation error.
Using that adaptive precision for the line would reproduce the error itself.
The plot does not contain sampled QPE measurement outcomes. Iteration 0 contains
the unmodified baseline errors. Incumbent averages stop at the last round completed
by every seed.
Older exported statistics without calibration budgets retain their original
chemical-accuracy reference; a baseline QPE allocation is never inferred from measured bias.

Both iteration plots overlay one rejected-trial mean per iteration. First average
available molecule measurements within each rejected seed, using each molecule's
own iteration-zero T95 denominator; then average those seed means with equal
weight. Only rejected seeds contribute, so their number can vary by iteration.
Faster seeds' completed trials can extend beyond the matched incumbent curves;
the shaded right-hand region marks that coverage. The top panel zooms around the
incumbent trajectories. A lower symmetric-log panel shows the full range of
rejected means, including large means and zero.

Amber markers show rejected means. A red outline flags any measured chemical
accuracy failure among that round's rejected proposals, even when the mean energy
error is below the limit. Hollow markers indicate missing metric values. Inaccurate
cases have no chemically accurate T95, so their diagnostic costs are excluded from
T95 means. Uncertified metrics remain null, never substituted with zero. Available
measurements from a rejected suite still contribute; means with partial coverage
are display summaries and cannot establish that the entire suite passed.
The exported `rejected_trial_means` give each mean, seed count, measurement coverage,
and failure counts. The original `proposal_trials` retain every archived proposal's
seed, iteration, acceptance, candidate checksum, and individual measurements for
inspection or alternate visualizations. Acceptance and incumbent curves are unchanged.

Download **Plot data** from the dashboard and reproduce the PNG and vector PDF:

```sh
uv run --extra plots python plots/plot_shared_protocol.py \
  --input statistics.json --output figures
```

The command also accepts the dashboard's `shared-protocol.json` telemetry snapshot.
LaTeX with Computer Modern and the AMS packages, plus `dvipng`, is required.

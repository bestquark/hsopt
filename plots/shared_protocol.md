The shared-protocol figure shows accepted incumbent costs relative to iteration 0.
Each gray curve averages one molecule over five independent seeds. The green curve
averages the five seed-level molecule means, with a band of ±1 sample standard
deviation across those five means (`ddof=1`). This is a standard-deviation band.
The per-molecule bars use the same five seeds, with per-molecule standard-deviation
whiskers. The energy-error plot averages absolute molecular errors, in mHa,
using the same five-seed means and standard-deviation band. Its red horizontal
line uses the recorded chemical-accuracy threshold. Iteration 0 contains the
unmodified baseline errors. All plots stop at the last round completed by every seed.

Download **Plot data** from the dashboard and reproduce the PNG and vector PDF:

```sh
uv run --extra plots python plots/plot_shared_protocol.py \
  --input statistics.json --output figures
```

The command also accepts the dashboard's `shared-protocol.json` telemetry snapshot.
LaTeX with Computer Modern and the AMS packages, plus `dvipng`, is required.

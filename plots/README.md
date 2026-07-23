# Plots

Publication-facing plotting source belongs in this directory.

Plot scripts should:

- read checksummed final exports or compact derived tables;
- write figures to an explicit caller-supplied output directory;
- use concise LaTeX-rendered labels and publication-readable typography;
- prefer vector PDF or SVG output for manuscripts;
- keep raw campaign databases, caches, logs, and generated figures out of this
  source directory.

Operational telemetry renderers remain beside their campaign runners under
`experiments/` because they are execution infrastructure rather than paper
plots.

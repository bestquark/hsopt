# Current shared-protocol paper figures

Figures are reproduced from frozen external campaign summaries. Keep the
summaries, generated PDFs/PNGs, and manuscript outside this checkout. Plotting
requires Matplotlib, NumPy, and real LaTeX fonts. It does not evaluate new policies
or change the running campaigns.

## Figure 2: precision and compiled-cost savings

See [elote_precision_summary.md](elote_precision_summary.md) for the exporter,
renderer, metric definitions, and provenance checks.

## Figure 3: four seeds through iteration 60

```bash
python plots/plot_protocol_trajectories_preview.py \
  --input /path/to/frozen/selected-seeds-snapshot.json \
  --last-iteration 60 \
  --output /path/to/external/figure3
```

The approved snapshot contains seeds 1, 2, 4, and 5 from the 115-Hamiltonian
tight campaign. These are selected by completion of round 60, not by score.
The input retains dashboard trajectories for these seeds, truncated to rounds
0--60, plus the selection rule and each seed's actual completed-round count
in `completed_at_snapshot`. The renderer checks matching case cohorts and
iteration grids, normalized baselines, and passing incumbent energy errors.
It writes `elote_trajectories_four_seeds_60.{pdf,png}` and `statistics.json`.

Green curves average Hamiltonians and seeds with equal weight. Gray curves
average each Hamiltonian over the selected seeds. Shading spans the seed-level
means; it is not a confidence interval. T95 includes QPE query/success factors.
The energy panel reports absolute compiled-unitary energy error. Its fixed
baseline calibration budget is a reference; candidate QPE precision adapts.

## Figure 5: the selected HamLib dataset

```bash
python plots/plot_hamlib_active_spaces.py \
  --selected-manifest /path/to/frozen/figure2/snapshot/molecules.csv \
  --full-census /path/to/hamlib_molecular_dataset_active_spaces.json \
  --output-dir /path/to/external/figure5
```

The census JSON contains `active_space_counts_by_nqubits`; CSV census inputs
are also supported. The selected CSV needs an `nqubits` or `n_qubits` column.
The current campaign contains 115 Hamiltonians at 4--16 qubits, including all
24 cases at 16 qubits; the context census contains 663 active-space instances.
The plot derives its cutoff and legend counts from the supplied data and
avoids assumptions about dense-unitary storage. It writes
`hamlib_molecular_dataset_active_spaces.{pdf,png,json}`, with input hashes and
count metadata in the JSON file.

# Current manuscript figures

This is the canonical figure-code directory for the 18-page ELOTE manuscript at
paper commit `4e62b9c501d03068279bd2c92935d144b61dc1b2`. The published PDFs in
[`rendered/`](rendered/) are byte-identical copies of the assets included by
that manuscript. Its text, captions, equations and table are unchanged.

| Figure | Generator | Released output |
| --- | --- | --- |
| 1: research-loop illustration | `plot_autoresearch_histograms.py`, `sources/hsopt_autoresearch_loop/banner.tex` | `hsopt_autoresearch_loop.pdf` |
| 2: precision and T95 savings | `assemble_elote_figure2.py` | `elote115_precision_and_compiled_savings.pdf` |
| 3: five-search progress and energy accuracy | `plot_elote_search_and_energy.py` | `elote115_search_and_energy.pdf` |
| 4: reaction cost and energy errors | `plot_reaction_elote_delta_radial.py`, `plot_reaction_elote_delta_energy_bars.py` | Radial cost panel, grouped energy-error bars, and `reaction_elote_delta_legend.pdf` |
| 5: HamLib census | `plot_hamlib_molecular_dataset_active_spaces.py` | `hamlib_molecular_dataset_active_spaces.pdf` |
| 6: baseline initialization | `plot_baseline_initialization_vs_qubits.py` | `baseline_initialization_vs_qubits.pdf` |

## Rebuild without Overleaf or a GPU

Install the plotting dependencies and a working LaTeX installation, then run
from the HSOpt root:

```bash
uv sync --locked --extra plots
uv run python figures/audit.py
uv run python figures/build.py --output-dir figures/generated
```

This recreates all eight included PDF assets from frozen inputs. `generated/`
is ignored; it does not overwrite the released PDFs or touch a paper checkout.
The illustration reuses its approved raster artwork, rebuilds the numerical
histogram inset, and assembles its original LaTeX banner. No image generation,
agent inference, Lanczos solve or rotation synthesis is rerun. Regenerated
PDF metadata can differ; all eight outputs were verified pixel-identical to
the manuscript assets after this migration.

All input data lives under `experiments/`, with the logical-to-physical path and
SHA-256 map in [`experiments/manuscript.json`](../experiments/manuscript.json).
`_paths.py` validates every loaded snapshot. The old Figure 2 provenance record
referenced a four-search version of the Figure 3 input; that exact historical
source is preserved solely to verify its recorded hash. Its per-Hamiltonian
selected-policy results are identical to the final five-search input. No plotted
values were changed to repair this historical dependency.

## Scientific trajectories and measured results

- [Molecular exports](../experiments/abs_protocol/results/elote115-paper/README.md):
  five searches, 100 mutations each, all proposed policy bytes and decisions.
- [Reaction exports](../experiments/diff_protocol/results/delta-elote-paper/README.md):
  five searches, 50 mutations each, initialized from the actual transferred ELOTE.
- [HamLib diagnostics](../experiments/abs_protocol/results/hamlib-diagnostics/README.md):
  the census, baseline-initialization sample and exact Figure 1 coefficient samples.

Every proposal retains all scored case metrics, including failed or uncertified
outcomes. Detailed compact receipts for each baseline and retained final policy
include literal T counts, QPE-inclusive T95, energy values and errors, compiler,
accelerator and evaluator provenance where recorded. Original result/decision
hashes are retained. Large expanded circuits, reference vectors, private agent
messages and mutable dispatcher state remain in the original external archive.
The frozen inputs are sufficient to reproduce the plots without accessing the
external campaign archive. One-time archive exporters are preserved at
[the frozen export commit](https://github.com/bestquark/hsopt/tree/c92def0822c7ce717c870795caa6d19ab44092db/figures),
and their hashes are listed in `experiments/manuscript.json`.

## Table 1

Its source is the complete five-search inference audit, with missing usage kept
missing. Regenerate its embedded rows outside the manuscript:

```bash
uv run python figures/export_elote_seed_effort.py \
  --snapshot experiments/abs_protocol/results/elote115-paper/elote_understanding/seed_effort.json \
  --output figures/generated/table1
```

The result exactly matches the archived table body. The manuscript embeds those
rows directly and needs no script during compilation.

## Current script scope and retired tools

Optional older figure frontends and superseded images are preserved on
[`manuscript-assets-archive-20261030`](https://github.com/bestquark/hsopt/tree/manuscript-assets-archive-20261030/plots).
Only the seven figure generators listed above, their two shared Python helpers,
`build.py`, `audit.py`, and the Table 1 exporter remain here. Figure 1 retains its
coefficient-histogram generator and original LaTeX banner assembly. Numerical
test support lives under `cli/tests/helpers/figure_numerics/`, outside this folder;
it does not add any manuscript figures.

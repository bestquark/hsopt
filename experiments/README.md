# Shared protocol experiments

| Folder | Purpose |
| --- | --- |
| [`abs_protocol/`](abs_protocol/README.md) | One shared ELOTE policy evaluated across the selected molecular suite. |
| [`diff_protocol/`](diff_protocol/README.md) | One shared δ-ELOTE policy evaluated across the selected reaction suite. |
| [`elote_search/`](elote_search/README.md) | Agent-only, PySR-only and hybrid comparisons. |
| [`baseline/`](baseline/README.md) | Input registries and starting-policy templates shared by the protocols. |

Source metadata lives in [`experiments/baseline/molecules.json`](baseline/molecules.json)
and [`experiments/baseline/reactions.json`](baseline/reactions.json). The
[baseline guide](baseline/README.md) describes input preparation, sealed
calibrations and starting-policy generation. One generic grouped PF2 template
serves both shared families; it is bound to each case's calibrated parameters.

Candidates, baselines generated for a run, exact references, full Hamiltonians,
GPU requests and mutation traces stay outside the checkout. The selected
[ELOTE release](../results/elote/README.md) is an explicit frozen
artifact with its own hashes and compact results.

Per-Hamiltonian `abs_energy/` and per-reaction `diff_energy/` workflows and their
184 generated baseline copies are preserved on the
[`single_molecule` branch](https://github.com/bestquark/hsopt/tree/single_molecule).
Run `uv run hsopt-layout --check` to validate the compact registries and current
shared-protocol entry points. Export input views with `hsopt-data export` into
an external workspace; do not recreate per-instance directories in `main`.

## Frozen manuscript data

The current figure-input map is [manuscript.json](manuscript.json). Molecular
[trajectory/data exports](../results/paper/elote115-paper/README.md), reaction
[trajectory/data exports](../results/delta_elote/README.md), and
[HamLib diagnostics](../results/paper/hamlib-diagnostics/README.md) are frozen
scientific artifacts. They include every evaluated proposal and original decision,
not live queues, credentials or private agent transcripts. Figure generation and
table processing live in [figures/](../figures/README.md).

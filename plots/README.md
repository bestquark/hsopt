# Paper figures

This directory has one entry-point script for each data figure currently
included by the T-gates manuscript. The scripts preserve the paper's filenames
and visual style while reading the new exact-unitary data contracts:

| Script | Paper figure | Output PDF(s) |
|---|---|---|
| `plot_hamlib_coefficients.py` | Four representative coefficient distributions | `hamlib_hamiltonian_coefficient_histograms.pdf` |
| `plot_hamlib_precision_energy.py` | Mixed precision and energy trajectories | `hamlib_pf_tolerance_angle_scatter.pdf`, `hamlib_pf_energy_error_trajectories.pdf` |
| `plot_hamlib_tgate_reduction.py` | Aggregate $T_{95}$ reduction | `hamlib_pf_tgate_reduction_histogram.pdf`, `hamlib_pf_tgate_trajectories.pdf` |
| `plot_reaction_comparison.py` | Ten-reaction seed-policy comparison | `reaction_cancellation_budget.pdf` |
| `plot_hamlib_active_spaces.py` | HamLib active-space census | `hamlib_molecular_dataset_active_spaces.pdf` |
| `plot_baseline_tolerance.py` | Initial tolerance versus Hamiltonian one-norm | `baseline_epsilon_vs_lambda.pdf` |
| `plot_policy_angle_tolerance_violins.py` | Initial-versus-best physical rotation distributions per experiment | `angle_tolerance_violin.pdf` and `.png` per Hamiltonian/reaction |

All result plots verify the strict export completion marker and every declared
SHA-256 checksum before reading data. They never read a live campaign database.
The current 91-instance manifest supplies stable dataset metadata, while the
coefficient plot reads the immutable prepared Hamiltonian JSON files. Scripts
refuse to write generated figures inside either immutable input tree.

## Usage

Set paths to a completed absolute-energy export, difference-energy export,
prepared HamLib input tree, and an external output directory:

```bash
export ABS_EXPORT=/path/to/final-abs-energy-export
export DIFF_EXPORT=/path/to/final-diff-energy-export
export HAMLIB_INPUTS=/path/to/prepared/abs_energy
export FIGURE_OUTPUT=/path/to/paper-figures
```

Generate the four main-text figures:

```bash
uv run python plots/plot_hamlib_coefficients.py \
  --hamlib-input-dir "$HAMLIB_INPUTS" \
  --output-dir "$FIGURE_OUTPUT"

uv run python plots/plot_hamlib_precision_energy.py \
  --export-dir "$ABS_EXPORT" \
  --output-dir "$FIGURE_OUTPUT"

uv run python plots/plot_hamlib_tgate_reduction.py \
  --export-dir "$ABS_EXPORT" \
  --output-dir "$FIGURE_OUTPUT"

uv run python plots/plot_reaction_comparison.py \
  --export-dir "$DIFF_EXPORT" \
  --output-dir "$FIGURE_OUTPUT"
```

Generate the two appendix figures:

```bash
uv run python plots/plot_hamlib_active_spaces.py \
  --output-dir "$FIGURE_OUTPUT"

uv run python plots/plot_baseline_tolerance.py \
  --export-dir "$ABS_EXPORT" \
  --output-dir "$FIGURE_OUTPUT"
```

Pass `--full-census /path/to/census.json` to
`plot_hamlib_active_spaces.py` to place the current 91-instance selection over
the full HamLib context. The accepted JSON format may contain either
`active_space_counts_by_nqubits` or a lane list; a CSV with an `nqubits` or
`n_qubits` column is also accepted.

Generate one angle/tolerance violin comparison for every experiment that has
an iteration-0 result and an incumbent in a live campaign:

```bash
uv run python plots/plot_policy_angle_tolerance_violins.py \
  --campaign-root /path/to/campaign \
  --baseline tight \
  --output-dir /path/to/policy-violins
```

By default, incomplete lanes are labeled as partial and the lowest feasible
current $T_{95}$ replicate is selected for each experiment. Add
`--require-complete` to export only experiments that reached all 50 iterations.
The violins weight physical rotations by product-formula repetitions and
reaction stoichiometry; QPE query repetitions are deliberately excluded.

The plots use a real LaTeX installation for typography and deliberately do not
fall back to Matplotlib mathtext. Generated PDFs, exports, raw campaigns,
databases, and logs remain outside the Git checkout.

## Independent iteration-0 T95 census

`compute_iteration0_t95.py` measures the plotted baseline costs outside the
autoresearch loops. It first verifies that the three absolute-energy lanes (or
two reaction lanes) have identical source, `initial_pf.py`, and exact-reference
hashes. It then snapshots one task per physical benchmark and runs that task
through the same fixed `hsopt.exact_queue` evaluator used at autoresearch
iteration 0. It neither reads nor changes an incumbent or candidate policy.

Prepare and submit the canonical 91 + 10 census as four balanced, single-A100
jobs:

```bash
uv run python plots/compute_iteration0_t95.py launch \
  --campaign-root "$HSOPT_DATA_ROOT/exact/production-cpu91-bound-eq34-v1" \
  --census-root "$HSOPT_DATA_ROOT/exact/iteration0-t95-census-v1" \
  --repo-dir "$PWD" \
  --jobs 4 \
  --partition polar4 \
  --backend cupynumeric \
  --accelerator gpu
```

The dense-eigensolve proxy is used only to balance the 101 immutable tasks
across jobs; it never enters the score. Collecting is idempotent, and export
refuses partial data:

```bash
uv run python plots/compute_iteration0_t95.py status \
  --census-root "$HSOPT_DATA_ROOT/exact/iteration0-t95-census-v1" \
  --remote

uv run python plots/compute_iteration0_t95.py collect \
  --census-root "$HSOPT_DATA_ROOT/exact/iteration0-t95-census-v1"

uv run python plots/compute_iteration0_t95.py export \
  --census-root "$HSOPT_DATA_ROOT/exact/iteration0-t95-census-v1"
```

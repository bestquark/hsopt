# ELOTE precision and compiled-cost figure

`plot_elote_precision_summary.py` freezes the lowest-scoring accepted shared
policy and renders the paper's Figure 2. It selects one policy across the whole
suite, rather than selecting a different policy for each Hamiltonian. An
unfinished campaign produces an explicitly labeled interim snapshot.

On the data host, use the campaign's pinned HSOpt implementation to resolve
the archived schedules and verify their hashes against the evaluation receipts:

```bash
PYTHONPATH=/path/to/pinned/hsopt/cli/src python plots/plot_elote_precision_summary.py export \
  --campaign /path/to/shared-campaign \
  --output /path/to/external/figure2/snapshot
```

Render that frozen snapshot locally with Matplotlib, NumPy, and a real LaTeX
installation:

```bash
python plots/plot_elote_precision_summary.py plot \
  --snapshot /path/to/external/figure2/snapshot \
  --output /path/to/external/figure2
```

The main output is `elote_precision_and_compiled_savings.{pdf,png}`. Panel (a)
shows relative synthesis tolerance against the compiler's absolute Rz angle,
with the uniform tight baseline as a dashed line and no grid. Larger tolerances
permit looser synthesis. Panel (b) shows one compiled T-count reduction per
Hamiltonian, including all product-formula repetitions but excluding QPE query
and success-amplification factors. A separate T95 variant includes those factors.
Savings reflect the complete learned policy, including splitting changes, and
do not isolate the effect of mixed precision.

The export retains the policy, per-Hamiltonian costs, rotation features, and
provenance in a checksummed snapshot; it does not copy raw Hamiltonian inputs.
The renderer verifies the exported checksums and records output hashes in
`figure_manifest.json`. Neither stage synthesizes gates or runs an eigensolver.
Keep snapshots and generated figures outside the checkout alongside campaign
data and manuscript assets.

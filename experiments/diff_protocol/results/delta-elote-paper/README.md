# Frozen reaction δ-ELOTE publication data

The two summary JSON files are the exact inputs for Figure 4's cost and error
panels. best_policy.py is the selected CUDA-Q/A100 policy, SHA256
6952b7d2728cbb4c9eb65562dac8ffa613fbc26f2780019817bbb0aeecf1370c.

trajectories/trajectories.jsonl.gz contains five initial transferred-policy
baselines plus 250 mutations, each scored across ten reactions. Exact proposal
bytes, decisions, aggregate results and every scored reaction metric are retained.
Detailed compact receipts for the initial and retained final policy of each
search preserve the recorded compiler, accelerator, literal T counts, QPE costs,
energies and cancellation/componentwise error information. No laptop/PyGridSynth
results are merged into this canonical export.

The actual transferred molecular initializer is the earlier 966d815a5dcbace3b9b5538ea450393a5207ebfaf228f99c2decbd4bae6e46d5
checkpoint, not the final d199 molecular policy. Figure 4's normalization remains
its measured tight-baseline cost, and all ten signed differences satisfy the
recorded chemical-accuracy contract. Plot data retains both the signed reaction
error and the no-cancellation triangle bound. Source result hashes identify
the separate tight-baseline/transfer validations used for that comparison.

Run python figures/audit.py to verify the transferred policy and globally
selected reaction policy against every evaluated proposal. Rendering and
scientific export code is in figures/, not this data directory.

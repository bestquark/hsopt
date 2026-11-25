# Frozen molecular ELOTE publication data

The plotting snapshots are copied byte-for-byte from the latest manuscript data.
They retain the final d199f6a4efef965aa015dbd73f69d14f628fd1b9eaa3653391dad8e3a75e472e
policy, 115 training Hamiltonians, 76 measured larger testing cases, original
compiler/energy provenance and the two explicit NaLi construction exclusions.
The testing baseline fit is frozen from training; HF-18's failed baseline remains
recorded. Data scope and path/hash mappings are in experiments/manuscript.json.

trajectories/trajectories.jsonl.gz contains one campaign header and 505 round
records: five unchanged initial baselines plus 500 evaluated mutations. Each
record preserves the original candidate source bytes/hash, original decision,
full aggregate result and all scored case metrics. Baselines and each search's
retained final policy also contain compact detailed scientific receipts, including
measured energies, literal circuit T counts, QPE T95 and accelerator/compiler
identity. Large expanded circuits remain in the original archive. Missing or
uncertified values are preserved as such; rejected policies are never substituted
for retained incumbents. Source hashes identify the original result records.

The elote_understanding/ inference audit separately accounts for 644 agent
attempts including 144 archived retries; those pre-scoring retries are not extra
scored mutations. Its token-coverage gaps remain missing, not zeros. The selected
Python policy itself is also available in results/elote/.

Use python figures/audit.py to independently replay accepted-policy progress,
verify its plotted means/ranges and verify the global selected policy. The
published dashboard percentages have six decimal places; the audit accounts
for that display quantization without modifying any scientific error threshold.

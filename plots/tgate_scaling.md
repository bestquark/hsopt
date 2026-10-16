# Small-system learning and large-system resource estimates

Train and select a frozen protocol on a fixed small-system suite. Larger
Hamiltonians can then be used for resource accounting without allocating their
statevectors, Hamiltonian matrices or unitaries. This answers a resource
question, separately from whether the larger system meets chemical accuracy.

Lanczos on sparse or matrix-free H can supply a numerical reference energy
without full diagonalization. Comparing the protocol's QPE energy with it still
requires applying the compiled protocol, extracting its selected phase and
checking convergence. An H energy alone does not evaluate PF error. Eventually
even those statevectors become infeasible, while circuit counting remains
possible.

## Count definition

For a stationary protocol with r repetitions:

\[
T_{U(t)}=r\sum_{j\in\text{one step}}
T_{\mathrm{CUDA-Q}}\!\left(R_z(-2\alpha_j),\epsilon_j\right).
\]

This counts **one physical evolution**, excluding QPE query repetitions,
confidence amplification and state preparation. It is not T95. Exact mode
uses the evaluator's rounded angles, synthesis options, tolerance checks and
CPU CUDA-Q compiler. Its tiny single-qubit outputs use a bounded cache. A
streaming path counts canonical Pauli PF2 without retaining all terms or gates.
Other protocols use the trusted schedule interface, supporting grouping,
splittings and Clifford frames without building a molecular unitary.
The benchmark uses `t = 1/sum_j |c_j|` for each Hamiltonian: these are its
normalized evolutions, not equal physical durations across different molecules.

The optional `analytic_ross_selinger` method is a **cost surrogate**, not a
literal circuit count or rigorous upper bound. It fits the supplied uniform
baseline counts to `max(0, gamma*log2(1/epsilon)+offset)`. It is never an
acceptance metric.

## Starting parameters on unseen systems

Tight calibration is Hamiltonian-specific. Small-case records do not provide a
numerically calibrated tight beta for larger systems. The analysis therefore:

1. Reuses sealed r, epsilon and beta for their original small Hamiltonians.
2. Fits log(beta times target time), the log reference step count, using only
   those records. Features are log qubits, log nonidentity term count and log
   nonidentity coefficient one-norm. A standardized ridge fit records both
   leave-one-row and leave-one-molecule-out diagnostics.
3. Predicts beta for an unseen Hamiltonian, with a minimum reference step count
   of one, then applies the existing Eq.34 allocation. Baseline and learned
   policy receive the **same** starting allocation.

The large-system curve uses **extrapolated parameters**, even when every count
comes from literal CUDA-Q synthesis. Its allocated error budget is not a
verified energy-error bound. No large-case energy selects the fit or enters
mutation feedback. Input rows, policy and model are hashed in the manifest.

The recovered precision-only policy was trained on all 91 JW instances through
14 qubits. Its 16-qubit extension needs either extrapolation or newly calibrated
Pauli records. The grouped ELOTE calibration is a different baseline and cannot
substitute for those records.

## Reproduction

Inputs remain external. The catalog is the HamLib electronic-structure JW
census CSV. `training.csv` contains these columns for each calibrated case:

```text
case_id,problem_name,collection,file,dataset,molecule,n_qubits,nonidentity_terms,
target_time,reference_trotter_number_beta,uniform_epsilon,
baseline_one_step_t_count,baseline_compiled_evolution_t_count,
learned_compiled_evolution_t_count,baseline_passed,learned_passed
```

`calibrations.json` maps case_id to its sealed flat policy parameters, including
`n_steps`, `uniform_epsilon`, `reference_trotter_number_beta` and baseline
profile. A frozen provenance JSON identifies the old policy by `policy_sha256`.

```bash
uv run python plots/plot_tgate_scaling.py count \
  --catalog /data/hamlib_electronic_active_space_instances.csv \
  --archive-root /data/hamlib/electronic \
  --training-table /analysis/training.csv \
  --calibrations /analysis/calibrations.json \
  --baseline-policy experiments/abs_protocol/initial_pf.py \
  --policy /analysis/frozen_policy.py \
  --policy-provenance /analysis/policy_provenance.json \
  --training-max-qubits 14 --max-qubits 120 --method cudaq \
  --output-dir /analysis/scaling

uv run python plots/plot_tgate_scaling.py plot \
  --input-dir /analysis/scaling --output-dir /analysis/figures
```

CUDA-Q synthesis uses the supported Linux environment. Plotting needs
Matplotlib and real LaTeX. The figure shows archived counts in the calibrated
region and computed counts under extrapolated parameters outside it, with
geometric means by qubit size. Its second panel shows paired relative T counts.
It does not label untested counts as chemically accurate or as T95.

The JSONL table checkpoints cases and resumes only with matching configuration
and input hashes. `--archive-cache /scratch/hdf5` extracts each verified HDF5
once; this derived cache is disposable after workers finish. Parallel CPU
workers use distinct `--shard-index` values with the same `--shard-count` and
output directory. Merge after all complete:

```bash
uv run python plots/plot_tgate_scaling.py merge --input-dir /analysis/scaling
```

Merge checks every selected instance ID and refuses an incomplete census.
Source code contains no site account, host or queue settings; deployment stays
in the external runtime.

## Reusing the analysis for grouped ELOTE

Pass the new frozen policy, its grouped baseline policy, and its own grouped
small-case calibration/count table. Select `--profile fermionic_nested
--full-schedule --allow-different-policy --policy-label ELOTE`. The baseline's
trusted forward decomposition and rotation count feed Eq.34, matching grouped
preparation. The learned schedule is counted as returned, including changed
repetitions, splitting, grouping and frames. Old precision-only archived counts
are never substituted for a new policy. The full-schedule path retains a gate
list: its memory scales with circuit size rather than Hilbert-space dimension.

Using all 115 instances through 16 qubits for training and larger instances for
resource studies would be a new fixed experimental scope. This analysis does
not change a running campaign or its existing train/test split.

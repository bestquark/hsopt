# Local sensitivity and mixed precision

These scripts explain a **frozen** shared absolute-energy policy and compare it
with a post-hoc sensitivity-based allocation. They do not change live proposals,
sealed baselines, or campaign acceptance. Keep all tables, Hamiltonians, cached
states, bound validation policies and figures outside the checkout.

## Model

Write one ideal product-formula step as `S = G_L ... G_1`, with
`G_j = exp(+i alpha_j P_j)` and `S v = exp(+i phi) v`. In HSOpt,
`E = c0 + phi / dt` locally on the tracked phase branch. An infinitesimal
left error `exp(+i delta Q_j)` after gate j has eigenphase derivative

`d phi / d delta = <v_j | Q_j | v_j>`, with `v_j = G_j ... G_1 v`.

This follows by differentiating the unitary eigenvalue equation. For a pure
Pauli-exponential angle error, `Q_j = P_j`. This delta is the alpha error,
not the Rz angle error: HSOpt synthesizes `Rz(-2 alpha)`.

An explicit derivation uses `S(delta) v(delta) = exp(i phi(delta)) v(delta)`.
Differentiate, multiply by `v^dagger S^dagger`, and use normalization. The
eigenvector-derivative terms cancel, leaving
`d phi/d delta = -i <v|S^dagger (dS/d delta)|v>`. Inserting a gate error
`exp(i delta Q_j)` gives the propagated-state expectation above. Finally divide
by dt to obtain the energy sensitivity. This assumes a smooth isolated phase.

### A simple chemical-structure assumption

For a short step, replace `v_j` by the H eigenstate, so that the angle weight
`dt s_j` is approximately `|<P_j>|`. A determinant has diagonal I/Z
expectations of magnitude one and off-diagonal X/Y expectations zero. Mixing
other determinants changes this distinction.

For a two-determinant real toy Hamiltonian `[[0, v], [v, Delta]]`, the ground
energy is `(Delta - sqrt(Delta^2 + 4v^2))/2`. Differentiation gives
`|<X>| = 2|v| / sqrt(Delta^2 + 4v^2)`. With `|v| << Delta`, this becomes
`2|v|/Delta`, so inverse sensitivity favors tolerances proportional to
`Delta/|v|`. When mixing is strong that sensitivity saturates at one. Diagonal
rotation sensitivities are instead near one in the weak-mixing limit.

This illustrates why a chemistry-aware allocation can differ from uniform
precision. It does not derive one universal coefficient exponent: `v` is a
combined determinant coupling, often involving several Pauli coefficients,
and excitation gaps and interference also matter. It follows the standard
[first-order perturbation expansion](https://www.ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/a0889c5ca8a479c3e56c544d646fb770_MIT8_06S18ch1.pdf).

The analyzer also conjugates all three X/Y/Z pivot-error axes through the
evaluator's actual Clifford scaffold. If their expectations are `b_j`, then
the worst first-order response to a phase-aligned SU(2) error of infinitesimal
operator norm epsilon is `||b_j||_2 epsilon / dt`. The angle-only model uses
`|b_j,z| epsilon / dt`. This distinguishes small angle sensitivity from
insensitivity to an arbitrary synthesized rotation-axis error.

For the cohort, the scripts substitute the **cached exact H ground input**
for `v`, the ideal PF step eigenvector. This is a perturbative approximation,
not an exact molecular-QPE sensitivity calculation. The report records its
step residual and survival probability for every molecule. Selected cases of
at most eight qubits additionally use an offline ideal-step Schur decomposition
to check that substitution and directly finite-difference its eigenphases at
two error sizes. Degenerate phases are flagged instead of assigning a unique
branch derivative. This small validation is not a campaign dense fallback.

## Conditional optimum

For independent occurrence errors and an additive **first-order** error model,
minimizing equal logarithmic synthesis costs gives `epsilon*_k = B/(L s_k)`.
If a term's `m_j` occurrences share one tolerance, define
`S_j = sum_occurrence s_k`; the corresponding optimum is
`epsilon*_j = m_j B/(L S_j) = B/(L mean_occurrence(s_k))`.

The figures compare **shapes** after dividing each molecule's tolerances by
their occurrence-weighted geometric mean, `GM(x) = exp(sum_j m_j log(x_j) / sum_j m_j)`.
The coefficient axis uses the same per-molecule normalization, so a value of ten
means ten times that molecule's geometric-mean coefficient magnitude. Thus the
theory shape is the normalized inverse mean sensitivity, and overall precision relaxation cannot
create apparent agreement. The CSV also provides absolute predictions matched
to the learned policy's own linearized budget. Those are not certified paper
budget allocations. Near-zero responses have no finite interior optimum in
this model. The uncapped diagnostic omits them, recording counts and cutoff
checks at 1e-6, 1e-8, and 1e-10. The bounded compilation comparison below retains
every term instead. No finite-error regularizer is fitted.

The coefficient/support power-law fits are **descriptive fits to calculated
responses**, with equal molecule weight and molecule intercepts removed. Their
exponents and in-sample R-squared are not an analytic derivation from chemical
structure. The binned figure bands are interquartile ranges across terms, not
seed uncertainty or confidence intervals.

The fitted model is `log(epsilon_shape_j) = A_m + a log|c_j| + b log w_j`.
Here `A_m` is a molecule intercept, not an additional optimized protocol
parameter. `sensitivity_power_law_fits` removes this intercept and fitted support
dependence to show the coefficient trend. Fits to inverse sensitivity and to
the learned tolerances are computed separately; the latter's high R-squared
primarily reflects the explicit power-law structure already present in the code.

`sensitivity_overlay` puts the derived angle allocation (blue) and learned
allocation (green) on one common axis, as two point clouds without trend curves.
It uses the same terms, coefficient normalization and geometric-mean tolerance
normalization for both; support is retained without series-specific corrections.
Points are a matched sample of terms. Tick numbers, axis labels and legend use
26, 30 and 24 point fonts. Use `--overlay-only` to regenerate this figure.
Epsilon is an allowed synthesis error, so a larger y value means **less**
precision. Approximate precision in bits is `-log2(epsilon)` and would reverse
the direction; the figure explicitly labels the quantity as error tolerance.

## Actual synthesis counts

`compile_angle_allocation.py` turns a selected response into concrete tolerances
and compiles both it and the frozen agent policy with the same CUDA-Q settings.
Select `--sensitivity-model angle` (the original default) for
`g_j = mean_occurrence(|b_z|)`, or `--sensitivity-model su2` for
`g_j = mean_occurrence(sqrt(b_x^2 + b_y^2 + b_z^2))`. The latter accounts for
all three local synthesis-error directions, not only an extra Rz angle error.
For each molecule, set `B_agent = sum_j m_j g_j epsilon_agent,j` using the
**same selected response model** for agent and derived allocations.
Minimize `sum_j m_j log(1/epsilon_j)`
subject to that same budget and `epsilon_j <= max_k epsilon_agent,k`.
The bounded solution is `epsilon_j = min(cap, C/g_j)`; bisection fixes C to
match the budget. Zero responses receive cap. No terms are discarded and no
unbounded or order-one error tolerance is introduced. This finite cap makes
the compilation comparison well-defined while retaining the small-error regime
of the original agent tolerances.

Counts include every physical rotation and product-formula repetition in one
full evolution. CUDA-Q counts are checked against the literal T characters in
the synthesized sequence. The report contains per-molecule counts and both a
summed cohort cost and a geometric-mean ratio. The synthesis report alone is a
compilation-only comparison; it does not evaluate chemical accuracy, QPE
probabilities or full T95. Matching a local error model does not establish
equality of actual energy errors.

Pass `--compilation-dir /path/to/synthesis-comparison` to the plotter to display
these concrete bounded tolerances, including all terms, instead of the uncapped
inverse-response diagnostic. The compiled table checksum and frozen policy
identity are verified before plotting.

Pass `--normalization baseline` together with `--compilation-dir` to plot
`|c_j|/c_max` against `epsilon_j/epsilon_0`, where epsilon_0 is the sealed
uniform baseline tolerance. Blue and green use exactly the same per-molecule
scales, preserving absolute relaxation relative to baseline. The dashed line
is the uniform baseline. The model label comes from the compilation report.
The default `--normalization shape` preserves the earlier geometric-mean view.

### Why the angle-only allocation can fail

An actual synthesized gate can have a local error
`exp(i(delta_0 I + delta_x X + delta_y Y + delta_z Z))`, instead of just an
extra Z rotation. Thus a small `|b_z|` does not justify loose tolerance when
`|b_x|` or `|b_y|` is large. Independent reconstruction of the frozen angle-only
comparison found this mechanism in its two accuracy failures: the missing
first-order X contribution was approximately -2.018 mHa for H4 R=0.7 and
-4.167 mHa for NaLi-14. Native synthesis stayed within its requested tolerance.

The three-axis comparison corrects this directional model, but is still not a
finite-error certificate. Its matched budget is the agent's **modeled** budget,
which may exceed chemical accuracy because the measured agent benefits from
cancellation and favorable compiler errors. Keep the same cached-state proxy
and cap, do not tune a scale after observing evaluation outcomes, and validate
every case through the unchanged scientific evaluator before claiming improvement.

Neither a local derivative nor its inverse certifies finite compilation error,
higher-order effects, branch switching, QPE overlap probability or optimal T95.
General U(2) scalar phase errors are outside the SU(2) model. Identical compiled
gates can have correlated errors; the absolute-sum model permits no cancellation.
Thus numerical agreement would support a mechanism, rather than prove that the
agent recovered a unique globally optimal allocation.

## Full energy and T95 validation

`validate_angle_allocation.py` evaluates the concrete derived tolerances through
the unchanged compiled-unitary evaluator with `cupy-lanczos`, matrix-free literal
circuits, and the cached exact ground-input reference. It preserves the frozen
policy's rotation schedule and repetitions, replacing only the synthesis-tolerance
method. There is no dense-diagonalization fallback or baseline recalibration.

Saved iteration-0 and frozen-agent results are reused only after cohort, policy,
Hamiltonian, compiler-settings and fresh-agent circuit-count checks. Every
derived count and rotation-schedule hash must match the compilation report.
Results record the actual energy bias, chemical-accuracy pass, QPE probability
and T95, alongside evaluator source hashes and accelerator metadata. Reports and
bound policies are written outside the live campaign. An interrupted run can
reuse receipts only with identical inputs, script and evaluator provenance.

The report distinguishes measured accuracy failures from uncertified evaluations.
Both prevent claiming a feasible whole-suite improvement. It reports the shared
geometric-mean baseline-relative T95 only when **every** molecule passes, rather
than silently excluding failures or confusing raw circuit counts with T95.

Related accuracy-allocation frameworks: [Meuli et al. (2020)](https://arxiv.org/abs/2003.08408)
and [Kuklinski et al. (2025), section IV.A](https://arxiv.org/abs/2508.03987).
The specific local response and allocation comparison here are our derivation;
neither reference establishes these molecular sensitivities.

## Reproduce

Use the same policy and manifest as `plot_shared_policy_precision.py`:

```bash
uv run python plots/analyze_policy_sensitivity.py \
  --run /path/to/campaign/seed-04 \
  --policy /path/to/precision-snapshot/policy.py \
  --snapshot /path/to/precision-snapshot/manifest.json \
  --output-dir /path/to/sensitivity \
  --validate-case 0265_hydrogen_data_H2_ES_H2_linear_R1.4_sto-6g_ham_JW

uv run python plots/plot_policy_sensitivity.py \
  --data /path/to/sensitivity/term_sensitivities.csv.gz \
  --manifest /path/to/sensitivity/sensitivity_manifest.json \
  --output-dir /path/to/sensitivity

uv run python plots/compile_angle_allocation.py \
  --run /path/to/campaign/seed-04 \
  --policy /path/to/precision-snapshot/policy.py \
  --snapshot /path/to/precision-snapshot/manifest.json \
  --sensitivities-dir /path/to/sensitivity \
  --output-dir /path/to/synthesis-comparison

uv run python plots/validate_angle_allocation.py \
  --run /path/to/campaign/seed-04 \
  --policy /path/to/precision-snapshot/policy.py \
  --compilation-dir /path/to/synthesis-comparison \
  --agent-iteration 20 \
  --output-dir /path/to/synthesis-comparison/validation

# Independent three-axis comparison; preserve the original angle-only results.
uv run python plots/compile_angle_allocation.py \
  --run /path/to/campaign/seed-04 \
  --policy /path/to/precision-snapshot/policy.py \
  --snapshot /path/to/precision-snapshot/manifest.json \
  --sensitivities-dir /path/to/sensitivity \
  --sensitivity-model su2 \
  --output-dir /path/to/sensitivity/su2

uv run python plots/validate_angle_allocation.py \
  --run /path/to/campaign/seed-04 \
  --policy /path/to/precision-snapshot/policy.py \
  --compilation-dir /path/to/sensitivity/su2 \
  --agent-iteration 20 \
  --output-dir /path/to/sensitivity/su2/validation

uv run python plots/plot_policy_sensitivity.py \
  --data /path/to/sensitivity/term_sensitivities.csv.gz \
  --manifest /path/to/sensitivity/sensitivity_manifest.json \
  --compilation-dir /path/to/sensitivity/su2 \
  --normalization baseline --overlay-only \
  --output-dir /path/to/sensitivity/su2
```

The analyzer validates source/policy hashes and uses only statevectors for the
cohort calculation. It does not compile any rotations. Plotting uses real
LaTeX and writes PNG/PDF figures plus the allocation table and summary.
Compilation and validation require CUDA-Q; validation additionally requires
CuPy on a GPU and the same scientific evaluator revision as the saved controls.

# Local sensitivity and mixed precision

These scripts explain a **frozen** shared absolute-energy policy. They do not
score new proposals, change tolerances, or alter campaign acceptance. Keep all
tables, Hamiltonians, cached states and figures outside the checkout.

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
their occurrence-weighted geometric mean. Thus the theory shape is the
normalized inverse mean sensitivity, and overall precision relaxation cannot
create apparent agreement. The CSV also provides absolute predictions matched
to the learned policy's own linearized budget. Those are not certified paper
budget allocations. Near-zero responses have no finite interior optimum in
this model; they are explicitly omitted, with counts and cutoff checks at
1e-6, 1e-8, and 1e-10. No finite-error regularizer is fitted.

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

`sensitivity_overlay` puts calculated inverse-angle allocation and learned
allocation on one common axis. It uses the same terms, coefficient normalization
and geometric-mean tolerance normalization for both; support is retained in the
data without series-specific corrections. Faint points are a matched sample of
terms, while curves are medians in identical logarithmic coefficient bins. These
curves describe the observed distributions and are not the conditional power-law
fits above. Use `--overlay-only` to regenerate only this figure from saved data.

Neither a local derivative nor its inverse certifies finite compilation error,
higher-order effects, branch switching, QPE overlap probability or optimal T95.
General U(2) scalar phase errors are outside the SU(2) model. Identical compiled
gates can have correlated errors; the absolute-sum model permits no cancellation.
Thus numerical agreement would support a mechanism, rather than prove that the
agent recovered a unique globally optimal allocation.

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
```

The analyzer validates source/policy hashes and uses only statevectors for the
cohort calculation. It does not compile any rotations. Plotting uses real
LaTeX and writes PNG/PDF figures plus the allocation table and summary.

# Resolving failed ground-input Lanczos checks

The ground-input contract requires a majority-weight compiled QPE peak whose
energy satisfies chemical accuracy. Successful solves still require the same
literal-unitary residual, independent-start energy/weight agreement, integer
QPE-cost stability and no-repetition-wrap checks. There is no dense fallback.

An inaccurate proposal can distribute the input over many compiled eigenstates.
Increasing the Krylov space does not guarantee a majority peak exists. The
numerical verifier therefore has three outcomes:

- `ok`: a resolved energy, overlap and T95 from the existing solver;
- `infeasible`: a numerical certificate excludes an accurate majority peak;
- `error`: verification remains unresolved, or infrastructure failed.

The shared policy is rejected if any case is certified infeasible. This does
not invent an eigenenergy or a T95 score for that case. Unresolved cases remain
explicitly listed, even when another case already proves the proposal infeasible.

## Rejection-only spectral filter

Let the literal compiled step have eigenphases theta_j and input probabilities
p_j. Its step count is r and physical evolution time is t. The accurate,
non-wrapped phase interval has center

    theta0 = (E0 - c0) t/r

and half-width delta = (epsilon_chem + epsilon_reference) t/r. The reference
allowance is 1e-9 Hartree. Define

    A_m = (1/m) sum_{k=0}^{m-1} exp(-ik theta0) U^k.

This is a QPE/Dirichlet filter, evaluated through state-vector actions of the
actual compiled circuit. Its squared norm is

    ||A_m psi||^2 = sum_j p_j |(1/m) sum_k exp(ik(theta_j-theta0))|^2.

For every phase in the accurate interval, the real part of the centered sum is
at least cos((m-1) delta/2), provided (m-1) delta/2 < pi/2. Consequently,

    P(accurate interval) <= ||A_m psi||^2 / cos((m-1) delta/2)^2.

A conservative upper bound below 1/2 excludes every accurate majority-weight
peak. This is a negative certificate for the current contract. A bound above
1/2 establishes neither accuracy nor failure and cannot accept a policy.

The numerical bound includes complex128 allowances for the literal synthesis
word, its Pauli-mask vector action, accumulation and norm reduction. Propagation
uses the circuit's per-action allowance. An inconsistent observed vector norm
invalidates the certificate. Missing roundoff metadata, an aliased/too-wide
window or the filter deadline also leave verification unresolved.

The filter runs after the first inconclusive Lanczos solve. If it cannot reject,
the original automatic refinements continue. Passing fast-path energies,
tolerances, compilation and T counts are unchanged.
The certifier version enters task identity, and the filter source enters the
evaluation fingerprint; older unresolved receipts cannot stand in for a repair.

Background on QPE filtering: [Sakuma et al., Phys. Rev. A 113, 012602 (2026)](https://arxiv.org/abs/2507.01361).
The conservative window bound above is derived here; it is not a claim that the
paper proves HSOpt's implementation or its floating-point allowance.

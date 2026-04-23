# Product-Formula Dynamics

`pfs/h2/simple_pf.py` estimates direct real-time dynamics with first- or
second-order product formulas. The evaluator keeps the H2 Hamiltonian fixed and
scores T count only after the dense product-formula error plus rotation
synthesis error bound is below chemical accuracy.

Editable policy knobs:

- `order`
- `trotter_steps`
- `rotation_synthesis_epsilon`

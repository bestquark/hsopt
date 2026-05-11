# Product-formula T-count experiments

This is the runnable exhibit in the current repo.

The experiment fixes a molecular Hamiltonian and asks an agent to reduce the
compiled T count of a product-formula simulation by editing only:

```text
examples/pfs/<molecule>/simple_pf.py
```

The evaluator fixes the Hamiltonian instance, basis, active space, localization,
target time, and accuracy criterion.

## Lanes

Small lanes are useful for smoke tests:

- `h2`
- `bh`
- `lih`
- `dmds`

The larger campaign grid is:

| Molecules | Constraints | Target iterations |
| --- | --- | --- |
| `n2`, `co`, `nh3`, `hcn` | `energy`, `fidelity` | 100 per lane |

That gives eight independent agent lanes for the large-molecule product-formula
experiments.

## What is computed

Each lane builds a fixed molecular problem, generates a Jordan-Wigner Pauli
Hamiltonian, and applies a product-formula evolution to a reference state. The
method file decides the synthesis precision policy for the Pauli rotations.

Two scoring contracts are exposed:

| Evaluator | Validity check | Lower-is-better cost |
| --- | --- | --- |
| `evaluate_energy.py` | chemical accuracy on the selected HF-reference spectral peak | compiled T count |
| `evaluate_fidelity.py` | final-state infidelity below the selected budget | compiled T count |

Invalid runs receive a large penalty added to the T count.

## Run the exhibit

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
uv run hsopt prompt examples/pfs/h2 --evaluator energy
```

Invoke the installed skill from Codex:

```text
$hsopt Run 20 sequential mutations on examples/pfs/h2 using the energy evaluator.
Bias toward broad epsilon policies over one-label tweaks.
```

Or from Claude Code when the client exposes slash commands:

```text
/hsopt Run 20 sequential mutations on examples/pfs/h2 using the energy evaluator.
Bias toward broad epsilon policies over one-label tweaks.
```

## What the agent should try

Good mutations are precision-policy changes that remain easy to audit:

- group Pauli labels by family, support size, or coefficient magnitude
- relax synthesis epsilon for low-impact rotations first
- tighten only terms implicated by failed accuracy diagnostics
- keep `n_steps` fixed unless diagnostics show product-formula error rather
  than synthesis error

The agent should not edit the evaluator, molecule definition, active space,
target time, geometry, basis, or scoring semantics.

## Direct scoring

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy
```

Print the exact fidelity command without running it:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator fidelity --epsilon 1e-3 --print-only
```

For the cost model itself, see [How T Gates Are Counted](t-gates.md).

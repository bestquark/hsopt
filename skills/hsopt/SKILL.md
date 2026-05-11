---
name: hsopt
description: Run sequential autoresearch-style T-gate optimization in hsopt. Use this whenever the user asks to reduce T gates, optimize synthesis precision, mutate product-formula precision policies, continue a lane, inspect T-count progress, or run the repo's `/hsopt` workflow.
---

# HSOpt

Use this skill for direct T-gate optimization runs where an agent edits a
benchmark lane's `simple_pf.py`, scores the mutation, reads the result, and
continues one scored step at a time.

Invocation note:
- In Codex, use this skill as `$hsopt` or through `/skills`.
- Some clients may expose it as `/hsopt`.
- The repo CLI is `uv run hsopt ...`.

## First Step

Inspect lanes and pick the active benchmark:

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
```

Render the exact agent prompt for a lane:

```bash
uv run hsopt prompt examples/pfs/h2
```

The primary runnable family is `examples/pfs/`. The CLI also
accepts legacy shorthand targets such as `pfs/h2`.

## Loop Contract

1. Read the lane status and choose `--evaluator energy` or `--evaluator fidelity`.
2. Edit only the lane's `simple_pf.py` unless the user explicitly changes the
   benchmark framework.
3. Make one coherent precision-policy mutation.
4. Run exactly one fixed evaluator command.
5. Read the JSON result before choosing the next mutation.
6. Preserve campaign history in the external run directory used by the caller.

## Canonical Commands

Print the evaluator command:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy --print-only
uv run hsopt evaluate examples/pfs/h2 --evaluator fidelity --epsilon 1e-3 --print-only
```

Score one mutation directly:

```bash
uv run hsopt evaluate examples/pfs/h2 --evaluator energy
```

## Optimization Target

Optimize the evaluator's lower-is-better `score`. In practice, this means
lowering compiled `t_gate_count` while keeping the fixed accuracy contract true:

- `energy`: chemical-accuracy estimate from the lowest HF-reference spectral
  peak in the fixed-electron sector.
- `fidelity`: final-state infidelity below the selected epsilon budget.

Treat these as fixed infrastructure:

- `chemistry.py`
- `evaluate_energy.py`
- `evaluate_fidelity.py`
- molecule-local `evaluate.py` wrappers

## Search Behavior

Prefer simple precision policies that are easy to audit:

- Pauli label families (`XXYY`, `ZZ`, single-`Z`, etc.)
- support size
- coefficient magnitude
- per-step angle magnitude
- grouped epsilon buckets
- targeted tightening for terms implicated by failed accuracy diagnostics

Avoid scripted sweeps, future-result batching, evaluator edits, chemistry
changes, and long stretches of tiny one-label tweaks after progress stalls.

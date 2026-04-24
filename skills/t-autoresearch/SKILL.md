---
name: t-autoresearch
description: Run sequential autoresearch-style T-gate optimization in trovolut. Use this whenever the user asks to reduce T gates, optimize synthesis precision, mutate product-formula precision policies, continue a lane, inspect T-count progress, or run the repo's `/t-autoresearch` workflow.
---

# T Autoresearch

Use this skill for direct T-gate optimization runs where an agent edits a
benchmark lane's `simple_pf.py`, scores the mutation, reads the result, and
continues one scored step at a time.

Invocation note:
- In Codex, use this skill as `$t-autoresearch` or through `/skills`.
- Some clients may expose it as `/t-autoresearch`.
- The repo CLI is `uv run t-autoresearch ...`.

## First Step

Inspect lanes and pick the active benchmark:

```bash
uv run t-autoresearch list
uv run t-autoresearch status examples/ft_protocols/pfs/h2
```

Render the exact agent prompt for a lane:

```bash
uv run t-autoresearch prompt examples/ft_protocols/pfs/h2
```

The primary runnable family is `examples/ft_protocols/pfs/`. The CLI also
accepts legacy shorthand targets such as `pfs/h2`.

## Loop Contract

1. Read the lane status and lane instruction file.
2. Restore the current best kept valid snapshot if the live file is not already
   best.
3. Edit only the lane's `simple_pf.py` unless the user explicitly changes the
   benchmark framework.
4. Make one coherent precision-policy mutation.
5. Run exactly one queued tracked evaluation.
6. Read the JSON result before choosing the next mutation.
7. If the result is `discard` or `crash`, restore the best kept valid snapshot
   before mutating again.
8. Preserve every scored proposal. Do not delete, renumber, or overwrite
   `snapshots/`.
9. Leave the live file restored to the best valid state before stopping.

## Canonical Commands

Print the evaluator command:

```bash
uv run t-autoresearch evaluate examples/ft_protocols/pfs/h2 --print-only
```

Score one mutation:

```bash
uv run t-autoresearch track \
  examples/ft_protocols/pfs/h2 \
  --description "relax low-coefficient ZZ precision buckets"
```

Restore best:

```bash
uv run t-autoresearch restore examples/ft_protocols/pfs/h2
```

## Optimization Target

Optimize the evaluator's lower-is-better `score`. In practice, this means
lowering compiled `t_gate_count` while keeping the fixed accuracy contract true:

- `pf1`: chemical-accuracy ground-state phase estimate.

Treat these as fixed infrastructure:

- `evaluate.py`
- `track_iteration.py`
- `queued_track_iteration.py`
- `restore_best_iteration.py`
- `chemistry.py`
- `model_registry.py`

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

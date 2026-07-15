---
name: hsgepa
description: Optimize the T-gate autoresearch prompt with GEPA `optimize_anything`. Use this whenever the user mentions GEPA, prompt optimization, `/hsgepa`, comparing GEPA to autoresearch, optimizing the agent prompt, or testing T-gate prompt strategies across hsopt lanes.
---

# HSGEPA

Use this skill when the object being optimized is the prompt or strategy text
that drives T-gate optimization, not a direct `simple_pf.py` mutation.

Invocation note:
- In Codex, use this skill as `$hsgepa` or through `/skills`.
- Some clients may expose it as `/hsgepa`.
- The repo CLI is `uv run hsgepa ...`.

## Design

`hsgepa` follows GEPA's `optimize_anything` pattern:

- candidate: a text prompt fragment stored as `{"prompt": "..."}`
- evaluator: run that prompt in an isolated benchmark copy
- dataset: T-gate benchmark lane examples
- score: higher-is-better valid fractional `t_gate_count` reduction
- side information: baseline/result summaries, agent trace tails, and trial
  paths for reflection

Each evaluator call creates a trial workspace under the caller's explicit
external `--run-root`, copies the needed benchmark family, archives a baseline,
launches Codex with the candidate strategy prompt, restores best, and scores
the resulting valid T-count improvement. This keeps GEPA trials out of the
repository and the canonical lane histories.

## First Step

Run a dry prompt-evaluation smoke test before a real GEPA run:

```bash
HSOPT_RUNS=/absolute/path/outside/hsopt
mkdir -p "$HSOPT_RUNS/hsgepa"
uv run hsgepa seed-prompt > /tmp/hsgepa_seed_prompt.md
uv run hsgepa evaluate-prompt \
  --prompt-file /tmp/hsgepa_seed_prompt.md \
  --family pf1 \
  --molecule h2 \
  --target-iterations 1 \
  --run-root "$HSOPT_RUNS/hsgepa/manual" \
  --dry-run
```

Write a default dataset:

```bash
uv run hsgepa dataset --smoke --output "$HSOPT_RUNS/hsgepa/dataset_smoke.json"
uv run hsgepa dataset --output "$HSOPT_RUNS/hsgepa/dataset_full.json"
```

Run GEPA once dependencies/API keys are available:

```bash
uv sync --extra gepa
uv run hsgepa optimize \
  --dataset "$HSOPT_RUNS/hsgepa/dataset_smoke.json" \
  --run-root "$HSOPT_RUNS/hsgepa/smoke" \
  --max-metric-calls 6 \
  --no-parallel
```

## Full Run Shape

Use smoke mode first (`h2`, one mutation) to validate wiring. Then broaden to
`h2`, `bh`, `lih`, and `dmds` with a larger metric-call budget.

Recommended progression:

```bash
uv run hsgepa optimize --smoke --run-root "$HSOPT_RUNS/hsgepa/smoke" \
  --max-metric-calls 6 --no-parallel
uv run hsgepa optimize --dataset "$HSOPT_RUNS/hsgepa/dataset_full.json" \
  --run-root "$HSOPT_RUNS/hsgepa/full" --max-metric-calls 24 --no-parallel
```

Only enable parallel GEPA workers after confirming local agent launches and
queued evaluators are stable:

```bash
uv run hsgepa optimize \
  --dataset "$HSOPT_RUNS/hsgepa/dataset_full.json" \
  --run-root "$HSOPT_RUNS/hsgepa/full_parallel" \
  --max-metric-calls 48 \
  --parallel \
  --max-workers 2
```

## Evaluator Semantics

The GEPA score is not the benchmark's raw lower-is-better `score`. It is a
higher-is-better prompt reward:

```text
(baseline_t_gate_count - best_valid_t_gate_count) / baseline_t_gate_count
```

Invalid, missing, or no-valid-result trials receive a small negative reward.
The structured side information is intentionally rich so GEPA can reflect on
why a prompt produced a crash, no mutation, invalid accuracy, or a real T-count
improvement.

## Prompt Search Policy

GEPA should learn prompt guidance that makes agents:

- inspect the previous scored result before mutating
- prefer broad precision-policy families over brittle one-label tweaks
- use diagnostics from failed accuracy runs to retighten specific terms
- keep evaluator and chemistry contracts fixed
- restore best after invalid or worse proposals
- produce concrete `--description` text for every archived mutation

Do not use `hsgepa` to directly mutate canonical benchmark files. Use
`$hsopt` for direct lane optimization.

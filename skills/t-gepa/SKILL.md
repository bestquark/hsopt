---
name: t-gepa
description: Optimize the T-gate autoresearch prompt with GEPA `optimize_anything`. Use this whenever the user mentions GEPA, prompt optimization, `/t-gepa`, comparing GEPA to autoresearch, optimizing the agent prompt, or testing T-gate prompt strategies across trovolut lanes.
---

# T GEPA

Use this skill when the object being optimized is the prompt or strategy text
that drives T-gate optimization, not a direct `simple_pf.py` mutation.

Invocation note:
- In Codex, use this skill as `$t-gepa` or through `/skills`.
- Some clients may expose it as `/t-gepa`.
- The repo CLI is `uv run t-gepa ...`.

## Design

`t-gepa` follows GEPA's `optimize_anything` pattern:

- candidate: a text prompt fragment stored as `{"prompt": "..."}`
- evaluator: run that prompt in an isolated benchmark copy
- dataset: T-gate benchmark lane examples
- score: higher-is-better valid fractional `t_gate_count` reduction
- side information: baseline/result summaries, agent trace tails, and trial
  paths for reflection

Each evaluator call creates a trial workspace under `runs/t_gepa/`, copies the
needed benchmark family, archives a baseline, launches Codex with the candidate
strategy prompt, restores best, and scores the resulting valid T-count
improvement. This keeps GEPA trials out of the canonical lane histories.

## First Step

Run a dry prompt-evaluation smoke test before a real GEPA run:

```bash
uv run t-gepa seed-prompt > /tmp/t_gepa_seed_prompt.md
uv run t-gepa evaluate-prompt \
  --prompt-file /tmp/t_gepa_seed_prompt.md \
  --family pf1 \
  --molecule h2 \
  --target-iterations 1 \
  --dry-run
```

Write a default dataset:

```bash
uv run t-gepa dataset --smoke --output runs/t_gepa/dataset_smoke.json
uv run t-gepa dataset --output runs/t_gepa/dataset_full.json
```

Run GEPA once dependencies/API keys are available:

```bash
uv sync --extra gepa
uv run t-gepa optimize \
  --dataset runs/t_gepa/dataset_smoke.json \
  --run-root runs/t_gepa/smoke \
  --max-metric-calls 6 \
  --no-parallel
```

## Full Run Shape

Use smoke mode first (`h2`, one mutation) to validate wiring. Then broaden to
`h2`, `bh`, `lih`, and `dmds` with a larger metric-call budget.

Recommended progression:

```bash
uv run t-gepa optimize --smoke --run-root runs/t_gepa/smoke --max-metric-calls 6 --no-parallel
uv run t-gepa optimize --dataset runs/t_gepa/dataset_full.json --run-root runs/t_gepa/full --max-metric-calls 24 --no-parallel
```

Only enable parallel GEPA workers after confirming local agent launches and
queued evaluators are stable:

```bash
uv run t-gepa optimize \
  --dataset runs/t_gepa/dataset_full.json \
  --run-root runs/t_gepa/full_parallel \
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

Do not use `t-gepa` to directly mutate canonical benchmark files. Use
`$t-autoresearch` for direct lane optimization.

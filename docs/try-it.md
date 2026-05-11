# Try the showroom

The site is a showroom first, but the product-formula exhibit is runnable from
the repository.

## Install the agent skills

```bash
npx skills add bestquark/hsopt
```

Install both skills into Claude Code:

```bash
npx skills add bestquark/hsopt --agent claude-code --skill '*' -y
```

The public skills are:

- `hsopt`: run an agent optimization lane.
- `hsgepa`: optimize the prompt that drives those agent runs.

## Run locally

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync
```

Inspect the product-formula lanes:

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
```

Run a direct evaluator:

```bash
uv run python examples/pfs/evaluate_energy.py \
  --molecule h2 \
  --script examples/pfs/h2/simple_pf.py
```

Render the agent prompt:

```bash
uv run hsopt prompt examples/pfs/h2 --evaluator energy
```

## Prompt optimization

Install the optional GEPA dependency when optimizing the agent strategy itself:

```bash
uv sync --extra gepa
uv run hsgepa dataset --smoke --output runs/hsgepa/dataset_smoke.json
uv run hsgepa optimize --smoke --run-root runs/hsgepa/smoke --max-metric-calls 6 --no-parallel
```

Each `hsgepa` trial copies the benchmark into an isolated workspace, launches an
agent with a candidate strategy prompt, restores the best valid state, and
scores valid T-count reduction.

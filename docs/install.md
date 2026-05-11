# Install

## Agent skills

Install the public hsopt skills from GitHub:

```bash
npx skills add bestquark/hsopt
```

Install both skills non-interactively:

```bash
npx skills add bestquark/hsopt --skill '*' -y
```

Install for Claude Code:

```bash
npx skills add bestquark/hsopt --agent claude-code --skill '*' -y
```

Check what will be installed:

```bash
npx skills add bestquark/hsopt --list
```

The package exposes two skills:

- `hsopt`: direct lane optimization.
- `hsgepa`: prompt optimization for hsopt agents.

## Local Python environment

Clone the repository when you also want to run the evaluators and CLIs:

```bash
git clone https://github.com/bestquark/hsopt.git
cd hsopt
uv sync
```

Optional GEPA dependencies:

```bash
uv sync --extra gepa
```

Optional cuPyNumeric dependencies for Legate-capable systems:

```bash
uv sync --extra cupynumeric
```

## Smoke checks

```bash
uv run hsopt list
uv run hsopt status examples/pfs/h2
uv run hsgepa --help
```

For a direct evaluator check:

```bash
uv run python examples/pfs/evaluate_energy.py \
  --molecule h2 \
  --script examples/pfs/h2/simple_pf.py
```

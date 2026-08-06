# HSOpt CLI

This directory contains the installable, tested runtime used to prepare and
run HSOpt campaigns. The repository root keeps `pyproject.toml` and `uv.lock`
so every command still runs as `uv run <command>` from the checkout root.

```text
cli/
├── src/hsopt/
│   ├── campaign/     Campaign setup, lanes, queue, supervisor, export, and parity
│   ├── cluster/      NVPark/Slurm submission adapter
│   ├── evaluation/   Fixed compiled-unitary evaluator and diagnostics
│   ├── sandbox/      OpenShell/OpenCode isolation and agent telemetry
│   ├── view/         Read-only mutation history and final report views
│   └── tgate_opt/    Generic mutation-agent execution helpers
└── tests/       Unit tests and compact executable evaluator fixtures
```

The scientific paper baseline itself intentionally lives in one visible file:
`../experiments/generate_initial_pf.py`. Campaign code loads that file when it
needs to calibrate or render an immutable iteration-0 policy; there is no
second baseline implementation hidden in this package.

The agent-facing `$hsopt` instructions stay separately in
`skills/hsopt/SKILL.md`. A mutation sandbox receives that compact skill and
the selected lane context; it does not receive this complete CLI source tree.

## OpenShell boundary and telemetry

`src/hsopt/sandbox/openshell.py` owns the full host-to-sandbox boundary:

- `hsopt-openshell configure` configures the host-side NVIDIA inference
  provider without copying `NVIDIA_API_KEY` into a sandbox;
- `hsopt-openshell mutate` creates one disposable OpenShell sandbox, installs
  the compact skill, exposes only `candidate.py`, the fixed evaluator, and the
  selected benchmark JSON, invokes OpenCode once, downloads the candidate, and
  removes the sandbox;
- every mutation automatically preserves raw `opencode.jsonl` and
  `openshell.log`, plus redacted `agent_actions.jsonl`,
  `opencode_metrics.json`, `openshell_metrics.json`, and `agent_audit.json`;
- `hsopt-openshell audit --run-dir <mutation-run>` deterministically rebuilds
  the redacted action and metric files from the two raw logs.

The redacted metrics count tools by name, online tool calls, web searches,
inference steps, tokens, OpenShell network/process events, and denied events.
They never copy tool outputs, credentials, or request payloads into the summary.

## Mutation-history view

`src/hsopt/view/mutation_history.py` reads the durable campaign directory
without modifying it. It exports a stable `mutation-history.json` model plus a
small dark-mode grid: grey cells are pending, pale green cells are accepted,
pale red cells are rejected, and a yellow star marks the current incumbent.
Selecting iteration 0 or a mutation shows the exact archived policy source.

```bash
uv run hsopt-view --campaign-root /path/to/campaign --serve
```

The JSON model is the reusable boundary for `hsopt.mantilla.ca`; the hosted
site does not need its own interpretation of lane state or decision files.

## Development

From the repository root:

```bash
uv sync
uv run ruff format --check cli/src cli/tests
uv run ruff check cli/src cli/tests
uv run pytest -q
```

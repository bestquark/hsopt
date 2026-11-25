# HSOpt CLI

`skills/hsopt/` is the installable, tested runtime used to prepare and run
HSOpt campaigns. The repository root keeps `pyproject.toml` and `uv.lock` so
every command still runs as `uv run <command>` from the checkout root.

```text
skills/hsopt/
├── SKILL.md          Agent-facing mutation contract
├── references/       This guide, the mutation brief and the reference docs
├── campaign/         Campaign setup, lanes, task records, exports, and validation
├── evaluation/       Fixed compiled-unitary evaluator and diagnostics
├── sandbox/          OpenShell/OpenCode isolation and agent telemetry
├── view/             Read-only mutation history and final report views
└── tgate_opt/        Generic mutation-agent execution helpers
tests/                Unit tests and compact executable evaluator fixtures
```

`experiments/baseline/generate_initial_pf.py` defines the compact PF object and emits
immutable iteration-0 policies. `skills/hsopt/campaign/calibration.py` owns the
Reiher calibration math, sealed records, and baseline parameter preparation;
the generator delegates its existing calibration API to that module.

The agent-facing `$hsopt` instructions are `skills/hsopt/SKILL.md`, beside this
package. A mutation sandbox receives only the compact mutation brief
(`references/mutation.md`) and the selected lane context; it does not receive
this complete source tree.

## OpenShell boundary and telemetry

`skills/hsopt/sandbox/openshell.py` owns the full host-to-sandbox boundary:

- `hsopt-openshell configure` configures the host-side NVIDIA inference
  provider without copying `NVIDIA_API_KEY` into a sandbox;
- `hsopt-openshell mutate` creates one disposable OpenShell sandbox, installs
  the compact skill, exposes only `candidate.py`, the fixed evaluator, and the
  selected benchmark JSON, runs a read-only preparation phase followed by an
  editing continuation in the same OpenCode session, downloads the candidate, and
  removes the sandbox;
- every mutation automatically preserves raw `opencode.jsonl` and
  `openshell.log`, plus redacted `agent_actions.jsonl`,
  `opencode_metrics.json`, `openshell_metrics.json`, and `agent_audit.json`;
- host `mkdir`/`chmod` setup operations record `setup_commands.jsonl`, including
  both the OpenShell exit code and a nonce-verified command-completion receipt.
  Missing receipts and actual command failures stop setup; this does not retry
  commands or change the agent's edit/attempt budgets;
- candidate downloads record `candidate_downloads.jsonl`. Known SSH probe/tar
  exit-status failures receive at most three transfers of the same file within
  the original download timeout. Success requires a zero exit code, followed by
  the unchanged policy and agent-action checks. Transfers do not rerun OpenCode,
  grant another edit, or reset a lane's proposal-attempt budget;
- `hsopt-openshell audit --run-dir <mutation-run>` deterministically rebuilds
  the redacted action and metric files from the raw logs.

The redacted metrics count tools by name, online tool calls, web searches,
inference steps, tokens, OpenShell network/process events, and denied events.
They never copy tool outputs, credentials, or request payloads into the summary.
Each round also records its actual OpenCode/OpenShell versions and requested model
in `runtime_versions.json`. Actions are categorized as file reads/edits, terminal,
web research, skill use, or job submission. Terminal and scheduler tools are denied
in the mutation sandbox; host dispatch events belong to the external runtime.

With OpenShell 0.1.3, inference calls use the native provider endpoint and a
proxy credential placeholder. Import the deployment provider profile before
`hsopt-openshell configure`; the retired `openshell inference` route is not used.
For full structured network/process/security events, enable OCSF JSON logging on
the gateway and set `HSOPT_AUDIT_COLLECTOR`. It receives the sandbox name and
mutation directory, writes `openshell_ocsf.jsonl`, and runs before sandbox deletion.
`audit_collection.json` records collection failures; absent events never imply
that a tool or network request occurred.

Before editing is enabled, the host verifies that the agent read the policy,
the generated `research_context.json`, and every required benchmark input.
Context includes recent lane feedback and the supported policy API. Duplicate
policies and formatting-only changes are rejected before evaluation.

Confirmed permission-blocked candidate writes are audit events, not fatal edits:
the session retains its reads and the host verifies unchanged candidate bytes
before enabling editing. `scratch/` permits disposable notes and test drafts in
either stage. Code Mode (`execute`) supports JavaScript data checks through
permitted tools; nested permissions still apply. Shell/Python execution remains
unavailable. Scratch edits do not count toward the single scored-policy edit.

Transient failures use bounded retries and backoff. A permanent inference
failure blocks new mutations while evaluation collection continues. After
repairing the cause, resume explicitly with an audit reason:

```bash
uv run hsopt-openshell resume --path /path/to/campaign --reason "Inference configuration repaired"
```

The backend-independent `hsopt-handoff` API delivers completed evaluator
receipts, records scientific results, and wakes the lane for the next mutation.
Machine-specific dispatch and monitoring live outside this package; see
[the runtime contract](runtime.md).


## Mutation-history view

`skills/hsopt/view/mutation_history.py` reads the durable campaign directory
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
uv run ruff format --check skills tests
uv run ruff check .
uv run pytest -q
```

# Three ELOTE search strategies

| Experiment | Proposals | Search space |
| --- | --- | --- |
| `agent_only/` | Ordinary OpenCode/OpenShell autoresearch | Complete permitted PF |
| `pysr_only/` | PySR custom fitness, no agent or regression target | Tolerance expression on the initial PF |
| `agent_pysr/` | Agent with optional `hsopt_pysr` tool | Agent proposes structure; PySR tunes its precision equation |

Each method/seed has its own external history, incumbent, queue, traces and held-out
report. Within a seed, all methods use the **same train/test split, iteration-0
policy, per-molecule calibrated parameters, chemical-accuracy constraints and
geometric-mean baseline-relative T95 objective**. Splitting/grouping expressivity
differs between agent and symbolic-only search; this is not a matched-expressivity
comparison. The current implementation compares shared absolute-energy protocols.
Split seeds and PySR RNG seeds are recorded. Agent replicates use independent
stochastic sessions; they do not claim deterministic model-inference seeding.

The budget is new training-suite candidate evaluations, including **every hybrid
PySR probe**, rather than outer agent turns. Baseline and final held-out evaluations
are reported separately. Cached identical-source evaluations do not spend the budget.
All selected training cases are evaluated; a numerical certification failure holds
the search instead of receiving a physical-inaccuracy penalty. Only certified,
accurate strict improvements can replace the incumbent. No test result enters
PySR, the agent context or model selection. Each final training incumbent is frozen
and tested once, even if its test performance is poor.

## Prepare

Use an external selection with cached references and sealed calibrations, as in
[`abs_protocol`](../abs_protocol/README.md). Dataset size, seeds and budget are
explicit, not inherited from a running campaign:

```bash
uv run hsopt-search prepare --selection /work/data/selection.json \
  --output /work/runs/elote-search --baseline tight \
  --policy-profile fermionic_nested --energy-target ground_branch \
  --train-size 80 --test-size 20 --budget 100 --seed 1 --seed 2 --seed 3
```

This prepares three independent runs per seed. It does not submit GPU jobs or
modify an existing campaign. The fixed production judge remains CUDA-Q plus GPU
Lanczos; the earlier tiny CPU pilot is not a production fallback.

## Execute

Install `uv sync --extra symbolic-search` in an appropriate environment, or pass
`--pysr-python /path/to/isolated/bin/python`. The optional dependency is pinned to
PySR 2.7.2; its first use installs/precompiles Julia. Julia runs in a separate child
process to avoid the LLVM collision with chemistry/compiler libraries. Scope Julia
depot/project environment variables to a runtime directory if desired.
Time limits are checked between physical evaluations; an in-flight evaluation is
collected before returning. Set `--agent-timeout` to cover sandbox preparation,
inference and any inner-search GPU queue/evaluation time for the chosen deployment.

```bash
uv run hsopt-search run --run-dir /work/runs/elote-search/agent_only/seed-01
uv run hsopt-search run --run-dir /work/runs/elote-search/pysr_only/seed-01 \
  --pysr-python /work/runtime/envs/pysr/bin/python --pysr-evals-per-call 20
```

The default `--worker external` writes the ordinary portable GPU requests to each
run's `iterations/NNNN/requests.json`, then waits for receipts. An external runtime
dispatches those requests and uses `hsopt.campaign.protocol.record_receipt` to
deliver each result. `--worker local` evaluates with GPU Lanczos on the same GPU
machine; it never selects a CPU evaluator. The final test creates another normal
queue below `held_out/`; resume the same command after its receipts arrive. Scientific
source hashes are pinned at preparation, and input hashes are verified by the
existing shared-protocol machinery.

For the hybrid, expose a run-scoped service address reachable **from the sandbox**:

```bash
uv run hsopt-search run --run-dir /work/runs/elote-search/agent_pysr/seed-01 \
  --pysr-python /work/runtime/envs/pysr/bin/python \
  --listen 127.0.0.1 --port 8787 --tool-url https://runtime.example
```

The deployment forwards `/search` to that service (the URL's host must route to the
specified listener). A random per-process capability token protects the service and
is supplied only to its plugin. OpenShell receives binary-scoped endpoint access,
a read-only OpenCode v2 plugin, and an explicit `hsopt_pysr` permission in the editing
stage. The controller explicitly sets `HSOPT_ENABLE_PYSR=1` for the hybrid arm;
agent-only runs disable and clear this capability even if inherited from the shell.
For a separately configured shared runtime, `mutate.py advance --pysr` and
`--no-pysr` override that switch for the next proposal. Enabling the flag requires
the budgeted service URL and run token; it does not launch PySR by itself.
Network placement, tunnels and GPU scheduling remain external runtime concerns.

The agent can pass a complete proposed `base_policy`, a starting arithmetic `guess`,
an operator subset, expression size and requested inner budget. The host caps the
request and replaces **only** `synthesis_epsilon`; it preserves the proposed
splitting/grouping/repetitions. The tool returns its best certified training source
without editing `candidate.py`. Every probe goes through ordinary acceptance and
is archived, including unsuccessful ones. The agent then makes its usual single
final file edit. A hybrid run may finish its evaluation budget inside a tool call;
its best accepted probe remains the training incumbent.

Features are `c=|coefficient|/c_ref`, `w=support/w_ref`,
`a=|angle| r/(c_ref target_time)`, `d=1` for I/Z-only,
`xy=(X+Y count)/w_ref` and `lc=log(c)`. References use the largest coefficient and
support in the trusted problem. PySR searches arithmetic `+,-,*,/,abs,sqrt` for
`delta`, then returns `epsilon0 * exp(clamp(delta,-8,16))`, clipped to `(0,1)`.
The precision family is explicit and bounded; it does not search arbitrary code.

## Inspect

```bash
uv run hsopt-search report --root /work/runs/elote-search
```

The report separates training and held-out scores, actual evaluations, agent turns
and symbolic calls, tokens, controller wall time and recorded evaluator time.
Use matching GPU resources for timing comparisons; placement is a runtime choice,
not assumed equivalent from the evaluation count. Each ordinary evaluation retains per-case results and acceptance;
`symbolic_searches/call-*/` additionally retains base source, requested limits,
every fitness response, equations, versions and wall time. `agent_turns/` retains
the standard model, token, tool and OpenShell audits. `hsopt_pysr` has its own
`symbolic_search` telemetry category. Compare score against both evaluation count
and elapsed time; equal outer rounds would hide the hybrid's inner-search cost.

# Runtime interface

HSOpt owns the scientific skill, input acquisition/preparation, evaluator,
acceptance rules, file-based trajectory ledger, exports, and `plots/`.
Your external `runtime/` owns compute selection, dispatch queues, scheduling,
staging, environments, retries, and dashboard hosting. It can run tasks on the
agent host or another machine. HSOpt never imports the runtime.

The public interface is `hsopt.campaign.interface`, also available as
`uv run hsopt-handoff`. It uses JSON, requires no scheduler library, and does
not require an additional database. Existing lane manifests, iteration files,
and `state.json` retain their format.

## Three operations

1. **Advance:** `advance_lane(lane)` reconciles completed feedback and proposes
   at most one new mutation. It returns `request` while an evaluation is pending,
   or `request: null` when the lane is complete or needs attention.
2. **Evaluate:** `run_evaluation(request, task_dir=staged_snapshot, backend=...)`
   verifies the task and runs the fixed scientific evaluator. It returns a
   receipt. This operation runs on the chosen evaluation computer.
3. **Deliver:** `deliver_result(lane, receipt)` records the result durably,
   applies acceptance, then invokes the agent for its next proposal. The return
   value includes the next pending request. This operation runs on the agent host.

The runtime can remain idle while compute is queued, use a callback, or poll.
It chooses how to wake up; the result-delivery operation advances the scientific
lane. Each lane stays sequential; different lanes can run independently.

Requests carry `schema: hsopt-evaluation-v1`, task metadata, file SHA256 hashes,
an evaluator-source hash, and `request_sha256`. `task_dir` is a host-local path
hint and is excluded from the fingerprint. Copy the entire task directory and
use `--task-dir` on a remote worker. Use the recorded HSOpt version and locked
dependencies there. Requests and receipts are JSON data, never shell commands.

Receipts bind the same task/request/evaluator identity to the result and retain
execution metadata. A wrong task, changed input, incompatible scoring contract,
or conflicting repeat is rejected. Retrying an identical delivery is idempotent.
The result is saved before invoking the agent. If the process stops during the
wake-up, call `advance` again to resume using the existing trajectory.

## Minimal local runtime

Save this as `hsopt_workdir/runtime/run_local.py` and execute it from your
installed HSOpt environment. The HSOpt lane must already be initialized.

```python
from pathlib import Path
from hsopt.campaign.interface import advance_lane, deliver_result, run_evaluation

lane = Path("../runs/my-campaign/lanes/abs_energy/my-molecule/run_01")
step = advance_lane(lane)
while step["request"] is not None:
    receipt = run_evaluation(step["request"], backend="numpy", accelerator="cpu")
    step = deliver_result(lane, receipt)
print(step["state"])
```

For asynchronous execution, replace the `run_evaluation` call with your submit
operation. Persist its job handle outside the source checkout, retrieve the
receipt after completion, and call `deliver_result`. Retain submission errors
and logs; a failed dispatch is not a scientific evaluation result. Use your
runtime's submission identity to recover an uncertain submission before retrying.

The same worker contract is available without importing Python:

```bash
uv run hsopt-handoff advance --lane /path/to/lane
uv run hsopt-handoff evaluate --request /path/to/request.json \
  --task-dir /path/to/staged-task --backend numpy --accelerator cpu \
  --output /path/to/receipt.json
uv run hsopt-handoff deliver --lane /path/to/lane --receipt /path/to/receipt.json
```

`advance` and `deliver` return JSON with a nested `request`; save that object as
the worker's `request.json`. Add `--no-mutate` to collect results without invoking
the next agent proposal. Evaluator errors remain recorded and use the existing
mutation-failure/backoff rules. Blocked lanes require repair and explicit resume.

## Existing deployments

Retain pinned code and environments for campaigns already running. Synchronizing
`main` does not migrate, restart, or reinterpret those campaigns. Scheduler
adapters and supervisor services must be installed in the external runtime;
their former `hsopt-nvpark` and `hsopt-supervisor` commands are no longer part of
the scientific package.

The old `hsopt-exact` batch commands and `hsopt-campaign-scope` maintenance
command are also outside the package. New runs use the single-task handoff.
The scientific ledger can still read historical batch snapshots; it never
creates, dispatches, merges, or retries batches.

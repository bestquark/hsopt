# Ten-reaction campaign runner

This runner executes the fixed paper comparison between equal-share and
reaction-aware synthesis-error contracts. It creates 20 sequential lanes (two
contracts for each of ten reactions), evaluates iteration 0 as the preserved
baseline, and then records exactly 50 Codex proposals per lane.

Codex mutation runs on Omni. Deterministic Slurm jobs each request one A100
80GB and evaluate five lanes serially through Legate/cuPyNumeric, yielding four
concurrent GPU batches per iteration. Stationary Trotter policies use compiled
one-step diagonalization; evaluator-seeded randomized Trotter, qDRIFT, and
custom schedules are propagated explicitly from the exact ground state. All
paths run on the allocated A100 and NumPy/CPU production scoring is rejected.
Custom ordering may be any exact permutation, and explicit schedule entries may
set a distinct synthesis tolerance per occurrence.

The 42 Hamiltonian cache files are intentionally external to Git. Preparation
requires their directory explicitly and verifies every file against
`../inputs/problem_cache.sha256` before copying it into a new campaign. The
destination must be empty and outside the repository. Preparation also refuses
a dirty checkout or detached HEAD, snapshots the benchmark, evaluator, inputs,
runner, lockfile, and hashes, and pins autoresearch to:

- `@openai/codex@0.144.0`
- model `gpt-5.5`
- reasoning effort `high`

Set the absolute paths once. `SOURCE_RUNNER` is the runner in a clean HSOpt
checkout; `CAMPAIGN_ROOT` and `CLUSTER_ROOT` must be outside that checkout:

```bash
SOURCE_RUNNER=/absolute/path/to/hsopt/experiments/reactions10/runner
SOURCE_ENV="$SOURCE_RUNNER/environment"
HSOPT_DATA_ROOT=/absolute/path/to/hsopt-data
CAMPAIGN_ID=reactions10_paper_YYYYMMDD_COMMIT
CAMPAIGN_ROOT="$HSOPT_DATA_ROOT/paper/reactions10/$CAMPAIGN_ID"
CLUSTER_ROOT=/cluster/path/hsopt/paper/reactions10/$CAMPAIGN_ID
```

Create the locked Python 3.12 environment from the runner's own project and
lockfile. The explicit `--project` makes this command independent of the
caller's current working directory:

```bash
uv sync --locked --project "$SOURCE_ENV"
```

Prepare a campaign with that environment. All paths and cluster settings are
explicit; this runner contains no personal host, account, partition, or
storage defaults.

```bash
uv run --locked --project "$SOURCE_ENV" python "$SOURCE_RUNNER/prepare_campaign.py" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --problem-cache /external/reaction_seed/problem_cache \
  --cluster-root "$CLUSTER_ROOT" \
  --cluster-python "$CLUSTER_ROOT/runner/environment/.venv/bin/python"
```

Preparation snapshots the runner and its lockfile into the campaign. Use that
immutable campaign snapshot—not the source checkout—for verification,
submission, supervision, and export:

```bash
CAMPAIGN_RUNNER="$CAMPAIGN_ROOT/runner"
CAMPAIGN_ENV="$CAMPAIGN_RUNNER/environment"
```

Before any evaluation can be submitted, verify the cached eigensystems and
the actual `ground_state` vectors used for phase matching. Verification also
reconstructs the identity and cached Pauli terms and requires them to reproduce
the dense Hamiltonian. `uv run --locked --project` binds the command to the
snapshotted environment regardless of the current directory:

```bash
uv run --locked --project "$CAMPAIGN_ENV" python \
  "$CAMPAIGN_RUNNER/verify_hamiltonians.py" --campaign-root "$CAMPAIGN_ROOT"
```

Sync iteration 0 without submitting, then create the locked environment on the
cluster from `runner/environment/uv.lock`:

```bash
uv run --locked --project "$CAMPAIGN_ENV" python \
  "$CAMPAIGN_RUNNER/submit_iteration.py" \
  --campaign-root "$CAMPAIGN_ROOT" --iteration 0 \
  --sync-host YOUR_SYNC_HOST --submit-host YOUR_SUBMIT_HOST \
  --account YOUR_SLURM_ACCOUNT --partition YOUR_GPU_PARTITION --workers 1 --sync-only

ssh YOUR_SUBMIT_HOST \
  "uv sync --locked --project '$CLUSTER_ROOT/runner/environment'"
```

If orchestration runs on a workstation where the sync and submit hosts are
reachable only through one SSH config alias, pass that route explicitly to
both runner commands and use the same route for the one-time environment
setup. For example:

```bash
uv run --locked --project "$CAMPAIGN_ENV" python \
  "$CAMPAIGN_RUNNER/submit_iteration.py" \
  --campaign-root "$CAMPAIGN_ROOT" --iteration 0 \
  --sync-host YOUR_SYNC_HOST --submit-host YOUR_SUBMIT_HOST \
  --proxy-jump-host YOUR_JUMP_HOST \
  --account YOUR_SLURM_ACCOUNT --partition YOUR_GPU_PARTITION --workers 1 --sync-only

ssh -J YOUR_JUMP_HOST YOUR_SUBMIT_HOST \
  "uv sync --locked --project '$CLUSTER_ROOT/runner/environment'"

uv run --locked --project "$CAMPAIGN_ENV" python \
  "$CAMPAIGN_RUNNER/supervise_campaign.py" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --sync-host YOUR_SYNC_HOST --submit-host YOUR_SUBMIT_HOST \
  --proxy-jump-host YOUR_JUMP_HOST \
  --account YOUR_SLURM_ACCOUNT --partition YOUR_GPU_PARTITION --evaluator-workers 1
```

The proxy-jump alias is recorded in each durable submission attempt. It is
transport provenance rather than a scientific campaign input, so preparation
and the immutable manifest remain host-independent.

Then run the restart-safe supervisor:

```bash
uv run --locked --project "$CAMPAIGN_ENV" python \
  "$CAMPAIGN_RUNNER/supervise_campaign.py" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --sync-host YOUR_SYNC_HOST \
  --submit-host YOUR_SUBMIT_HOST \
  --account YOUR_SLURM_ACCOUNT \
  --partition YOUR_GPU_PARTITION \
  --evaluator-workers 1
```

The supervisor derives its next action from durable campaign state. Mutation
agents edit only disposable iteration work directories under a
`workspace-write` sandbox. Codex runs with `exec --json`. A close-on-exec
handshake records the low-level wrapper exec, but that alone never consumes a
proposal: the launcher must also observe Codex's own `thread.started` event,
and the completed process evidence must contain the same thread plus exactly
one later `turn.completed` event, nonnegative usage, return code zero, and a
nonempty final message. The launch nonce, lifecycle-event hashes, output-file
hashes, PID/process group, distinct exec PID, command, and work directory are
cross-linked through the immutable intent, launch, process, and mutation
records. Failed npm/npx resolution, startup, timeout, or incomplete turns are
archived with exact file and directory evidence and retried as the same
proposal. Recovery terminates the exact work-dir process group before reading
its files. A lane exception is reported with its lane ID and does not cancel
healthy sibling mutations; rerunning the same iteration skips completed lane
records and resumes only incomplete attempts. The supervisor streams child
output directly, so a nonzero exit retains the actual failure record instead
of collapsing it into an opaque status code. Canonical lane files are never
mutation targets.

Every production Python entry point redirects bytecode into a fresh private
owner-only cache prefix before importing campaign code. Immutable source and
runner snapshots reject local `__pycache__` trees, symlinks, special entries,
undeclared files, and undeclared directories, preventing stale bytecode from
substituting for hashed source. Candidate and problem-cache directories are
likewise exact regular-file trees. Mutation records hash every work file, bind
every work/archive directory, and recompute `changed` from the parent and work
bytes. Evaluation always loads an immutable candidate snapshot. Results bind
the campaign ID and manifest hash and include portable candidate and cache
paths, so export replay does not depend on the cluster's absolute mount point.

Submission uses non-deleting `rsync`, durable local batch intents, and a remote
receipt protected by `flock`. Preparation creates a campaign ID from 32 random
bytes. Before the first sync, the sync host atomically claims the configured
cluster root for that campaign ID and manifest hash; the submit host verifies
the exact same claim before touching receipts or Slurm. A different campaign
cannot reuse that root, while retries from the same campaign are idempotent.
Submission keys, Slurm job names, and receipt paths include the campaign
identity. `squeue`/`sacct` reconciliation prevents a disconnected SSH session
from creating an orphan duplicate. Missing or evaluator-error results enter
`retry_ready` without remutating. Present `status=ok` results with identity,
hash, provenance, or arithmetic drift are fatal.

Collection validates all 20 candidates and results before changing any lane.
Its replayable transaction writes the new state last, so a crash during derived
`best/` or `simple_pf.py` copies is safe. Immutable candidates, results,
decisions, and state—not the convenience copies—are the source of truth.

The campaign is complete only after iterations 0 through 50 are collected for
all 20 lanes. `export_results.py` performs the independent strict export used
for paper artifacts. Export with the same snapshotted locked environment, and
keep the first export external until its completion gates pass:

```bash
uv run --locked --project "$CAMPAIGN_ENV" python \
  "$CAMPAIGN_RUNNER/export_results.py" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --output-dir /external/reactions10_export
```

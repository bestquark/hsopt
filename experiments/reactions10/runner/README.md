# Ten-reaction campaign runner

This runner executes the fixed paper comparison between equal-share and
reaction-aware synthesis-error contracts. It creates 20 sequential lanes (two
contracts for each of ten reactions), evaluates iteration 0 as the preserved
baseline, and then records exactly 50 Codex proposals per lane.

The 42 Hamiltonian cache files are intentionally external to Git. Preparation
requires their directory explicitly and verifies every file against
`../inputs/problem_cache.sha256` before copying it into a new campaign. The
destination must be empty and outside the repository. Preparation also refuses
a dirty checkout or detached HEAD, snapshots the benchmark, evaluator, inputs,
runner, lockfile, and hashes, and pins autoresearch to:

- `@openai/codex@0.144.0`
- model `gpt-5.5`
- reasoning effort `high`

Create the Python 3.12 environment first:

```bash
cd experiments/reactions10/runner/environment
uv sync --frozen
```

Prepare a campaign. All paths and cluster settings are explicit; this runner
contains no personal host, account, partition, or storage defaults.

```bash
python experiments/reactions10/runner/prepare_campaign.py \
  --campaign-root /external/reactions10_run \
  --problem-cache /external/reaction_seed/problem_cache \
  --cluster-root /cluster/path/reactions10_run \
  --cluster-python /cluster/path/reactions10_run/runner/environment/.venv/bin/python
```

Before any evaluation can be submitted, verify the cached eigensystems and the
actual `ground_state` vectors used for phase matching. Verification also
reconstructs the identity and cached Pauli terms and requires them to reproduce
the dense Hamiltonian:

```bash
python /external/reactions10_run/runner/verify_hamiltonians.py \
  --campaign-root /external/reactions10_run
```

Sync iteration 0 without submitting, then create the locked environment on the
cluster from `runner/environment/uv.lock`:

```bash
python /external/reactions10_run/runner/submit_iteration.py \
  --campaign-root /external/reactions10_run --iteration 0 \
  --sync-host YOUR_SYNC_HOST --submit-host YOUR_SUBMIT_HOST \
  --account YOUR_SLURM_ACCOUNT --partition YOUR_CPU_PARTITION --sync-only

ssh YOUR_SUBMIT_HOST \
  'cd /cluster/path/reactions10_run/runner/environment && uv sync --frozen'
```

Then run the restart-safe supervisor:

```bash
python /external/reactions10_run/runner/supervise_campaign.py \
  --campaign-root /external/reactions10_run \
  --sync-host YOUR_SYNC_HOST \
  --submit-host YOUR_SUBMIT_HOST \
  --account YOUR_SLURM_ACCOUNT \
  --partition YOUR_CPU_PARTITION
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
its files. Canonical lane files are never mutation targets.

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
for paper artifacts.

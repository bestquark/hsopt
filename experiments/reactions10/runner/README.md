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
actual `ground_state` vectors used for phase matching:

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
`workspace-write` sandbox. A self-written launcher receipt identifies any live
Codex process group after a crash; recovery terminates that exact work-dir
process before reading its files. Canonical lane files are never mutation
targets. Evaluation always loads an immutable candidate snapshot.

Submission uses non-deleting `rsync`, durable local batch intents, and a remote
receipt protected by `flock`. A unique Slurm job name plus `squeue`/`sacct`
reconciliation prevents a disconnected SSH session from creating an orphan
duplicate. Missing or evaluator-error results enter `retry_ready` without
remutating. Present `status=ok` results with identity, hash, provenance, or
arithmetic drift are fatal.

Collection validates all 20 candidates and results before changing any lane.
Its replayable transaction writes the new state last, so a crash during derived
`best/` or `simple_pf.py` copies is safe. Immutable candidates, results,
decisions, and state—not the convenience copies—are the source of truth.

The campaign is complete only after iterations 0 through 50 are collected for
all 20 lanes. `export_results.py` performs the independent strict export used
for paper artifacts.

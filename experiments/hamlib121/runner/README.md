# HamLib121 Runner

This is a self-contained runner for the fixed five-run HamLib experiment. Copy
it into external campaign workspaces; do not run campaigns inside the Git
checkout.

## Environment

The runner is a nested uv project because its NumPy and pygridsynth versions
differ from the root hsopt environment. Its own `uv.lock` must be committed and
used without re-resolution:

```bash
uv sync --locked
uv sync --extra gpu --locked  # only where the GPU baseline is computed
```

The paper protocol pins the mutation client to `@openai/codex@0.144.0`, the
configured model identifier to `gpt-5.5`, and reasoning effort to `high`.
Campaign provenance records the observed CLI version plus runner and lockfile
hashes. The model identifier is configuration provenance; it is not a claim
that the hosted backend is an immutable dated snapshot.

Create external workspaces named `run_01` through `run_05`, each containing a
copy of this directory. Keep their SQLite databases, lane policies, iterations,
logs, telemetry, virtual environments, and Slurm output external.

Run every command below from inside the copied workspace. In other words,
`scripts/`, `slurm/`, `pyproject.toml`, and `uv.lock` sit directly inside each
`run_XX` directory.

## Prepare a run

Set site-specific values explicitly. Do not encode them in this repository:

```bash
REPLICATE_ROOT="/absolute/path/to/external-workspaces"
CAMPAIGN_ROOT="$REPLICATE_ROOT/run_01"
CLUSTER_ROOT="cluster/path/to/hamlib121/run_01"
HAMLIB_DATA_ROOT="cluster/path/to/hamlib-electronic-data"
SLURM_ACCOUNT="site-account"
GPU_PARTITION="gpu-partition"
CPU_PARTITION="cpu-partition"
SYNC_HOST="data-transfer-host"
SUBMIT_HOST="slurm-submit-host"
SLURM_QUERY_HOST="slurm-query-host"

cd "$CAMPAIGN_ROOT"

uv run --locked python scripts/prepare_campaign.py \
  --source-csv manifests/hamlib_electronic_active_space_instances_le32_qubits.csv \
  --campaign-root "$CAMPAIGN_ROOT" \
  --cluster-root "$CLUSTER_ROOT" \
  --hamlib-data-root "$HAMLIB_DATA_ROOT"
```

Preparation fails unless the exact 401-to-121 manifest invariant holds. Repeat
with distinct campaign and cluster roots for all five runs.

## Establish iteration 0

Before any mutation, submit and collect the deterministic initial policy:

```bash
uv run --locked python scripts/submit_eval_batch.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --iteration 0 \
  --batch-id initial_baseline \
  --all-active \
  --compute-baseline \
  --policy-snapshot initial \
  --account "$SLURM_ACCOUNT" \
  --partition "$GPU_PARTITION" \
  --cpu-partition "$CPU_PARTITION" \
  --sync-host "$SYNC_HOST" \
  --submit-host "$SUBMIT_HOST"
```

The command prints a Slurm job ID. Wait for that job to reach a terminal state
and confirm its evaluator output before collecting:

```bash
uv run --locked python scripts/collect_iteration.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --iteration 0 \
  --sync-host "$SYNC_HOST"
```

After preparing `run_02` through `run_05`, reuse the deterministic baseline and
collect those local copies without contacting cluster storage:

```bash
uv run --locked python scripts/replicate_initial_baseline.py \
  --source-campaign "$REPLICATE_ROOT/run_01" \
  --target-campaign "$REPLICATE_ROOT/run_02" \
  --target-campaign "$REPLICATE_ROOT/run_03" \
  --target-campaign "$REPLICATE_ROOT/run_04" \
  --target-campaign "$REPLICATE_ROOT/run_05"

for run in run_02 run_03 run_04 run_05; do
  uv run --locked python scripts/collect_iteration.py \
    --campaign-root "$REPLICATE_ROOT/$run" \
    --iteration 0 \
    --skip-rsync
done
```

Mutation supervision refuses to start until all 121 baselines are accepted in
that run's database.

## Run 50 proposals per lane

Pass account, hosts, and partitions explicitly for every run:

```bash
uv run --locked python scripts/supervise_streaming_batches.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --start-iteration 1 \
  --iterations 50 \
  --account "$SLURM_ACCOUNT" \
  --partition "$GPU_PARTITION" \
  --cpu-partition "$CPU_PARTITION" \
  --sync-host "$SYNC_HOST" \
  --submit-host "$SUBMIT_HOST" \
  --slurm-query-host "$SLURM_QUERY_HOST" \
  --model gpt-5.5 \
  --reasoning-effort high \
  --codex-bin "npx -y @openai/codex@0.144.0"
```

The supervisor holds a single-process campaign lock and persists every
evaluation batch before contacting Slurm. Restart it with the same protocol
arguments after an interruption: queued candidates, ambiguous submissions,
active job IDs, collection, and evaluation retries resume from the campaign
database without remutating in-flight lanes. Provenance drift fails closed.

Partition fallback is disabled unless `--enable-partition-fallback` is passed
with an explicit `--fallback-partition`. It applies to pending CPU evaluation
batches, so choose a compatible alternate partition.
Only job IDs persisted by that campaign are eligible for rescue.

## Export final results

After all five workspaces reach iteration 50, run the strict exporter without
`--allow-incomplete` and write to a new external destination path:

```bash
uv run --locked python scripts/export_replicate_results.py \
  --replicate-root "$REPLICATE_ROOT" \
  --output-dir "/absolute/path/to/external-final-export" \
  --runs 5 \
  --expected-lanes 121 \
  --expected-iterations 50
```

`finalize_replicates_when_complete.py` can wait for the same fixed gate and
invoke the strict exporter automatically. Both paths produce the same
checksummed `FINAL_EXPORT_COMPLETE` marker. Promote only an export whose
`summary.json` has `complete: true` into the repository `results/` directory.
The strict gate also requires identical model, reasoning, Codex CLI, runner,
and lockfile provenance across all five runs and emits it as `provenance.json`.
`--diagnostic-shape` exists only for external test fixtures and can never
produce a complete paper artifact.

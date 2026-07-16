# HamLib121 Runner

This is a self-contained runner for the fixed paper experiment: five independent
autoresearch replicates over the same A100-memory-selected set of 121 HamLib
Hamiltonians, with 50 proposals per lane. Copy it into external campaign
workspaces; do not run campaigns inside the Git checkout.

Only the shared iteration-0 dense-sector baseline is computed on one NVIDIA
A100 with cuPyNumeric. That verified baseline is copied, with provenance, from
`run_01` into `run_02` through `run_05`. All 30,250 proposal evaluations
(`5 * 121 * 50`) are CPU/NumPy policy evaluations. This is not five full A100
runs.

## Environment

The runner is a nested uv project because its NumPy and pygridsynth versions
differ from the root hsopt environment. Its own `uv.lock` must be committed and
used without re-resolution:

```bash
uv sync --locked
```

Mutation launch requires a preinstalled Node.js 24.x runtime. Put its `bin`
directory first on `PATH`; the runner records the exact Node version, requires
major version 24, pins Codex to `@openai/codex@0.144.0`, and forces npm to the
campaign-local `.codex_npm` cache:

```bash
export PATH="/absolute/path/to/node-v24/bin:$PATH"
node --version  # must be v24.x
mkdir -p -m 700 "$CAMPAIGN_ROOT/.codex_npm"
npx -y @openai/codex@0.144.0 --version
```

Local staging and collection require `rsync` 3.x with a working
`--protect-args` parser. macOS `/usr/bin/rsync` 2.6.9 is not compatible. Install
a current rsync (for example, Homebrew's rsync), then either put it first on
`PATH` or bind it explicitly:

```bash
export HSOPT_RSYNC_BIN="/absolute/path/to/rsync"
RSYNC_BIN="${HSOPT_RSYNC_BIN:-$(command -v rsync)}"
"$RSYNC_BIN" --version
"$RSYNC_BIN" --protect-args --help >/dev/null
```

`submit_eval_batch.py`, `collect_iteration.py`, and
`supervise_streaming_batches.py` also accept `--rsync-bin`. If neither the flag
nor `HSOPT_RSYNC_BIN` is set, they probe `PATH` and common Homebrew locations,
skip incompatible candidates, and fail before any staging or collection. The
resolved real path, version, protocol version, and executable SHA-256 are bound
in `manifests/rsync_provenance.json`; later transfers and supervisor restarts
must match that receipt. Preflight includes a real local copy of a filename with
spaces and shell metacharacters under `--protect-args`, verifies the executable
identity and hash before and after that copy, and repeats the check immediately
before each runner-managed transfer.

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
HSOPT_DATA_ROOT="/absolute/path/to/hsopt-data"
CAMPAIGN_ID="hamlib121_paper_YYYYMMDD_COMMIT"
REPLICATE_ROOT="$HSOPT_DATA_ROOT/paper/hamlib121/$CAMPAIGN_ID"
CAMPAIGN_ROOT="$REPLICATE_ROOT/run_01"
CLUSTER_ROOT="/absolute/cluster/path/to/hsopt/paper/hamlib121/$CAMPAIGN_ID/run_01"
HAMLIB_DATA_ROOT="/absolute/cluster/path/to/hamlib-electronic-data"
HAMLIB_ZIP_SHA256="manifests/hamlib_zip_sha256.txt"
SLURM_ACCOUNT="site-account"
GPU_PARTITION="gpu-partition"
CPU_PARTITION="cpu-partition"
SYNC_HOST="data-transfer-host"
SUBMIT_HOST="slurm-submit-host"
SLURM_QUERY_HOST="slurm-query-host"

cd "$CAMPAIGN_ROOT"

uv run --locked python scripts/prepare_campaign.py \
  --source-csv manifests/hamlib_electronic_active_space_instances_le32_qubits.csv \
  --hamlib-zip-checksums "$HAMLIB_ZIP_SHA256" \
  --campaign-root "$CAMPAIGN_ROOT" \
  --cluster-root "$CLUSTER_ROOT" \
  --hamlib-data-root "$HAMLIB_DATA_ROOT"
```

When the campaign workspace is local and those cluster aliases are reachable
only through one SSH config alias, append `--proxy-jump-host omni` to
`submit_eval_batch.py`, `collect_iteration.py`, and
`supervise_streaming_batches.py`. The supervisor propagates the validated alias
to submission, result collection, `squeue`/`sacct`, and optional partition
rescue children, and records it in `manifests/supervisor_provenance.json`.
Use `ssh -J omni` and `"$RSYNC_BIN" -e 'ssh -J omni'` for the manual
prebuild and archive-preflight commands below. Omitting the flag preserves the
direct-host route.

Preparation fails unless the exact 72,200-byte canonical 401-row source
manifest (`SHA256
ff7f9acce5f5140d65f84ebe33df92276c9058d13a8de4a0ed254b7ad868e489`),
its 401-to-121 selection invariant, and the
checksum file is exactly the canonical 83-line, 9,383-byte
`hamlib_zip_sha256.txt` (`SHA256
2a5958a86a6adaa30bd98b325701830f618f7663ddcd925509213ad3ae187def`).
Workers resolve every ZIP through its bound data-root-relative path, verify each
archive at most once per worker process, and bind its digest into every result.
Repeat preparation with distinct campaign and cluster roots for all five runs;
each run receives a fresh random campaign ID.

## Establish iteration 0

Before any mutation, claim and stage the campaign root without creating a
durable Slurm batch intent:

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
  --submit-host "$SUBMIT_HOST" \
  --sync-only
```

On the sync/login host, prebuild both locked evaluator environments exactly
once. Slurm jobs never resolve or install dependencies; they require these
environments and verify their recorded `uv.lock` digest:

```bash
ssh "$SYNC_HOST" \
  "cd '$CLUSTER_ROOT' && python3 scripts/prebuild_cluster_environments.py --campaign-root '$CLUSTER_ROOT'"
```

On that same host, hash and open the exact 83 bound archives and all 121
selected members. Copy the campaign-bound receipt back before submission:

```bash
ssh "$SYNC_HOST" \
  "cd '$CLUSTER_ROOT' && python3 scripts/verify_hamlib_archives.py \
    --campaign-root '$CLUSTER_ROOT' --hamlib-data-root '$HAMLIB_DATA_ROOT'"

"$RSYNC_BIN" -ah --protect-args \
  "$SYNC_HOST:$CLUSTER_ROOT/manifests/hamlib_archive_preflight.json" \
  "$CAMPAIGN_ROOT/manifests/"
```

Then rerun the same submission command without `--sync-only`:

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

The submit command atomically verifies the campaign ID/manifest claim on both
sync and submit hosts, then prints a Slurm job ID. Wait for that job to reach a terminal state
and confirm its evaluator output before collecting:

```bash
uv run --locked python scripts/collect_iteration.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --iteration 0 \
  --batch-id initial_baseline \
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
that run's database and replays the A100, cuPyNumeric, policy-byte, archive,
batch-intent, and baseline-cache provenance for every lane.

## Run 50 proposals per lane

Pass account, hosts, and partitions explicitly for every run:

```bash
uv run --locked python scripts/supervise_streaming_batches.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --start-iteration 1 \
  --iterations 50 \
  --batch-size 8 \
  --max-concurrent-mutators 4 \
  --max-running-eval-jobs 8 \
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
The paper runner rejects any change from 50 iterations, batch size 8, four
concurrent mutators, eight running evaluator jobs, model `gpt-5.5`, reasoning
`high`, or the pinned Codex command. The complete execution configuration is
recorded in `manifests/supervisor_provenance.json`.

Each Codex proposal runs in a fresh disposable directory containing only
`simple_pf.py`, under `workspace-write` with user configuration and repository
rules disabled. The runner requires a complete JSON lifecycle, terminates the
whole proposal process group, rejects any extra workspace entry, validates the
candidate with the restricted policy loader, and only then atomically promotes
its bytes into the lane. The policy loader permits deterministic math and the
explicit dataclass/typing names used by the API; filesystem, process, network,
reflection, private-attribute, and arbitrary-module capabilities are rejected.

Trust boundary: receipts and strict export detect accidental drift and
single-record tampering, but campaign-owner access is trusted. An owner who
rewrites mutually consistent SQLite rows, result JSON, or installed environment
bytes can defeat local replay checks; the export is not a cryptographic
signature or a full re-evaluation of every proposal.

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

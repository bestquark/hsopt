# HamLib121 Runner

This is a self-contained runner for the fixed paper experiment: five independent
autoresearch replicates over the same A100-memory-selected set of 121 HamLib
Hamiltonians, with 50 proposals per lane. Copy it into external campaign
workspaces; do not run campaigns inside the Git checkout.

Every baseline and proposal is evaluated on an NVIDIA A100 through
Legate/cuPyNumeric. OpenCode/Nemotron mutation agents run on Omni, then invoke the
deterministic submitter; Slurm jobs contain no mutation logic. Each job requests
one A100 and evaluates its HamLib lane subset serially. Each streaming
supervisor accepts a per-replicate limit from one through four; orchestration
must keep the sum across simultaneous HamLib and reaction supervisors at or
below four. No job places concurrent state-vector simulations on one device.

The acceptance metric is the measured HF-reference spectral ground-energy
error of the compiled proposed product formula. Analytic product-formula and
synthesis bounds are retained only as diagnostics.

The default stationary Trotter policy uses a fast path: the evaluator applies
one compiled slice on the A100, measures its spectral phase over
`tau / n_steps`, and scales that phase to the configured base unitary while
retaining physical rotation and T counts for every repetition. Policies may
instead select evaluator-seeded randomized Trotter, qDRIFT, or a deterministic
custom explicit schedule. Explicit schedules are propagated in full, may use
any exact term permutation, and may choose a distinct synthesis epsilon for
each occurrence.

## Environment

The runner is a nested uv project because its NumPy and pygridsynth versions
differ from the root hsopt environment. Its own `uv.lock` must be committed and
used without re-resolution:

```bash
uv sync --locked
```

Mutation launch requires a preinstalled Node.js 24.x runtime. Put its `bin`
directory first on `PATH`; the runner records the exact Node version, requires
major version 24, pins OpenCode to `opencode-ai@1.4.11`, and forces npm to the
campaign-local cache. Load `NVIDIA_API_KEY` from an owner-readable environment
file; never write it into this workspace or its logs:

```bash
export PATH="/absolute/path/to/node-v24/bin:$PATH"
node --version  # must be v24.x
mkdir -p -m 700 "$CAMPAIGN_ROOT/.codex_npm"
npx -y opencode-ai@1.4.11 --version
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

The paper protocol pins the mutation client to `opencode-ai@1.4.11` and the
configured model identifier to `nvidia/nvidia/nemotron-3-super-v3` through
NVIDIA's OpenAI-compatible API.
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
SYNC_HOST="data-transfer-host"
SUBMIT_HOST="slurm-submit-host"

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
  --sync-host "$SYNC_HOST" \
  --submit-host "$SUBMIT_HOST" \
  --sync-only
```

On the sync/login host, prebuild the locked evaluator environment exactly
once. The prebuild keeps uv's mutable cache under the external campaign root,
so a small cluster home-directory quota cannot prevent environment creation.
Slurm jobs never resolve or install dependencies; they require this environment
and verify its recorded `uv.lock` digest:

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
  --max-running-eval-jobs 4 \
  --account "$SLURM_ACCOUNT" \
  --partition "$GPU_PARTITION" \
  --sync-host "$SYNC_HOST" \
  --submit-host "$SUBMIT_HOST" \
  --model nvidia/nvidia/nemotron-3-super-v3
```

The supervisor holds a single-process campaign lock and persists every
evaluation batch before contacting Slurm. Restart it with the same protocol
arguments after an interruption: queued candidates, ambiguous submissions,
active job IDs, collection, and evaluation retries resume from the campaign
database without remutating in-flight lanes. Provenance drift fails closed.
The paper runner rejects any change from 50 iterations, batch size 8, four
concurrent mutators, one-to-four running evaluator jobs, the pinned Nemotron
model, or the pinned OpenCode command. When campaigns run together,
their evaluator-job limits must sum to at most four; the standard layout is
three HamLib supervisors at one job each plus the reaction campaign's single
20-lane batch. The complete execution configuration is recorded in
`manifests/supervisor_provenance.json`.

Each OpenCode proposal runs in a fresh disposable directory containing only
`simple_pf.py`, with every capability denied except read/edit access to that
file. The runner preserves raw OpenCode JSON events inside its compatibility
envelope, requires a complete lifecycle, terminates the
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

Evaluation submission uses blocking `sbatch --wait` calls. Up to four such
calls may be active, one per A100 job; the supervisor performs no `squeue`,
`sacct`, or device-monitoring loop.

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
The strict gate also requires identical model, OpenCode CLI, runner,
and lockfile provenance across all five runs and emits it as `provenance.json`.
`--diagnostic-shape` exists only for external test fixtures and can never
produce a complete paper artifact.

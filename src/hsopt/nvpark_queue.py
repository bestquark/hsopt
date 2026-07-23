"""Submit and collect exact-evaluation batches on NVPark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .exact_queue import merge_batch, prepare_batch, queue_status


DEFAULT_HOST = "nvlogin"
DEFAULT_REMOTE_ROOT = "/lustre/fs12/portfolios/nvr/users/lmantilla/hsopt-exact"
DEFAULT_ACCOUNT = "nvr_qualg_lmbm"
DEFAULT_PARTITION = "polar4"
DEFAULT_QOS = "normal"
DEFAULT_CPUS = 2
DEFAULT_MEMORY = "16G"
DEFAULT_WALLTIME = "01:00:00"
DEFAULT_BACKEND = "cupynumeric"
DEFAULT_ACCELERATOR = "gpu"
DEFAULT_SYSTEM_MEMORY_MIB = 12_000
DEFAULT_GPU_MEMORY_MIB = 8_192


def _run(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _recover_job_id(host: str, job_name: str) -> str | None:
    """Find a uniquely named submitted job after a lost SSH response."""

    queued = _run(
        ["ssh", host, "squeue", "-h", "--name", job_name, "--format=%A"],
    )
    job_ids = [line.strip() for line in queued.stdout.splitlines() if line.strip().isdigit()]
    if job_ids:
        return job_ids[0]
    accounting = _run(
        [
            "ssh",
            host,
            "sacct",
            "-X",
            "-n",
            "--starttime=now-1days",
            "--name",
            job_name,
            "--format=JobIDRaw",
        ],
    )
    job_ids = [
        line.strip()
        for line in accounting.stdout.splitlines()
        if line.strip().isdigit()
    ]
    return job_ids[0] if job_ids else None


def _source_id(repo_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = [repo_dir / "pyproject.toml", repo_dir / "uv.lock", repo_dir / "README.md"]
    paths.extend(sorted((repo_dir / "src").rglob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(repo_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def _evaluator_source_id(repo_dir: Path) -> str:
    """Fingerprint only the fixed scientific evaluator and its locked dependencies."""

    digest = hashlib.sha256()
    paths = [
        repo_dir / "src" / "hsopt" / "compiled_unitary.py",
        repo_dir / "src" / "hsopt" / "exact_queue.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(repo_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def submit_batch(
    batch_dir: Path,
    *,
    repo_dir: Path,
    host: str = DEFAULT_HOST,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    account: str = DEFAULT_ACCOUNT,
    partition: str = DEFAULT_PARTITION,
    qos: str = DEFAULT_QOS,
    cpus: int = DEFAULT_CPUS,
    memory: str = DEFAULT_MEMORY,
    walltime: str = DEFAULT_WALLTIME,
    backend: str = DEFAULT_BACKEND,
    accelerator: str = DEFAULT_ACCELERATOR,
    runtime_python: str | None = None,
) -> dict[str, Any]:
    """Upload one batch and submit an evaluator with explicit CPU/GPU placement."""

    batch_dir = batch_dir.resolve()
    repo_dir = repo_dir.resolve()
    if backend not in {"numpy", "cupynumeric"}:
        raise ValueError("backend must be numpy or cupynumeric")
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be cpu or gpu")
    use_gpu = accelerator == "gpu"
    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    source_id = _source_id(repo_dir)
    evaluator_source_id = _evaluator_source_id(repo_dir)
    remote_source = f"{remote_root}/sources/{source_id}"
    remote_batch = f"{remote_root}/batches/{manifest['batch_id']}"
    submission_path = batch_dir / "submission.json"
    job_name = f"hsopt-{'gpu' if use_gpu else 'cpu'}-{manifest['batch_id']}"
    if submission_path.is_file():
        previous = json.loads(submission_path.read_text(encoding="utf-8"))
        if previous.get("job_id"):
            return previous
        recovered_job_id = _recover_job_id(host, job_name)
        if recovered_job_id:
            previous.update(
                {
                    "status": "submitted",
                    "job_id": recovered_job_id,
                    "recovered_submission": True,
                }
            )
            _write_json(submission_path, previous)
            return previous

    _run(["ssh", host, "mkdir", "-p", remote_source, remote_batch])
    _run(
        [
            "rsync",
            "-azR",
            "./pyproject.toml",
            "./uv.lock",
            "./README.md",
            "./src",
            f"{host}:{remote_source}/",
        ],
        cwd=repo_dir,
    )
    _run(["rsync", "-az", f"{batch_dir}/", f"{host}:{remote_batch}/"])

    legate_cpus = cpus if use_gpu else max(1, cpus - 1)
    framebuffer_memory_mib = DEFAULT_GPU_MEMORY_MIB if use_gpu else 0
    legate_config = (
        f"--cpus={legate_cpus} --gpus={int(use_gpu)} "
        f"--sysmem={DEFAULT_SYSTEM_MEMORY_MIB} --fbmem={framebuffer_memory_mib}"
    )
    environment_parts = [
        "set -euo pipefail;",
        f"cd {shlex.quote(remote_source)};",
    ]
    if runtime_python is None:
        environment_parts.extend(
            [
                "export UV_PROJECT_ENVIRONMENT=.venv;",
                f"export UV_CACHE_DIR={shlex.quote(f'{remote_root}/uv-cache')};",
                f"uv sync {'--extra cupynumeric ' if backend == 'cupynumeric' else ''}--frozen;",
            ]
        )
    else:
        environment_parts.append(f"test -x {shlex.quote(runtime_python)};")
    environment_script = " ".join(environment_parts)
    _run(["ssh", host, f"bash -lc {shlex.quote(environment_script)}"])

    batch_parts = [
        "set -euo pipefail;",
        f"cd {shlex.quote(remote_source)};",
    ]
    if runtime_python is None:
        batch_parts.extend(
            [
                "export UV_PROJECT_ENVIRONMENT=.venv;",
                f"export UV_CACHE_DIR={shlex.quote(f'{remote_root}/uv-cache')};",
            ]
        )
        evaluator_python = "uv run --no-sync python"
    else:
        batch_parts.append(
            f"export PYTHONPATH={shlex.quote(f'{remote_source}/src')};"
        )
        evaluator_python = shlex.quote(runtime_python)
    if use_gpu:
        monitor_path = f"{remote_batch}/gpu-utilization.csv"
        batch_parts.extend(
            [
                "nvidia-smi -L;",
                "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader;",
                (
                    "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,"
                    "utilization.memory,memory.used,power.draw --format=csv -l 30 "
                    f"> {shlex.quote(monitor_path)} 2>&1 &"
                ),
                "HSOPT_GPU_MONITOR_PID=$!;",
                "trap 'kill \"$HSOPT_GPU_MONITOR_PID\" 2>/dev/null || true' EXIT;",
            ]
        )
    if backend == "cupynumeric":
        batch_parts.extend(
            [
                "export LEGATE_AUTO_CONFIG=0;",
                f"export LEGATE_CONFIG={shlex.quote(legate_config)};",
                "export CUPYNUMERIC_FORCE_THUNK=deferred;",
            ]
        )
    batch_parts.append(
        f"{evaluator_python} -m hsopt.exact_queue run-batch "
        f"--batch-dir {shlex.quote(remote_batch)} --backend {backend}"
    )
    batch_script = " ".join(batch_parts)
    job_command = f"bash -lc {shlex.quote(batch_script)}"
    submission = {
        "status": "submitting",
        "job_id": None,
        "job_name": job_name,
        "host": host,
        "remote_root": remote_root,
        "remote_source": remote_source,
        "remote_batch": remote_batch,
        "source_id": source_id,
        "evaluator_source_id": evaluator_source_id,
        "backend": backend,
        "accelerator": accelerator,
        "runtime_python": runtime_python,
        "resources": {
            "nodes": 1,
            "gpus": int(use_gpu),
            "cpus": cpus,
            "memory": memory,
            "walltime": walltime,
            "partition": partition,
            "account": account,
            "qos": qos,
            "legate_config": legate_config if backend == "cupynumeric" else None,
        },
    }
    _write_json(submission_path, submission)
    sbatch = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--partition={partition}",
        f"--qos={qos}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={walltime}",
        f"--job-name={job_name}",
        f"--output={remote_batch}/slurm-%j.out",
        "--wrap",
        job_command,
    ]
    if use_gpu:
        sbatch.insert(7, "--gpus-per-node=1")
    remote_sbatch = " ".join(shlex.quote(part) for part in sbatch)
    submitted = _run(["ssh", host, remote_sbatch])
    job_id = submitted.stdout.strip().split(";", maxsplit=1)[0]
    submission.update({"status": "submitted", "job_id": job_id})
    _write_json(submission_path, submission)
    return submission


def submit_ready(
    queue_dir: Path,
    *,
    repo_dir: Path,
    threshold: int = 40,
    force: bool = False,
    **submit_options: Any,
) -> dict[str, Any]:
    """Submit at most 40 tasks once the local queue reaches the threshold."""

    status = queue_status(queue_dir)
    if status["pending"] < threshold and not force:
        return {
            "status": "waiting",
            "pending": status["pending"],
            "threshold": threshold,
            "needed": threshold - status["pending"],
        }
    prepared = prepare_batch(queue_dir, limit=40)
    submitted = submit_batch(
        Path(prepared["batch_dir"]),
        repo_dir=repo_dir,
        **submit_options,
    )
    return {**prepared, **submitted}


def remote_status(batch_dir: Path) -> dict[str, Any]:
    submission = json.loads((batch_dir.resolve() / "submission.json").read_text(encoding="utf-8"))
    completed = _run(
        [
            "ssh",
            submission["host"],
            "sacct",
            "-X",
            "-j",
            submission["job_id"],
            "--noheader",
            "--parsable2",
            "--format=JobID,State,Elapsed,Start,End",
        ]
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    return {"status": "queried", "job_id": submission["job_id"], "rows": rows}


def collect_batch(queue_dir: Path, batch_dir: Path) -> dict[str, Any]:
    """Download available results and merge them into local external history."""

    batch_dir = batch_dir.resolve()
    submission = json.loads((batch_dir / "submission.json").read_text(encoding="utf-8"))
    host = submission["host"]
    remote_batch = submission["remote_batch"]
    (batch_dir / "results").mkdir(exist_ok=True)
    _run(["rsync", "-az", f"{host}:{remote_batch}/results/", f"{batch_dir}/results/"])
    _run(["rsync", "-az", f"{host}:{remote_batch}/batch.json", f"{batch_dir}/batch.json"])
    logs = subprocess.run(
        ["rsync", "-az", f"{host}:{remote_batch}/slurm-*.out", f"{batch_dir}/"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    utilization_downloaded = False
    if submission.get("accelerator", "gpu") == "gpu":
        utilization = subprocess.run(
            [
                "rsync",
                "-az",
                f"{host}:{remote_batch}/gpu-utilization.csv",
                f"{batch_dir}/",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        utilization_downloaded = utilization.returncode == 0
    merged = merge_batch(queue_dir, batch_dir)
    return {
        **merged,
        "job_id": submission["job_id"],
        "slurm_log_downloaded": logs.returncode == 0,
        "gpu_utilization_downloaded": utilization_downloaded,
    }


def _common_submit_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--qos", default=DEFAULT_QOS)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument(
        "--backend",
        choices=("cupynumeric", "numpy"),
        default=DEFAULT_BACKEND,
    )
    parser.add_argument(
        "--accelerator",
        choices=("gpu", "cpu"),
        default=DEFAULT_ACCELERATOR,
    )
    parser.add_argument(
        "--runtime-python",
        help=(
            "optional prebuilt Python interpreter; evaluates the uploaded source "
            "through PYTHONPATH instead of the source snapshot's uv environment"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit-batch")
    submit.add_argument("--batch-dir", type=Path, required=True)
    _common_submit_flags(submit)

    ready = subparsers.add_parser("submit-ready")
    ready.add_argument("--queue-dir", type=Path, required=True)
    ready.add_argument("--threshold", type=int, default=40)
    ready.add_argument("--force", action="store_true")
    _common_submit_flags(ready)

    status = subparsers.add_parser("status")
    status.add_argument("--batch-dir", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--queue-dir", type=Path, required=True)
    collect.add_argument("--batch-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    values = vars(args)
    command = values.pop("command")
    if command == "submit-batch":
        batch_dir = values.pop("batch_dir")
        result = submit_batch(batch_dir, **values)
    elif command == "submit-ready":
        queue_dir = values.pop("queue_dir")
        result = submit_ready(queue_dir, **values)
    elif command == "status":
        result = remote_status(values["batch_dir"])
    else:
        result = collect_batch(values["queue_dir"], values["batch_dir"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

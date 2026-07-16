#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-runner-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

import numpy as np  # noqa: E402

from campaign_common import (  # noqa: E402
    candidate_path,
    evaluate_candidate,
    ensure_campaign_runner_runtime,
    exact_flat_regular_file_names,
    lane_map,
    mutation_dir,
    now_iso,
    read_json,
    require_campaign,
    result_path,
    result_validation_error,
    sha256_file,
    validate_iteration,
    write_json,
)


def evaluate_one(
    root_text: str,
    lane: dict[str, Any],
    iteration: int,
    campaign_id: str,
    campaign_manifest_sha256: str,
    gpu_preflight: dict[str, Any],
    execution_configuration: dict[str, Any],
) -> dict[str, Any]:
    root = Path(root_text)
    try:
        return evaluate_candidate(
            root,
            lane,
            iteration,
            campaign_id=campaign_id,
            campaign_manifest_sha256=campaign_manifest_sha256,
            gpu_preflight=gpu_preflight,
            execution_configuration=execution_configuration,
        )
    except Exception as exc:
        candidate = candidate_path(root, iteration, str(lane["lane_id"]))
        return {
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "lane_id": lane["lane_id"],
            "reaction_key": lane["reaction_key"],
            "contract": lane["contract"],
            "iteration": int(iteration),
            "status": "error",
            "policy_sha256": sha256_file(candidate) if candidate.is_file() else None,
            "candidate_relative_path": candidate.relative_to(root.resolve()).as_posix(),
            "problem_cache_relative_path": "problem_cache",
            "array_backend": "cupynumeric",
            "gpu_preflight": gpu_preflight,
            "execution_configuration": execution_configuration,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "evaluated_at": now_iso(),
        }


def import_array_backend():
    requested = os.environ.get("CAMPAIGN_ARRAY_BACKEND", "cupynumeric").strip().lower()
    if requested in {"numpy", "cpu"}:
        raise RuntimeError(
            "reaction proposal evaluations require cuPyNumeric/Legate on an allocated A100; "
            "NumPy/CPU proposal scoring is disabled"
        )
    if requested not in {"cupynumeric", "cunumeric", "gpu"}:
        raise RuntimeError(f"unsupported CAMPAIGN_ARRAY_BACKEND={requested!r}")
    try:
        import cupynumeric as xp
    except Exception as exc:
        raise RuntimeError(
            "cuPyNumeric could not be imported; run the worker through Legate on an A100"
        ) from exc
    return xp


def gpu_backend_preflight(*, xp) -> dict[str, Any]:
    backend_module = str(getattr(xp, "__name__", "")).lower()
    if backend_module not in {"cupynumeric", "cunumeric"}:
        raise RuntimeError(f"expected the real cuPyNumeric module, got {backend_module!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(visible_devices) != 1:
        raise RuntimeError(
            f"reaction evaluation requires exactly one allocated GPU; got {visible!r}"
        )
    smi_path_text = os.environ.get("CAMPAIGN_NVIDIA_SMI", "")
    if not smi_path_text or not os.path.isabs(smi_path_text):
        raise RuntimeError("CAMPAIGN_NVIDIA_SMI must name a trusted absolute system tool")
    smi_path = Path(smi_path_text)
    metadata = smi_path.stat()
    if (
        smi_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(smi_path, os.X_OK)
    ):
        raise RuntimeError(f"untrusted NVIDIA system tool: {smi_path}")
    smi = subprocess.run(
        [
            str(smi_path),
            "-i",
            visible_devices[0],
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if smi.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {smi.stderr.strip()}")
    lines = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one queryable GPU, got {lines!r}")
    fields = [field.strip() for field in lines[0].split(",", 3)]
    if len(fields) != 4:
        raise RuntimeError(f"could not parse GPU evidence: {lines[0]!r}")
    gpu_index, gpu_name, gpu_uuid, memory_text = fields
    memory_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s+MiB", memory_text)
    memory_mib = float(memory_match.group(1)) if memory_match else -1.0
    normalized_name = re.sub(r"\s+", "", gpu_name.upper())
    if "A100" not in normalized_name or "80GB" not in normalized_name or memory_mib < 80_000:
        raise RuntimeError(f"reaction campaign requires an A100 80GB; got {lines[0]!r}")
    started = time.perf_counter()
    probe = xp.eye(512, dtype=xp.float64)
    first = float(np.asarray(xp.linalg.eigh(probe)[0])[0])
    if abs(first - 1.0) > 1.0e-12:
        raise RuntimeError(f"cuPyNumeric eigensolver preflight returned {first}")
    return {
        "array_backend": "cupynumeric",
        "array_backend_module": backend_module,
        "cuda_visible_devices": visible,
        "allocated_gpu_count": 1,
        "nvidia_smi": lines,
        "nvidia_smi_path": str(smi_path),
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_memory_mib": memory_mib,
        "accelerator_family": "NVIDIA A100 80GB",
        "preflight_eigh_seconds": time.perf_counter() - started,
        "preflight_array_type": f"{type(probe).__module__}.{type(probe).__name__}",
    }


def lane_ids_from_file(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError("lane ID file is empty")
    if len(values) != len(set(values)):
        raise ValueError("lane ID file contains duplicates")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one immutable reaction batch shard.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--lane-ids-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    validate_iteration(args.iteration)

    root = args.campaign_root.resolve()
    manifest, state = require_campaign(root)
    campaign_id = str(manifest["campaign_id"])
    campaign_manifest_sha256 = str(state["manifest_sha256"])
    if state.get("campaign_id") != campaign_id:
        raise RuntimeError("campaign identity drift")
    ensure_campaign_runner_runtime(root, Path(__file__), manifest)
    by_id = lane_map(manifest)
    lane_ids = lane_ids_from_file(args.lane_ids_file)
    unknown = sorted(set(lane_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown lane IDs: {unknown}")
    lanes = [by_id[lane_id] for lane_id in lane_ids]
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        candidate = candidate_path(root, args.iteration, lane_id)
        record_path = mutation_dir(root, args.iteration, lane_id) / "record.json"
        if not candidate.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"candidate/record missing for {lane_id}")
        if exact_flat_regular_file_names(candidate.parent, "immutable candidate directory") != [
            "simple_pf.py"
        ]:
            raise RuntimeError(f"candidate directory is not exact for {lane_id}")
        record = read_json(record_path)
        if record.get("candidate_policy_sha256") != sha256_file(candidate):
            raise RuntimeError(f"candidate hash drift for {lane_id}")

    lanes.sort(
        key=lambda lane: (len(lane["species"]), int(lane["seed_total_t_gate_count"])),
        reverse=True,
    )
    if args.workers != 1:
        parser.error("one-A100 reaction jobs evaluate lanes serially; --workers must be 1")
    xp = import_array_backend()
    gpu_preflight = gpu_backend_preflight(xp=xp)
    execution_configuration = {
        "execution_mode": "gpu_reaction",
        "array_backend": "cupynumeric",
        "batch_id": os.environ.get("BATCH_ID", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "legate_fbmem": os.environ.get("CAMPAIGN_LEGATE_FBMEM", ""),
    }
    print(
        f"pid={os.getpid()} iteration={args.iteration} lanes={len(lanes)} workers=1 "
        f"gpu={gpu_preflight['gpu_name']}",
        flush=True,
    )
    errors = 0
    pending: list[dict[str, Any]] = []
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        candidate_hash = sha256_file(candidate_path(root, args.iteration, lane_id))
        existing = result_path(root, args.iteration, lane_id)
        if (
            existing.is_file()
            and result_validation_error(
                existing,
                lane=lane,
                iteration=args.iteration,
                candidate_hash=candidate_hash,
                campaign_id=campaign_id,
                campaign_manifest_sha256=campaign_manifest_sha256,
            )
            is None
        ):
            print(f"{lane_id} status=already_valid", flush=True)
            continue
        pending.append(lane)

    for lane in pending:
        result = evaluate_one(
            str(root),
            lane,
            args.iteration,
            campaign_id,
            campaign_manifest_sha256,
            gpu_preflight,
            execution_configuration,
        )
        if result.get("status") != "ok":
            errors += 1
        destination = result_path(root, args.iteration, str(result["lane_id"]))
        write_json(destination, result)
        print(
            f"{result['lane_id']} status={result.get('status')} "
            f"pass={result.get('passes_contract')} T={result.get('total_t_gate_count')}",
            flush=True,
        )
    if errors:
        raise SystemExit(f"{errors} of {len(pending)} attempted lane evaluations failed")


if __name__ == "__main__":
    main()

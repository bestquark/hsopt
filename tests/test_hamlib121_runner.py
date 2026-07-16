from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_ROOT = REPO_ROOT / "experiments" / "hamlib121" / "runner"
SCRIPTS_ROOT = RUNNER_ROOT / "scripts"
SOURCE_MANIFEST = (
    RUNNER_ROOT / "manifests" / "hamlib_electronic_active_space_instances_le32_qubits.csv"
)
ACTIVE_MANIFEST = RUNNER_ROOT / "manifests" / "active_lanes.csv"
ZIP_CHECKSUM_MANIFEST = RUNNER_ROOT / "manifests" / "hamlib_zip_sha256.txt"
SOURCE_MANIFEST_SHA256 = "ff7f9acce5f5140d65f84ebe33df92276c9058d13a8de4a0ed254b7ad868e489"
ACTIVE_MANIFEST_SHA256 = "32c9c2c117f11894df9e4999f5ee1a9bffd94ef41558a158a17ca0c634cc47fe"


def _load_runner_module(stem: str) -> ModuleType:
    path = SCRIPTS_ROOT / f"{stem}.py"
    module_name = f"_hsopt_hamlib121_test_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_common = sys.modules.get("campaign_common")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(SCRIPTS_ROOT))
        if stem != "campaign_common" and "CAMPAIGN_COMMON" in globals():
            sys.modules["campaign_common"] = globals()["CAMPAIGN_COMMON"]
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        if stem != "campaign_common" and "CAMPAIGN_COMMON" in globals():
            if previous_common is None:
                sys.modules.pop("campaign_common", None)
            else:
                sys.modules["campaign_common"] = previous_common
    return module


CAMPAIGN_COMMON = _load_runner_module("campaign_common")
EVALUATE_WORKER = _load_runner_module("evaluate_worker")
PREPARE_CAMPAIGN = _load_runner_module("prepare_campaign")
COLLECT_ITERATION = _load_runner_module("collect_iteration")
MUTATE_ITERATION = _load_runner_module("mutate_iteration")
STREAM_SUPERVISOR = _load_runner_module("supervise_streaming_batches")
FINALIZE_REPLICATES = _load_runner_module("finalize_replicates_when_complete")
EXPORT_REPLICATES = _load_runner_module("export_replicate_results")
RESCUE_PARTITIONS = _load_runner_module("rescue_pending_eval_partitions")
SUBMIT_EVAL = _load_runner_module("submit_eval_batch")
REPLICATE_BASELINE = _load_runner_module("replicate_initial_baseline")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = "directory" if path.is_dir() else _sha256(path)
    return snapshot


def _copy_runner_provenance_sources(root: Path) -> None:
    for source in [*CAMPAIGN_COMMON.runner_bundle_paths(RUNNER_ROOT), RUNNER_ROOT / "uv.lock"]:
        relative = source.relative_to(RUNNER_ROOT)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_archive_preflight_receipt(root: Path) -> None:
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(root)
    runner = CAMPAIGN_COMMON.validate_runner_provenance(root)
    CAMPAIGN_COMMON.write_json(
        root / "manifests" / "hamlib_archive_preflight.json",
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_sha256": identity["campaign_manifest_sha256"],
            "runner_provenance": runner,
            "hamlib_zip_checksums_sha256": manifest["hamlib_zip_checksums_sha256"],
            "active_lanes_verified": 121,
            "archives_verified": 83,
            "archive_keys": sorted(
                str(record["archive_key"]) for record in manifest["hamlib_archives"]
            ),
        },
    )


def _write_synthetic_export_run(root: Path) -> None:
    _copy_runner_provenance_sources(root)
    runner_provenance = CAMPAIGN_COMMON.build_runner_provenance(root)
    CAMPAIGN_COMMON.write_json(root / "manifests" / "runner_provenance.json", runner_provenance)
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    CAMPAIGN_COMMON.ensure_rsync_provenance(root, rsync_provenance)
    CAMPAIGN_COMMON.write_json(
        root / "manifests" / "autoresearch_provenance.json",
        {
            "schema_version": 1,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "codex_cli_version": "0.144.0",
            "node_version": "24.11.1",
            "npm_cache": ".codex_npm",
            "codex_exec_flags": list(CAMPAIGN_COMMON.PAPER_CODEX_EXEC_FLAGS),
            **{key: runner_provenance[key] for key in ("runner_bundle_sha256", "uv_lock_sha256")},
        },
    )
    lane_id = "0000_test_lane"
    initial_policy = root / "lanes" / lane_id / "initial_simple_pf.py"
    candidate_policy = (
        root / "iterations" / "iter_0001" / "mutations" / lane_id / "after_simple_pf.py"
    )
    initial_policy.parent.mkdir(parents=True)
    candidate_policy.parent.mkdir(parents=True)
    initial_policy.write_text("# initial\n", encoding="utf-8")
    candidate_policy.write_text("# improved\n", encoding="utf-8")

    nonce = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    campaign_id = CAMPAIGN_COMMON.derive_campaign_id(nonce)
    archive_record = {
        "archive_key": "synthetic:lane.zip",
        "relative_path": "synthetic/lane.zip",
        "sha256": "a" * 64,
    }
    campaign_manifest = {
        "schema_version": 2,
        "preparation_nonce": nonce,
        "campaign_id": campaign_id,
        "hamlib_archives": [archive_record],
        "lanes": [
            {
                "lane_id": lane_id,
                "collection": "synthetic",
                "file": "lane.hdf5",
                "dataset": "hamiltonian",
                "molecule": "H2",
                "nqubits": 4,
                "electrons": 2,
                "terms": 15,
                "one_norm": 1.0,
                "exact_fit": True,
            }
        ],
    }
    campaign_manifest_path = root / "manifests" / "campaign_manifest.json"
    CAMPAIGN_COMMON.write_json(campaign_manifest_path, campaign_manifest)
    campaign_manifest_sha256 = _sha256(campaign_manifest_path)
    CAMPAIGN_COMMON.write_json(
        root / "manifests" / "campaign_identity.json",
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
        },
    )
    archive_preflight_path = root / "manifests" / "hamlib_archive_preflight.json"
    CAMPAIGN_COMMON.write_json(
        archive_preflight_path,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "archives_verified": 1,
        },
    )
    archive_preflight_sha256 = _sha256(archive_preflight_path)
    CAMPAIGN_COMMON.write_json(
        root / "manifests" / "supervisor_provenance.json",
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "start_iteration": 1,
            "proposal_iterations": 1,
            "batch_size": 1,
            "max_running_eval_jobs": 1,
            "max_concurrent_mutators": 1,
            "max_batch_wait_seconds": 1,
            "max_eval_retries": 1,
            "max_mutation_retries": 1,
            "mutation_retry_base_seconds": 1,
            "mutation_retry_max_seconds": 1,
            "poll_seconds": 1,
            "rescue_poll_seconds": 1,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "codex_bin": "npx -y @openai/codex@0.144.0",
            "codex_exec_flags": list(CAMPAIGN_COMMON.PAPER_CODEX_EXEC_FLAGS),
            "npm_cache": ".codex_npm",
            "python_environments": [".venv", ".venv_cpu"],
            "account": "test-account",
            "gpu_partition": "test-gpu",
            "cpu_partition": "test-cpu",
            "fallback_enabled": False,
            "fallback_partition": None,
            "sync_host": "test-sync",
            "submit_host": "test-submit",
            "slurm_query_host": "test-query",
            "proxy_jump_host": None,
            "rsync": rsync_provenance,
            "legate_fbmem": "70000",
            "uv_base_dir": None,
            "uv_cache_dir": None,
            "uv_python_install_dir": None,
        },
    )

    baseline_execution = {
        "schema_version": 1,
        "execution_mode": "gpu_baseline",
        "array_backend": "cupynumeric",
        "compute_baseline": True,
        "policy_snapshot": "initial",
        "batch_id": "initial_baseline",
        "submission_key": "0" * 12,
        "job_name": f"hpf_{'0' * 12}",
        "slurm_job_id": "1000",
        "slurm_account": "test-account",
        "slurm_partition": "test-gpu",
        "legate_fbmem": "70000",
        "uv_base_dir": None,
        "uv_cache_dir": None,
        "uv_python_install_dir": None,
        "hamlib_archive_preflight_sha256": archive_preflight_sha256,
        "worker_id": 0,
        "num_workers": 1,
    }
    baseline = {
        "baseline_schema_version": 2,
        "baseline_status": "computed",
        "baseline_backend": "cupynumeric",
        "baseline_accelerator": "NVIDIA A100",
        "baseline_build_backend": "numpy_host_dense_then_cupynumeric_transfer",
        "baseline_array_type": "cupynumeric.ndarray",
        "ground_state_energy_hartree": -1.0,
        "sector_dim": 6,
        "hamlib_archive": archive_record,
        "initial_policy_sha256": _sha256(initial_policy),
        "baseline_computed_campaign_id": campaign_id,
        "baseline_computed_manifest_sha256": campaign_manifest_sha256,
        "gpu_preflight": {
            "accelerator_family": "NVIDIA A100 80GB",
            "allocated_gpu_count": 1,
            "array_backend": "cupynumeric",
            "array_backend_module": "cupynumeric",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "gpu_memory_mib": 81920.0,
            "nvidia_smi_path": "/usr/bin/nvidia-smi",
            "preflight_array_type": "cupynumeric.ndarray",
        },
        "baseline_execution_configuration": baseline_execution,
        "baseline_runner_provenance": runner_provenance,
    }
    baseline_path = root / "baselines" / f"{lane_id}.json"
    CAMPAIGN_COMMON.write_json(baseline_path, baseline)

    def execution(iteration: int) -> tuple[dict[str, object], dict[str, object], str]:
        is_baseline = iteration == 0
        key = str(iteration) * 12
        batch_id = "initial_baseline" if is_baseline else "iter0001_batch0001"
        mode = "gpu_baseline" if is_baseline else "cpu_policy"
        backend = "cupynumeric" if is_baseline else "numpy"
        snapshot = "initial" if is_baseline else "current"
        partition = "test-gpu" if is_baseline else "test-cpu"
        policy = initial_policy if is_baseline else candidate_policy
        worker = {
            "schema_version": 1,
            "execution_mode": mode,
            "array_backend": backend,
            "compute_baseline": is_baseline,
            "policy_snapshot": snapshot,
            "batch_id": batch_id,
            "submission_key": key,
            "job_name": f"hpf_{key}",
            "slurm_job_id": str(1000 + iteration),
            "slurm_account": "test-account",
            "slurm_partition": partition,
            "legate_fbmem": "70000" if is_baseline else None,
            "uv_base_dir": None,
            "uv_cache_dir": None,
            "uv_python_install_dir": None,
            "hamlib_archive_preflight_sha256": archive_preflight_sha256,
            "worker_id": 0,
            "num_workers": 1,
        }
        submitted = {
            "schema_version": 1,
            "execution_mode": mode,
            "array_backend": backend,
            "compute_baseline": is_baseline,
            "policy_snapshot": snapshot,
            "account": "test-account",
            "partition": partition,
            "slurm_script": (
                "slurm/evaluate_batch.sbatch" if is_baseline else "slurm/evaluate_batch_cpu.sbatch"
            ),
            "legate_fbmem": "70000" if is_baseline else None,
            "uv_base_dir": None,
            "uv_cache_dir": None,
            "uv_python_install_dir": None,
            "hamlib_archive_preflight_sha256": archive_preflight_sha256,
        }
        batch = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "batch_id": batch_id,
            "iteration": iteration,
            "lane_ids": [lane_id],
            "policy_snapshot": snapshot,
            "policy_sha256s": {lane_id: _sha256(policy)},
            "submission_key": key,
            "job_name": f"hpf_{key}",
            "execution_configuration": submitted,
            "runner_provenance": runner_provenance,
        }
        batch_path = (
            root
            / "iterations"
            / f"iter_{iteration:04d}"
            / "batches"
            / batch_id
            / "batch_manifest.json"
        )
        CAMPAIGN_COMMON.write_json(batch_path, batch)
        return worker, submitted, _sha256(batch_path)

    def result(total_t: int, *, incumbent: bool, iteration: int) -> dict[str, object]:
        policy = initial_policy if iteration == 0 else candidate_policy
        worker_execution, _submitted, batch_hash = execution(iteration)
        payload: dict[str, object] = {
            "lane_id": lane_id,
            "iteration": iteration,
            "status": "ok",
            "passes_chemical_accuracy_bound": True,
            "total_t_gates_qpe": total_t,
            "total_error_bound_hartree": 0.0005,
            "pf_error_bound_hartree": 0.0002,
            "synthesis_error_bound_hartree": 0.0002,
            "qpe_grid_spacing_hartree": 0.0001,
            "qpe_bits": 5,
            "qpe_unitary_applications": 31,
            "policy_path": str(policy),
            "policy_snapshot": "initial" if iteration == 0 else "current",
            "policy_sha256": _sha256(policy),
            "initial_policy_sha256": _sha256(initial_policy),
            "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "runner_provenance": runner_provenance,
            "hamlib_archive": archive_record,
            "baseline_cache_sha256": _sha256(baseline_path),
            "execution_configuration": worker_execution,
            "batch_manifest_sha256": batch_hash,
            **baseline,
            "config": {
                "formula_order": 2,
                "n_steps": 10 if not incumbent else 9,
                "term_order_strategy": "coefficient_desc",
                "uniform_epsilon": 1e-6,
                "zero_angle_threshold": 0.0,
                "name": "synthetic",
            },
        }
        if incumbent:
            payload.update(
                {
                    "incumbent_iteration": 0,
                    "incumbent_t_gates_qpe": 100,
                }
            )
        return payload

    conn = CAMPAIGN_COMMON.db_connect(root)
    CAMPAIGN_COMMON.init_db(conn)
    conn.execute(
        """
        INSERT INTO lanes(
          lane_id, manifest_index, collection, file, dataset, molecule, nqubits,
          electrons, terms, one_norm, exact_fit, fit_reason, sector_dim,
          estimated_state_workspace_gib, estimated_sector_workspace_gib, active,
          iteration, current_policy_path, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lane_id,
            0,
            "synthetic",
            "lane.hdf5",
            "hamiltonian",
            "H2",
            4,
            2,
            15,
            1.0,
            1,
            "fits",
            6,
            0.01,
            0.01,
            1,
            1,
            str(candidate_policy),
            "2026-07-15T00:00:00Z",
        ),
    )
    for iteration in (0, 1):
        key = str(iteration) * 12
        batch_id = "initial_baseline" if iteration == 0 else "iter0001_batch0001"
        conn.execute(
            """
            INSERT INTO eval_batches(
              batch_id, iteration, lane_ids_json, submission_key, job_name, job_id,
              state, created_at, updated_at, submitted_at, collected_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'collected', ?, ?, ?, ?)
            """,
            (
                batch_id,
                iteration,
                json.dumps([lane_id]),
                key,
                f"hpf_{key}",
                str(1000 + iteration),
                "2026-07-15T00:00:00Z",
                "2026-07-15T00:00:00Z",
                "2026-07-15T00:00:00Z",
                "2026-07-15T00:00:00Z",
            ),
        )
    mutation_summary = f"See [candidate](file://{candidate_policy})."
    for iteration, total_t, accepted, policy, incumbent in (
        (0, 100, 1, initial_policy, False),
        (1, 90, 1, candidate_policy, True),
    ):
        payload = result(total_t, incumbent=incumbent, iteration=iteration)
        result_path = root / "iterations" / f"iter_{iteration:04d}" / "results" / f"{lane_id}.json"
        CAMPAIGN_COMMON.write_json(result_path, payload)
        conn.execute(
            """
            INSERT INTO evaluations(
              lane_id, iteration, status, total_t_gates_qpe, result_json,
              passes_accuracy, accepted, incumbent_iteration,
              incumbent_t_gates_qpe, decision_reason, policy_path,
              mutation_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lane_id,
                iteration,
                "ok",
                total_t,
                json.dumps(payload, sort_keys=True),
                1,
                accepted,
                0 if incumbent else None,
                100 if incumbent else None,
                "strict_t_gate_improvement" if incumbent else "initial_baseline",
                str(policy),
                mutation_summary if incumbent else "deterministic baseline",
                "2026-07-15T00:00:00Z",
            ),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def prepared_campaign(tmp_path: Path) -> Path:
    campaign_root = tmp_path / "campaign"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_campaign.py"),
            "--source-csv",
            str(SOURCE_MANIFEST),
            "--campaign-root",
            str(campaign_root),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--hamlib-data-root",
            "/data/hamlib-test",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    _copy_runner_provenance_sources(campaign_root)
    _write_archive_preflight_receipt(campaign_root)
    return campaign_root


def test_fixed_manifest_selects_exact_121_lanes_from_401_rows() -> None:
    assert _sha256(SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA256
    assert _sha256(ACTIVE_MANIFEST) == ACTIVE_MANIFEST_SHA256

    source_rows = CAMPAIGN_COMMON.read_manifest_rows(SOURCE_MANIFEST, max_qubits=32)
    lanes = [
        CAMPAIGN_COMMON.fit_lane(row, index, memory_gib=64.0, workspace_factor=4.0)
        for index, row in enumerate(source_rows)
    ]
    active_lanes = [lane for lane in lanes if lane.exact_fit]
    expected_rows = list(csv.DictReader(ACTIVE_MANIFEST.open(encoding="utf-8")))

    assert len(source_rows) == 401
    assert len(active_lanes) == 121
    assert len(expected_rows) == 121
    assert [lane.lane_id for lane in active_lanes] == [row["lane_id"] for row in expected_rows]
    PREPARE_CAMPAIGN.validate_benchmark_selection(
        source_rows,
        active_lanes,
        ACTIVE_MANIFEST,
    )


def test_canonical_83_zip_manifest_is_exact_and_immutable(tmp_path: Path) -> None:
    assert ZIP_CHECKSUM_MANIFEST.stat().st_size == 9_383
    assert _sha256(ZIP_CHECKSUM_MANIFEST) == (
        "2a5958a86a6adaa30bd98b325701830f618f7663ddcd925509213ad3ae187def"
    )
    with ACTIVE_MANIFEST.open(encoding="utf-8") as handle:
        lanes = list(csv.DictReader(handle))
    records = CAMPAIGN_COMMON.parse_hamlib_zip_checksums(ZIP_CHECKSUM_MANIFEST, lanes)
    assert len(records) == 83
    assert [record["archive_key"] for record in records] == sorted(
        record["archive_key"] for record in records
    )
    CAMPAIGN_COMMON.validate_canonical_hamlib_zip_checksum_manifest(ZIP_CHECKSUM_MANIFEST)

    drifted = tmp_path / "hamlib_zip_sha256.txt"
    payload = ZIP_CHECKSUM_MANIFEST.read_text(encoding="utf-8")
    drifted.write_text(("0" if payload[0] != "0" else "1") + payload[1:], encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum manifest drift"):
        CAMPAIGN_COMMON.validate_canonical_hamlib_zip_checksum_manifest(drifted)

    unsafe = tmp_path / "unsafe.txt"
    unsafe.write_text("a" * 64 + "  ../escape.zip\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe/noncanonical"):
        CAMPAIGN_COMMON.parse_hamlib_zip_checksums(unsafe, lanes)


def test_archive_verification_is_memoized_and_checks_member_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "hamlib"
    archive = data_root / "synthetic" / "lane.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("lane.hdf5", b"synthetic hdf5 payload")
    lane = {"collection": "synthetic", "file": "lane.hdf5"}
    record = {
        "archive_key": "synthetic:lane.zip",
        "relative_path": "synthetic/lane.zip",
        "sha256": _sha256(archive),
    }
    CAMPAIGN_COMMON._VERIFIED_ARCHIVE_CACHE.clear()
    CAMPAIGN_COMMON._VERIFIED_ARCHIVE_MEMBER_CACHE.clear()
    real_sha256 = CAMPAIGN_COMMON.sha256_file
    calls = 0

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(path)

    monkeypatch.setattr(CAMPAIGN_COMMON, "sha256_file", counted)
    first = CAMPAIGN_COMMON.validate_hamlib_archive(
        data_root=data_root, lane=lane, archive_records=[record]
    )
    second = CAMPAIGN_COMMON.validate_hamlib_archive(
        data_root=data_root, lane=lane, archive_records=[record]
    )
    assert first == second
    assert calls == 1

    stored = data_root / "synthetic" / "stored.zip"
    with zipfile.ZipFile(stored, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("stored.hdf5", b"payload")
    stored_lane = {"collection": "synthetic", "file": "stored.hdf5"}
    stored_record = {
        "archive_key": "synthetic:stored.zip",
        "relative_path": "synthetic/stored.zip",
        "sha256": real_sha256(stored),
    }
    with pytest.raises(RuntimeError, match="unencrypted deflated"):
        CAMPAIGN_COMMON.validate_hamlib_archive(
            data_root=data_root,
            lane=stored_lane,
            archive_records=[stored_record],
        )


def test_runner_and_agent_provenance_are_pinned_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "@openai/codex@0.144.0" in CAMPAIGN_COMMON.DEFAULT_CODEX_BIN
    assert "@latest" not in CAMPAIGN_COMMON.DEFAULT_CODEX_BIN
    assert CAMPAIGN_COMMON.parse_codex_cli_version("codex-cli 0.144.4") == "0.144.4"
    assert CAMPAIGN_COMMON.parse_codex_cli_version("OpenAI Codex v0.144.0") == "0.144.0"
    with pytest.raises(RuntimeError, match="could not parse"):
        CAMPAIGN_COMMON.parse_codex_cli_version("unknown build")

    run_a = tmp_path / "copy-a"
    run_b = tmp_path / "copy-b"
    _copy_runner_provenance_sources(run_a)
    _copy_runner_provenance_sources(run_b)
    original = CAMPAIGN_COMMON.build_runner_provenance(run_a)
    assert CAMPAIGN_COMMON.build_runner_provenance(run_b) == original

    script = run_b / "scripts" / "campaign_common.py"
    script.write_text(
        script.read_text(encoding="utf-8") + "\n# provenance drift\n", encoding="utf-8"
    )
    script_drift = CAMPAIGN_COMMON.build_runner_provenance(run_b)
    assert script_drift["runner_bundle_sha256"] != original["runner_bundle_sha256"]
    assert script_drift["uv_lock_sha256"] == original["uv_lock_sha256"]

    _copy_runner_provenance_sources(run_b)
    lock = run_b / "uv.lock"
    lock.write_text(lock.read_text(encoding="utf-8") + "\n# lock drift\n", encoding="utf-8")
    lock_drift = CAMPAIGN_COMMON.build_runner_provenance(run_b)
    assert lock_drift["runner_bundle_sha256"] == original["runner_bundle_sha256"]
    assert lock_drift["uv_lock_sha256"] != original["uv_lock_sha256"]

    campaign = tmp_path / "campaign"
    _copy_runner_provenance_sources(campaign)
    CAMPAIGN_COMMON.ensure_runner_provenance(campaign_root=campaign, runner_root=campaign)
    monkeypatch.setattr(
        CAMPAIGN_COMMON,
        "probe_codex_cli_version",
        lambda *_args, **_kwargs: "0.144.0",
    )
    monkeypatch.setattr(CAMPAIGN_COMMON, "probe_node_version", lambda: "24.11.1")
    recorded = CAMPAIGN_COMMON.ensure_autoresearch_provenance(
        campaign_root=campaign,
        model="gpt-5.5",
        reasoning_effort="high",
        codex_bin=CAMPAIGN_COMMON.DEFAULT_CODEX_BIN,
    )
    assert recorded["npm_cache"] == ".codex_npm"
    assert (
        CAMPAIGN_COMMON.ensure_autoresearch_provenance(
            campaign_root=campaign,
            model="gpt-5.5",
            reasoning_effort="high",
            codex_bin=CAMPAIGN_COMMON.DEFAULT_CODEX_BIN,
        )
        == recorded
    )
    with pytest.raises(RuntimeError, match="provenance drifted"):
        CAMPAIGN_COMMON.ensure_autoresearch_provenance(
            campaign_root=campaign,
            model="different-model",
            reasoning_effort="high",
            codex_bin=CAMPAIGN_COMMON.DEFAULT_CODEX_BIN,
        )

    (campaign / "manifests" / "autoresearch_provenance.json").unlink()
    evidence = campaign / "iterations" / "iter_0001" / "mutations" / "lane"
    evidence.mkdir(parents=True)
    (evidence / "after_simple_pf.py").write_text("# candidate\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to backfill"):
        CAMPAIGN_COMMON.ensure_autoresearch_provenance(
            campaign_root=campaign,
            model="gpt-5.5",
            reasoning_effort="high",
            codex_bin=CAMPAIGN_COMMON.DEFAULT_CODEX_BIN,
        )


def test_prepare_campaign_creates_121_active_lanes_and_policies(
    prepared_campaign: Path,
) -> None:
    manifest = json.loads(
        (prepared_campaign / "manifests" / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requested_count"] == 401
    assert manifest["target_fit_count"] == 121
    assert manifest["active_fit_count"] == 121
    assert manifest["cluster_root"] == "/cluster/hsopt-test"
    assert manifest["hamlib_data_root"] == "/data/hamlib-test"
    assert len(manifest["hamlib_archives"]) == 83
    assert manifest["hamlib_zip_checksums_sha256"] == _sha256(ZIP_CHECKSUM_MANIFEST)
    assert (
        CAMPAIGN_COMMON.derive_campaign_id(manifest["preparation_nonce"]) == manifest["campaign_id"]
    )
    validated_manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(prepared_campaign)
    assert validated_manifest == manifest
    assert identity["campaign_id"] == manifest["campaign_id"]

    database = prepared_campaign / "campaign.sqlite"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
        total_lanes, active_lanes = conn.execute(
            "SELECT COUNT(*), SUM(active) FROM lanes"
        ).fetchone()
        policy_paths = [
            Path(row[0])
            for row in conn.execute(
                "SELECT current_policy_path FROM lanes WHERE active=1 ORDER BY lane_id"
            )
        ]
    assert (total_lanes, active_lanes) == (121, 121)
    assert len(policy_paths) == 121
    assert len(set(policy_paths)) == 121
    assert all(path.is_file() for path in policy_paths)
    assert len(list((prepared_campaign / "lanes").glob("*/simple_pf.py"))) == 121
    assert len(list((prepared_campaign / "lanes").glob("*/initial_simple_pf.py"))) == 121
    assert json.loads(
        (prepared_campaign / "manifests" / "runner_provenance.json").read_text(encoding="utf-8")
    ) == CAMPAIGN_COMMON.build_runner_provenance(RUNNER_ROOT)
    EXPORT_REPLICATES.validate_canonical_campaign_manifest(prepared_campaign)


def test_prepare_campaign_resumes_after_identity_with_missing_database_and_policies(
    prepared_campaign: Path,
) -> None:
    identity_path = prepared_campaign / "manifests" / "campaign_identity.json"
    identity_before = identity_path.read_bytes()
    shutil.rmtree(prepared_campaign / "lanes")
    (prepared_campaign / "campaign.sqlite").unlink()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_campaign.py"),
            "--source-csv",
            str(SOURCE_MANIFEST),
            "--campaign-root",
            str(prepared_campaign),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--hamlib-data-root",
            "/data/hamlib-test",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert identity_path.read_bytes() == identity_before
    assert len(list((prepared_campaign / "lanes").glob("*/simple_pf.py"))) == 121
    assert len(list((prepared_campaign / "lanes").glob("*/initial_simple_pf.py"))) == 121
    with sqlite3.connect(prepared_campaign / "campaign.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM lanes WHERE active=1").fetchone()[0] == 121


def test_prepare_campaign_reconstructs_identity_after_manifest_only_crash(
    prepared_campaign: Path,
) -> None:
    manifest_path = prepared_campaign / "manifests" / "campaign_manifest.json"
    identity_path = prepared_campaign / "manifests" / "campaign_identity.json"
    manifest_before = manifest_path.read_bytes()
    identity_before = json.loads(identity_path.read_text(encoding="utf-8"))
    identity_path.unlink()
    shutil.rmtree(prepared_campaign / "lanes")
    (prepared_campaign / "campaign.sqlite").unlink()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_campaign.py"),
            "--source-csv",
            str(SOURCE_MANIFEST),
            "--campaign-root",
            str(prepared_campaign),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--hamlib-data-root",
            "/data/hamlib-test",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert manifest_path.read_bytes() == manifest_before
    assert json.loads(identity_path.read_text(encoding="utf-8")) == identity_before
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(prepared_campaign)
    assert manifest["campaign_id"] == identity["campaign_id"]
    with sqlite3.connect(prepared_campaign / "campaign.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM lanes WHERE active=1").fetchone()[0] == 121


def test_prepare_campaign_rejects_stray_identity_without_manifest(
    prepared_campaign: Path,
) -> None:
    (prepared_campaign / "manifests" / "campaign_manifest.json").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_campaign.py"),
            "--source-csv",
            str(SOURCE_MANIFEST),
            "--campaign-root",
            str(prepared_campaign),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--hamlib-data-root",
            "/data/hamlib-test",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "identity but no manifest; refusing ambiguity" in completed.stderr


def test_campaign_identity_fails_closed_on_manifest_drift(prepared_campaign: Path) -> None:
    manifest_path = prepared_campaign / "manifests" / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cluster_root"] = "/cluster/rebound-to-another-run"
    CAMPAIGN_COMMON.write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="identity or manifest hash drift"):
        CAMPAIGN_COMMON.validate_campaign_identity(prepared_campaign)


def test_remote_claim_is_atomic_identity_bound_and_path_safe() -> None:
    campaign_id = "a" * 64
    manifest_hash = "b" * 64
    payload = json.loads(
        SUBMIT_EVAL.remote_claim_payload(
            campaign_id=campaign_id,
            manifest_sha256=manifest_hash,
        )
    )
    assert payload == {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": manifest_hash,
    }
    command = SUBMIT_EVAL.remote_claim_command(
        cluster_root="/cluster/paper/run_01",
        campaign_id=campaign_id,
        manifest_sha256=manifest_hash,
    )
    assert "flock -x" in command
    assert "set -C" in command
    assert "${tmp##*/}" in command
    assert "refusing to claim non-empty campaign root" in command
    assert "remote campaign claim mismatch" in command
    with pytest.raises(ValueError, match="absolute POSIX path"):
        SUBMIT_EVAL.remote_claim_command(
            cluster_root="relative/path",
            campaign_id=campaign_id,
            manifest_sha256=manifest_hash,
        )
    with pytest.raises(ValueError, match="safe SSH host alias"):
        SUBMIT_EVAL.validate_ssh_host("-oProxyCommand=bad", "sync_host")
    with pytest.raises(ValueError, match="unsafe for Slurm"):
        SUBMIT_EVAL.slurm_export_safe("70000,BASH_ENV=/tmp/payload", "legate_fbmem")


def test_proxy_jump_transport_builders_are_exact_and_option_safe() -> None:
    assert CAMPAIGN_COMMON.ssh_command("nvlogin", "squeue -h") == [
        "ssh",
        "nvlogin",
        "squeue -h",
    ]
    assert CAMPAIGN_COMMON.ssh_command(
        "nvlogin",
        "squeue -h",
        proxy_jump_host="omni",
    ) == [
        "ssh",
        "-J",
        "omni",
        "nvlogin",
        "squeue -h",
    ]
    assert CAMPAIGN_COMMON.rsync_proxy_jump_args() == []
    assert CAMPAIGN_COMMON.rsync_proxy_jump_args("omni") == ["-e", "ssh -J omni"]
    assert CAMPAIGN_COMMON.proxy_jump_cli_args() == []
    assert CAMPAIGN_COMMON.proxy_jump_cli_args("omni") == ["--proxy-jump-host", "omni"]
    for unsafe in ("-oProxyCommand=bad", "omni other", "omni,other", "user@omni", ""):
        with pytest.raises(ValueError, match="safe SSH host alias"):
            CAMPAIGN_COMMON.ssh_command("nvlogin", "true", proxy_jump_host=unsafe)


def test_rsync_resolver_executes_real_version_and_option_parsers() -> None:
    provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    binary = Path(provenance["binary"])
    version = subprocess.run(
        [str(binary), "--version"],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
    )
    option_parser = subprocess.run(
        [str(binary), "--protect-args", "--help"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert version.returncode == 0
    assert provenance["version"] in (version.stdout + version.stderr)
    assert option_parser.returncode == 0
    assert provenance["binary_sha256"] == _sha256(binary)
    assert provenance["protect_args_supported"] is True


def test_rsync_resolver_detects_binary_change_during_real_copy_smoke(tmp_path: Path) -> None:
    real_rsync = CAMPAIGN_COMMON.resolve_rsync_binary()["binary"]
    wrapper = tmp_path / "mutating-rsync"
    wrapper.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = '-a' ]; then printf '\\n# changed\\n' >>\"$0\"; fi\n"
        f'exec {shlex.quote(str(real_rsync))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    with pytest.raises(RuntimeError, match="changed during compatibility preflight"):
        CAMPAIGN_COMMON.resolve_rsync_binary(str(wrapper))


def test_rsync_resolver_rejects_real_legacy_macos_parser() -> None:
    legacy = Path("/usr/bin/rsync")
    if not legacy.is_file():
        pytest.skip("macOS system rsync is unavailable")
    parser_probe = subprocess.run(
        [str(legacy), "--protect-args", "--help"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if parser_probe.returncode == 0:
        pytest.skip("/usr/bin/rsync is already compatible on this platform")

    with pytest.raises(RuntimeError, match="failed compatibility preflight"):
        CAMPAIGN_COMMON.resolve_rsync_binary(str(legacy))


def test_rsync_discovery_skips_real_legacy_path_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = Path("/usr/bin/rsync")
    homebrew = Path("/opt/homebrew/bin/rsync")
    if not legacy.is_file() or not homebrew.is_file():
        pytest.skip("macOS legacy and Homebrew rsync pair is unavailable")
    legacy_parser = subprocess.run(
        [str(legacy), "--protect-args", "--help"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if legacy_parser.returncode == 0:
        pytest.skip("/usr/bin/rsync is already compatible on this platform")

    monkeypatch.setenv("PATH", "/usr/bin")
    provenance = CAMPAIGN_COMMON.resolve_rsync_binary()

    assert Path(provenance["binary"]) == homebrew.resolve()


def test_campaign_rsync_provenance_is_immutable(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir()
    provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    receipt = CAMPAIGN_COMMON.ensure_rsync_provenance(tmp_path, provenance)
    assert json.loads(receipt.read_text(encoding="utf-8")) == provenance

    drifted = dict(provenance)
    drifted["binary_sha256"] = "0" * 64
    CAMPAIGN_COMMON.write_json(receipt, drifted)
    with pytest.raises(RuntimeError, match="executable or version drifted"):
        CAMPAIGN_COMMON.ensure_rsync_provenance(tmp_path, provenance)


def test_missing_rsync_receipt_cannot_be_recreated_after_campaign_activity(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "manifests").mkdir(parents=True)
    provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    receipt = CAMPAIGN_COMMON.ensure_rsync_provenance(campaign, provenance)
    batch = campaign / "iterations" / "iter_0000" / "batches" / "initial_baseline"
    batch.mkdir(parents=True)
    receipt.unlink()

    wrapper = tmp_path / "alternate-rsync"
    wrapper.write_text(
        f'#!/bin/sh\nexec {shlex.quote(str(provenance["binary"]))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    alternate = CAMPAIGN_COMMON.resolve_rsync_binary(str(wrapper))

    with pytest.raises(RuntimeError, match="receipt is missing after transfer or batch activity"):
        SUBMIT_EVAL.require_or_allow_new_rsync_binding(campaign, alternate)
    with pytest.raises(RuntimeError, match="provenance receipt is missing"):
        CAMPAIGN_COMMON.require_existing_rsync_provenance(campaign, alternate)
    assert not receipt.exists()


def test_rsync_retry_revalidates_immediately_before_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    revalidations: list[int] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command[0])
        if len(calls) == 1:
            raise subprocess.CalledProcessError(23, command, stderr="transient")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(SUBMIT_EVAL.subprocess, "run", fake_run)
    completed = SUBMIT_EVAL.run_with_retries(
        ["rsync", "source", "target"],
        attempts=2,
        timeout_seconds=10,
        retry_delay_seconds=0,
        before_attempt=lambda: revalidations.append(len(calls)),
    )

    assert completed.stdout == "ok"
    assert calls == ["rsync", "rsync"]
    assert revalidations == [0, 1]


def test_rsync_retry_stops_if_binary_changes_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    revalidations = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        raise subprocess.CalledProcessError(23, command, stderr="transient")

    def revalidate() -> None:
        nonlocal revalidations
        revalidations += 1
        if revalidations == 2:
            raise RuntimeError("rsync executable changed")

    monkeypatch.setattr(SUBMIT_EVAL.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="rsync executable changed"):
        SUBMIT_EVAL.run_with_retries(
            ["rsync", "source", "target"],
            attempts=3,
            timeout_seconds=10,
            retry_delay_seconds=0,
            before_attempt=revalidate,
        )

    assert calls == 1
    assert revalidations == 2


def test_rsync_receipt_rejects_symlinked_manifests_parent(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    outside = tmp_path / "outside-manifests"
    campaign.mkdir()
    outside.mkdir()
    (campaign / "manifests").symlink_to(outside, target_is_directory=True)
    provenance = CAMPAIGN_COMMON.resolve_rsync_binary()

    with pytest.raises(RuntimeError, match="manifests parent chain contains a symlink"):
        CAMPAIGN_COMMON.validate_existing_rsync_provenance(campaign, provenance)

    real_campaign = tmp_path / "real-campaign"
    (real_campaign / "manifests").mkdir(parents=True)
    campaign_alias = tmp_path / "campaign-alias"
    campaign_alias.symlink_to(real_campaign, target_is_directory=True)
    with pytest.raises(RuntimeError, match="manifests parent chain contains a symlink"):
        CAMPAIGN_COMMON.validate_existing_rsync_provenance(campaign_alias, provenance)


def test_collect_and_slurm_queries_use_the_same_proxy_jump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    collect_calls: list[list[str]] = []
    revalidation_calls: list[dict[str, object]] = []

    def fake_collect_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        collect_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        COLLECT_ITERATION,
        "subprocess",
        SimpleNamespace(run=fake_collect_run, DEVNULL=subprocess.DEVNULL),
    )
    monkeypatch.setattr(
        COLLECT_ITERATION,
        "revalidate_rsync_binary",
        lambda provenance: revalidation_calls.append(provenance) or provenance,
    )
    COLLECT_ITERATION.rsync_from_cluster(
        "/cluster/hamlib121/run_01",
        tmp_path / "run_01",
        1,
        "nvdata",
        rsync_provenance=rsync_provenance,
        proxy_jump_host="omni",
    )
    assert collect_calls[0][:7] == [
        rsync_provenance["binary"],
        "-ah",
        "--partial",
        "--protect-args",
        "-e",
        "ssh -J omni",
        "nvdata:/cluster/hamlib121/run_01/iterations/iter_0001/results/",
    ]
    assert collect_calls[1] == ["ssh", "-J", "omni", "nvdata", "bash", "-s"]
    assert collect_calls[2][4:6] == ["-e", "ssh -J omni"]
    assert revalidation_calls == [rsync_provenance, rsync_provenance]

    slurm_calls: list[list[str]] = []

    def fake_slurm_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        slurm_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="123|RUNNING\n", stderr="")

    monkeypatch.setattr(STREAM_SUPERVISOR.subprocess, "run", fake_slurm_run)
    assert not STREAM_SUPERVISOR.job_is_done("123", "nvlogin", proxy_jump_host="omni")
    assert slurm_calls == [["ssh", "-J", "omni", "nvlogin", "squeue -h -j 123 -o '%A|%T'"]]


def test_supervisor_propagates_proxy_jump_to_every_transport_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_calls: list[list[str]] = []

    def fake_child_run(command: list[str], *, cwd: Path) -> str:
        assert cwd == tmp_path
        child_calls.append(command)
        return "123456"

    monkeypatch.setattr(STREAM_SUPERVISOR, "run", fake_child_run)
    STREAM_SUPERVISOR.submit_batch(
        root=tmp_path,
        iteration=1,
        batch_id="iter0001_batch0001",
        lane_ids=["lane-1"],
        account="test-account",
        partition="test-gpu",
        cpu_partition="test-cpu",
        sync_host="nvdata",
        submit_host="nvlogin",
        proxy_jump_host="omni",
        rsync_binary="/verified/rsync",
        legate_fbmem=None,
        uv_base_dir=None,
        uv_cache_dir=None,
        uv_python_install_dir=None,
    )
    STREAM_SUPERVISOR.collect_iteration(
        tmp_path,
        1,
        "nvdata",
        ["lane-1"],
        "iter0001_batch0001",
        "omni",
        "/verified/rsync",
    )
    assert all(command[command.index("--proxy-jump-host") + 1] == "omni" for command in child_calls)
    assert all(
        command[command.index("--rsync-bin") + 1] == "/verified/rsync" for command in child_calls
    )

    rescue_calls: list[list[str]] = []

    def fake_rescue_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        rescue_calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(STREAM_SUPERVISOR.subprocess, "run", fake_rescue_run)
    STREAM_SUPERVISOR.rescue_pending(
        tmp_path,
        job_ids=["123456"],
        query_host="nvlogin",
        primary_partition="test-cpu",
        fallback_partition="test-fallback",
        proxy_jump_host="omni",
    )
    assert rescue_calls[0][rescue_calls[0].index("--proxy-jump-host") + 1] == "omni"

    remote_calls: list[list[str]] = []

    def fake_remote_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        remote_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(RESCUE_PARTITIONS.subprocess, "run", fake_remote_run)
    RESCUE_PARTITIONS.run_remote("squeue -h", query_host="nvlogin", proxy_jump_host="omni")
    assert remote_calls == [["ssh", "-J", "omni", "nvlogin", "squeue -h"]]


def test_strict_export_rejects_unsafe_proxy_jump_provenance(tmp_path: Path) -> None:
    root = tmp_path / "unsafe-transport"
    _write_synthetic_export_run(root)
    path = root / "manifests" / "supervisor_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proxy_jump_host"] = "-oProxyCommand=bad"
    CAMPAIGN_COMMON.write_json(path, payload)
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(root)
    with pytest.raises(RuntimeError, match="unsafe supervisor transport provenance"):
        EXPORT_REPLICATES.validate_supervisor_provenance(
            root,
            manifest=manifest,
            identity=identity,
            require_paper_protocol=False,
        )


def test_strict_export_accepts_exact_enumerated_legacy_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "legacy-direct-transport"
    _write_synthetic_export_run(root)
    path = root / "manifests" / "supervisor_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("proxy_jump_host")
    payload.pop("rsync")
    CAMPAIGN_COMMON.write_json(path, payload)
    (root / "manifests" / "rsync_provenance.json").unlink()
    legacy_hash = next(iter(EXPORT_REPLICATES.LEGACY_RSYNC_RUNNER_BUNDLE_SHA256S))
    monkeypatch.setattr(
        EXPORT_REPLICATES,
        "validate_runner_provenance",
        lambda _root: {"runner_bundle_sha256": legacy_hash},
    )
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(root)
    protocol = EXPORT_REPLICATES.validate_supervisor_provenance(
        root,
        manifest=manifest,
        identity=identity,
        require_paper_protocol=False,
    )
    assert protocol["batch_size"] == 1


def test_strict_export_rejects_rsync_key_deletion_for_current_runner(tmp_path: Path) -> None:
    root = tmp_path / "current-runner-downgrade"
    _write_synthetic_export_run(root)
    path = root / "manifests" / "supervisor_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("rsync")
    CAMPAIGN_COMMON.write_json(path, payload)
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(root)

    with pytest.raises(RuntimeError, match="current runner requires supervisor rsync provenance"):
        EXPORT_REPLICATES.validate_supervisor_provenance(
            root,
            manifest=manifest,
            identity=identity,
            require_paper_protocol=False,
        )


def test_strict_export_binds_supervisor_to_rsync_receipt(tmp_path: Path) -> None:
    root = tmp_path / "rsync-transport"
    _write_synthetic_export_run(root)
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    CAMPAIGN_COMMON.ensure_rsync_provenance(root, rsync_provenance)
    path = root / "manifests" / "supervisor_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rsync"] = rsync_provenance
    CAMPAIGN_COMMON.write_json(path, payload)
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(root)

    protocol = EXPORT_REPLICATES.validate_supervisor_provenance(
        root,
        manifest=manifest,
        identity=identity,
        require_paper_protocol=False,
    )
    assert protocol["batch_size"] == 1

    receipt = root / "manifests" / "rsync_provenance.json"
    drifted = dict(rsync_provenance)
    drifted["binary_sha256"] = "0" * 64
    CAMPAIGN_COMMON.write_json(receipt, drifted)
    with pytest.raises(RuntimeError, match="supervisor rsync provenance drift"):
        EXPORT_REPLICATES.validate_supervisor_provenance(
            root,
            manifest=manifest,
            identity=identity,
            require_paper_protocol=False,
        )


def test_prepare_campaign_rejects_fixed_protocol_drift(tmp_path: Path) -> None:
    campaign_root = tmp_path / "drifted"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_campaign.py"),
            "--source-csv",
            str(SOURCE_MANIFEST),
            "--campaign-root",
            str(campaign_root),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--hamlib-data-root",
            "/data/hamlib-test",
            "--memory-gib",
            "65",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "fixed HamLib121 benchmark parameters changed" in completed.stderr
    assert not campaign_root.exists()


def test_prepare_campaign_rejects_same_size_drift_in_an_inactive_source_row(
    tmp_path: Path,
) -> None:
    rows = CAMPAIGN_COMMON.read_manifest_rows(SOURCE_MANIFEST, max_qubits=32)
    inactive = next(
        row
        for index, row in enumerate(rows)
        if not CAMPAIGN_COMMON.fit_lane(
            row,
            index,
            memory_gib=64.0,
            workspace_factor=4.0,
        ).exact_fit
    )
    needle = str(inactive["file"]).encode("utf-8")
    replacement = bytes([needle[0] + 1]) + needle[1:]
    source_bytes = SOURCE_MANIFEST.read_bytes()
    assert source_bytes.count(needle) == 1
    drifted_bytes = source_bytes.replace(needle, replacement, 1)
    assert len(drifted_bytes) == len(source_bytes)
    drifted = tmp_path / SOURCE_MANIFEST.name
    drifted.write_bytes(drifted_bytes)

    campaign_root = tmp_path / "inactive-row-drift"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_campaign.py"),
            "--source-csv",
            str(drifted),
            "--campaign-root",
            str(campaign_root),
            "--cluster-root",
            "/cluster/hsopt-test",
            "--hamlib-data-root",
            "/data/hamlib-test",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "canonical HamLib source manifest drift" in completed.stderr
    assert not (campaign_root / "manifests" / "campaign_identity.json").exists()
    assert not (campaign_root / "campaign.sqlite").exists()


def test_decide_candidate_accepts_only_feasible_strict_improvements(tmp_path: Path) -> None:
    proposal = tmp_path / "candidate.py"
    proposal.write_text("# candidate\n", encoding="utf-8")

    def result(total_t_gates: int, *, feasible: bool = True) -> dict[str, object]:
        return {
            "status": "ok",
            "passes_chemical_accuracy_bound": feasible,
            "total_t_gates_qpe": total_t_gates,
        }

    baseline = COLLECT_ITERATION.decide_candidate(
        iteration=0,
        result=result(100),
        incumbent=None,
        proposal_path=proposal,
    )
    assert baseline == (True, "initial_baseline", None)

    incumbent = {"iteration": 0, "total_t_gates_qpe": 100}
    infeasible = COLLECT_ITERATION.decide_candidate(
        iteration=1,
        result=result(90, feasible=False),
        incumbent=incumbent,
        proposal_path=proposal,
    )
    higher = COLLECT_ITERATION.decide_candidate(
        iteration=1,
        result=result(101),
        incumbent=incumbent,
        proposal_path=proposal,
    )
    equal = COLLECT_ITERATION.decide_candidate(
        iteration=1,
        result=result(100),
        incumbent=incumbent,
        proposal_path=proposal,
    )
    lower = COLLECT_ITERATION.decide_candidate(
        iteration=1,
        result=result(90),
        incumbent=incumbent,
        proposal_path=proposal,
    )

    assert infeasible[:2] == (False, "chemical_accuracy_failed")
    assert higher[:2] == (False, "no_t_gate_improvement")
    assert equal[:2] == (False, "no_t_gate_improvement")
    assert lower[:2] == (True, "strict_t_gate_improvement")
    assert lower[2] == pytest.approx(10.0)


def test_custom_order_is_an_exact_permutation_and_cannot_mutate_source_terms() -> None:
    original = [
        {
            "label": "IX",
            "coefficient": 0.25,
            "support_size": 1,
            "x_count": 1,
            "y_count": 0,
            "z_count": 0,
        },
        {
            "label": "ZI",
            "coefficient": -0.5,
            "support_size": 1,
            "x_count": 0,
            "y_count": 0,
            "z_count": 1,
        },
    ]

    class ReversingPolicy:
        @staticmethod
        def order_product_formula_terms(terms: list[dict], _config: object) -> list[dict]:
            return list(reversed(terms))

    ordered = CAMPAIGN_COMMON.order_terms(original, "custom", ReversingPolicy, object())
    assert ordered == list(reversed(original))

    class MutatingPolicy:
        @staticmethod
        def order_product_formula_terms(terms: list[dict], _config: object) -> list[dict]:
            terms[0]["coefficient"] = 1_000_000.0
            return terms

    with pytest.raises(ValueError, match="exact Hamiltonian term dictionaries"):
        CAMPAIGN_COMMON.order_terms(original, "custom", MutatingPolicy, object())
    assert original[0]["coefficient"] == 0.25

    class RetainingPolicy:
        retained: list[dict] = []

        @classmethod
        def order_product_formula_terms(cls, terms: list[dict], _config: object) -> list[dict]:
            cls.retained = terms
            return terms

    detached = CAMPAIGN_COMMON.order_terms(original, "custom", RetainingPolicy, object())
    RetainingPolicy.retained[0]["coefficient"] = -999.0
    assert detached == original


def test_safe_policy_loader_accepts_the_fixed_policy_and_rejects_host_capabilities(
    prepared_campaign: Path,
    tmp_path: Path,
) -> None:
    initial_policy = next((prepared_campaign / "lanes").glob("*/initial_simple_pf.py"))
    loaded = CAMPAIGN_COMMON.load_policy_module(initial_policy)
    assert loaded.get_method_config().n_steps > 0

    representative = tmp_path / "representative.py"
    representative.write_text(
        """
from dataclasses import dataclass
from typing import Dict
import math

@dataclass(frozen=True)
class Config:
    n_steps: int = 2

def choose(term: Dict[str, object], config: Config) -> float:
    return math.sqrt(abs(float(term["coefficient"]))) / config.n_steps
""".lstrip(),
        encoding="utf-8",
    )
    representative_module = CAMPAIGN_COMMON.load_policy_module(representative)
    assert representative_module.choose(
        {"coefficient": -4.0}, representative_module.Config()
    ) == pytest.approx(1.0)

    sentinel = tmp_path / "policy-escaped.txt"
    malicious_sources = {
        "os import": "import os\n",
        "subprocess import": "import subprocess\n",
        "pathlib import": "from pathlib import Path\n",
        "socket import": "import socket\n",
        "dataclasses module escape": (
            "import dataclasses\n"
            f"value = dataclasses.sys.modules['os'].system('touch {sentinel}')\n"
        ),
        "typing module escape": (
            f"import typing\nvalue = typing.sys.modules['os'].system('touch {sentinel}')\n"
        ),
        "private math import": "from math import __loader__\n",
        "open builtin": f"open({str(sentinel)!r}, 'w').write('escaped')\n",
        "eval builtin": "eval('1 + 1')\n",
        "exec builtin": "exec('value = 1')\n",
        "dynamic import": "__import__('os')\n",
        "dunder reflection": "value = (1).__class__\n",
        "private module reflection": "import math\nvalue = math.__dict__\n",
        "unbounded loop": "def spin():\n    while True:\n        pass\n",
    }
    for label, source in malicious_sources.items():
        policy = tmp_path / f"malicious-{label.replace(' ', '-')}.py"
        policy.write_text(source, encoding="utf-8")
        with pytest.raises((ImportError, ValueError)):
            CAMPAIGN_COMMON.load_policy_module(policy)
        assert not sentinel.exists(), label


def test_omitted_rotations_are_charged_to_the_triangle_inequality_budget() -> None:
    config = SimpleNamespace(formula_order=1, n_steps=2, zero_angle_threshold=1.0)
    terms = [{"label": "X", "coefficient": 0.2}]

    rotations, scheduled, nonzero, total_error, omitted_error = EVALUATE_WORKER.aggregate_rotations(
        terms=terms,
        config=config,
        choose_synthesis_epsilon=lambda *_: (_ for _ in ()).throw(
            AssertionError("dropped rotations must not be synthesized")
        ),
        tau=1.0,
    )

    expected = 2 * (2.0 * abs(math.sin(0.1 / 2.0)))
    assert rotations == {}
    assert (scheduled, nonzero) == (2, 0)
    assert omitted_error == pytest.approx(expected)
    assert total_error == pytest.approx(expected)

    bad_config = SimpleNamespace(formula_order=1, n_steps=1, zero_angle_threshold=-1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        EVALUATE_WORKER.aggregate_rotations(
            terms=terms,
            config=bad_config,
            choose_synthesis_epsilon=lambda *_: 1e-6,
            tau=1.0,
        )

    for invalid_threshold in (float("inf"), float("nan")):
        invalid_config = SimpleNamespace(
            formula_order=1,
            n_steps=1,
            zero_angle_threshold=invalid_threshold,
        )
        with pytest.raises(ValueError, match="finite and nonnegative"):
            EVALUATE_WORKER.aggregate_rotations(
                terms=terms,
                config=invalid_config,
                choose_synthesis_epsilon=lambda *_: 1e-6,
                tau=1.0,
            )


def test_streaming_progress_helpers_resume_and_number_batches(tmp_path: Path) -> None:
    lanes = [
        {"lane_id": "new", "iteration": None},
        {"lane_id": "in_progress", "iteration": 4},
        {"lane_id": "finished", "iteration": 5},
    ]
    lane_state, next_iteration = STREAM_SUPERVISOR.initialize_lane_progress(
        lanes,
        start_iteration=2,
        final_iteration=5,
    )
    assert lane_state == {"new": "idle", "in_progress": "idle", "finished": "done"}
    assert next_iteration == {"new": 2, "in_progress": 5, "finished": 6}

    for relative in (
        "iterations/iter_0002/batches/iter0002_batch0007",
        "iterations/iter_0004/batches/iter0004_batch0042",
        "iterations/iter_0005/batches/iter0005_batchoops",
    ):
        (tmp_path / relative).mkdir(parents=True)
    assert STREAM_SUPERVISOR.highest_existing_batch_index(tmp_path) == 42

    tools = STREAM_SUPERVISOR.prepare_codex_tools(tmp_path)
    for name in ("python", "python3"):
        shim = tools / name
        assert shim.is_file()
        assert str(Path(sys.executable)) in shim.read_text(encoding="utf-8")

    first_lock = STREAM_SUPERVISOR.acquire_supervisor_lock(tmp_path)
    with pytest.raises(RuntimeError, match="another streaming supervisor"):
        STREAM_SUPERVISOR.acquire_supervisor_lock(tmp_path)
    first_lock.close()
    resumed_lock = STREAM_SUPERVISOR.acquire_supervisor_lock(tmp_path)
    resumed_lock.close()


def test_mutation_runs_in_isolated_workspace_and_promotes_only_exact_closure(
    prepared_campaign: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads(
        (prepared_campaign / "manifests" / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    lane = next(lane for lane in manifest["lanes"] if lane["exact_fit"])
    live_policy = prepared_campaign / "lanes" / lane["lane_id"] / "simple_pf.py"
    original = live_policy.read_bytes()
    calls: list[list[str]] = []
    work_dirs: list[Path] = []
    add_extra_file = False

    class FakeProcess:
        pid = 4242

        async def wait(self) -> int:
            return 0

    async def fake_create_subprocess_exec(
        *command: str,
        **kwargs: object,
    ) -> FakeProcess:
        args = list(command)
        calls.append(args)
        work_dir = Path(args[args.index("-C") + 1])
        work_dirs.append(work_dir)
        candidate = work_dir / "simple_pf.py"
        candidate.write_bytes(candidate.read_bytes() + b"\n# isolated mutation\n")
        if add_extra_file:
            (work_dir / "undeclared.txt").write_text("escape", encoding="utf-8")
        stdout = kwargs["stdout"]
        assert hasattr(stdout, "write")
        stdout.write('{"type":"thread.started"}\n')
        stdout.write('{"type":"turn.completed"}\n')
        stdout.flush()
        last_message = Path(args[args.index("-o") + 1])
        last_message.write_text("changed the policy", encoding="utf-8")
        return FakeProcess()

    terminated: list[int] = []

    async def fake_terminate_process_group(process_group_id: int) -> None:
        terminated.append(process_group_id)

    monkeypatch.setattr(
        MUTATE_ITERATION.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        MUTATE_ITERATION,
        "terminate_process_group",
        fake_terminate_process_group,
    )
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()

    lane_id, code, succeeded, error = asyncio.run(
        MUTATE_ITERATION.run_one(
            root=prepared_campaign,
            lane=lane,
            iteration=1,
            model="gpt-5.5",
            reasoning_effort="high",
            codex_bin="codex-test",
            timeout_seconds=30,
            tools_dir=tools_dir,
            semaphore=asyncio.Semaphore(1),
        )
    )
    assert (lane_id, code, succeeded, error) == (lane["lane_id"], 0, True, None)
    promoted = live_policy.read_bytes()
    assert promoted != original
    assert promoted.endswith(b"# isolated mutation\n")
    assert calls[0][1 : 1 + len(CAMPAIGN_COMMON.PAPER_CODEX_EXEC_FLAGS)] == list(
        CAMPAIGN_COMMON.PAPER_CODEX_EXEC_FLAGS
    )
    assert "--dangerously-bypass-approvals-and-sandbox" not in calls[0]
    assert work_dirs[0] != live_policy.parent
    assert not work_dirs[0].exists()
    assert terminated == [4242]

    add_extra_file = True
    lane_id, code, succeeded, error = asyncio.run(
        MUTATE_ITERATION.run_one(
            root=prepared_campaign,
            lane=lane,
            iteration=2,
            model="gpt-5.5",
            reasoning_effort="high",
            codex_bin="codex-test",
            timeout_seconds=30,
            tools_dir=tools_dir,
            semaphore=asyncio.Semaphore(1),
        )
    )
    assert (lane_id, code, succeeded) == (lane["lane_id"], 0, False)
    assert error is not None and "workspace closure changed" in error
    assert live_policy.read_bytes() == promoted
    assert not work_dirs[1].exists()
    assert terminated == [4242, 4242]


def test_process_group_teardown_kills_background_child_before_it_can_write(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "background-child-survived.txt"
    ready = tmp_path / "background-child-ready.txt"
    child_source = (
        "import time; from pathlib import Path; "
        f"time.sleep(1.5); Path({str(sentinel)!r}).write_text('survived')"
    )
    parent_source = (
        "import subprocess, sys; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}]); "
        f"Path({str(ready)!r}).write_text('ready')"
    )

    async def exercise() -> int:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            parent_source,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        process_group_id = process.pid
        await process.wait()
        assert ready.is_file()
        try:
            await MUTATE_ITERATION.terminate_process_group(process_group_id)
            await asyncio.sleep(1.7)
        finally:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return process_group_id

    assert asyncio.run(exercise()) > 0
    assert not sentinel.exists()


def test_eval_batches_persist_submission_and_retry_state(prepared_campaign: Path) -> None:
    conn = CAMPAIGN_COMMON.db_connect(prepared_campaign)
    CAMPAIGN_COMMON.init_db(conn)
    lane_ids = [
        str(row[0])
        for row in conn.execute(
            "SELECT lane_id FROM lanes WHERE active=1 ORDER BY lane_id LIMIT 3"
        ).fetchall()
    ]
    batch_id = "iter0001_batch0042"
    submission_key, job_name = CAMPAIGN_COMMON.eval_submission_identity(
        cluster_root="/cluster/hsopt-test",
        iteration=1,
        batch_id=batch_id,
        policy_snapshot="current",
        lane_ids=lane_ids[:2],
    )

    prepared = CAMPAIGN_COMMON.register_eval_batch(
        conn,
        batch_id=batch_id,
        iteration=1,
        lane_ids=lane_ids[:2],
        submission_key=submission_key,
        job_name=job_name,
    )
    assert prepared.state == "prepared"
    assert prepared.job_id is None
    assert CAMPAIGN_COMMON.load_open_eval_batches(conn) == [prepared]
    assert STREAM_SUPERVISOR.highest_existing_batch_index(prepared_campaign, conn) == 42
    lane_state = conn.execute(
        "SELECT pending_iteration, status FROM lanes WHERE lane_id=?", (lane_ids[0],)
    ).fetchone()
    assert tuple(lane_state) == (1, "submitting")

    with pytest.raises(RuntimeError, match="already belong to open batch"):
        other_key, other_name = CAMPAIGN_COMMON.eval_submission_identity(
            cluster_root="/cluster/hsopt-test",
            iteration=1,
            batch_id="iter0001_batch0043",
            policy_snapshot="current",
            lane_ids=[lane_ids[1], lane_ids[2]],
        )
        CAMPAIGN_COMMON.register_eval_batch(
            conn,
            batch_id="iter0001_batch0043",
            iteration=1,
            lane_ids=[lane_ids[1], lane_ids[2]],
            submission_key=other_key,
            job_name=other_name,
        )

    submitted = CAMPAIGN_COMMON.mark_eval_batch_submitted(
        conn,
        batch_id=batch_id,
        job_id="123456",
    )
    assert submitted.state == "submitted"
    assert submitted.job_id == "123456"
    assert (
        CAMPAIGN_COMMON.mark_eval_batch_submitted(
            conn,
            batch_id=batch_id,
            job_id="123456",
        )
        == submitted
    )
    with pytest.raises(RuntimeError, match="Slurm job changed"):
        CAMPAIGN_COMMON.mark_eval_batch_submitted(
            conn,
            batch_id=batch_id,
            job_id="654321",
        )

    CAMPAIGN_COMMON.mark_eval_batch_collected(
        conn,
        batch_id=batch_id,
        successful_lane_ids=[lane_ids[0]],
        retry_lane_ids=[lane_ids[1]],
    )
    assert CAMPAIGN_COMMON.load_open_eval_batches(conn) == []
    success = conn.execute(
        "SELECT pending_iteration, eval_retry_count FROM lanes WHERE lane_id=?",
        (lane_ids[0],),
    ).fetchone()
    retry = conn.execute(
        "SELECT pending_iteration, eval_retry_count, status FROM lanes WHERE lane_id=?",
        (lane_ids[1],),
    ).fetchone()
    assert tuple(success) == (None, 0)
    assert tuple(retry) == (1, 1, "eval_retry")
    conn.close()


def test_submit_dry_run_has_no_writes_or_network_transfer(
    prepared_campaign: Path,
    tmp_path: Path,
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    manifest = json.loads(
        (prepared_campaign / "manifests" / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    active_lane_ids = [lane["lane_id"] for lane in manifest["lanes"] if lane["exact_fit"]]
    empty_path = tmp_path / "no-external-tools"
    empty_path.mkdir()
    before = _tree_snapshot(prepared_campaign)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "submit_eval_batch.py"),
            "--campaign-root",
            str(prepared_campaign),
            "--iteration",
            "0",
            "--batch-id",
            "dry-run-batch",
            "--all-active",
            "--compute-baseline",
            "--policy-snapshot",
            "initial",
            "--account",
            "test-account",
            "--partition",
            "test-gpu",
            "--cpu-partition",
            "test-cpu",
            "--sync-host",
            "test-sync",
            "--submit-host",
            "test-submit",
            "--rsync-bin",
            str(rsync_provenance["binary"]),
            "--dry-run",
        ],
        cwd=RUNNER_ROOT,
        env={
            **os.environ,
            "PATH": str(empty_path),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    plan = json.loads(completed.stdout)
    assert plan["dry_run"] is True
    assert plan["lane_ids"] == active_lane_ids
    assert plan["execution_mode"] == "gpu_baseline"
    assert plan["slurm_script"] == "slurm/evaluate_batch.sbatch"
    assert plan["rsync"] == rsync_provenance
    assert _tree_snapshot(prepared_campaign) == before
    assert not (
        prepared_campaign / "iterations" / "iter_0001" / "batches" / "dry-run-batch"
    ).exists()

    absent_campaign = tmp_path / "absent-campaign"
    absent = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "submit_eval_batch.py"),
            "--campaign-root",
            str(absent_campaign),
            "--iteration",
            "1",
            "--batch-id",
            "dry-run-absent",
            "--lane-id",
            active_lane_ids[0],
            "--account",
            "test-account",
            "--partition",
            "test-gpu",
            "--cpu-partition",
            "test-cpu",
            "--sync-host",
            "test-sync",
            "--submit-host",
            "test-submit",
            "--rsync-bin",
            str(rsync_provenance["binary"]),
            "--dry-run",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PATH": str(empty_path), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert absent.returncode != 0
    assert not absent_campaign.exists()


def test_submit_rejects_legacy_rsync_before_campaign_writes(
    prepared_campaign: Path,
) -> None:
    legacy = Path("/usr/bin/rsync")
    if not legacy.is_file():
        pytest.skip("macOS system rsync is unavailable")
    parser_probe = subprocess.run(
        [str(legacy), "--protect-args", "--help"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if parser_probe.returncode == 0:
        pytest.skip("/usr/bin/rsync is already compatible on this platform")
    before = _tree_snapshot(prepared_campaign)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "submit_eval_batch.py"),
            "--campaign-root",
            str(prepared_campaign),
            "--iteration",
            "0",
            "--batch-id",
            "legacy-rsync",
            "--all-active",
            "--compute-baseline",
            "--policy-snapshot",
            "initial",
            "--account",
            "test-account",
            "--partition",
            "test-gpu",
            "--cpu-partition",
            "test-cpu",
            "--sync-host",
            "test-sync",
            "--submit-host",
            "test-submit",
            "--rsync-bin",
            str(legacy),
            "--sync-only",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failed compatibility preflight" in completed.stderr
    assert _tree_snapshot(prepared_campaign) == before


def test_collect_rejects_drifted_rsync_receipt_before_ensure_dirs(
    prepared_campaign: Path,
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    receipt = CAMPAIGN_COMMON.ensure_rsync_provenance(prepared_campaign, rsync_provenance)
    drifted = dict(rsync_provenance)
    drifted["binary_sha256"] = "0" * 64
    CAMPAIGN_COMMON.write_json(receipt, drifted)
    logs = prepared_campaign / "logs"
    shutil.rmtree(logs)
    before = _tree_snapshot(prepared_campaign)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "collect_iteration.py"),
            "--campaign-root",
            str(prepared_campaign),
            "--iteration",
            "1",
            "--sync-host",
            "test-sync",
            "--rsync-bin",
            str(rsync_provenance["binary"]),
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "campaign rsync executable or version drifted" in completed.stderr
    assert not logs.exists()
    assert _tree_snapshot(prepared_campaign) == before


def test_supervisor_rejects_drifted_rsync_receipt_before_campaign_side_effects(
    prepared_campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    receipt = CAMPAIGN_COMMON.ensure_rsync_provenance(prepared_campaign, rsync_provenance)
    drifted = dict(rsync_provenance)
    drifted["binary_sha256"] = "0" * 64
    CAMPAIGN_COMMON.write_json(receipt, drifted)
    before = _tree_snapshot(prepared_campaign)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("supervisor reached a campaign write or external side effect")

    monkeypatch.setattr(STREAM_SUPERVISOR, "ensure_dirs", forbidden)
    monkeypatch.setattr(STREAM_SUPERVISOR, "acquire_supervisor_lock", forbidden)
    monkeypatch.setattr(STREAM_SUPERVISOR, "db_connect", forbidden)
    monkeypatch.setattr(STREAM_SUPERVISOR, "ensure_autoresearch_provenance", forbidden)

    with pytest.raises(RuntimeError, match="campaign rsync executable or version drifted"):
        asyncio.run(
            STREAM_SUPERVISOR.supervise(
                SimpleNamespace(
                    campaign_root=prepared_campaign,
                    rsync_provenance=rsync_provenance,
                )
            )
        )

    assert _tree_snapshot(prepared_campaign) == before


def test_supervisor_does_not_persist_missing_receipt_before_binary_revalidation(
    prepared_campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    receipt = prepared_campaign / "manifests" / "rsync_provenance.json"
    assert not receipt.exists()
    before = _tree_snapshot(prepared_campaign)
    monkeypatch.setattr(
        STREAM_SUPERVISOR,
        "revalidate_rsync_binary",
        lambda _provenance: (_ for _ in ()).throw(RuntimeError("local rsync revalidation failed")),
    )

    with pytest.raises(RuntimeError, match="local rsync revalidation failed"):
        asyncio.run(
            STREAM_SUPERVISOR.supervise(
                SimpleNamespace(
                    campaign_root=prepared_campaign,
                    rsync_provenance=rsync_provenance,
                )
            )
        )

    assert not receipt.exists()
    assert _tree_snapshot(prepared_campaign) == before


def test_initial_baseline_submission_is_durable_before_network_and_identity_bound(
    prepared_campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    calls: list[tuple[list[str], str | None]] = []
    revalidation_at_call_counts: list[int] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        before_attempt = kwargs.get("before_attempt")
        if before_attempt is not None:
            assert callable(before_attempt)
            before_attempt()
        input_text = kwargs.get("input_text")
        assert input_text is None or isinstance(input_text, str)
        calls.append((command, input_text))
        is_submit = command[0] == "ssh" and input_text is not None and "sbatch" in input_text
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="123456\n" if is_submit else "",
            stderr="",
        )

    monkeypatch.setattr(SUBMIT_EVAL, "run_with_retries", fake_run)
    monkeypatch.setattr(
        SUBMIT_EVAL,
        "revalidate_rsync_binary",
        lambda _provenance: revalidation_at_call_counts.append(len(calls)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_eval_batch.py",
            "--campaign-root",
            str(prepared_campaign),
            "--iteration",
            "0",
            "--batch-id",
            "initial_baseline",
            "--all-active",
            "--compute-baseline",
            "--policy-snapshot",
            "initial",
            "--account",
            "test-account",
            "--partition",
            "test-gpu",
            "--cpu-partition",
            "test-cpu",
            "--sync-host",
            "test-sync",
            "--submit-host",
            "test-submit",
            "--proxy-jump-host",
            "omni",
            "--rsync-bin",
            str(rsync_provenance["binary"]),
        ],
    )
    SUBMIT_EVAL.main()
    assert capsys.readouterr().out.strip() == "123456"
    assert revalidation_at_call_counts == [1]
    assert [command[0][0] for command in calls] == [
        "ssh",
        rsync_provenance["binary"],
        "ssh",
    ]
    assert calls[0][1] is not None and "hsopt-campaign-claim" in calls[0][1]

    with sqlite3.connect(prepared_campaign / "campaign.sqlite") as conn:
        row = conn.execute(
            "SELECT state, job_id, lane_ids_json, submission_key, job_name "
            "FROM eval_batches WHERE batch_id='initial_baseline'"
        ).fetchone()
    assert row is not None
    state, job_id, lane_ids_json, submission_key, job_name = row
    assert (state, job_id, job_name) == ("submitted", "123456", f"hpf_{submission_key}")
    assert len(json.loads(lane_ids_json)) == 121
    batch_manifest = json.loads(
        (
            prepared_campaign
            / "iterations"
            / "iter_0000"
            / "batches"
            / "initial_baseline"
            / "batch_manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest, identity = CAMPAIGN_COMMON.validate_campaign_identity(prepared_campaign)
    assert batch_manifest["campaign_id"] == manifest["campaign_id"]
    assert batch_manifest["campaign_manifest_sha256"] == identity["campaign_manifest_sha256"]
    assert batch_manifest["runner_provenance"] == CAMPAIGN_COMMON.validate_runner_provenance(
        prepared_campaign
    )
    assert len(batch_manifest["policy_sha256s"]) == 121
    assert batch_manifest["execution_configuration"]["execution_mode"] == "gpu_baseline"
    assert batch_manifest["transport_configuration"] == {
        "sync_host": "test-sync",
        "submit_host": "test-submit",
        "proxy_jump_host": "omni",
        "rsync": rsync_provenance,
    }
    assert (
        json.loads(
            (prepared_campaign / "manifests" / "rsync_provenance.json").read_text(encoding="utf-8")
        )
        == rsync_provenance
    )
    assert "--protect-args" in calls[1][0]
    assert calls[0][0] == ["ssh", "-J", "omni", "test-sync", "bash", "-s"]
    assert calls[1][0][4:6] == ["-e", "ssh -J omni"]
    assert calls[1][0][calls[1][0].index(".venv_cpu/") - 1] == "--exclude"
    assert calls[2][1] is not None
    assert "--export='ALL," not in calls[2][1]
    assert "flock -x 8" in calls[2][1]
    assert calls[2][0] == ["ssh", "-J", "omni", "test-submit", "bash", "-s"]


def test_sync_only_rsync_excludes_both_cluster_environments(
    prepared_campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rsync_provenance = CAMPAIGN_COMMON.resolve_rsync_binary()
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        input_text = kwargs.get("input_text")
        assert input_text is None or isinstance(input_text, str)
        calls.append((command, input_text))
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(SUBMIT_EVAL, "run_with_retries", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_eval_batch.py",
            "--campaign-root",
            str(prepared_campaign),
            "--iteration",
            "0",
            "--batch-id",
            "sync-only-baseline",
            "--all-active",
            "--compute-baseline",
            "--policy-snapshot",
            "initial",
            "--account",
            "test-account",
            "--partition",
            "test-gpu",
            "--cpu-partition",
            "test-cpu",
            "--sync-host",
            "test-sync",
            "--submit-host",
            "test-submit",
            "--rsync-bin",
            str(rsync_provenance["binary"]),
            "--sync-only",
        ],
    )

    SUBMIT_EVAL.main()
    assert [command[0] for command, _input in calls] == [
        "ssh",
        rsync_provenance["binary"],
    ]
    rsync_command = calls[1][0]
    assert rsync_command[rsync_command.index(".venv/") - 1] == "--exclude"
    assert rsync_command[rsync_command.index(".venv_cpu/") - 1] == "--exclude"
    with sqlite3.connect(prepared_campaign / "campaign.sqlite") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM eval_batches WHERE batch_id='sync-only-baseline'"
            ).fetchone()[0]
            == 0
        )


def test_partition_rescue_is_scoped_to_campaign_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert RESCUE_PARTITIONS.is_eval_job_name("hpf_0123456789ab")
    assert RESCUE_PARTITIONS.is_eval_job_name("hamlib_pf_batch")
    assert not RESCUE_PARTITIONS.is_eval_job_name("hpf_short")
    assert not RESCUE_PARTITIONS.is_eval_job_name("unrelated_job")

    commands: list[str] = []

    def fake_run_remote(command: str, **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="123|test-cpu|hpf_0123456789ab|Priority\n",
            stderr="",
        )

    monkeypatch.setattr(RESCUE_PARTITIONS, "run_remote", fake_run_remote)
    pending = RESCUE_PARTITIONS.pending_eval_tasks(
        "test-cpu",
        query_host="test-query",
        job_ids=["123"],
    )

    assert pending == [("123", "Priority")]
    assert commands == ["timeout 20s squeue -h -j 123 -t PD -o '%i|%P|%j|%R'"]
    assert "-u $USER" not in commands[0]

    with pytest.raises(ValueError, match="invalid Slurm job ID"):
        RESCUE_PARTITIONS.pending_eval_tasks(
            "test-cpu",
            query_host="test-query",
            job_ids=["123; scontrol show job"],
        )


def test_collector_cannot_close_a_batch_with_a_missing_lane_result(tmp_path: Path) -> None:
    root = tmp_path / "missing-result"
    _write_synthetic_export_run(root)
    with sqlite3.connect(root / "campaign.sqlite") as conn:
        conn.execute(
            "UPDATE eval_batches SET lane_ids_json=?, state='submitted', collected_at=NULL "
            "WHERE batch_id='iter0001_batch0001'",
            (json.dumps(["0000_test_lane", "0001_missing_lane"]),),
        )
        conn.commit()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "collect_iteration.py"),
            "--campaign-root",
            str(root),
            "--iteration",
            "1",
            "--cluster-root",
            "/cluster/synthetic",
            "--batch-id",
            "iter0001_batch0001",
            "--skip-rsync",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "result closure is incomplete or ambiguous" in completed.stderr
    assert "0001_missing_lane" in completed.stderr
    with sqlite3.connect(root / "campaign.sqlite") as conn:
        state, collected_at = conn.execute(
            "SELECT state, collected_at FROM eval_batches WHERE batch_id='iter0001_batch0001'"
        ).fetchone()
    assert (state, collected_at) == ("submitted", None)


def test_campaign_progress_reports_missing_databases_as_incomplete(tmp_path: Path) -> None:
    replicate_root = tmp_path / "replicates"
    progress = FINALIZE_REPLICATES.campaign_progress(
        replicate_root,
        runs=2,
        expected_lanes=121,
        expected_iterations=50,
    )

    assert progress["complete"] is False
    assert progress["total_proposals"] == 0
    assert progress["expected_total_proposals"] == 12_100
    assert [row["status"] for row in progress["runs"]] == [
        "missing_campaign_database",
        "missing_campaign_database",
    ]
    assert not replicate_root.exists()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "finalize_replicates_when_complete.py"),
            "--replicate-root",
            str(replicate_root),
            "--output-dir",
            str(tmp_path / "final-export"),
            "--once",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert not replicate_root.exists()


def test_partial_initial_baseline_replication_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_synthetic_export_run(source)
    _write_synthetic_export_run(target)
    lane_id = "0000_test_lane"
    source_result = source / "iterations" / "iter_0000" / "results" / f"{lane_id}.json"
    payload = json.loads(source_result.read_text(encoding="utf-8"))
    payload.update(
        {
            "accepted": True,
            "decision_reason": "initial_baseline",
            "relative_gain_percent": None,
            "incumbent_iteration": None,
            "incumbent_t_gates_qpe": None,
            "mutation_summary": "deterministic baseline",
        }
    )
    CAMPAIGN_COMMON.write_json(source_result, payload)

    target_result = target / "iterations" / "iter_0000" / "results" / f"{lane_id}.json"
    target_baseline = target / "baselines" / f"{lane_id}.json"
    target_result.unlink()
    target_baseline.unlink()
    shutil.copy2(source / "baselines" / f"{lane_id}.json", target_baseline)
    monkeypatch.setattr(REPLICATE_BASELINE, "EXPECTED_LANES", 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replicate_initial_baseline.py",
            "--source-campaign",
            str(source),
            "--target-campaign",
            str(target),
        ],
    )

    REPLICATE_BASELINE.main()
    receipt = target / "manifests" / "baseline_replication_receipt.json"
    assert target_result.is_file()
    assert receipt.is_file()
    replicated = json.loads(target_result.read_text(encoding="utf-8"))
    source_manifest, source_identity = CAMPAIGN_COMMON.validate_campaign_identity(source)
    assert replicated["baseline_replication"] == {
        "schema_version": 1,
        "source_campaign_id": source_manifest["campaign_id"],
        "source_manifest_sha256": source_identity["campaign_manifest_sha256"],
        "source_result_sha256": _sha256(source_result),
    }
    snapshot = _tree_snapshot(target)
    REPLICATE_BASELINE.main()
    assert _tree_snapshot(target) == snapshot


def test_export_helpers_sanitize_paths_and_checksum_outputs(tmp_path: Path) -> None:
    run_root = Path("/home/researcher/external/replicates/run_01")
    summary = (
        f"See [policy](file://{run_root}/iterations/iter_0001/mutations/lane/after_simple_pf.py)."
    )
    sanitized = EXPORT_REPLICATES.sanitize_summary(
        summary,
        run_root=run_root,
        run_name="run_01",
    )
    assert sanitized == (
        "See [policy](run_01/iterations/iter_0001/mutations/lane/after_simple_pf.py)."
    )

    output = tmp_path / "export"
    (output / "policies" / "lane").mkdir(parents=True)
    (output / "summary.json").write_text('{"complete": true}\n', encoding="utf-8")
    policy = output / "policies" / "lane" / "simple_pf.py"
    policy.write_text("# portable policy\n", encoding="utf-8")
    EXPORT_REPLICATES.write_checksum_manifest(output)
    EXPORT_REPLICATES.assert_portable_export(output)

    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksums == [
        f"{_sha256(policy)}  policies/lane/simple_pf.py",
        f"{_sha256(output / 'summary.json')}  summary.json",
    ]

    (output / "summary.json").write_text(
        '{"source": "/Users/researcher/private/run"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="machine-specific absolute paths"):
        EXPORT_REPLICATES.assert_portable_export(output)


def test_exporter_requires_the_canonical_hamlib121_lane_set() -> None:
    with ACTIVE_MANIFEST.open(encoding="utf-8") as handle:
        canonical = list(csv.DictReader(handle))
    EXPORT_REPLICATES.validate_canonical_lanes(canonical)

    drifted = [dict(row) for row in canonical]
    drifted[0]["molecule"] = "not-the-paper-lane"
    with pytest.raises(RuntimeError, match="canonical HamLib121 manifest"):
        EXPORT_REPLICATES.validate_canonical_lanes(drifted)


def test_complete_paper_export_requires_the_fixed_agent_configuration() -> None:
    correct = {
        "model": CAMPAIGN_COMMON.PAPER_MODEL,
        "reasoning_effort": CAMPAIGN_COMMON.PAPER_REASONING_EFFORT,
        "codex_cli_version": CAMPAIGN_COMMON.PAPER_CODEX_CLI_VERSION,
        "node_version": "24.11.1",
        "npm_cache": ".codex_npm",
        "codex_exec_flags": list(CAMPAIGN_COMMON.PAPER_CODEX_EXEC_FLAGS),
    }
    EXPORT_REPLICATES.validate_paper_autoresearch_provenance(correct)
    for key in correct:
        drifted = {**correct, key: "wrong"}
        with pytest.raises(RuntimeError, match="paper autoresearch provenance changed"):
            EXPORT_REPLICATES.validate_paper_autoresearch_provenance(drifted)


def test_strict_exporter_validates_and_normalizes_a_complete_fixture(
    tmp_path: Path,
) -> None:
    replicate_root = tmp_path / "replicates"
    run_root = replicate_root / "run_01"
    output = tmp_path / "normalized"
    _write_synthetic_export_run(run_root)

    command = [
        sys.executable,
        str(SCRIPTS_ROOT / "export_replicate_results.py"),
        "--replicate-root",
        str(replicate_root),
        "--output-dir",
        str(output),
        "--runs",
        "1",
        "--expected-lanes",
        "1",
        "--expected-iterations",
        "1",
    ]
    rejected = subprocess.run(
        command,
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "final HamLib121 exports require" in rejected.stderr
    assert not output.exists()

    completed = subprocess.run(
        [
            *command,
            "--diagnostic-shape",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is False
    assert summary["paper_contract_shape"] is False
    assert summary["diagnostic_export"] is True
    assert summary["source_layout"] == "external run_01 through run_01 campaign workspaces"
    assert summary["exported_candidate_rows"] == 1
    assert summary["exported_lane_run_rows"] == 1
    assert summary["exported_lane_aggregate_rows"] == 1
    assert summary["exported_baseline_provenance_rows"] == 1
    assert summary["exported_policy_sources"] == 2
    assert summary["hamlib_zip_checksum_manifest"] == "hamlib_zip_sha256.txt"
    assert summary["confidence_interval_95_percent"] is None
    assert summary["provenance_file"] == "provenance.json"
    assert "replicate_root" not in summary

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["model"] == "gpt-5.5"
    assert provenance["reasoning_effort"] == "high"
    assert provenance["codex_cli_version"] == "0.144.0"
    assert provenance["node_version"] == "24.11.1"
    assert provenance["replicate_runs_verified"] == 1
    assert provenance["supervisor_protocol"]["batch_size"] == 1

    candidate_log = (output / "candidate_log.csv").read_text(encoding="utf-8")
    lane_summary = (output / "lane_run_summary.csv").read_text(encoding="utf-8")
    assert str(run_root) not in candidate_log
    assert str(run_root) not in lane_summary
    assert "run_01/iterations/iter_0001" in candidate_log
    assert "run_01/iterations/iter_0001" in lane_summary

    baseline_provenance = list(
        csv.DictReader((output / "baseline_provenance.csv").open(encoding="utf-8"))
    )
    assert len(baseline_provenance) == 1
    assert baseline_provenance[0]["lane_id"] == "0000_test_lane"
    assert baseline_provenance[0]["gpu_uuid"].startswith("GPU-")
    exported_zip_manifest = output / "hamlib_zip_sha256.txt"
    assert exported_zip_manifest.read_bytes() == ZIP_CHECKSUM_MANIFEST.read_bytes()

    checksum_lines = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksum_lines
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        assert _sha256(output / relative) == expected
    EXPORT_REPLICATES.assert_portable_export(output)

    leaky_output = tmp_path / "leaky-normalized"
    candidate_policy = (
        run_root
        / "iterations"
        / "iter_0001"
        / "mutations"
        / "0000_test_lane"
        / "after_simple_pf.py"
    )
    candidate_policy.write_text("# leaked /scratch/private/campaign\n", encoding="utf-8")
    failed = subprocess.run(
        [
            *command[:5],
            str(leaky_output),
            *command[6:],
            "--diagnostic-shape",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "evaluated policy bytes changed" in failed.stderr
    assert not leaky_output.exists()
    assert not list(tmp_path.glob(".leaky-normalized.staging-*"))


def test_strict_export_rejects_an_open_submitted_batch(tmp_path: Path) -> None:
    replicate_root = tmp_path / "replicates"
    run_root = replicate_root / "run_01"
    output = tmp_path / "normalized"
    _write_synthetic_export_run(run_root)
    with sqlite3.connect(run_root / "campaign.sqlite") as conn:
        conn.execute(
            "UPDATE eval_batches SET state='submitted', collected_at=NULL "
            "WHERE batch_id='iter0001_batch0001'"
        )
        conn.commit()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "export_replicate_results.py"),
            "--replicate-root",
            str(replicate_root),
            "--output-dir",
            str(output),
            "--runs",
            "1",
            "--expected-lanes",
            "1",
            "--expected-iterations",
            "1",
            "--diagnostic-shape",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "evaluation batch(es) are still open" in completed.stderr
    assert "strict export requires every durable submission to be collected" in completed.stderr
    assert not output.exists()


def test_exporter_rejects_cross_run_provenance_drift(tmp_path: Path) -> None:
    replicate_root = tmp_path / "replicates"
    for run_name in ("run_01", "run_02"):
        _write_synthetic_export_run(replicate_root / run_name)
    drift_path = replicate_root / "run_02" / "manifests" / "autoresearch_provenance.json"
    drifted = json.loads(drift_path.read_text(encoding="utf-8"))
    drifted["model"] = "different-model"
    CAMPAIGN_COMMON.write_json(drift_path, drifted)

    output = tmp_path / "normalized"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "export_replicate_results.py"),
            "--replicate-root",
            str(replicate_root),
            "--output-dir",
            str(output),
            "--runs",
            "2",
            "--expected-lanes",
            "1",
            "--expected-iterations",
            "1",
            "--diagnostic-shape",
        ],
        cwd=RUNNER_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "provenance differs from run_01" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("policy", "evaluated policy bytes changed"),
        ("baseline", "baseline cache bytes changed"),
        ("batch", "batch manifest bytes changed"),
    ),
)
def test_result_replay_rejects_mutable_evidence(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    root = tmp_path / tamper
    _write_synthetic_export_run(root)
    lane_id = "0000_test_lane"
    result = json.loads(
        (root / "iterations" / "iter_0001" / "results" / f"{lane_id}.json").read_text(
            encoding="utf-8"
        )
    )
    policy = root / "iterations" / "iter_0001" / "mutations" / lane_id / "after_simple_pf.py"
    if tamper == "policy":
        policy.write_text("# policy tampered after evaluation\n", encoding="utf-8")
    elif tamper == "baseline":
        baseline = root / "baselines" / f"{lane_id}.json"
        baseline.write_text(baseline.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        batch = (
            root
            / "iterations"
            / "iter_0001"
            / "batches"
            / "iter0001_batch0001"
            / "batch_manifest.json"
        )
        batch.write_text(batch.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest, _identity = CAMPAIGN_COMMON.validate_campaign_identity(root)
    with pytest.raises(RuntimeError, match=message):
        CAMPAIGN_COMMON.validate_evaluation_result_provenance(
            campaign_root=root,
            lane=manifest["lanes"][0],
            iteration=1,
            result=result,
            policy_path=policy,
        )


def test_baseline_semantics_require_verified_a100_cupynumeric_evidence(tmp_path: Path) -> None:
    root = tmp_path / "a100-drift"
    _write_synthetic_export_run(root)
    lane_id = "0000_test_lane"
    result_path = root / "iterations" / "iter_0000" / "results" / f"{lane_id}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    baseline_path = root / "baselines" / f"{lane_id}.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["baseline_accelerator"] = "NVIDIA H100"
    baseline["gpu_preflight"]["accelerator_family"] = "NVIDIA H100"
    CAMPAIGN_COMMON.write_json(baseline_path, baseline)
    result["baseline_accelerator"] = "NVIDIA H100"
    result["gpu_preflight"] = baseline["gpu_preflight"]
    result["baseline_cache_sha256"] = _sha256(baseline_path)
    manifest, _identity = CAMPAIGN_COMMON.validate_campaign_identity(root)
    with pytest.raises(RuntimeError, match="not verified A100 evidence"):
        CAMPAIGN_COMMON.validate_evaluation_result_provenance(
            campaign_root=root,
            lane=manifest["lanes"][0],
            iteration=0,
            result=result,
            policy_path=root / "lanes" / lane_id / "initial_simple_pf.py",
        )


def test_campaign_runner_snapshot_rejects_bytecode_and_undeclared_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _copy_runner_provenance_sources(snapshot)
    CAMPAIGN_COMMON.write_json(
        snapshot / "manifests" / "campaign_manifest.json",
        {"campaign_id": "synthetic"},
    )
    pycache = snapshot / "scripts" / "__pycache__"
    pycache.mkdir()
    with pytest.raises(RuntimeError, match="forbidden runner bytecode"):
        CAMPAIGN_COMMON.build_runner_provenance(snapshot)
    pycache.rmdir()
    (snapshot / "scripts" / "undeclared.py").write_text("# drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="closure changed"):
        CAMPAIGN_COMMON.build_runner_provenance(snapshot)


def test_runner_has_no_personal_roots_and_slurm_scripts_parse() -> None:
    text_suffixes = {".csv", ".json", ".lock", ".md", ".py", ".sbatch", ".toml"}
    generated_parts = {".codex_npm", ".venv", ".venv_cpu", "__pycache__"}
    personal_root_patterns = (
        re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+(?=/|\s|[\"'])"),
        re.compile(r"/lustre/[^\s\"']*/users/[A-Za-z0-9._-]+(?=/|\s|[\"'])"),
    )
    violations: list[str] = []
    for path in sorted(RUNNER_ROOT.rglob("*")):
        relative = path.relative_to(RUNNER_ROOT)
        if (
            not path.is_file()
            or path.suffix not in text_suffixes
            or generated_parts.intersection(relative.parts)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in personal_root_patterns:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0)}")
    assert not violations, "personal absolute roots found:\n" + "\n".join(violations)

    runner_readme = (RUNNER_ROOT / "README.md").read_text(encoding="utf-8")
    assert "\npython3 scripts/" not in runner_readme
    assert "uv run --locked python scripts/" in runner_readme
    for required in (
        "Node.js 24.x",
        "hamlib_zip_sha256.txt",
        "2a5958a86a6adaa30bd98b325701830f618f7663ddcd925509213ad3ae187def",
        "prebuild_cluster_environments.py",
        "--batch-size 8",
        "--max-concurrent-mutators 4",
        "--max-running-eval-jobs 8",
        "not five full A100",
    ):
        assert required in runner_readme

    for entrypoint in sorted((RUNNER_ROOT / "scripts").glob("*.py")):
        if entrypoint.name == "campaign_common.py":
            continue
        source = entrypoint.read_text(encoding="utf-8")
        assert "sys.pycache_prefix" in source
        assert 'os.environ["PYTHONPYCACHEPREFIX"]' in source

    bash = shutil.which("bash")
    assert bash is not None
    for script in sorted((RUNNER_ROOT / "slurm").glob("*.sbatch")):
        completed = subprocess.run(
            [bash, "-n", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"
        source = script.read_text(encoding="utf-8")
        assert "uv sync" not in source
        assert ".hsopt-uv-lock-sha256" in source


def test_slurm_workers_pin_and_verify_the_locked_python_runtime() -> None:
    gpu_source = (RUNNER_ROOT / "slurm" / "evaluate_batch.sbatch").read_text(encoding="utf-8")
    cpu_source = (RUNNER_ROOT / "slurm" / "evaluate_batch_cpu.sbatch").read_text(encoding="utf-8")
    for source in (gpu_source, cpu_source):
        assert 'export CAMPAIGN_LOCKED_PYTHON="$VIRTUAL_ENV/bin/python"' in source
    assert '--run-mode exec "$CAMPAIGN_LOCKED_PYTHON" scripts/evaluate_worker.py' in gpu_source
    assert '"$CAMPAIGN_LOCKED_PYTHON" scripts/evaluate_worker.py' in cpu_source

    worker_source = (SCRIPTS_ROOT / "evaluate_worker.py").read_text(encoding="utf-8")
    guard_call = 'if os.environ.get("SLURM_JOB_ID") or os.environ.get('
    assert worker_source.index(guard_call) < worker_source.index("import numpy as np")


def test_locked_python_runtime_guard_rejects_interpreter_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(sys.executable)
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setenv("CAMPAIGN_LOCKED_PYTHON", str(executable))
    EVALUATE_WORKER.validate_locked_python_runtime()

    wrong_runtime = shutil.which("sh")
    assert wrong_runtime is not None
    monkeypatch.setattr(EVALUATE_WORKER.sys, "executable", wrong_runtime)
    with pytest.raises(RuntimeError, match="Python runtime drift"):
        EVALUATE_WORKER.validate_locked_python_runtime()

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
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
SOURCE_MANIFEST_SHA256 = "ff7f9acce5f5140d65f84ebe33df92276c9058d13a8de4a0ed254b7ad868e489"
ACTIVE_MANIFEST_SHA256 = "32c9c2c117f11894df9e4999f5ee1a9bffd94ef41558a158a17ca0c634cc47fe"


def _load_runner_module(stem: str) -> ModuleType:
    path = SCRIPTS_ROOT / f"{stem}.py"
    module_name = f"_hsopt_hamlib121_test_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(SCRIPTS_ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


CAMPAIGN_COMMON = _load_runner_module("campaign_common")
EVALUATE_WORKER = _load_runner_module("evaluate_worker")
PREPARE_CAMPAIGN = _load_runner_module("prepare_campaign")
COLLECT_ITERATION = _load_runner_module("collect_iteration")
STREAM_SUPERVISOR = _load_runner_module("supervise_streaming_batches")
FINALIZE_REPLICATES = _load_runner_module("finalize_replicates_when_complete")
EXPORT_REPLICATES = _load_runner_module("export_replicate_results")
RESCUE_PARTITIONS = _load_runner_module("rescue_pending_eval_partitions")


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


def _write_synthetic_export_run(root: Path) -> None:
    _copy_runner_provenance_sources(root)
    runner_provenance = CAMPAIGN_COMMON.build_runner_provenance(root)
    CAMPAIGN_COMMON.write_json(root / "manifests" / "runner_provenance.json", runner_provenance)
    CAMPAIGN_COMMON.write_json(
        root / "manifests" / "autoresearch_provenance.json",
        {
            "schema_version": 1,
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "codex_cli_version": "0.144.0",
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

    def result(total_t: int, *, incumbent: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "ok",
            "passes_chemical_accuracy_bound": True,
            "total_t_gates_qpe": total_t,
            "total_error_bound_hartree": 0.0005,
            "pf_error_bound_hartree": 0.0002,
            "synthesis_error_bound_hartree": 0.0002,
            "qpe_grid_spacing_hartree": 0.0001,
            "qpe_bits": 5,
            "qpe_unitary_applications": 31,
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
    mutation_summary = f"See [candidate](file://{candidate_policy})."
    for iteration, total_t, accepted, policy, incumbent in (
        (0, 100, 1, initial_policy, False),
        (1, 90, 1, candidate_policy, True),
    ):
        payload = result(total_t, incumbent=incumbent)
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
    monkeypatch.setattr(CAMPAIGN_COMMON, "probe_codex_cli_version", lambda _command: "0.144.0")
    recorded = CAMPAIGN_COMMON.ensure_autoresearch_provenance(
        campaign_root=campaign,
        model="gpt-5.5",
        reasoning_effort="high",
        codex_bin=CAMPAIGN_COMMON.DEFAULT_CODEX_BIN,
    )
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


def test_submit_dry_run_has_no_writes_or_network_tools(
    prepared_campaign: Path,
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (prepared_campaign / "manifests" / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    lane_id = next(lane["lane_id"] for lane in manifest["lanes"] if lane["exact_fit"])
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
            "1",
            "--batch-id",
            "dry-run-batch",
            "--lane-id",
            lane_id,
            "--compute-baseline",
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
    assert plan["lane_ids"] == [lane_id]
    assert plan["execution_mode"] == "gpu_baseline"
    assert plan["slurm_script"] == "slurm/evaluate_batch.sbatch"
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
            lane_id,
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
    assert summary["exported_policy_sources"] == 2
    assert summary["confidence_interval_95_percent"] is None
    assert summary["provenance_file"] == "provenance.json"
    assert "replicate_root" not in summary

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["model"] == "gpt-5.5"
    assert provenance["reasoning_effort"] == "high"
    assert provenance["codex_cli_version"] == "0.144.0"
    assert provenance["replicate_runs_verified"] == 1

    candidate_log = (output / "candidate_log.csv").read_text(encoding="utf-8")
    lane_summary = (output / "lane_run_summary.csv").read_text(encoding="utf-8")
    assert str(run_root) not in candidate_log
    assert str(run_root) not in lane_summary
    assert "run_01/iterations/iter_0001" in candidate_log
    assert "run_01/iterations/iter_0001" in lane_summary

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
    assert "machine-specific absolute paths" in failed.stderr
    assert not leaky_output.exists()
    assert not list(tmp_path.glob(".leaky-normalized.staging-*"))


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

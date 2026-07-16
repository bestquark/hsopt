#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PRIVATE_PYCACHE_PREFIX = tempfile.mkdtemp(prefix="hsopt-hamlib121-pycache-")
_private_pycache_stat = os.stat(_PRIVATE_PYCACHE_PREFIX, follow_symlinks=False)
if (
    not stat.S_ISDIR(_private_pycache_stat.st_mode)
    or _private_pycache_stat.st_uid != os.getuid()
    or stat.S_IMODE(_private_pycache_stat.st_mode) & 0o077
):
    raise RuntimeError("private Python bytecode prefix is not an owner-only real directory")
sys.pycache_prefix = _PRIVATE_PYCACHE_PREFIX
os.environ["PYTHONPYCACHEPREFIX"] = _PRIVATE_PYCACHE_PREFIX

from campaign_common import (  # noqa: E402
    PAPER_CODEX_CLI_VERSION,
    PAPER_CODEX_EXEC_FLAGS,
    PAPER_MODEL,
    PAPER_REASONING_EFFORT,
    EXPECTED_HAMLIB_ARCHIVES,
    load_json,
    parse_hamlib_zip_checksums,
    validate_evaluation_result_provenance,
    validate_manifests_parent_chain,
    validate_rsync_provenance,
    validate_result_submission_receipt,
    validate_runner_provenance,
    validate_ssh_host,
    validate_campaign_identity,
    validate_canonical_hamlib_source_manifest,
    validate_canonical_hamlib_zip_checksum_manifest,
    write_json,
)


T_CRITICAL_95_DF4 = 2.7764451051977987
RUNNER_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ACTIVE_MANIFEST = RUNNER_ROOT / "manifests" / "active_lanes.csv"
CANONICAL_SOURCE_MANIFEST = (
    RUNNER_ROOT / "manifests" / "hamlib_electronic_active_space_instances_le32_qubits.csv"
)
FORBIDDEN_ABSOLUTE_PATH_MARKERS = (
    "/home/",
    "/Users/",
    "/lustre/",
    "/scratch/",
    "/tmp/",
    "/var/tmp/",
    "/private/var/",
    "/mnt/",
    "file://",
)
FORBIDDEN_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/][^\s,;]+"),
    re.compile(r"\\\\[^\\\s]+\\[^\\\s]+"),
)
PAPER_RUNS = 5
PAPER_LANES = 121
PAPER_ITERATIONS = 50
LEGACY_RSYNC_RUNNER_BUNDLE_SHA256S = frozenset(
    {
        # Last runner bundle before local-rsync provenance became mandatory.
        "5121425c79901417a12884a196a052b75d9ddf5c13075e5c0023080e7af80cf6",
    }
)
LEGACY_SUPERVISOR_PROVENANCE_KEYS = {
    "schema_version",
    "campaign_id",
    "campaign_manifest_sha256",
    "start_iteration",
    "proposal_iterations",
    "batch_size",
    "max_running_eval_jobs",
    "max_concurrent_mutators",
    "max_batch_wait_seconds",
    "max_eval_retries",
    "max_mutation_retries",
    "mutation_retry_base_seconds",
    "mutation_retry_max_seconds",
    "poll_seconds",
    "rescue_poll_seconds",
    "model",
    "reasoning_effort",
    "codex_bin",
    "codex_exec_flags",
    "npm_cache",
    "python_environments",
    "account",
    "gpu_partition",
    "cpu_partition",
    "fallback_enabled",
    "fallback_partition",
    "sync_host",
    "submit_host",
    "slurm_query_host",
    "legate_fbmem",
    "uv_base_dir",
    "uv_cache_dir",
    "uv_python_install_dir",
}
PROXY_SUPERVISOR_PROVENANCE_KEYS = LEGACY_SUPERVISOR_PROVENANCE_KEYS | {"proxy_jump_host"}
SUPERVISOR_PROVENANCE_KEYS = PROXY_SUPERVISOR_PROVENANCE_KEYS | {"rsync"}


def validate_paper_autoresearch_provenance(provenance: dict[str, Any]) -> None:
    expected = {
        "model": PAPER_MODEL,
        "reasoning_effort": PAPER_REASONING_EFFORT,
        "codex_cli_version": PAPER_CODEX_CLI_VERSION,
        "npm_cache": ".codex_npm",
        "codex_exec_flags": list(PAPER_CODEX_EXEC_FLAGS),
    }
    drift = {
        key: {"actual": provenance.get(key), "expected": value}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if drift:
        raise RuntimeError(f"paper autoresearch provenance changed: {drift}")
    node_version = str(provenance.get("node_version", ""))
    if re.fullmatch(r"24\.\d+\.\d+", node_version) is None:
        raise RuntimeError(
            f"paper autoresearch provenance changed: Node.js 24.x required, got {node_version!r}"
        )


def validate_supervisor_provenance(
    root: Path,
    *,
    manifest: dict[str, Any],
    identity: dict[str, Any],
    require_paper_protocol: bool,
) -> dict[str, Any]:
    root = validate_manifests_parent_chain(root)
    path = root / "manifests" / "supervisor_provenance.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{root.name}: missing or unsafe supervisor provenance")
    payload = load_json(path)
    if not isinstance(payload, dict) or set(payload) not in (
        LEGACY_SUPERVISOR_PROVENANCE_KEYS,
        PROXY_SUPERVISOR_PROVENANCE_KEYS,
        SUPERVISOR_PROVENANCE_KEYS,
    ):
        raise RuntimeError(f"{root.name}: malformed supervisor provenance")
    if "rsync" not in payload:
        runner_bundle_sha256 = validate_runner_provenance(root).get("runner_bundle_sha256")
        if runner_bundle_sha256 not in LEGACY_RSYNC_RUNNER_BUNDLE_SHA256S:
            raise RuntimeError(
                f"{root.name}: current runner requires supervisor rsync provenance; "
                "legacy schema is allowed only for an enumerated pre-rsync runner bundle"
            )
    if (
        payload.get("schema_version") != 1
        or payload.get("campaign_id") != manifest["campaign_id"]
        or payload.get("campaign_manifest_sha256") != identity["campaign_manifest_sha256"]
        or payload.get("npm_cache") != ".codex_npm"
        or payload.get("python_environments") != [".venv", ".venv_cpu"]
    ):
        raise RuntimeError(f"{root.name}: supervisor identity/runtime provenance drift")
    try:
        for field in ("sync_host", "submit_host", "slurm_query_host"):
            validate_ssh_host(payload.get(field), f"supervisor provenance {field}")
        proxy_jump_host = payload.get("proxy_jump_host")
        if proxy_jump_host is not None:
            validate_ssh_host(proxy_jump_host, "supervisor provenance proxy_jump_host")
    except ValueError as exc:
        raise RuntimeError(f"{root.name}: unsafe supervisor transport provenance: {exc}") from exc
    if "rsync" in payload:
        try:
            rsync_provenance = validate_rsync_provenance(payload["rsync"])
        except RuntimeError as exc:
            raise RuntimeError(f"{root.name}: malformed supervisor rsync provenance") from exc
        rsync_path = root / "manifests" / "rsync_provenance.json"
        if rsync_path.is_symlink() or not rsync_path.is_file():
            raise RuntimeError(f"{root.name}: missing or unsafe rsync provenance receipt")
        if load_json(rsync_path) != rsync_provenance:
            raise RuntimeError(f"{root.name}: supervisor rsync provenance drift")
    protocol = {
        key: payload[key]
        for key in (
            "start_iteration",
            "proposal_iterations",
            "batch_size",
            "max_running_eval_jobs",
            "max_concurrent_mutators",
            "model",
            "reasoning_effort",
            "codex_bin",
            "codex_exec_flags",
            "npm_cache",
            "python_environments",
        )
    }
    if require_paper_protocol:
        expected = {
            "start_iteration": 1,
            "proposal_iterations": 50,
            "batch_size": 8,
            "max_running_eval_jobs": 8,
            "max_concurrent_mutators": 4,
            "model": PAPER_MODEL,
            "reasoning_effort": PAPER_REASONING_EFFORT,
            "codex_bin": "npx -y @openai/codex@0.144.0",
            "codex_exec_flags": list(PAPER_CODEX_EXEC_FLAGS),
            "npm_cache": ".codex_npm",
            "python_environments": [".venv", ".venv_cpu"],
        }
        if protocol != expected:
            raise RuntimeError(f"{root.name}: fixed paper supervisor protocol changed")
    return protocol


def parse_result(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(str(row["result_json"] or "{}"))


def exact_t_count(result: dict[str, Any]) -> int:
    value = result.get("total_t_gates_qpe")
    if value is None:
        raise ValueError(f"missing total_t_gates_qpe for {result.get('lane_id')}")
    return int(value)


def policy_path(root: Path, lane_id: str, iteration: int) -> Path:
    if iteration == 0:
        initial_policy = root / "lanes" / lane_id / "initial_simple_pf.py"
        if initial_policy.is_file():
            return initial_policy
    mutation = root / "iterations" / f"iter_{max(1, iteration):04d}" / "mutations" / lane_id
    return mutation / ("before_simple_pf.py" if iteration == 0 else "after_simple_pf.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def sanitize_summary(value: object, *, run_root: Path, run_name: str) -> str:
    """Replace the external campaign root with a stable public run label."""
    text = "" if value is None else str(value)
    roots = {str(run_root), str(run_root.resolve())}
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(f"file://{root}", run_name)
        text = text.replace(root, run_name)
    return text


def assert_portable_export(output_dir: Path) -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        violations = [marker for marker in FORBIDDEN_ABSOLUTE_PATH_MARKERS if marker in text]
        violations.extend(
            pattern.pattern for pattern in FORBIDDEN_ABSOLUTE_PATH_PATTERNS if pattern.search(text)
        )
        if violations:
            offenders[path.relative_to(output_dir).as_posix()] = violations
    if offenders:
        raise RuntimeError(f"export contains machine-specific absolute paths: {offenders}")


def write_checksum_manifest(output_dir: Path) -> None:
    manifest = output_dir / "SHA256SUMS"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file())
        if path != manifest
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_fields(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config") or {}
    return {
        "formula_order": config.get("formula_order"),
        "n_steps": config.get("n_steps"),
        "term_order_strategy": config.get("term_order_strategy"),
        "uniform_epsilon": config.get("uniform_epsilon"),
        "zero_angle_threshold": config.get("zero_angle_threshold"),
        "policy_name": config.get("name"),
    }


def metric_fields(result: dict[str, Any]) -> dict[str, Any]:
    def to_mha(value: object) -> float | None:
        return None if value is None else 1000.0 * float(value)

    return {
        "total_t_gates_qpe": exact_t_count(result),
        "total_error_mha": to_mha(result.get("total_error_bound_hartree")),
        "pf_error_mha": to_mha(result.get("pf_error_bound_hartree")),
        "synthesis_error_mha": to_mha(result.get("synthesis_error_bound_hartree")),
        "zero_angle_error_mha": to_mha(result.get("zero_angle_error_bound_hartree")),
        "qpe_grid_mha": to_mha(result.get("qpe_grid_spacing_hartree")),
        "qpe_bits": result.get("qpe_bits"),
        "qpe_unitary_applications": result.get("qpe_unitary_applications"),
    }


def canonical_lane_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["lane_id"]),
        str(row["collection"]),
        str(row["file"]),
        str(row["dataset"]),
        str(row["molecule"]),
        int(float(row["nqubits"])),
        int(float(row["electrons"])),
        int(float(row["terms"])),
        float(row["one_norm"]),
    )


def validate_canonical_lanes(lanes: list[dict[str, Any]]) -> None:
    with CANONICAL_ACTIVE_MANIFEST.open(encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    actual_keys = [canonical_lane_key(row) for row in lanes]
    expected_keys = [canonical_lane_key(row) for row in expected]
    if actual_keys != expected_keys:
        differing = [
            index + 1
            for index, (actual, wanted) in enumerate(zip(actual_keys, expected_keys, strict=False))
            if actual != wanted
        ]
        if len(actual_keys) != len(expected_keys):
            differing.append(min(len(actual_keys), len(expected_keys)) + 1)
        raise RuntimeError(
            f"active lanes differ from the canonical HamLib121 manifest at rows {differing[:10]}"
        )


def validate_canonical_campaign_manifest(root: Path) -> None:
    validate_canonical_hamlib_source_manifest(CANONICAL_SOURCE_MANIFEST)
    for canonical in (CANONICAL_SOURCE_MANIFEST, CANONICAL_ACTIVE_MANIFEST):
        campaign_copy = root / "manifests" / canonical.name
        if not campaign_copy.is_file() or sha256_file(campaign_copy) != sha256_file(canonical):
            raise RuntimeError(f"{root.name}: campaign manifest copy differs from {canonical.name}")

    payload, identity = validate_campaign_identity(root)
    fixed_fields = {
        "requested_count": 401,
        "target_fit_count": PAPER_LANES,
        "active_fit_count": PAPER_LANES,
        "accelerator_label": "a100",
        "memory_gib": 64.0,
        "workspace_factor": 4.0,
        "active_min_memory_gib": None,
        "schema_version": 2,
    }
    drift = {
        key: {"actual": payload.get(key), "expected": expected}
        for key, expected in fixed_fields.items()
        if payload.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"{root.name}: fixed campaign manifest fields changed: {drift}")
    expected_paper_protocol = {
        "replicate_runs": 5,
        "active_lanes": PAPER_LANES,
        "proposal_iterations": PAPER_ITERATIONS,
        "baseline_backend": "cupynumeric",
        "baseline_accelerator": "NVIDIA A100",
        "baseline_replicated_from_run_01": True,
        "policy_backend": "numpy",
        "streaming_batch_size": 8,
        "max_concurrent_mutators": 4,
        "max_running_eval_jobs": 8,
        "node_major": 24,
        "codex_cli_version": PAPER_CODEX_CLI_VERSION,
        "model": PAPER_MODEL,
        "reasoning_effort": PAPER_REASONING_EFFORT,
    }
    if payload.get("paper_protocol") != expected_paper_protocol:
        raise RuntimeError(f"{root.name}: fixed paper protocol manifest changed")
    checksum_path = root / str(payload.get("hamlib_zip_checksums_file", ""))
    if (
        checksum_path.is_symlink()
        or not checksum_path.is_file()
        or payload.get("hamlib_zip_checksums_sha256") != sha256_file(checksum_path)
    ):
        raise RuntimeError(f"{root.name}: HamLib ZIP checksum manifest drift")
    validate_canonical_hamlib_zip_checksum_manifest(checksum_path)
    archive_records = parse_hamlib_zip_checksums(
        checksum_path,
        active_rows := [lane for lane in payload.get("lanes", []) if lane.get("exact_fit")],
    )
    if (
        archive_records != payload.get("hamlib_archives")
        or len(archive_records) != EXPECTED_HAMLIB_ARCHIVES
    ):
        raise RuntimeError(f"{root.name}: immutable 83-ZIP archive binding drift")
    if identity.get("campaign_id") != payload.get("campaign_id"):
        raise RuntimeError(f"{root.name}: campaign identity drift")

    with CANONICAL_ACTIVE_MANIFEST.open(encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    fields = list(expected_rows[0])
    normalized = [{field: str(row.get(field, "")) for field in fields} for row in active_rows]
    if normalized != expected_rows:
        raise RuntimeError(
            f"{root.name}: full campaign manifest differs from canonical HamLib121 metadata"
        )


def load_run(
    root: Path,
    *,
    expected_iterations: int,
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest, _identity = validate_campaign_identity(root)
    manifest_lane_by_id = {
        str(lane["lane_id"]): lane for lane in manifest["lanes"] if lane.get("exact_fit")
    }
    conn = sqlite3.connect(root / "campaign.sqlite")
    conn.row_factory = sqlite3.Row
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(evaluations)")}
    required_columns = {
        "accepted",
        "decision_reason",
        "incumbent_iteration",
        "incumbent_t_gates_qpe",
        "mutation_summary",
        "passes_accuracy",
    }
    missing_columns = required_columns - columns
    if missing_columns:
        raise RuntimeError(f"{root}: missing decision columns {sorted(missing_columns)}")
    open_batches = int(
        conn.execute("SELECT COUNT(*) FROM eval_batches WHERE state!='collected'").fetchone()[0]
    )
    if not allow_incomplete and open_batches:
        raise RuntimeError(
            f"{root.name}: {open_batches} evaluation batch(es) are still open; "
            "strict export requires every durable submission to be collected"
        )

    lane_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT lane_id, molecule, nqubits, electrons, terms, one_norm, collection, file, dataset
            FROM lanes WHERE active=1 ORDER BY lane_id
            """
        )
    ]
    evaluation_rows = conn.execute(
        """
        SELECT * FROM evaluations
        ORDER BY lane_id, iteration
        """
    ).fetchall()

    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        item = dict(row)
        item["result"] = parse_result(row)
        lane_id = str(row["lane_id"])
        iteration = int(row["iteration"])
        result_path = root / "iterations" / f"iter_{iteration:04d}" / "results" / f"{lane_id}.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise RuntimeError(
                f"{root.name}/{lane_id}/iter {iteration}: result artifact is missing or unsafe"
            )
        artifact = load_json(result_path)
        if artifact != item["result"]:
            raise RuntimeError(
                f"{root.name}/{lane_id}/iter {iteration}: SQLite result_json differs from "
                "the checksummed iteration result artifact"
            )
        validate_result_submission_receipt(
            conn,
            lane_id=lane_id,
            iteration=iteration,
            result=artifact,
        )
        by_lane[lane_id].append(item)

    expected_iteration_set = set(range(0, expected_iterations + 1))
    for lane in lane_rows:
        lane_id = str(lane["lane_id"])
        rows = by_lane.get(lane_id, [])
        seen = {int(row["iteration"]) for row in rows}
        if not allow_incomplete and seen != expected_iteration_set:
            missing = sorted(expected_iteration_set - seen)
            extra = sorted(seen - expected_iteration_set)
            raise RuntimeError(f"{root.name}/{lane_id}: missing={missing}, extra={extra}")
        baseline = next((row for row in rows if int(row["iteration"]) == 0), None)
        if baseline is not None and int(baseline["accepted"] or 0) != 1:
            raise RuntimeError(f"{root.name}/{lane_id}: iteration-0 baseline was not accepted")
        if baseline is not None:
            baseline_result = baseline["result"]
            if str(baseline["status"]) != "ok" or str(baseline_result.get("status")) != "ok":
                raise RuntimeError(f"{root.name}/{lane_id}: iteration-0 baseline status was not ok")
            if not bool(baseline_result.get("passes_chemical_accuracy_bound", False)):
                raise RuntimeError(f"{root.name}/{lane_id}: iteration-0 baseline was infeasible")
            baseline_snapshot = policy_path(root, lane_id, 0)
            if not baseline_snapshot.is_file():
                raise RuntimeError(
                    f"{root.name}/{lane_id}: missing iteration-0 policy {baseline_snapshot}"
                )
            validate_evaluation_result_provenance(
                campaign_root=root,
                lane=manifest_lane_by_id[lane_id],
                iteration=0,
                result=baseline_result,
                policy_path=baseline_snapshot,
            )

        incumbent_t = None if baseline is None else exact_t_count(baseline["result"])
        incumbent_iteration = None if baseline is None else 0
        for row in rows:
            iteration = int(row["iteration"])
            if iteration == 0:
                continue
            if not allow_incomplete and str(row["status"]) != "ok":
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: status={row['status']}"
                )
            result = row["result"]
            if str(row["status"]) != str(result.get("status")):
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: database/result status mismatch"
                )
            snapshot = policy_path(root, lane_id, iteration)
            if not snapshot.is_file():
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: missing policy {snapshot}"
                )
            if str(row["status"]) == "ok":
                validate_evaluation_result_provenance(
                    campaign_root=root,
                    lane=manifest_lane_by_id[lane_id],
                    iteration=iteration,
                    result=result,
                    policy_path=snapshot,
                )
            passes = bool(result.get("passes_chemical_accuracy_bound", False))
            if bool(row["passes_accuracy"]) != passes:
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: database/result feasibility mismatch"
                )
            if (
                result.get("incumbent_iteration") is not None
                and int(result["incumbent_iteration"]) != incumbent_iteration
            ):
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: incumbent iteration mismatch"
                )
            if (
                result.get("incumbent_t_gates_qpe") is not None
                and int(result["incumbent_t_gates_qpe"]) != incumbent_t
            ):
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: incumbent T-count mismatch"
                )

            candidate_t = exact_t_count(result)
            should_accept = (
                str(row["status"]) == "ok"
                and passes
                and incumbent_t is not None
                and candidate_t > 0
                and candidate_t < incumbent_t
            )
            accepted = bool(row["accepted"])
            if accepted != should_accept:
                raise RuntimeError(
                    f"{root.name}/{lane_id}/iter {iteration}: accepted={accepted}, "
                    f"expected={should_accept}, candidate={candidate_t}, incumbent={incumbent_t}, passes={passes}"
                )
            if accepted:
                incumbent_t = candidate_t
                incumbent_iteration = iteration
    conn.close()
    return lane_rows, by_lane


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and validate five HamLib replicate campaigns."
    )
    parser.add_argument("--replicate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=PAPER_RUNS)
    parser.add_argument("--expected-lanes", type=int, default=PAPER_LANES)
    parser.add_argument("--expected-iterations", type=int, default=PAPER_ITERATIONS)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--diagnostic-shape",
        action="store_true",
        help=(
            "Allow non-paper run/lane/iteration counts for diagnostics. "
            "Such exports are always marked incomplete."
        ),
    )
    args = parser.parse_args()

    paper_shape = (
        args.runs == PAPER_RUNS
        and args.expected_lanes == PAPER_LANES
        and args.expected_iterations == PAPER_ITERATIONS
    )
    complete_paper_export = paper_shape and not args.allow_incomplete and not args.diagnostic_shape
    if not paper_shape and not args.diagnostic_shape:
        parser.error(
            "final HamLib121 exports require --runs 5, --expected-lanes 121, "
            "and --expected-iterations 50; use --diagnostic-shape only for external diagnostics"
        )

    replicate_root = args.replicate_root.resolve()
    final_output_dir = args.output_dir.resolve()
    if final_output_dir.exists():
        raise RuntimeError(f"output path must not already exist: {final_output_dir}")
    run_roots = [replicate_root / f"run_{index:02d}" for index in range(1, args.runs + 1)]
    for root in run_roots:
        if not (root / "campaign.sqlite").is_file():
            raise FileNotFoundError(root / "campaign.sqlite")

    all_candidate_rows: list[dict[str, Any]] = []
    lane_run_rows: list[dict[str, Any]] = []
    campaign_rows: list[dict[str, Any]] = []
    baseline_provenance_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    lane_run_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_lane_ids: list[str] | None = None
    canonical_baselines: dict[str, dict[str, Any]] = {}
    canonical_autoresearch_provenance: dict[str, Any] | None = None
    canonical_input_binding: dict[str, Any] | None = None
    canonical_supervisor_protocol: dict[str, Any] | None = None
    source_campaign_identity: dict[str, str] | None = None
    campaign_identities: list[dict[str, str]] = []

    for run_index, root in enumerate(run_roots, start=1):
        run_name = f"run_{run_index:02d}"
        campaign_manifest, campaign_identity = validate_campaign_identity(root)
        current_input_binding = {
            "hamlib_zip_checksums_sha256": campaign_manifest.get("hamlib_zip_checksums_sha256"),
            "hamlib_archives": campaign_manifest.get("hamlib_archives"),
        }
        if canonical_input_binding is None:
            canonical_input_binding = current_input_binding
        elif current_input_binding != canonical_input_binding:
            raise RuntimeError(f"{run_name}: immutable HamLib ZIP binding differs from run_01")
        current_identity = {
            "run": run_name,
            "campaign_id": str(campaign_manifest["campaign_id"]),
            "campaign_manifest_sha256": str(campaign_identity["campaign_manifest_sha256"]),
        }
        if any(
            item["campaign_id"] == current_identity["campaign_id"] for item in campaign_identities
        ):
            raise RuntimeError(f"{run_name}: replicate campaign ID is not independent")
        campaign_identities.append(current_identity)
        if run_index == 1:
            source_campaign_identity = current_identity
        supervisor_protocol = validate_supervisor_provenance(
            root,
            manifest=campaign_manifest,
            identity=campaign_identity,
            require_paper_protocol=complete_paper_export,
        )
        if canonical_supervisor_protocol is None:
            canonical_supervisor_protocol = supervisor_protocol
        elif supervisor_protocol != canonical_supervisor_protocol:
            raise RuntimeError(f"{run_name}: supervisor protocol differs from run_01")
        runner_provenance = validate_runner_provenance(root)
        autoresearch_path = root / "manifests" / "autoresearch_provenance.json"
        if not autoresearch_path.is_file():
            raise RuntimeError(f"{run_name}: missing autoresearch_provenance.json")
        autoresearch_provenance = load_json(autoresearch_path)
        expected_provenance_keys = {
            "schema_version",
            "model",
            "reasoning_effort",
            "codex_cli_version",
            "node_version",
            "npm_cache",
            "runner_bundle_sha256",
            "uv_lock_sha256",
            "codex_exec_flags",
        }
        if set(autoresearch_provenance) != expected_provenance_keys:
            raise RuntimeError(f"{run_name}: malformed autoresearch provenance")
        if autoresearch_provenance["schema_version"] != 1 or any(
            not isinstance(autoresearch_provenance[key], str) or not autoresearch_provenance[key]
            for key in (
                "model",
                "reasoning_effort",
                "codex_cli_version",
                "node_version",
                "npm_cache",
            )
        ):
            raise RuntimeError(f"{run_name}: malformed autoresearch provenance values")
        if autoresearch_provenance["codex_exec_flags"] != list(PAPER_CODEX_EXEC_FLAGS):
            raise RuntimeError(f"{run_name}: mutation sandbox flags differ from paper protocol")
        if (
            autoresearch_provenance["runner_bundle_sha256"]
            != runner_provenance["runner_bundle_sha256"]
            or autoresearch_provenance["uv_lock_sha256"] != runner_provenance["uv_lock_sha256"]
        ):
            raise RuntimeError(f"{run_name}: autoresearch and runner provenance differ")
        if complete_paper_export:
            validate_paper_autoresearch_provenance(autoresearch_provenance)
        if canonical_autoresearch_provenance is None:
            canonical_autoresearch_provenance = autoresearch_provenance
        elif autoresearch_provenance != canonical_autoresearch_provenance:
            raise RuntimeError(f"{run_name}: autoresearch provenance differs from run_01")
        lanes, by_lane = load_run(
            root,
            expected_iterations=args.expected_iterations,
            allow_incomplete=args.allow_incomplete,
        )
        lane_ids = [str(lane["lane_id"]) for lane in lanes]
        if len(lane_ids) != args.expected_lanes:
            raise RuntimeError(
                f"{run_name}: expected {args.expected_lanes} lanes, found {len(lane_ids)}"
            )
        if paper_shape:
            validate_canonical_lanes(lanes)
            validate_canonical_campaign_manifest(root)
        if canonical_lane_ids is None:
            canonical_lane_ids = lane_ids
        elif lane_ids != canonical_lane_ids:
            raise RuntimeError(f"{run_name}: lane set differs from run_01")

        accepted_mutations = 0
        rejected_mutations = 0
        final_reductions: list[float] = []
        trajectories: dict[int, list[float]] = defaultdict(list)

        for lane in lanes:
            lane_id = str(lane["lane_id"])
            rows = by_lane[lane_id]
            if not rows:
                continue
            baseline_row = next(row for row in rows if int(row["iteration"]) == 0)
            baseline_result = baseline_row["result"]
            baseline_result_path = root / "iterations" / "iter_0000" / "results" / f"{lane_id}.json"
            replication = baseline_result.get("baseline_replication")
            if run_index == 1:
                if replication is not None:
                    raise RuntimeError(
                        f"{run_name}/{lane_id}: source baseline cannot be replicated"
                    )
            else:
                assert source_campaign_identity is not None
                source_result_path = (
                    run_roots[0] / "iterations" / "iter_0000" / "results" / f"{lane_id}.json"
                )
                expected_replication = {
                    "schema_version": 1,
                    "source_campaign_id": source_campaign_identity["campaign_id"],
                    "source_manifest_sha256": source_campaign_identity["campaign_manifest_sha256"],
                    "source_result_sha256": sha256_file(source_result_path),
                }
                if replication != expected_replication:
                    raise RuntimeError(
                        f"{run_name}/{lane_id}: baseline replication provenance drift"
                    )
            if not baseline_result_path.is_file():
                raise RuntimeError(f"{run_name}/{lane_id}: baseline result file is missing")
            baseline_t = exact_t_count(baseline_result)
            baseline_snapshot = policy_path(root, lane_id, 0)
            baseline_fingerprint = {
                "policy_sha256": sha256_file(baseline_snapshot),
                "config": config_fields(baseline_result),
                "metrics": metric_fields(baseline_result),
            }
            if lane_id not in canonical_baselines:
                canonical_baselines[lane_id] = baseline_fingerprint
            elif baseline_fingerprint != canonical_baselines[lane_id]:
                raise RuntimeError(f"{run_name}/{lane_id}: baseline differs from run_01")
            if run_index == 1:
                gpu = baseline_result.get("gpu_preflight") or {}
                execution = baseline_result.get("execution_configuration") or {}
                archive = baseline_result.get("hamlib_archive") or {}
                baseline_provenance_rows.append(
                    {
                        "lane_id": lane_id,
                        "campaign_id": baseline_result.get("campaign_id"),
                        "campaign_manifest_sha256": baseline_result.get("campaign_manifest_sha256"),
                        "result_sha256": sha256_file(baseline_result_path),
                        "baseline_cache_sha256": baseline_result.get("baseline_cache_sha256"),
                        "policy_sha256": baseline_result.get("policy_sha256"),
                        "initial_policy_sha256": baseline_result.get("initial_policy_sha256"),
                        "batch_manifest_sha256": baseline_result.get("batch_manifest_sha256"),
                        "runner_bundle_sha256": (
                            baseline_result.get("runner_provenance") or {}
                        ).get("runner_bundle_sha256"),
                        "archive_key": archive.get("archive_key"),
                        "archive_relative_path": archive.get("relative_path"),
                        "archive_sha256": archive.get("sha256"),
                        "baseline_backend": baseline_result.get("baseline_backend"),
                        "baseline_accelerator": baseline_result.get("baseline_accelerator"),
                        "baseline_array_type": baseline_result.get("baseline_array_type"),
                        "gpu_name": gpu.get("gpu_name"),
                        "gpu_uuid": gpu.get("gpu_uuid"),
                        "gpu_memory_mib": gpu.get("gpu_memory_mib"),
                        "cuda_visible_devices": gpu.get("cuda_visible_devices"),
                        "preflight_array_type": gpu.get("preflight_array_type"),
                        "nvidia_smi_path": gpu.get("nvidia_smi_path"),
                        "slurm_job_id": execution.get("slurm_job_id"),
                        "slurm_account": execution.get("slurm_account"),
                        "slurm_partition": execution.get("slurm_partition"),
                        "ground_state_energy_hartree": baseline_result.get(
                            "ground_state_energy_hartree"
                        ),
                    }
                )
            incumbent_row = baseline_row
            incumbent_t = baseline_t
            for iteration in range(0, args.expected_iterations + 1):
                row = next((item for item in rows if int(item["iteration"]) == iteration), None)
                if row is not None and iteration > 0:
                    accepted = bool(row["accepted"])
                    accepted_mutations += int(accepted)
                    rejected_mutations += int(not accepted)
                    if accepted:
                        incumbent_row = row
                        incumbent_t = exact_t_count(row["result"])

                    result = row["result"]
                    snapshot = policy_path(root, lane_id, iteration)
                    all_candidate_rows.append(
                        {
                            "run": run_name,
                            **lane,
                            "iteration": iteration,
                            "accepted": int(accepted),
                            "decision_reason": row.get("decision_reason"),
                            "relative_gain_percent": row.get("relative_gain_percent"),
                            "status": row.get("status"),
                            "passes_accuracy": int(
                                bool(result.get("passes_chemical_accuracy_bound", False))
                            ),
                            **config_fields(result),
                            **metric_fields(result),
                            "mutation_summary": sanitize_summary(
                                row.get("mutation_summary") or result.get("mutation_summary"),
                                run_root=root,
                                run_name=run_name,
                            ),
                            "policy_sha256": sha256_file(snapshot) if snapshot.is_file() else "",
                        }
                    )
                if row is not None or iteration == 0:
                    trajectories[iteration].append(100.0 * (baseline_t - incumbent_t) / baseline_t)

            best_result = incumbent_row["result"]
            best_iteration = int(incumbent_row["iteration"])
            reduction = 100.0 * (baseline_t - incumbent_t) / baseline_t
            final_reductions.append(reduction)
            lane_run_row = {
                "run": run_name,
                **lane,
                "accepted": sum(
                    int(row["accepted"] or 0) for row in rows if int(row["iteration"]) > 0
                ),
                "rejected": sum(
                    1 - int(row["accepted"] or 0) for row in rows if int(row["iteration"]) > 0
                ),
                "baseline_iteration": 0,
                **{
                    f"baseline_{key}": value
                    for key, value in config_fields(baseline_result).items()
                },
                **{
                    f"baseline_{key}": value
                    for key, value in metric_fields(baseline_result).items()
                },
                "baseline_policy_sha256": sha256_file(policy_path(root, lane_id, 0)),
                "best_iteration": best_iteration,
                **{f"best_{key}": value for key, value in config_fields(best_result).items()},
                **{f"best_{key}": value for key, value in metric_fields(best_result).items()},
                "best_mutation_summary": sanitize_summary(
                    incumbent_row.get("mutation_summary") or best_result.get("mutation_summary"),
                    run_root=root,
                    run_name=run_name,
                ),
                "reduction_percent": reduction,
                "best_policy_sha256": sha256_file(policy_path(root, lane_id, best_iteration)),
            }
            lane_run_rows.append(lane_run_row)
            lane_run_index[lane_id].append(lane_run_row)

        for iteration in sorted(trajectories):
            values = trajectories[iteration]
            trajectory_rows.append(
                {
                    "run": run_name,
                    "iteration": iteration,
                    "mean_reduction_percent": statistics.fmean(values),
                    "median_reduction_percent": statistics.median(values),
                    "lanes": len(values),
                }
            )
        campaign_rows.append(
            {
                "run": run_name,
                "lanes": len(lanes),
                "iterations": max(
                    (int(row["iteration"]) for rows in by_lane.values() for row in rows), default=-1
                ),
                "accepted": accepted_mutations,
                "rejected": rejected_mutations,
                "acceptance_percent": 100.0
                * accepted_mutations
                / max(1, accepted_mutations + rejected_mutations),
                "mean_reduction_percent": statistics.fmean(final_reductions),
                "median_reduction_percent": statistics.median(final_reductions),
            }
        )

    lane_aggregate_rows: list[dict[str, Any]] = []
    exported_policy_paths: set[Path] = set()
    policy_export_plan: list[tuple[Path, Path, str]] = []
    run_root_by_name = {f"run_{index:02d}": root for index, root in enumerate(run_roots, start=1)}
    for lane_id in canonical_lane_ids or []:
        rows = lane_run_index[lane_id]
        best = min(rows, key=lambda row: int(row["best_total_t_gates_qpe"]))
        reductions = [float(row["reduction_percent"]) for row in rows]
        best_iteration = int(best["best_iteration"])
        initial_source = policy_path(run_roots[0], lane_id, 0)
        best_source = policy_path(run_root_by_name[str(best["run"])], lane_id, best_iteration)
        initial_relative = Path("policies") / lane_id / "initial_simple_pf.py"
        best_relative = (
            Path("policies")
            / lane_id
            / f"best_{best['run']}_iter_{best_iteration:04d}_simple_pf.py"
        )
        exported_policy_paths.update((initial_relative, best_relative))
        initial_hash = sha256_file(initial_source)
        best_hash = sha256_file(best_source)
        if initial_hash != str(best["baseline_policy_sha256"]):
            raise RuntimeError(f"{lane_id}: exported initial policy hash mismatch")
        if best_hash != str(best["best_policy_sha256"]):
            raise RuntimeError(f"{lane_id}: exported best policy hash mismatch")
        policy_export_plan.extend(
            (
                (initial_source, initial_relative, initial_hash),
                (best_source, best_relative, best_hash),
            )
        )

        lane_aggregate_rows.append(
            {
                "lane_id": lane_id,
                "molecule": best["molecule"],
                "nqubits": best["nqubits"],
                "electrons": best["electrons"],
                "terms": best["terms"],
                "one_norm": best["one_norm"],
                "accepted": sum(int(row["accepted"]) for row in rows),
                "rejected": sum(int(row["rejected"]) for row in rows),
                "acceptance_percent": 100.0
                * sum(int(row["accepted"]) for row in rows)
                / max(1, sum(int(row["accepted"]) + int(row["rejected"]) for row in rows)),
                "mean_reduction_percent": statistics.fmean(reductions),
                "std_reduction_percent": statistics.stdev(reductions)
                if len(reductions) > 1
                else 0.0,
                "best_run": best["run"],
                **{
                    key: value
                    for key, value in best.items()
                    if key.startswith("baseline_") or key.startswith("best_")
                },
                "baseline_policy_file": initial_relative.as_posix(),
                "best_policy_file": best_relative.as_posix(),
            }
        )

    run_means = [float(row["mean_reduction_percent"]) for row in campaign_rows]
    grand_mean = statistics.fmean(run_means)
    run_standard_deviation = statistics.stdev(run_means) if len(run_means) > 1 else 0.0
    standard_error = (
        run_standard_deviation / math.sqrt(len(run_means)) if len(run_means) > 1 else 0.0
    )
    if len(run_means) == 5:
        half_width = T_CRITICAL_95_DF4 * standard_error
        confidence_interval: list[float] | None = [
            grand_mean - half_width,
            grand_mean + half_width,
        ]
        confidence_interval_method = (
            "two-sided Student-t interval with 4 degrees of freedom over five independent "
            "campaign-level means; each campaign mean averages its Hamiltonians"
        )
    else:
        confidence_interval = None
        confidence_interval_method = (
            "not computed; the paper confidence interval requires exactly five campaigns"
        )
    expected_candidate_rows = args.runs * args.expected_lanes * args.expected_iterations
    expected_lane_run_rows = args.runs * args.expected_lanes
    expected_policy_sources = 2 * args.expected_lanes
    if not args.allow_incomplete:
        completion_counts = {
            "candidate_rows": (len(all_candidate_rows), expected_candidate_rows),
            "lane_run_rows": (len(lane_run_rows), expected_lane_run_rows),
            "lane_aggregate_rows": (len(lane_aggregate_rows), args.expected_lanes),
            "baseline_provenance_rows": (
                len(baseline_provenance_rows),
                args.expected_lanes,
            ),
            "policy_sources": (
                len(exported_policy_paths),
                expected_policy_sources,
            ),
        }
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in completion_counts.items()
            if actual != expected
        }
        if mismatches:
            raise RuntimeError(f"final export count mismatch: {mismatches}")
    summary = {
        "source_layout": (f"external run_01 through run_{args.runs:02d} campaign workspaces"),
        "runs": len(run_roots),
        "lanes_per_run": args.expected_lanes,
        "iterations_per_lane": args.expected_iterations,
        "complete": not args.allow_incomplete and paper_shape and not args.diagnostic_shape,
        "paper_contract_shape": paper_shape,
        "diagnostic_export": bool(args.diagnostic_shape),
        "run_mean_reductions_percent": run_means,
        "mean_reduction_percent": grand_mean,
        "run_standard_deviation_percent": run_standard_deviation,
        "standard_error_percent": standard_error,
        "confidence_interval_95_percent": confidence_interval,
        "confidence_interval_method": confidence_interval_method,
        "accepted_mutations": sum(int(row["accepted"]) for row in campaign_rows),
        "rejected_mutations": sum(int(row["rejected"]) for row in campaign_rows),
        "exported_candidate_rows": len(all_candidate_rows),
        "exported_lane_run_rows": len(lane_run_rows),
        "exported_lane_aggregate_rows": len(lane_aggregate_rows),
        "exported_baseline_provenance_rows": len(baseline_provenance_rows),
        "exported_policy_sources": len(exported_policy_paths),
        "hamlib_zip_checksum_manifest": "hamlib_zip_sha256.txt",
        "expected_complete_counts": {
            "candidate_rows": expected_candidate_rows,
            "lane_run_rows": expected_lane_run_rows,
            "lane_aggregate_rows": args.expected_lanes,
            "baseline_provenance_rows": args.expected_lanes,
            "policy_sources": expected_policy_sources,
        },
        "provenance_file": "provenance.json",
    }
    if canonical_autoresearch_provenance is None:
        raise RuntimeError("no autoresearch provenance was loaded")
    export_provenance = {
        **canonical_autoresearch_provenance,
        "replicate_runs_verified": len(run_roots),
        "campaign_identities": campaign_identities,
        "hamlib_zip_checksums_sha256": None
        if canonical_input_binding is None
        else canonical_input_binding["hamlib_zip_checksums_sha256"],
        "hamlib_archive_count": 0
        if canonical_input_binding is None
        else len(canonical_input_binding["hamlib_archives"]),
        "supervisor_protocol": canonical_supervisor_protocol,
    }
    baseline_steps = [float(row["baseline_n_steps"]) for row in lane_aggregate_rows]
    best_steps = [float(row["best_n_steps"]) for row in lane_aggregate_rows]
    policy_distribution_summary = {
        "hamiltonians": len(lane_aggregate_rows),
        "baseline_formula_order_counts": dict(
            sorted(
                Counter(
                    str(int(float(row["baseline_formula_order"]))) for row in lane_aggregate_rows
                ).items()
            )
        ),
        "best_formula_order_counts": dict(
            sorted(
                Counter(
                    str(int(float(row["best_formula_order"]))) for row in lane_aggregate_rows
                ).items()
            )
        ),
        "baseline_term_order_counts": dict(
            sorted(
                Counter(
                    str(row["baseline_term_order_strategy"]) for row in lane_aggregate_rows
                ).items()
            )
        ),
        "best_term_order_counts": dict(
            sorted(
                Counter(str(row["best_term_order_strategy"]) for row in lane_aggregate_rows).items()
            )
        ),
        "best_run_counts": dict(
            sorted(Counter(str(row["best_run"]) for row in lane_aggregate_rows).items())
        ),
        "baseline_step_count": numeric_summary(baseline_steps),
        "best_step_count": numeric_summary(best_steps),
        "formula_order_changed": sum(
            int(float(row["baseline_formula_order"])) != int(float(row["best_formula_order"]))
            for row in lane_aggregate_rows
        ),
        "term_order_changed": sum(
            str(row["baseline_term_order_strategy"]) != str(row["best_term_order_strategy"])
            for row in lane_aggregate_rows
        ),
        "step_count_changed": sum(
            int(float(row["baseline_n_steps"])) != int(float(row["best_n_steps"]))
            for row in lane_aggregate_rows
        ),
    }

    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging-",
            dir=final_output_dir.parent,
        )
    )
    try:
        for source, relative, expected_sha256 in policy_export_plan:
            destination = staging_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != expected_sha256:
                raise RuntimeError(
                    f"policy changed while exporting: {source} -> {relative.as_posix()}"
                )
        checksum_manifest_source = RUNNER_ROOT / "manifests" / "hamlib_zip_sha256.txt"
        checksum_manifest_destination = staging_dir / "hamlib_zip_sha256.txt"
        shutil.copy2(checksum_manifest_source, checksum_manifest_destination)
        if sha256_file(checksum_manifest_destination) != sha256_file(checksum_manifest_source):
            raise RuntimeError("HamLib ZIP checksum manifest changed while exporting")
        write_csv(
            staging_dir / "baseline_provenance.csv",
            baseline_provenance_rows,
            list(baseline_provenance_rows[0]) if baseline_provenance_rows else [],
        )
        write_csv(
            staging_dir / "candidate_log.csv",
            all_candidate_rows,
            list(all_candidate_rows[0]) if all_candidate_rows else [],
        )
        write_csv(
            staging_dir / "lane_run_summary.csv",
            lane_run_rows,
            list(lane_run_rows[0]) if lane_run_rows else [],
        )
        write_csv(
            staging_dir / "lane_aggregate_summary.csv",
            lane_aggregate_rows,
            list(lane_aggregate_rows[0]) if lane_aggregate_rows else [],
        )
        write_csv(
            staging_dir / "campaign_summary.csv",
            campaign_rows,
            list(campaign_rows[0]) if campaign_rows else [],
        )
        write_csv(
            staging_dir / "trajectory_summary.csv",
            trajectory_rows,
            list(trajectory_rows[0]) if trajectory_rows else [],
        )
        write_json(staging_dir / "policy_distribution_summary.json", policy_distribution_summary)
        write_json(staging_dir / "provenance.json", export_provenance)
        write_json(staging_dir / "summary.json", summary)
        if summary["complete"]:
            (staging_dir / "FINAL_EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
        write_checksum_manifest(staging_dir)
        assert_portable_export(staging_dir)
        staging_dir.replace(final_output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import secrets
import shutil
import stat
import sys
import tempfile
from pathlib import Path

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
    DEFAULT_CLUSTER_ROOT,
    DEFAULT_HAMLIB_DATA_ROOT,
    Lane,
    db_connect,
    ensure_dirs,
    fit_lane,
    derive_campaign_id,
    ensure_runner_provenance,
    init_db,
    load_json,
    log_event,
    policy_template,
    read_manifest_rows,
    write_json,
    now_iso,
    parse_hamlib_zip_checksums,
    sha256_file,
    validate_absolute_posix_path,
    validate_campaign_identity,
    validate_canonical_hamlib_source_manifest,
    validate_canonical_hamlib_zip_checksum_manifest,
)


RUNNER_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE_MANIFEST = (
    RUNNER_ROOT / "manifests" / "hamlib_electronic_active_space_instances_le32_qubits.csv"
)
CANONICAL_ACTIVE_MANIFEST = RUNNER_ROOT / "manifests" / "active_lanes.csv"
EXPECTED_SOURCE_ROWS = 401
EXPECTED_ACTIVE_ROWS = 121
EXPECTED_MAX_QUBITS = 32
EXPECTED_MEMORY_GIB = 64.0
EXPECTED_WORKSPACE_FACTOR = 4.0


def _csv_value(value: object) -> str:
    return str(value)


def validate_benchmark_selection(
    source_rows: list[dict[str, str]],
    active_lanes: list[Lane],
    expected_manifest: Path,
) -> None:
    if len(source_rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"fixed HamLib source must contain {EXPECTED_SOURCE_ROWS} rows, found {len(source_rows)}"
        )
    with expected_manifest.open(encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    if len(expected_rows) != EXPECTED_ACTIVE_ROWS:
        raise RuntimeError(
            f"canonical active manifest must contain {EXPECTED_ACTIVE_ROWS} rows, "
            f"found {len(expected_rows)}"
        )
    if len(active_lanes) != EXPECTED_ACTIVE_ROWS:
        raise RuntimeError(
            f"fixed 64 GiB/workspace-factor-4 filter selected {len(active_lanes)} lanes; "
            f"expected {EXPECTED_ACTIVE_ROWS}"
        )
    for index, (lane, expected) in enumerate(zip(active_lanes, expected_rows, strict=True)):
        actual = {key: _csv_value(value) for key, value in lane.__dict__.items()}
        if actual != expected:
            differing = sorted(
                key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
            )
            raise RuntimeError(
                f"canonical lane mismatch at row {index + 1} ({lane.lane_id}): {differing}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a HamLib autoresearch campaign.")
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument(
        "--hamlib-zip-checksums",
        type=Path,
        default=RUNNER_ROOT / "manifests" / "hamlib_zip_sha256.txt",
        help=(
            "GNU SHA256SUMS-format file binding exactly the 83 ZIP archives used by "
            "the canonical 121 lanes; paths are relative to --hamlib-data-root"
        ),
    )
    parser.add_argument("--campaign-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-qubits", type=int, default=32)
    parser.add_argument(
        "--a100-memory-gib",
        "--memory-gib",
        dest="memory_gib",
        type=float,
        default=64.0,
        help="Memory fit threshold in GiB. The legacy --a100-memory-gib spelling is kept for compatibility.",
    )
    parser.add_argument("--accelerator-label", default="a100")
    parser.add_argument(
        "--active-min-memory-gib",
        type=float,
        default=None,
        help=(
            "Only mark lanes active if at least one workspace estimate exceeds this lower memory threshold. "
            "Useful for preparing an H100-only expansion beyond an A100-fit set."
        ),
    )
    parser.add_argument("--workspace-factor", type=float, default=4.0)
    parser.add_argument("--cluster-root", default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--hamlib-data-root", default=DEFAULT_HAMLIB_DATA_ROOT)
    parser.add_argument(
        "--include-nonfit", action="store_true", help="Create inactive lanes for non-fit rows too."
    )
    args = parser.parse_args()

    if not args.cluster_root:
        parser.error("pass --cluster-root or set HSOPT_HAMLIB_CLUSTER_ROOT")
    if not args.hamlib_data_root:
        parser.error("pass --hamlib-data-root or set HSOPT_HAMLIB_DATA_ROOT")
    args.cluster_root = validate_absolute_posix_path(args.cluster_root, "cluster_root")
    args.hamlib_data_root = validate_absolute_posix_path(args.hamlib_data_root, "hamlib_data_root")
    fixed_parameters = {
        "max_qubits": (args.max_qubits, EXPECTED_MAX_QUBITS),
        "memory_gib": (args.memory_gib, EXPECTED_MEMORY_GIB),
        "workspace_factor": (args.workspace_factor, EXPECTED_WORKSPACE_FACTOR),
    }
    drift = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in fixed_parameters.items()
        if actual != expected
    }
    if args.accelerator_label.lower() != "a100":
        drift["accelerator_label"] = {
            "actual": args.accelerator_label,
            "expected": "a100",
        }
    if args.active_min_memory_gib is not None:
        drift["active_min_memory_gib"] = {
            "actual": args.active_min_memory_gib,
            "expected": None,
        }
    if drift:
        parser.error(f"fixed HamLib121 benchmark parameters changed: {drift}")

    root = args.campaign_root.resolve()
    campaign_manifest_path = root / "manifests" / "campaign_manifest.json"
    campaign_identity_path = root / "manifests" / "campaign_identity.json"
    manifest_exists = campaign_manifest_path.exists()
    identity_exists = campaign_identity_path.exists()
    if identity_exists and not manifest_exists:
        raise RuntimeError(
            "campaign preparation has an identity but no manifest; refusing ambiguity"
        )
    resume_preparation = manifest_exists
    paths = ensure_dirs(root)
    ensure_runner_provenance(campaign_root=root, runner_root=RUNNER_ROOT)
    validate_canonical_hamlib_source_manifest(args.source_csv)
    source_copy = paths["manifests"] / CANONICAL_SOURCE_MANIFEST.name
    if args.source_csv.resolve() != source_copy.resolve():
        shutil.copy2(args.source_csv, source_copy)

    rows = read_manifest_rows(source_copy, max_qubits=args.max_qubits)
    lanes = [
        fit_lane(row, index, memory_gib=args.memory_gib, workspace_factor=args.workspace_factor)
        for index, row in enumerate(rows)
    ]
    target_fit_lanes = [lane for lane in lanes if lane.exact_fit]
    if args.active_min_memory_gib is None:
        active_lanes = target_fit_lanes
    else:
        active_lanes = [
            lane
            for lane in target_fit_lanes
            if (
                lane.estimated_state_workspace_gib > args.active_min_memory_gib
                or lane.estimated_sector_workspace_gib > args.active_min_memory_gib
            )
        ]
    active_lane_ids = {lane.lane_id for lane in active_lanes}
    validate_benchmark_selection(rows, active_lanes, CANONICAL_ACTIVE_MANIFEST)
    validate_canonical_hamlib_zip_checksum_manifest(args.hamlib_zip_checksums)
    archive_records = parse_hamlib_zip_checksums(args.hamlib_zip_checksums, active_lanes)
    checksum_copy = paths["manifests"] / "hamlib_zip_sha256.txt"
    if args.hamlib_zip_checksums.resolve() != checksum_copy.resolve():
        shutil.copy2(args.hamlib_zip_checksums, checksum_copy)
    active_manifest_copy = paths["manifests"] / CANONICAL_ACTIVE_MANIFEST.name
    if CANONICAL_ACTIVE_MANIFEST.resolve() != active_manifest_copy.resolve():
        shutil.copy2(CANONICAL_ACTIVE_MANIFEST, active_manifest_copy)

    paper_protocol = {
        "replicate_runs": 5,
        "active_lanes": EXPECTED_ACTIVE_ROWS,
        "proposal_iterations": 50,
        "baseline_backend": "cupynumeric",
        "baseline_accelerator": "NVIDIA A100",
        "baseline_replicated_from_run_01": True,
        "policy_backend": "cupynumeric",
        "streaming_batch_size": 8,
        "max_concurrent_mutators": 4,
        "max_running_eval_jobs": 4,
        "node_major": 24,
        "mutation_agent": "opencode",
        "mutation_package": "opencode-ai@1.4.11",
        "codex_cli_version": "1.4.11",
        "model": "nvidia/nvidia/nemotron-3-super-v3",
        "reasoning_effort": "not_applicable",
    }
    fixed_manifest_fields = {
        "source_csv": str(source_copy),
        "source_csv_sha256": sha256_file(source_copy),
        "hamlib_zip_checksums_file": "manifests/hamlib_zip_sha256.txt",
        "hamlib_zip_checksums_sha256": sha256_file(checksum_copy),
        "hamlib_archives": archive_records,
        "requested_max_qubits": args.max_qubits,
        "requested_count": len(lanes),
        "target_fit_count": len(target_fit_lanes),
        "active_fit_count": len(active_lanes),
        "active_a100_fit_count": len(active_lanes),
        "accelerator_label": args.accelerator_label,
        "memory_gib": args.memory_gib,
        "a100_memory_gib": args.memory_gib,
        "active_min_memory_gib": args.active_min_memory_gib,
        "workspace_factor": args.workspace_factor,
        "paper_protocol": paper_protocol,
        "cluster_root": args.cluster_root,
        "hamlib_data_root": args.hamlib_data_root,
        "lanes": [lane.__dict__ for lane in lanes],
    }
    if resume_preparation:
        if identity_exists:
            manifest, _identity = validate_campaign_identity(root)
        else:
            if campaign_manifest_path.is_symlink() or not campaign_manifest_path.is_file():
                raise RuntimeError("partial campaign manifest is missing or unsafe")
            manifest = load_json(campaign_manifest_path)
            nonce = manifest.get("preparation_nonce")
            campaign_id = manifest.get("campaign_id")
            if (
                not isinstance(nonce, str)
                or not isinstance(campaign_id, str)
                or derive_campaign_id(nonce) != campaign_id
            ):
                raise RuntimeError("partial campaign manifest identity is malformed")
        drift = {
            key: {"actual": manifest.get(key), "expected": value}
            for key, value in fixed_manifest_fields.items()
            if manifest.get(key) != value
        }
        if manifest.get("schema_version") != 2 or drift:
            raise RuntimeError(f"existing partial campaign preparation differs: {drift}")
        campaign_id = str(manifest["campaign_id"])
        if not identity_exists:
            write_json(
                paths["identity"],
                {
                    "schema_version": 1,
                    "campaign_id": campaign_id,
                    "campaign_manifest_sha256": sha256_file(campaign_manifest_path),
                },
            )
    else:
        preparation_nonce = secrets.token_hex(32)
        campaign_id = derive_campaign_id(preparation_nonce)
        manifest = {
            "schema_version": 2,
            "created_at": now_iso(),
            "preparation_nonce": preparation_nonce,
            "campaign_id": campaign_id,
            **fixed_manifest_fields,
        }
        campaign_manifest_path = paths["manifests"] / "campaign_manifest.json"
        write_json(campaign_manifest_path, manifest)
        write_json(
            paths["identity"],
            {
                "schema_version": 1,
                "campaign_id": campaign_id,
                "campaign_manifest_sha256": sha256_file(campaign_manifest_path),
            },
        )

    conn = db_connect(root)
    init_db(conn)
    # Re-preparing a campaign with a narrower fit set must not leave older lanes
    # active in SQLite after the manifest/policies have been regenerated.
    conn.execute("UPDATE lanes SET active=0, status='inactive', updated_at=?", (now_iso(),))
    for lane in lanes:
        is_active = lane.lane_id in active_lane_ids
        if not is_active and not args.include_nonfit:
            continue
        lane_dir = paths["lanes"] / lane.lane_id
        lane_dir.mkdir(parents=True, exist_ok=True)
        policy = lane_dir / "simple_pf.py"
        initial_policy = lane_dir / "initial_simple_pf.py"
        expected_policy = policy_template(lane).encode("utf-8")
        for target in (initial_policy, policy):
            if target.exists():
                if target.is_symlink() or not target.is_file():
                    raise RuntimeError(f"campaign policy path is unsafe: {target}")
                if target.read_bytes() != expected_policy:
                    raise RuntimeError(
                        f"existing campaign preparation has changed policy bytes: {target}"
                    )
                continue
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(expected_policy)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        write_json(lane_dir / "lane.json", lane.__dict__)
        conn.execute(
            """
            INSERT INTO lanes(
              lane_id, manifest_index, collection, file, dataset, molecule, nqubits, electrons,
              terms, one_norm, exact_fit, fit_reason, sector_dim, estimated_state_workspace_gib,
              estimated_sector_workspace_gib, active, current_policy_path, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lane_id) DO UPDATE SET
              exact_fit=excluded.exact_fit,
              fit_reason=excluded.fit_reason,
              active=excluded.active,
              current_policy_path=excluded.current_policy_path,
              updated_at=excluded.updated_at
            """,
            (
                lane.lane_id,
                lane.index,
                lane.collection,
                lane.file,
                lane.dataset,
                lane.molecule,
                lane.nqubits,
                lane.electrons,
                lane.terms,
                lane.one_norm,
                1 if lane.exact_fit else 0,
                lane.fit_reason,
                lane.sector_dim,
                lane.estimated_state_workspace_gib,
                lane.estimated_sector_workspace_gib,
                1 if is_active else 0,
                str(policy),
                now_iso(),
            ),
        )
    conn.commit()
    log_event(
        conn,
        kind="prepare",
        message=(
            f"prepared {len(active_lanes)} active {args.accelerator_label}-fit lanes "
            f"from {len(target_fit_lanes)} target-fit lanes and {len(lanes)} requested rows"
        ),
        payload={
            "campaign_id": campaign_id,
            "requested": len(lanes),
            "target_fit": len(target_fit_lanes),
            "active": len(active_lanes),
            "accelerator_label": args.accelerator_label,
            "active_min_memory_gib": args.active_min_memory_gib,
        },
    )
    print(
        f"prepared requested={len(lanes)} active_{args.accelerator_label}_fit={len(active_lanes)} "
        f"target_fit={len(target_fit_lanes)} memory_gib={args.memory_gib} root={root}"
    )


if __name__ == "__main__":
    main()

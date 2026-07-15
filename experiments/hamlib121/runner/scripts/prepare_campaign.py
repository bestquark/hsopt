#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from campaign_common import (
    DEFAULT_CLUSTER_ROOT,
    DEFAULT_HAMLIB_DATA_ROOT,
    Lane,
    db_connect,
    ensure_dirs,
    fit_lane,
    ensure_runner_provenance,
    init_db,
    log_event,
    policy_template,
    read_manifest_rows,
    write_json,
    now_iso,
)


RUNNER_ROOT = Path(__file__).resolve().parents[1]
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
    paths = ensure_dirs(root)
    ensure_runner_provenance(campaign_root=root, runner_root=RUNNER_ROOT)
    source_copy = paths["manifests"] / args.source_csv.name
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

    manifest = {
        "created_at": now_iso(),
        "source_csv": str(source_copy),
        "requested_max_qubits": args.max_qubits,
        "requested_count": len(lanes),
        "target_fit_count": len(target_fit_lanes),
        "active_fit_count": len(active_lanes),
        "active_a100_fit_count": len(active_lanes)
        if args.accelerator_label.lower() == "a100"
        else None,
        "accelerator_label": args.accelerator_label,
        "memory_gib": args.memory_gib,
        "a100_memory_gib": args.memory_gib,
        "active_min_memory_gib": args.active_min_memory_gib,
        "workspace_factor": args.workspace_factor,
        "cluster_root": args.cluster_root,
        "hamlib_data_root": args.hamlib_data_root,
        "lanes": [lane.__dict__ for lane in lanes],
    }
    write_json(paths["manifests"] / "campaign_manifest.json", manifest)

    with (paths["manifests"] / "active_lanes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(active_lanes[0].__dict__.keys()) if active_lanes else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for lane in active_lanes:
            writer.writerow(lane.__dict__)

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
        if not initial_policy.exists():
            initial_policy.write_text(policy_template(lane), encoding="utf-8")
        if not policy.exists():
            policy.write_bytes(initial_policy.read_bytes())
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

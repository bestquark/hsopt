#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from campaign_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse one deterministic iteration-0 baseline evaluation in replicate campaigns."
    )
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--target-campaign", type=Path, action="append", required=True)
    args = parser.parse_args()

    source = args.source_campaign.resolve()
    source_results = source / "iterations" / "iter_0000" / "results"
    result_paths = sorted(source_results.glob("*.json"))
    if not result_paths:
        raise RuntimeError(f"no iteration-0 results found under {source_results}")

    for target_arg in args.target_campaign:
        target = target_arg.resolve()
        target_results = target / "iterations" / "iter_0000" / "results"
        target_results.mkdir(parents=True, exist_ok=True)
        for source_path in result_paths:
            result = json.loads(source_path.read_text(encoding="utf-8"))
            lane_id = str(result["lane_id"])
            result.pop("accepted", None)
            result.pop("decision_reason", None)
            result.pop("relative_gain_percent", None)
            result.pop("incumbent_iteration", None)
            result.pop("incumbent_t_gates_qpe", None)
            result.pop("mutation_summary", None)
            result["iteration"] = 0
            result["policy_path"] = str(
                target / "iterations" / "iter_0001" / "mutations" / lane_id / "before_simple_pf.py"
            )
            result["policy_snapshot"] = "initial"
            result["baseline_result_reused_from"] = str(source)
            write_json(target_results / source_path.name, result)

        source_baselines = source / "baselines"
        target_baselines = target / "baselines"
        if source_baselines.is_dir():
            target_baselines.mkdir(parents=True, exist_ok=True)
            for baseline_path in source_baselines.glob("*.json"):
                shutil.copy2(baseline_path, target_baselines / baseline_path.name)
        print(f"reused {len(result_paths)} initial baseline results in {target}")


if __name__ == "__main__":
    main()

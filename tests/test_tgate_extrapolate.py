from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hsopt.tgate_opt.extrapolate import (
    build_extrapolate_parser,
    extrapolate_gain,
    load_campaign_gain_records,
    load_gain_table,
    run_extrapolate_cli,
)


def _write_gain_table(path: Path) -> None:
    rows = [
        {
            "molecule": "h2",
            "baseline_t_count": 1200,
            "optimized_t_count": 1000,
            "active_orbitals": 2,
            "total_pauli_rotation_terms": 14,
        },
        {
            "molecule": "lih",
            "baseline_t_count": 6000,
            "optimized_t_count": 3000,
            "active_orbitals": 4,
            "total_pauli_rotation_terms": 104,
        },
        {
            "molecule": "n2",
            "baseline_t_count": 12000,
            "optimized_t_count": 4000,
            "active_orbitals": 5,
            "total_pauli_rotation_terms": 391,
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_loads_explicit_gain_table_and_extrapolates(tmp_path: Path) -> None:
    table = tmp_path / "gains.csv"
    _write_gain_table(table)

    records = load_gain_table(table)
    result = extrapolate_gain(
        records,
        {
            "molecule": "femoco",
            "active_orbitals": 54,
            "total_pauli_rotation_terms": 54**4,
        },
        target_baseline_t_count=1e12,
    )

    assert len(records) == 3
    assert result["gain_factor"]["lower"] > 0
    assert result["gain_factor"]["upper"] > result["gain_factor"]["lower"]
    assert result["target_optimized_t_count"]["upper_worst_case"] > 0
    assert result["target_t_gate_savings"]["median"] < 1e12


def test_loads_campaign_events_as_first_valid_to_best_valid_gain(tmp_path: Path) -> None:
    run_dir = tmp_path / "campaign" / "h2" / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    events = [
        {
            "type": "checkpoint_evaluated",
            "iteration": 1,
            "result": {
                "molecule": "h2",
                "passes_accuracy_constraint": False,
                "t_gate_count": 500,
            },
        },
        {
            "type": "checkpoint_evaluated",
            "iteration": 2,
            "result": {
                "molecule": "h2",
                "passes_accuracy_constraint": True,
                "t_gate_count": 400,
            },
        },
        {
            "type": "checkpoint_evaluated",
            "iteration": 3,
            "result": {
                "molecule": "h2",
                "passes_accuracy_constraint": True,
                "t_gate_count": 250,
            },
        },
    ]
    with (run_dir / "events.jsonl").open("w") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    records = load_campaign_gain_records(tmp_path / "campaign")

    assert len(records) == 1
    assert records[0].molecule == "h2"
    assert records[0].baseline_t_count == 400
    assert records[0].optimized_t_count == 250


def test_extrapolate_cli_json_output(tmp_path: Path, capsys) -> None:
    table = tmp_path / "gains.csv"
    plot_path = tmp_path / "uncertainty.svg"
    _write_gain_table(table)
    parser = build_extrapolate_parser(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--no-use-default-campaign-root",
            "--gain-table",
            str(table),
            "--target-baseline-t-count",
            "1000000",
            "--plot",
            str(plot_path),
            "--json",
        ]
    )

    assert run_extrapolate_cli(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"]["molecule"] == "femoco"
    assert payload["target_baseline_t_count"] == 1000000
    assert payload["training"]["count"] == 3
    assert payload["plot_path"] == str(plot_path)
    assert plot_path.read_text().startswith("<?xml")
    assert "<svg" in plot_path.read_text()

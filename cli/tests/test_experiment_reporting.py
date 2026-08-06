from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from hsopt.evaluation.compiled_unitary import DIFFERENCE_ERROR_CONTRACTS

from hsopt.view import reporting as reports


def _hamlib_summary_row() -> dict[str, str]:
    return {
        "run": "run_01",
        "lane_id": "0000_fixture",
        "molecule": "H2",
        "nqubits": "4",
        "electrons": "2",
        "terms": "15",
        "accepted": "1",
        "rejected": "49",
        "baseline_formula_order": "2",
        "baseline_n_steps": "8",
        "baseline_schedule_strategy": "stationary_trotter",
        "baseline_term_order_strategy": "input",
        "baseline_qdrift_samples": "",
        "baseline_uniform_epsilon": "1e-8",
        "baseline_zero_angle_threshold": "0",
        "baseline_total_t_gates_qpe": "1000",
        "baseline_ground_state_energy_hartree": "-1.0",
        "baseline_ground_state_energy_estimate_hartree": "-0.999",
        "baseline_ground_state_energy_error_hartree": "0.001",
        "best_iteration": "7",
        "best_formula_order": "2",
        "best_n_steps": "6",
        "best_schedule_strategy": "stationary_trotter",
        "best_term_order_strategy": "custom",
        "best_qdrift_samples": "",
        "best_uniform_epsilon": "2e-8",
        "best_zero_angle_threshold": "1e-12",
        "best_total_t_gates_qpe": "800",
        "best_ground_state_energy_hartree": "-1.0",
        "best_ground_state_energy_estimate_hartree": "-0.9992",
        "best_ground_state_energy_error_hartree": "0.0008",
        "reduction_percent": "20",
    }


def _hamlib_candidate_row() -> dict[str, str]:
    return {
        "run": "run_01",
        "lane_id": "0000_fixture",
        "iteration": "7",
        "accepted": "1",
        "decision_reason": "strict_t_gate_improvement",
        "passes_accuracy": "1",
        "formula_order": "2",
        "n_steps": "6",
        "schedule_strategy": "stationary_trotter",
        "term_order_strategy": "custom",
        "qdrift_samples": "",
        "total_t_gates_qpe": "800",
        "ground_state_energy_hartree": "-1.0",
        "ground_state_energy_estimate_hartree": "-0.9992",
        "ground_state_energy_error_hartree": "0.0008",
        "mutation_summary": "Changed order to custom & reduced repetitions by 2.",
    }


def test_latex_escape_and_custom_order_are_safe() -> None:
    escaped = reports.latex_escape(r"A_B & 50% -> ΔE")
    assert r"A\_B \& 50\%" in escaped
    assert "DeltaE" in escaped
    assert reports._order("custom") == "custom"
    assert reports._schedule({"schedule_strategy": "custom", "n_steps": "9"}) == "custom"


def test_verify_strict_export_requires_exact_checksum_coverage(tmp_path: Path) -> None:
    marker = tmp_path / "FINAL_EXPORT_COMPLETE"
    marker.write_text("complete\n", encoding="utf-8")
    data = tmp_path / "candidate_log.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    checksum = tmp_path / "SHA256SUMS"
    checksum.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in (marker, data)
        )
        + "\n",
        encoding="utf-8",
    )
    reports.verify_strict_export(tmp_path)
    data.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(reports.ReportValidationError, match="checksum mismatch"):
        reports.verify_strict_export(tmp_path)


def test_hamlib_report_recomputes_exact_error() -> None:
    row = _hamlib_candidate_row()
    assert reports._assert_energy(row) == pytest.approx(0.0008)
    row["ground_state_energy_error_hartree"] = "0.0007"
    with pytest.raises(reports.ReportValidationError, match="arithmetic"):
        reports._assert_energy(row)


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is not installed")
def test_fixture_summary_and_extended_reports_compile(tmp_path: Path) -> None:
    summary = _hamlib_summary_row()
    candidate = _hamlib_candidate_row()
    outputs = {
        "summary.pdf": reports._hamlib_summary("0000_fixture", [summary]),
        "extended_log.pdf": reports._hamlib_extended("0000_fixture", [summary], [candidate]),
    }
    for name, tex in outputs.items():
        output = tmp_path / name
        reports._compile_pdf(tex, output, shutil.which("latexmk") or "latexmk")
        assert output.read_bytes().startswith(b"%PDF-")
        assert output.stat().st_size > 10_000
    repeated = tmp_path / "summary-repeated.pdf"
    reports._compile_pdf(outputs["summary.pdf"], repeated, shutil.which("latexmk") or "latexmk")
    assert (
        hashlib.sha256(repeated.read_bytes()).digest()
        == hashlib.sha256((tmp_path / "summary.pdf").read_bytes()).digest()
    )


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is not installed")
def test_reaction_summary_and_extended_reports_compile(tmp_path: Path) -> None:
    species = json.dumps(
        {
            "H2": {
                "formula_order": 2,
                "n_steps": 8,
                "schedule_strategy": "stationary_trotter",
                "term_order_strategy": "custom",
                "t_gate_count": 100,
            }
        }
    )
    rows = []
    for contract in DIFFERENCE_ERROR_CONTRACTS:
        for iteration, t_count in ((0, 1000), (1, 800)):
            rows.append(
                {
                    "reaction_key": "fixture",
                    "source_contract": contract,
                    "source_lane": f"fixture_{contract}",
                    "iteration": str(iteration),
                    "accepted": "true",
                    "decision_reason": (
                        "initial_baseline" if iteration == 0 else "strict_t_gate_improvement"
                    ),
                    "mutation_summary": "Baseline" if iteration == 0 else "Reduced repetitions",
                    "species_json": species,
                    "total_t_gate_count": str(t_count),
                    "exact_reaction_energy_hartree": "-1.0",
                    "pf_reaction_energy_hartree": "-0.999",
                    "signed_reaction_error_hartree": "0.001",
                    "absolute_reaction_error_hartree": "0.001",
                    "sum_absolute_species_error_hartree": "0.0014",
                    "contract_error_hartree": (
                        "0.0014" if contract == "individual_errors" else "0.001"
                    ),
                    "max_species_absolute_error_hartree": "0.0005",
                    "passes_contract": "true",
                }
            )
    selection = {
        "selected_source_contract": DIFFERENCE_ERROR_CONTRACTS[0],
        "selected_source_iteration": 1,
        "total_t_gate_count": 800,
        "absolute_reaction_error_hartree": 0.001,
        "sum_absolute_species_error_hartree": 0.0014,
        "contract_error_hartree": 0.0014,
        "max_species_absolute_error_hartree": 0.0005,
    }
    pooled = {
        "reaction": "H2 -> 2H",
        DIFFERENCE_ERROR_CONTRACTS[0]: selection,
        DIFFERENCE_ERROR_CONTRACTS[1]: {
            **selection,
            "selected_source_contract": DIFFERENCE_ERROR_CONTRACTS[1],
            "contract_error_hartree": 0.001,
        },
    }
    outputs = {
        "reaction-summary.pdf": reports._reaction_summary("fixture", rows, pooled),
        "reaction-extended.pdf": reports._reaction_extended("fixture", rows, pooled),
    }
    for name, tex in outputs.items():
        output = tmp_path / name
        reports._compile_pdf(tex, output, shutil.which("latexmk") or "latexmk")
        assert output.read_bytes().startswith(b"%PDF-")
        assert output.stat().st_size > 10_000

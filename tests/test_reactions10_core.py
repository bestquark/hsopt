from __future__ import annotations

from collections import Counter
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
BENCHMARK_PATH = REPO_ROOT / "experiments" / "reactions10" / "benchmark.py"
SEED_MANIFEST = REPO_ROOT / "experiments" / "reactions10" / "inputs" / "seed_manifest.json"


def _load_benchmark() -> ModuleType:
    module_name = "_hsopt_reactions10_core_test"
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.modules[module_name] = module
    try:
        sys.path.insert(0, str(EXAMPLES_ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


BENCHMARK = _load_benchmark()
EVALUATOR = BENCHMARK.reaction_evaluate
TARGET = 1.0 / 627.5094740631


def _atom_counts(geometry: str) -> Counter[str]:
    return Counter(line.split()[0] for line in geometry.splitlines() if line.strip())


def test_reaction_contract_is_local_exact_and_full_space() -> None:
    assert EVALUATOR.REACTION_TARGET_HARTREE == TARGET
    assert EVALUATOR.REACTION_TARGET_MHA == 1000.0 * TARGET
    assert len(EVALUATOR.ARCHIVED_EVALUATOR_SHA256) == 64
    assert EVALUATOR.ARCHIVED_EVALUATOR_SHA256 == (
        "b81d8e936f71c0cbab76ba8f0b134b4ecbf2c610d95aaca04136b7aeb5309425"
    )
    assert BENCHMARK.ARCHIVED_BENCHMARK_SHA256 == (
        "09b713ebd784614ec774c73c22c0228b084d82bd9a3ea176589caf06918d965e"
    )
    assert (
        "project_fixed_electron_sector"
        not in inspect.signature(EVALUATOR.evaluate_species).parameters
    )


def test_registry_matches_seed_manifest_and_balances_every_reaction() -> None:
    seed = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    assert list(BENCHMARK.REACTIONS) == seed["reaction_keys"]
    canonical_species = {}

    for reaction_key, reaction in BENCHMARK.REACTIONS.items():
        expected = seed["reactions"][reaction_key]
        assert reaction.key == reaction_key
        assert reaction.label == expected["label"]
        assert reaction.rationale == expected["rationale"]
        assert list(spec.key for spec in reaction.species) == expected["species"]
        assert [
            {"coefficient": coefficient, "species": species}
            for coefficient, species in reaction.terms
        ] == expected["terms"]

        by_key = {spec.key: spec for spec in reaction.species}
        atom_balance: Counter[str] = Counter()
        charge_balance = 0.0
        for coefficient, species in reaction.terms:
            spec = by_key[species]
            for atom, count in _atom_counts(spec.geometry).items():
                atom_balance[atom] += coefficient * count
            charge_balance += coefficient * spec.charge
            previous = canonical_species.setdefault(spec.key, spec)
            assert spec == previous
        assert all(value == 0 for value in atom_balance.values())
        assert charge_balance == 0.0

    assert len(canonical_species) == 21


def test_reaction_arithmetic_uses_coefficients_but_t_count_uses_unique_species(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "simple_pf.py"
    policy_path.write_text("# synthetic policy\n", encoding="utf-8")
    policy = SimpleNamespace(
        set_active_species=lambda _species: None,
        get_method_config=lambda: SimpleNamespace(
            formula_order=1,
            n_steps=1,
            zero_angle_threshold=0.0,
        ),
        choose_synthesis_epsilon=lambda *_args: 1.0e-6,
    )
    monkeypatch.setattr(BENCHMARK, "_load_policy", lambda _path, _hash: policy)
    monkeypatch.setattr(EVALUATOR, "validate_policy_module", lambda *_args: None)

    signed_errors = {
        "n2": 0.16 * TARGET,
        "h2": -0.16 * TARGET,
        "nh3": -0.16 * TARGET,
    }
    t_counts = {"n2": 10, "h2": 20, "nh3": 30}

    def fake_species(spec, _policy, _config, _cache_root):
        signed = signed_errors[spec.key]
        return {
            "species": spec.key,
            "exact_energy_hartree": 0.0,
            "pf_energy_estimate_hartree": signed,
            "signed_energy_error_hartree": signed,
            "absolute_energy_error_hartree": abs(signed),
            "t_gate_count": t_counts[spec.key],
            "scheduled_rotation_count": 1,
        }

    monkeypatch.setattr(EVALUATOR, "evaluate_species", fake_species)
    result = BENCHMARK.evaluate_policy(
        policy_path=policy_path,
        reaction_key="n2_h2_nh3",
        contract="equal_share",
        cache_root=tmp_path,
        iteration=7,
        lane_id="synthetic",
    )

    expected_signed = -signed_errors["n2"] - 3.0 * signed_errors["h2"] + 2.0 * signed_errors["nh3"]
    expected_triangle = (
        abs(signed_errors["n2"]) + 3.0 * abs(signed_errors["h2"]) + 2.0 * abs(signed_errors["nh3"])
    )
    assert math.isclose(
        result["signed_reaction_error_hartree"],
        expected_signed,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )
    assert math.isclose(
        result["triangle_bound_hartree"],
        expected_triangle,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )
    assert result["equal_share_error_limit_hartree"] == TARGET / 6.0
    assert result["passes_equal_share_constraint"] is True
    assert result["passes_reaction_constraint"] is True
    assert result["total_t_gate_count"] == 60
    assert result["t_count_aggregation"] == "sum_each_unique_species_once"


def test_missing_cache_never_silently_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = BENCHMARK.REACTIONS["hcn_hnc"].species[0]

    def unexpected_build(*_args, **_kwargs):
        raise AssertionError("cache rebuild must be explicit")

    monkeypatch.setattr(EVALUATOR, "_build_problem_cache", unexpected_build)
    with pytest.raises(FileNotFoundError, match="prebuilt cache"):
        EVALUATOR._load_problem_data(spec, tmp_path, rebuild_cache=False)


def test_zero_angle_skip_records_exact_operator_error_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EVALUATOR, "import_pygridsynth_backend", lambda: (None, object))
    problem = {
        "pauli_terms": ({"label": "Z", "coefficient": 0.2},),
        "target_time": 1.0,
        "time_sign": 1.0,
        "compile_pauli_rotation": lambda *_args: pytest.fail("rotation should be skipped"),
        "n_qubits": 1,
    }
    policy = SimpleNamespace(choose_synthesis_epsilon=lambda *_args: 1.0e-6)
    config = SimpleNamespace(formula_order=1, n_steps=2, zero_angle_threshold=0.1)
    unitary, metrics = EVALUATOR._build_compiled_one_step_unitary(
        problem,
        policy,
        config,
    )

    alpha = 0.1
    assert np.array_equal(unitary, np.eye(2, dtype=np.complex128))
    assert metrics["skipped_rotation_count"] == 2
    assert metrics["skipped_rotation_operator_error_bound"] == (
        2 * 2.0 * abs(math.sin(alpha / 2.0))
    )
    assert metrics["unitary_construction"].endswith("in_full_hilbert_space")

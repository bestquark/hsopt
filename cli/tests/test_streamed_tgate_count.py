"""Bounded CPU synthesis counts match the evaluator's literal rotation schedule."""

import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from hsopt.evaluation import compiled_unitary as ev


PLOTS = Path(__file__).parents[2] / "figures" / "validation"
sys.path.insert(0, str(PLOTS))
try:
    spec = importlib.util.spec_from_file_location(
        "streamed_tgate_count", PLOTS / "_streamed_tgate_count.py"
    )
    stream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stream)
    from _hamlib_scaling import inspect_text, iter_terms
    from _tgate_scaling import count_policy
finally:
    sys.path.pop(0)


class Formula:
    def __init__(self, repetitions=3, epsilon=1e-5):
        self.r = repetitions
        self.epsilon = epsilon

    def repetitions(self):
        return self.r

    def synthesis_epsilon(self, term, angle):
        return self.epsilon


@pytest.fixture
def fake_compiler(monkeypatch):
    calls = []

    def compile_rotation(theta, epsilon):
        calls.append((theta, epsilon))
        return {"t_count": int(round(-math.log2(epsilon))) + int(theta != 0)}

    def forbidden(*args, **kwargs):
        raise AssertionError("count-only analysis attempted physical evaluation")

    monkeypatch.setattr(ev, "_cudaq_synth", lambda: None)
    monkeypatch.setattr(ev, "_compile_rz", compile_rotation)
    for name in (
        "array_backend",
        "evaluate_exact_reference",
        "evaluate_molecule",
        "_build_compiled_step",
    ):
        monkeypatch.setattr(ev, name, forbidden)
    return calls


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("chunk_size", [1, 2, 100])
@pytest.mark.parametrize("n_qubits", [4, 120])
def test_literal_counts_match_generic_trusted_schedule(
    tmp_path, fake_compiler, workers, chunk_size, n_qubits
):
    text = "0.1 [Y1] +\n0.2 [Z0] +\n0.3 [X0] +\n0.4 [X0 Z1] +\n4.0 []"
    stats = inspect_text(text, n_qubits)
    time = 1 / stats["one_norm_nonidentity_hartree"]
    baseline, candidate = Formula(3, 1e-6), Formula(5, 2e-5)
    actual = stream.streamed_cudaq_pair(
        text,
        n_qubits,
        stats["center_label"],
        baseline,
        candidate,
        time,
        workers=workers,
        chunk_size=chunk_size,
    )
    policy = tmp_path / "candidate.py"
    policy.write_text("""class ProductFormula:
    parameters = {}
    def __init__(self, problem):
        self.r, self.epsilon = self.parameters[problem["name"]]
    def repetitions(self):
        return self.r
    def build_step(self, terms, step_time):
        ordered = sorted(terms, key=lambda term: term["label"])
        half = [(term, 0.5 * term["coefficient"] * step_time) for term in ordered[:-1]]
        center = [(ordered[-1], ordered[-1]["coefficient"] * step_time)]
        return half + center + list(reversed(half))
    def synthesis_epsilon(self, term, angle):
        return self.epsilon
""")
    problem = {
        "name": "toy",
        "n_qubits": n_qubits,
        "identity_coefficient": 4.0,
        "target_time": time,
        "terms": list(iter_terms(text, n_qubits)),
    }
    for name, formula in (("baseline", baseline), ("policy", candidate)):
        expected = count_policy(
            problem,
            policy,
            {
                "n_steps": formula.r,
                "uniform_epsilon": formula.epsilon,
                "reference_trotter_number_beta": 1,
            },
        )
        assert actual[name + "_t_count"] == expected["compiled_evolution_t_count"]
        assert actual[name + "_repetitions"] == expected["n_steps"]
    assert actual["rotations_per_step"] == 7
    assert actual["exact_compiled_count"] and not actual["analytic_estimate"]
    assert not actual["accuracy_certified"] and not actual["energy_evaluated"]


def test_rounding_and_cache_keys_include_actual_tolerance(fake_compiler):
    text = "0.123456789012345678 [X0] +\n0.3 [Z0]"
    baseline, candidate = Formula(2, 1e-5), Formula(2, 2e-5)
    result = stream.streamed_cudaq_pair(text, 1, "Z", baseline, candidate, 0.1, chunk_size=1)
    expected_angle = round(-2 * 0.123456789012345678 * 0.1 / 2 / 2, 15)
    assert (expected_angle, 1e-5) in fake_compiler
    assert (expected_angle, 2e-5) in fake_compiler
    assert result["unique_compilation_count"] == 4
    assert len(fake_compiler) == 4


def test_half_angle_operation_order_matches_canonical_pf2_at_rounding_boundary(fake_compiler):
    coefficient, target_time, steps = -0.2826566382775233, 0.8280432609232662, 3
    center_coefficient = 1 / target_time - abs(coefficient)
    text = f"{coefficient!r} [X0] +\n{center_coefficient!r} [Z0]"
    canonical = round(-2 * (0.5 * coefficient * (target_time / steps)), 15)
    reordered = round(-2 * (coefficient * target_time / steps / 2), 15)
    assert canonical != reordered  # Chosen to straddle the compiler key's 15-digit rounding.
    stream.streamed_cudaq_pair(text, 1, "Z", Formula(steps), Formula(steps), target_time, workers=1)
    assert (canonical, 1e-5) in fake_compiler
    assert (reordered, 1e-5) not in fake_compiler


def test_rotation_terms_have_same_empty_frame_metadata_as_trusted_schedule(fake_compiler):
    class MetadataFormula(Formula):
        def synthesis_epsilon(self, term, angle):
            assert term["original_label"] == term["label"]
            assert term["frame_sign"] == 1
            return self.epsilon

    result = stream.streamed_cudaq_pair(
        ".1 [X0] +\n.2 [Z0]", 1, "Z", MetadataFormula(), MetadataFormula(), 1
    )
    assert result["exact_compiled_count"]


def test_identical_formulas_share_compilation_across_chunks(fake_compiler):
    # Same angle and tolerance for these distinct labels: count both gate
    # occurrences while compiling their key only once.
    text = "0.2 [X0] +\n0.2 [Y0] +\n0.1 [Z0]"
    result = stream.streamed_cudaq_pair(text, 1, "Z", Formula(), Formula(), 0.5, chunk_size=1)
    # The center's full angle also matches each outer half angle.
    assert len(fake_compiler) == 1
    assert result["unique_compilation_count"] == 1
    t_per_rotation = 18
    assert result["baseline_t_count"] == result["policy_t_count"] == t_per_rotation * 5 * 3


def test_production_unbounded_cache_is_bypassed_and_local_cache_bounded(monkeypatch):
    calls = []

    @lru_cache(maxsize=None)
    def production_compiler(theta, epsilon):
        calls.append((theta, epsilon))
        return {"t_count": 8}

    monkeypatch.setattr(ev, "_cudaq_synth", lambda: None)
    monkeypatch.setattr(ev, "_compile_rz", production_compiler)
    monkeypatch.setattr(stream, "COMPILER_CACHE_MAXSIZE", 2)
    text = ".1 [X0] +\n.2 [Y0] +\n.3 [Z0]"
    result = stream.streamed_cudaq_pair(text, 1, "Z", Formula(), Formula(), 1, chunk_size=1)
    assert result["compiler_cache_maxsize"] == result["compiler_cache_peak_entries"] == 2
    assert production_compiler.cache_info().currsize == 0
    assert len(calls) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_qubits", 0),
        ("n_qubits", True),
        ("workers", 0),
        ("workers", 1.5),
        ("chunk_size", 0),
        ("chunk_size", True),
        ("target_time", 0),
        ("target_time", math.inf),
        ("center_label", "I"),
        ("center_label", "ZZ"),
    ],
)
def test_invalid_parameters_fail_before_synthesis(fake_compiler, field, value):
    kwargs = {
        "text": ".1 [Z0]",
        "n_qubits": 1,
        "center_label": "Z",
        "baseline": Formula(),
        "policy": Formula(),
        "target_time": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        stream.streamed_cudaq_pair(**kwargs)
    assert not fake_compiler


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5])
def test_invalid_repetitions_fail_before_synthesis(fake_compiler, repetitions):
    with pytest.raises(ValueError, match="repetitions"):
        stream.streamed_cudaq_pair(".1 [Z0]", 1, "Z", Formula(), Formula(repetitions), 1)
    assert not fake_compiler


@pytest.mark.parametrize("epsilon", [0, 1, math.nan, math.inf, -1])
def test_invalid_tolerance_is_not_compiled(fake_compiler, epsilon):
    with pytest.raises(ValueError, match="synthesis_epsilon"):
        stream.streamed_cudaq_pair(".1 [Z0]", 1, "Z", Formula(epsilon=epsilon), Formula(), 1)
    assert not fake_compiler


@pytest.mark.parametrize(
    "text,center", [(".1 [X0] +\n.1 [Z0]", "X"), (".1 [X0]", "Z"), (".1 [Z0] +\n.1 [Z0]", "Z")]
)
def test_wrong_or_duplicate_center_rejected(fake_compiler, text, center):
    with pytest.raises(ValueError, match="lexicographic maximum"):
        stream.streamed_cudaq_pair(text, 1, center, Formula(), Formula(), 1)


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_h2_term_partitions_sum_to_literal_full_count(fake_compiler, stride):
    # A standard two-qubit H2 Pauli shape, including a negative center term.
    text = "-1.05 [] +\n.3979 [Z0] +\n-.3979 [Z1] +\n.1809 [X0 X1] +\n-.01128 [Z0 Z1]"
    stats = inspect_text(text, 2)

    class MixedPrecision(Formula):
        def synthesis_epsilon(self, term, angle):
            return self.epsilon * (term["support_size"] / abs(term["coefficient"])) ** 0.3

    baseline, policy = Formula(3, 1e-6), MixedPrecision(2, 1e-5)
    time = 1 / stats["one_norm_nonidentity_hartree"]
    full = stream.streamed_cudaq_pair(text, 2, stats["center_label"], baseline, policy, time)
    assert "partial_count" not in full
    pieces = [
        stream.streamed_cudaq_pair(
            text,
            2,
            stats["center_label"],
            baseline,
            policy,
            time,
            term_stride=stride,
            term_offset=offset,
            chunk_size=1,
        )
        for offset in range(stride)
    ]
    assert all(p["partial_count"] for p in pieces)
    assert sum(p["selected_terms"] for p in pieces) == full["nonidentity_terms"] == 4
    assert sum(p["selected_center_occurrences"] for p in pieces) == 1
    assert sum(p["selected_rotations_per_step"] for p in pieces) == full["rotations_per_step"]
    for name in ("baseline", "policy"):
        assert sum(p[name + "_t_count"] for p in pieces) == full[name + "_t_count"]
        assert min(p[name + "_epsilon_min"] for p in pieces) == full[name + "_epsilon_min"]
        assert max(p[name + "_epsilon_max"] for p in pieces) == full[name + "_epsilon_max"]


def test_term_partitions_keep_center_weight_negative_angles_and_zero_t(monkeypatch):
    seen = []

    def compiler(theta, epsilon):
        seen.append(theta)
        return {"t_count": 0 if math.isclose(abs(theta), math.pi / 2, abs_tol=1e-14) else 7}

    monkeypatch.setattr(ev, "_cudaq_synth", lambda: None)
    monkeypatch.setattr(ev, "_compile_rz", compiler)
    # Y is a half rotation with zero T; the negative Z center is a full
    # rotation with zero T. The exactly zero coefficient is dropped globally.
    text = f".25 [X0] +\n{math.pi / 2!r} [Y0] +\n{-math.pi / 4!r} [Z0] +\n0 [X1]"
    formula = Formula(1)
    full = stream.streamed_cudaq_pair(text, 2, "ZI", formula, formula, 1, workers=1)
    pieces = [
        stream.streamed_cudaq_pair(
            text,
            2,
            "ZI",
            formula,
            formula,
            1,
            workers=1,
            term_stride=3,
            term_offset=i,
        )
        for i in range(3)
    ]
    assert full["baseline_t_count"] == full["policy_t_count"] == 14
    assert [p["baseline_t_count"] for p in pieces] == [14, 0, 0]
    assert sum(p["selected_terms"] for p in pieces) == 3
    assert [p["selected_center_occurrences"] for p in pieces] == [0, 0, 1]
    assert any(theta > 0 for theta in seen) and any(theta < 0 for theta in seen)


@pytest.mark.parametrize("stride,offset", [(0, 0), (True, 0), (2, -1), (2, 2), (1, 1), (2, False)])
def test_invalid_term_partition_fails_before_compilation(fake_compiler, stride, offset):
    with pytest.raises(ValueError):
        stream.streamed_cudaq_pair(
            ".1 [Z0]", 1, "Z", Formula(), Formula(), 1, term_stride=stride, term_offset=offset
        )
    assert not fake_compiler


def test_partial_count_does_not_bypass_global_center_verification(fake_compiler):
    with pytest.raises(ValueError, match="lexicographic maximum"):
        stream.streamed_cudaq_pair(
            ".1 [X0] +\n.2 [Z0]", 1, "X", Formula(), Formula(), 1, term_stride=2, term_offset=1
        )


def test_contiguous_term_ranges_sum_without_changing_normalization(fake_compiler):
    text = ".4 [X0] +\n-.3 [Y0] +\n.2 [Z0] +\n.1 [X1]"
    full = stream.streamed_cudaq_pair(text, 2, "ZI", Formula(2), Formula(3), 0.4)
    pieces = [
        stream.streamed_cudaq_pair(
            text,
            2,
            "ZI",
            Formula(2),
            Formula(3),
            0.4,
            term_start=start,
            term_stop=stop,
        )
        for start, stop in [(0, 1), (1, 3), (3, 4)]
    ]
    for name in ("baseline", "policy"):
        assert sum(p[name + "_t_count"] for p in pieces) == full[name + "_t_count"]
    assert sum(p["selected_center_occurrences"] for p in pieces) == 1
    assert sum(p["selected_rotations_per_step"] for p in pieces) == 7
    assert [p["selected_terms"] for p in pieces] == [1, 2, 1]
    assert pieces[1]["term_partition"] == {
        "scheme": "global_nonidentity_index_range",
        "start": 1,
        "stop": 3,
    }


@pytest.mark.parametrize(
    "start,stop,stride", [(-1, 2, 1), (True, 2, 1), (2, 2, 1), (0, 4, 1), (0, 2, 2)]
)
def test_invalid_contiguous_ranges_are_rejected(fake_compiler, start, stop, stride):
    with pytest.raises(ValueError):
        stream.streamed_cudaq_pair(
            ".1 [X0] +\n.2 [Z0]",
            1,
            "Z",
            Formula(),
            Formula(),
            1,
            term_start=start,
            term_stop=stop,
            term_stride=stride,
        )

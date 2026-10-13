"""Literal-gate masks and state-targeted Lanczos against independent oracles."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from hsopt.evaluation import compiled_unitary as ev
from hsopt.evaluation.gpu_circuit import certify_qpe_repetitions, gate_parameters, ground_lanczos


def test_qpe_count_rejects_an_overlap_at_an_integer_repetition_boundary():
    confidence = ev.QPE_TARGET_CONFIDENCE
    boundary = (1 - (1 - confidence) ** (1 / 8)) / ev.QPE_SINGLE_RUN_PHASE_SUCCESS_LOWER_BOUND
    assert boundary > 0.5
    with pytest.raises(RuntimeError, match="overlap uncertainty"):
        certify_qpe_repetitions(boundary, confidence)
    certify_qpe_repetitions(0.9897665188, confidence)


@pytest.mark.parametrize("label", ["X", "Y", "Z", "IIX", "XYZ", "ZYY", "XIZ", "ZYX"])
def test_constant_masks_apply_the_entire_compiled_matrix(label):
    matrix = ev._gate_sequence_matrix("HTSXTHTTT")
    dim = 1 << len(label)
    source = np.random.RandomState(34).normal(size=dim) + 1j * np.random.RandomState(89).normal(
        size=dim
    )
    expected, _ = ev._apply_compiled_rotation(
        np.eye(dim, dtype=complex),
        np.empty((dim, dim), dtype=complex),
        label=label,
        compiled_rz=matrix,
        xp=np,
    )
    args = gate_parameters(label, matrix)
    coefficients = np.array(args[6:]).reshape(4, 2) @ [1, 1j]
    actual = coefficients[0] * source
    for index in range(3):
        x, z = (int(n) for n in args[2 * index : 2 * index + 2])
        locations = np.arange(dim) ^ x
        signs = np.array([(-1) ** (int(i & z).bit_count()) for i in locations])
        actual += coefficients[index + 1] * signs * source[locations]
    np.testing.assert_allclose(actual, expected @ source, atol=2e-14, rtol=2e-14)


def test_input_lanczos_finds_an_interior_dominant_peak_without_ground_energy_lookup(monkeypatch):
    # NumPy supplies an independent known operator; production graphs execute on GPU.
    class Scalar(float):
        def get(self):
            return float(self)

    class Complex(complex):
        @property
        def real(self):
            return Scalar(super().real)

    def norm(*args, **kwargs):
        value = np.linalg.norm(*args, **kwargs)
        return Scalar(value) if np.ndim(value) == 0 else value

    cp = ModuleType("cupy")
    for name in ("empty", "asarray", "sum", "abs", "angle", "complex128"):
        setattr(cp, name, getattr(np, name))
    cp.asnumpy = np.asarray
    cp.random = np.random
    cp.vdot = lambda a, b: Complex(np.vdot(a, b))
    cp.max = lambda v: Scalar(np.max(v))
    cp.linalg = SimpleNamespace(norm=norm)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    phases = np.linspace(-0.6, 0.6, 64)
    values = np.exp(1j * phases)

    class Operator:
        shape = (64, 64)

        def __matmul__(self, v):
            return values * v if v.ndim == 1 else values[:, None] * v

        def adjoint(self, v):
            return values.conj() * v if v.ndim == 1 else values.conj()[:, None] * v

    target = np.zeros(64, dtype=complex)
    target[27] = np.sqrt(0.999)
    target[30] = np.sqrt(0.001)
    rays, vectors, diagnostics = ground_lanczos(
        Operator(), {"target_time": 1.0, "identity_coefficient": 0.0}, {"n_steps": 1}, target
    )
    assert np.angle(rays[0]) == pytest.approx(phases[27], abs=1e-10)
    assert abs(vectors[:, 0].conj() @ target) ** 2 == pytest.approx(0.999, abs=1e-11)
    assert diagnostics["spectrum_complete"] is False
    assert diagnostics["full_diagonalization_permitted"] is False
    assert {r["attempt"] for r in diagnostics["trials"]} == {0, 1}
    assert diagnostics["trials"][-1]["residual_hartree"] <= 1e-10

"""Exercise convergence control with known Ritz vectors, without a GPU dependency."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from hsopt.evaluation import gpu_spectrum


def run_known_spectrum(monkeypatch, *, peak_index, corrupt_at=(), never_resolve=False):
    dimension = 128
    phases = np.linspace(-0.4, 0.4, dimension)
    unitary = np.diag(np.exp(1j * phases))
    calls = []

    def eigsh(operator, *, k, **kwargs):
        calls.append((k, kwargs["v0"].copy()))
        vectors = np.eye(dimension, k, dtype=np.complex128)
        if k in corrupt_at or never_resolve:
            if peak_index < k:
                vectors[-1, peak_index] = 1e-4
        return np.sin(phases[:k]), vectors

    cupy = ModuleType("cupy")
    for name in ("empty_like", "sum", "angle", "abs", "complex128"):
        setattr(cupy, name, getattr(np, name))
    cupy.asnumpy = np.asarray
    cupy.random = np.random
    cupy.linalg = np.linalg
    cupy.get_default_memory_pool = lambda: SimpleNamespace(free_all_blocks=lambda: None)
    sparse = ModuleType("cupyx.scipy.sparse.linalg")
    sparse.eigsh = eigsh
    sparse.LinearOperator = lambda *args, **kwargs: kwargs["matvec"]
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    for name in ("cupyx", "cupyx.scipy", "cupyx.scipy.sparse"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse.linalg", sparse)
    from hsopt.evaluation import compiled_unitary

    monkeypatch.setattr(compiled_unitary, "_hartree_fock_basis_index", lambda p: peak_index)
    problem = {"target_time": 1.0, "identity_coefficient": 0.0}
    metrics = {
        "n_steps": 1,
        "compiled_phase_enclosure": {"sin_monotone": True, "no_repetition_wrap": True},
    }
    return lambda: gpu_spectrum.eigensolve(unitary, problem, metrics, iterative=True), calls


def test_last_converged_search_gets_independent_confirmation(monkeypatch):
    solve, calls = run_known_spectrum(monkeypatch, peak_index=30, corrupt_at=(32,))
    _, _, diagnostics = solve()
    assert [k for k, _ in calls] == [4, 8, 16, 32, 64, 64]
    assert not np.array_equal(calls[-2][1], calls[-1][1])
    assert not diagnostics["trials"][-3]["resolved"]
    assert all(t["resolved"] for t in diagnostics["trials"][-2:])
    assert all(t["seconds"] >= 0 for t in diagnostics["trials"])


def test_easy_peak_stops_before_large_search(monkeypatch):
    solve, calls = run_known_spectrum(monkeypatch, peak_index=0)
    solve()
    assert [k for k, _ in calls] == [4, 8]


def test_unresolved_peak_still_fails(monkeypatch):
    solve, _ = run_known_spectrum(monkeypatch, peak_index=30, never_resolve=True)
    with pytest.raises(RuntimeError, match="did not resolve a stable selected peak"):
        solve()

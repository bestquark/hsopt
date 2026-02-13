"""Shared fixtures for the trovolut test suite."""

import numpy as np
import pytest

import trovolut as trv


@pytest.fixture(autouse=True, scope="session")
def backend_setup():
    """Activate the gpu-mp-emulator backend once for the entire test session."""
    trv.set_backend("gpu-mp-emulator")


# ── Tiny Hamiltonians (2x2, hand-computable) ──────────────────

@pytest.fixture
def identity_2x2():
    return np.eye(2, dtype=complex)


@pytest.fixture
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)


@pytest.fixture
def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)


# ── Hardware ──────────────────────────────────────────────────

@pytest.fixture
def a100_gpu():
    from trovolut.backends.gpu_mp_emulator.hardware import gpus
    return gpus.A100


@pytest.fixture
def a100_cluster(a100_gpu):
    return 5 * a100_gpu

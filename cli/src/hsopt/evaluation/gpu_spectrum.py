"""GPU compiled-unitary eigensolvers with mandatory Lanczos and explicit convergence failures.

Lanczos solves Im(U) only when a compiled-circuit phase enclosure proves
monotonicity and rules out repetition aliasing. It returns a partial spectrum,
not a count of all HF-accessible peaks. It fails visibly on unresolved phases,
Ritz residuals or HF weights; it never invokes full diagonalization.
"""

from __future__ import annotations

import math
import time

import numpy as np


def phase_enclosure(matrices, n_steps):
    """Enclose phases of a product using the actual 2x2 compiled gates.

    Each conjugated rotation has the same two eigenphases as its literal gate.
    Removing their midpoint leaves a unitary at geodesic distance at most its
    half-width from I. Distances add; global phases add separately. This bound
    uses no exact energy or reference spectrum.
    """
    center = radius = 0.0
    for matrix in matrices:
        phases = np.angle(np.linalg.eigvals(matrix))
        delta = math.atan2(
            math.sin(float(phases[1] - phases[0])), math.cos(float(phases[1] - phases[0]))
        )
        center += float(phases[0]) + delta / 2
        radius += abs(delta) / 2 + 1e-12
    center = math.atan2(math.sin(center), math.cos(center))
    edge = abs(center) + radius
    return {
        "center_radians": center,
        "radius_radians": radius,
        "sin_monotone": edge < math.pi / 2 - 1e-6,
        "no_repetition_wrap": int(n_steps) * edge < math.pi - 1e-6,
        "method": "compiled_gate_geodesic_sum",
    }


def _compiled_lanczos_sizes(dim):
    """Grow the partial search, leaving an independent confirmation at its cap."""
    if dim < 3:
        raise RuntimeError("Compiled Lanczos requires a dimension of at least three")
    sizes = tuple(dict.fromkeys(min(k, dim - 2) for k in (4, 8, 16, 32, 64)))
    return (*sizes, sizes[-1])


def eigensolve(unitary, problem, metrics, *, iterative, target_state=None):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
    from . import compiled_unitary as ev

    diagnostics = {
        "method": "gpu_dense_lanczos",
        "spectrum_complete": False,
        "full_diagonalization_permitted": False,
    }
    if not iterative:
        raise ValueError("This GPU evaluator requires Lanczos; full diagonalization is disabled")
    guard = metrics["compiled_phase_enclosure"]
    dim = unitary.shape[0]
    matrix_free = hasattr(unitary, "adjoint")
    if not guard["sin_monotone"] or not guard["no_repetition_wrap"]:
        if matrix_free:
            raise RuntimeError("Matrix-free Lanczos phase-order/repetition guard is not certified")
        # Cancellation between substeps can make the circuit-sum enclosure loose.
        # Gershgorin on Re(U) bounds *every* eigenphase from the constructed U,
        # independently of reference energies or a sampled Ritz subspace.
        lower = 1.0
        for first in range(0, dim, 128):
            last = min(first + 128, dim)
            block = 0.5 * (unitary[first:last] + unitary[:, first:last].conj().T)
            local = cp.arange(last - first)
            diagonal = block[local, cp.arange(first, last)].real
            rows = cp.sum(cp.abs(block), axis=1) - cp.abs(diagonal)
            lower = min(lower, float(cp.min(diagonal - rows).get()))
        limit = min(math.pi / 2, math.pi / metrics["n_steps"])
        threshold = math.cos(limit)
        diagnostics["phase_guard"] = {
            "method": "Gershgorin_Re_compiled_U",
            "lower_bound": lower,
            "required_lower_bound": threshold,
        }
        if lower <= threshold + 1e-10:
            raise RuntimeError("Lanczos phase-order/repetition guard is not certified")
    scale = metrics["n_steps"] / problem["target_time"]
    if matrix_free:
        from .gpu_circuit import ground_lanczos

        return ground_lanczos(unitary, problem, metrics, target_state)
    hermitian = cp.empty_like(unitary)
    for first in range(0, dim, 128):
        last = min(first + 128, dim)
        hermitian[first:last] = (-0.5j * scale) * (
            unitary[first:last] - unitary[:, first:last].conj().T
        )
    hf_index = ev._hartree_fock_basis_index(problem)
    trials = []
    started = time.monotonic()
    calls = 0

    def action(vector):
        nonlocal calls
        calls += 1
        if time.monotonic() - started > 120:
            raise TimeoutError("Lanczos exceeded 120 seconds")
        return hermitian @ vector

    try:
        for attempt, k in enumerate(_compiled_lanczos_sizes(dim)):
            trial_started = time.monotonic()
            seed = 42 + 95 * attempt
            rng = cp.random.RandomState(seed)
            start = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            start /= cp.linalg.norm(start)
            operator = LinearOperator((dim, dim), matvec=action, dtype=cp.complex128)
            values, vectors = eigsh(
                operator,
                k=k,
                ncv=min(max(48, 2 * k + 1), dim),
                which="SA",
                v0=start,
                maxiter=1200,
                tol=1e-11,
            )
            vectors /= cp.linalg.norm(vectors, axis=0, keepdims=True)
            uv = unitary @ vectors
            rays = cp.sum(vectors.conj() * uv, axis=0)
            residuals = cp.asnumpy(cp.linalg.norm(uv - vectors * rays[None, :], axis=0)) * scale
            phases = cp.asnumpy(cp.angle(rays))
            weights = cp.asnumpy(cp.abs(vectors[hf_index, :]) ** 2)
            if target_state is None:
                try:
                    peak = ev._select_lowest_hf_accessible_peak(
                        phases,
                        weights,
                        n_steps=metrics["n_steps"],
                        target_time=problem["target_time"],
                        identity_coefficient=problem["identity_coefficient"],
                    )
                except RuntimeError:
                    trials.append(
                        {"k": k, "resolved": False, "seconds": time.monotonic() - trial_started}
                    )
                    continue
                selected = peak["selected_eigen_index"]
                energy = peak["compiled_pf_energy_hartree"]
                probability = peak["qpe_success_probability_from_hf"]
            else:
                overlaps = cp.asnumpy(cp.abs(vectors.conj().T @ target_state) ** 2)
                from .ground_branch import select_peak

                try:
                    peak = select_peak(phases, overlaps, n_steps=metrics["n_steps"],
                                       target_time=problem["target_time"],
                                       identity_coefficient=problem["identity_coefficient"])
                except RuntimeError:
                    trials.append(
                        {"k": k, "resolved": False, "seconds": time.monotonic() - trial_started}
                    )
                    continue
                selected = peak["selected_eigen_index"]
                probability = peak["qpe_success_probability_from_ground_input"]
                energy = peak["compiled_pf_energy_hartree"]
            selected_residual = (
                max(residuals[i] for i in peak["selected_eigen_indices"])
                if target_state is not None else residuals[selected]
            )
            trial = {
                "k": k,
                "selected": selected,
                "resolved": float(selected_residual) <= 1e-10,
                "energy": energy,
                "probability": probability,
                "residual_hartree": float(selected_residual),
                "max_residual_hartree": float(max(residuals)),
                "seconds": time.monotonic() - trial_started,
            }
            trials.append(trial)
            if len(trials) < 2:
                continue
            a, b = trials[-2:]
            if not a["resolved"] or not b["resolved"]:
                continue
            if abs(a["energy"] - b["energy"]) > 1e-10:
                continue
            if abs(a["probability"] - b["probability"]) > 1e-11:
                continue
            diagnostics.update(trials=trials, matvec_calls=calls)
            return rays, vectors, diagnostics
        raise RuntimeError("Lanczos did not resolve a stable selected peak: " + str(trials))
    finally:
        hermitian = None
        cp.get_default_memory_pool().free_all_blocks()


def reference_eigensolve(hamiltonian, *, ground_atol):
    """Resolve the fixed-electron ground subspace with GPU Lanczos.

    At least one higher Ritz value must bracket the ground cluster. Independent
    starts verify energy and the basis-independent ground projector. No dense
    eigensolver is used when the partial search cannot establish convergence.
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh

    dimension = hamiltonian.shape[0]
    if dimension < 3:
        raise RuntimeError("GPU reference Lanczos requires a sector dimension of at least three")
    previous = None
    for attempt, requested in enumerate((4, 8, 16, 32, 64)):
        k = min(requested, dimension - 2)
        rng = cp.random.RandomState(42 + 95 * attempt)
        start = rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension)
        start /= cp.linalg.norm(start)
        values, vectors = eigsh(
            hamiltonian,
            k=k,
            ncv=min(dimension, max(48, 2 * k + 1)),
            which="SA",
            tol=1e-12,
            maxiter=1200,
            v0=start,
        )
        order = cp.argsort(values)
        values, vectors = values[order], vectors[:, order]
        host = cp.asnumpy(values)
        ground = np.flatnonzero(abs(host - host[0]) <= ground_atol)
        if len(ground) == k:
            previous = None
            continue
        basis = vectors[:, ground.tolist()]
        residual = float(
            cp.max(
                cp.linalg.norm(
                    hamiltonian @ basis - basis * values[ground.tolist()][None, :], axis=0
                )
            ).get()
        )
        if residual > 1e-10:
            previous = None
            continue
        if previous is not None:
            old_energy, old_basis = previous
            projectors_overlap = float(cp.sum(cp.abs(old_basis.conj().T @ basis) ** 2).get())
            if (
                old_basis.shape[1] == basis.shape[1]
                and abs(old_energy - host[0]) <= 1e-10
                and abs(projectors_overlap - basis.shape[1]) <= 1e-9
            ):
                return values, vectors
        previous = float(host[0]), basis
    raise RuntimeError("GPU reference ground subspace did not converge")


def reference_hf_eigensolve(
    hamiltonian,
    *,
    hf_index,
    identity_coefficient,
    target_time,
    backend,
):
    """Certify a stable lowest HF-supported cluster using partial eigensolves.

    CPU Lanczos is available for one-time reference preparation. Production
    cupy-lanczos remains on GPU and has no dense diagonalization fallback.
    A higher Ritz energy must bracket the selected cluster; independent starts
    must agree on its energy and basis-independent HF probability.
    """
    from . import compiled_unitary as ev

    if backend == "cupy-lanczos":
        import cupy as xp
        from cupyx.scipy.sparse.linalg import eigsh
    else:
        from scipy.sparse.linalg import eigsh

        xp = np
        hamiltonian = ev._host_array(hamiltonian, dtype=np.complex128)
    dimension = hamiltonian.shape[0]
    if dimension < 3:
        raise RuntimeError("HF reference Lanczos requires at least three sector states")
    previous = None
    trials = []
    for attempt, requested in enumerate((4, 8, 16, 32, 64)):
        k = min(requested, dimension - 2)
        rng = xp.random.RandomState(42 + 95 * attempt)
        start = rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension)
        start /= xp.linalg.norm(start)
        values, vectors = eigsh(
            hamiltonian,
            k=k,
            ncv=min(dimension, max(48, 2 * k + 1)),
            which="SA",
            tol=1e-12,
            maxiter=1200,
            v0=start,
        )
        vectors /= xp.linalg.norm(vectors, axis=0, keepdims=True)
        residuals = ev._host_array(
            xp.linalg.norm(hamiltonian @ vectors - vectors * values[None, :], axis=0)
        )
        energies = ev._host_array(values)
        weights = ev._host_array(xp.abs(vectors[hf_index, :]) ** 2)
        try:
            peak = ev._select_lowest_hf_accessible_peak(
                (energies - identity_coefficient) * target_time,
                weights,
                n_steps=1,
                target_time=target_time,
                identity_coefficient=identity_coefficient,
            )
        except RuntimeError:
            previous = None
            trials.append({"k": k, "resolved": False})
            continue
        energy = peak["compiled_pf_energy_hartree"]
        probability = peak["qpe_success_probability_from_hf"]
        selected = peak["selected_eigen_indices"]
        resolved = max(residuals) <= 1e-10 and max(energies) > energy + 1e-9
        trials.append(
            {
                "k": k,
                "energy": energy,
                "probability": probability,
                "resolved": bool(resolved),
                "selected_residual_hartree": float(max(residuals[selected])),
            }
        )
        if (
            resolved
            and previous is not None
            and abs(energy - previous[0]) <= 1e-10
            and abs(probability - previous[1]) <= 1e-11
        ):
            return (
                energy,
                probability,
                {
                    "method": "hamiltonian_lanczos_hf_peak",
                    "backend": backend,
                    "full_diagonalization_permitted": False,
                    "trials": trials,
                },
            )
        previous = (energy, probability) if resolved else None
    raise RuntimeError("Exact HF-accessible reference did not converge: " + str(trials))

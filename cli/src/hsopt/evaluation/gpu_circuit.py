"""Matrix-free action of the same literal compiled circuit, captured on GPU.

Forward and adjoint graphs apply every actual 2x2 CUDA-Q synthesis matrix and
its Clifford scaffold. No recovered-angle approximation or dense U is used.
"""

import numpy as np

PROBABILITY_ATOL = 1e-9
CERTIFIER_VERSION = "matrix_free_ground_lanczos_target_centered_retry_v4"


class GroundLanczosInconclusive(RuntimeError):
    """The fixed verifier has not certified a peak; no energy violation is implied."""

    def __init__(self, trials, calls):
        self.trials = trials
        self.calls = calls
        super().__init__(
            "Ground-input Lanczos did not certify a stable literal-U peak: " + str(trials)
        )


def certify_qpe_repetitions(probability, target_confidence):
    """Refuse a score whose integer cost depends on overlap roundoff."""
    from . import compiled_unitary as ev

    bounds = (max(0.0, probability - PROBABILITY_ATOL), min(1.0, probability + PROBABILITY_ATOL))
    if bounds[0] <= 0.5:
        raise RuntimeError("Ground-input peak majority is numerically ambiguous")
    repetitions = [
        ev._qpe_repetitions(
            p * ev.QPE_SINGLE_RUN_PHASE_SUCCESS_LOWER_BOUND, target_confidence=target_confidence
        )
        for p in bounds
    ]
    if repetitions[0] != repetitions[1]:
        raise RuntimeError("Ground-input QPE repetition count depends on overlap uncertainty")


def certify_selected_phase(rays, n_steps):
    """Certify no repetition alias for the literal-U peak actually selected.

    This is used only after literal-U residuals, majority input overlap and an
    independent-start comparison pass. Ordering the *entire* spectrum of Im(U)
    is unnecessary for that state-targeted certificate. The HF lowest-peak
    solver still needs its global phase-order guard.
    """
    from . import compiled_unitary as ev

    values = ev._host_array(rays, dtype=np.complex128)
    if (
        type(n_steps) is not int
        or n_steps < 1
        or values.ndim != 1
        or not values.size
        or not np.isfinite(values).all()
        or np.any(np.abs(values) == 0)
    ):
        raise ValueError(
            "selected phase certificate needs finite nonzero rays and positive repetitions"
        )
    phases = np.angle(values)
    repeated = n_steps * phases
    margin = float(np.min(np.pi - np.abs(repeated)))
    if margin <= 1e-6:
        raise RuntimeError("Selected literal-unitary QPE peak has an uncertified repetition wrap")
    return {
        "method": "selected_literal_U_peak_no_repetition_wrap",
        "selected_step_phase_min": float(np.min(phases)),
        "selected_step_phase_max": float(np.max(phases)),
        "selected_repeated_phase_min": float(np.min(repeated)),
        "selected_repeated_phase_max": float(np.max(repeated)),
        "wrap_margin_radians": margin,
        "global_spectrum_ordering_required": False,
    }


VECTOR_KERNEL = r"""
struct C { double x,y; };
__device__ __forceinline__ C mul(C a,C b) {
  return C{a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};
}
extern "C" __global__ void compiled_gate(
  const C* in,C* out,unsigned int size,
  unsigned int x0,unsigned int z0,unsigned int x1,unsigned int z1,
  unsigned int x2,unsigned int z2,double ar,double ai,
  double br,double bi,double cr,double ci,double dr,double di) {
  unsigned int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=size)return;
  C value=mul(C{ar,ai},in[row]);
  unsigned int xs[3]={x0,x1,x2},zs[3]={z0,z1,z2};
  C coeff[3]={{br,bi},{cr,ci},{dr,di}};
  #pragma unroll
  for(int p=0;p<3;p++){
    if(coeff[p].x==0. && coeff[p].y==0.)continue;
    unsigned int src=row^xs[p];
    C v=mul(coeff[p],in[src]);
    double sign=(__popc(src&zs[p])&1)?-1.:1.;
    value.x+=sign*v.x;value.y+=sign*v.y;
  }
  out[row]=value;
}
"""


def gate_parameters(label, matrix, frame=()):
    """Constant-size Pauli masks, exactly matching the literal Clifford scaffold."""
    from . import compiled_unitary as ev

    coefficients = [complex(0.5 * np.trace(p.conj().T @ matrix)) for p in ev._PAULIS.values()]
    coefficients = [c if abs(c) > 1e-15 else 0j for c in coefficients]
    masks = []
    for index, axis in enumerate("XYZ", 1):
        sign, transformed = ev._conjugated_pivot_pauli(label, axis, frame)
        x = sum(1 << i for i, c in enumerate(reversed(transformed)) if c in "XY")
        z = sum(1 << i for i, c in enumerate(reversed(transformed)) if c in "YZ")
        masks.extend((np.uint32(x), np.uint32(z)))
        coefficients[index] *= sign * (1j ** transformed.count("Y"))
    return (*masks, *(np.float64(a) for c in coefficients for a in (c.real, c.imag)))


class CompiledCircuit:
    def __init__(self, dimension):
        self.shape = (dimension, dimension)
        self.gates = []

    def add(self, label, matrix, *, frame=()):
        self.gates.append((label, np.asarray(matrix, dtype=np.complex128), frame))

    def finish(self):
        import cupy as cp

        self.kernel = cp.RawKernel(VECTOR_KERNEL, "compiled_gate", options=("--fmad=false",))
        self.kernel.compile()
        self.input = cp.empty((self.shape[0], 1), dtype=cp.complex128)
        self.workspace = cp.empty_like(self.input)
        stream = cp.cuda.Stream(non_blocking=True)
        self.graphs = []
        for gates in (
            self.gates,
            [(label, matrix.conj().T, frame) for label, matrix, frame in reversed(self.gates)],
        ):
            parameters = [gate_parameters(label, matrix, frame) for label, matrix, frame in gates]

            def execute():
                value, workspace = self.input, self.workspace
                for args in parameters:
                    self.kernel(
                        ((self.shape[0] + 255) // 256,),
                        (256,),
                        (value, workspace, np.uint32(self.shape[0]), *args),
                    )
                    value, workspace = workspace, value
                return value

            # Maps and kernels must exist before capture. Graph buffers are
            # fixed; solver vectors are copied into them outside capture.
            with stream:
                output = execute()
            stream.synchronize()
            with stream:
                stream.begin_capture()
                output = execute()
                graph = stream.end_capture()
            self.graphs.append((graph, output))
        self.gates = []

    def apply(self, value, *, adjoint=False):
        import cupy as cp

        if value.ndim == 2:
            return cp.column_stack(
                [self.apply(value[:, i], adjoint=adjoint) for i in range(value.shape[1])]
            )
        cp.copyto(self.input[:, 0], value)
        graph, output = self.graphs[int(adjoint)]
        graph.launch(cp.cuda.get_current_stream())
        return output[:, 0].copy()

    def __matmul__(self, value):
        return self.apply(value)

    def adjoint(self, value):
        return self.apply(value, adjoint=True)


def ground_lanczos(unitary, problem, metrics, target_state):
    """Refine an inconclusive solve, then try a target-centered Hermitian view.

    The successful fast path is unchanged. On an uncertified result with a
    residual-qualified first start, rerun the same literal operator and stabilize
    that start's energy AND weight before the independent comparison. This is
    evaluator work, not an agent repair; it never changes the candidate, energy
    target, tolerances, majority requirement, or compiled gate sequence.
    """
    try:
        rays, selected, diagnostics = _ground_lanczos(
            unitary, problem, metrics, target_state, refine_initial=False
        )
    except GroundLanczosInconclusive as original:
        failures = [original]
        rays = None
        if any(t["attempt"] == 0 and t["residual_hartree"] <= 1e-10 for t in original.trials):
            try:
                rays, selected, diagnostics = _ground_lanczos(
                    unitary, problem, metrics, target_state, refine_initial=True
                )
            except (GroundLanczosInconclusive, TimeoutError) as retry:
                failures.append(retry)
        centered = rays is None
        if centered:
            try:
                rays, selected, diagnostics = _ground_lanczos(
                    unitary, problem, metrics, target_state, refine_initial=True, center_input=True
                )
            except (GroundLanczosInconclusive, TimeoutError) as retry:
                raise RuntimeError(
                    "Ground-input Lanczos verification remains inconclusive after automatic "
                    "first-start refinement and target-centered projection; this is not a "
                    "measured energy-accuracy failure. " + str(retry)
                ) from retry
        if centered:
            # Near-degenerate peaks can be ill-conditioned in different
            # Hermitian views. Do not let a numerically stable retry silently
            # change the energy or integer QPE cost of earlier qualified peaks.
            from . import compiled_unitary as ev

            qualified = [
                trial
                for failure in failures
                for trial in getattr(failure, "trials", [])
                if trial["residual_hartree"] <= 1e-10
            ] + diagnostics["trials"][-1:]
            energies = [t["energy"] for t in qualified]
            probabilities = [t["probability"] for t in qualified]
            for probability in probabilities:
                certify_qpe_repetitions(probability, ev.QPE_TARGET_CONFIDENCE)
            counts = {
                ev._qpe_repetitions(
                    p * ev.QPE_SINGLE_RUN_PHASE_SUCCESS_LOWER_BOUND,
                    target_confidence=ev.QPE_TARGET_CONFIDENCE,
                )
                for p in probabilities
            }
            if energies and (max(energies) - min(energies) > 1e-10 or len(counts) != 1):
                raise RuntimeError(
                    "Target-centered Lanczos disagrees with prior qualified energy/QPE cost"
                )
            diagnostics["projection_comparison"] = {
                "qualified_peaks": len(qualified),
                "energy_spread_hartree": max(energies) - min(energies) if energies else 0.0,
                "probability_spread": max(probabilities) - min(probabilities)
                if probabilities
                else 0.0,
                "integer_qpe_cost_agrees": True,
            }
        diagnostics["verification_retry"] = {
            "attempted": True,
            "reason": "residual_qualified_first_start_not_independently_confirmed",
            "initial_matvec_calls": original.calls,
            "initial_trials": original.trials,
            "refinement": "stabilize_first_start_energy_and_overlap_before_independent_check",
            "target_centered_projection": centered,
            "previous_attempts": [
                {"trials": getattr(f, "trials", []), "calls": getattr(f, "calls", 0)}
                for f in failures
            ],
        }
        diagnostics["matvec_calls"] += sum(getattr(f, "calls", 0) for f in failures)
    else:
        diagnostics["verification_retry"] = {"attempted": False}
    diagnostics["certifier_version"] = CERTIFIER_VERSION
    return rays, selected, diagnostics


def _ground_lanczos(unitary, problem, metrics, target_state, *, refine_initial, center_input=False):
    """Ground-input Krylov search with full reorthogonalization and Ritz checks.

    Lanczos starts near the fixed input, rather than spending time resolving
    unrelated low-energy Fock sectors. Every returned peak needs majority input
    weight, a literal-U residual, and agreement between independent starts.
    """
    import time
    import cupy as cp
    from . import ground_branch

    if target_state is None:
        raise ValueError("matrix-free ground Lanczos requires its fixed input")
    dim = unitary.shape[0]
    scale = metrics["n_steps"] / problem["target_time"]
    started, calls, trials = time.monotonic(), 0, []
    phase_shift = 0.0
    if center_input:
        # The fixed input and compiled U determine this numerical coordinate;
        # cached exact energies never select a phase branch or solver shift.
        # A majority spectral component lies within pi/2 of this Rayleigh
        # phase. The final acceptance still uses the literal U, not arcsin.
        overlap = cp.vdot(target_state, unitary @ target_state)
        phase_shift = float(cp.asnumpy(cp.angle(overlap)))
    forward_phase, backward_phase = np.exp(-1j * phase_shift), np.exp(1j * phase_shift)

    def action(v):
        nonlocal calls
        calls += 1
        if time.monotonic() - started > 120:
            raise TimeoutError("Ground-input Lanczos exceeded 120 seconds")
        if center_input:
            return (-0.5j * scale) * (
                forward_phase * (unitary @ v) - backward_phase * unitary.adjoint(v)
            )
        return (-0.5j * scale) * (unitary @ v - unitary.adjoint(v))

    previous = None
    for attempt in range(2):
        rng = cp.random.RandomState(42 + 95 * attempt)
        jitter = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        jitter /= cp.linalg.norm(jitter)
        v = target_state + 1e-5 * jitter
        v /= cp.linalg.norm(v)
        maximum = min(dim, 1024)
        basis = cp.empty((dim, maximum), dtype=cp.complex128)
        alphas, betas = [], []
        for index in range(maximum):
            basis[:, index] = v
            w = action(v)
            alpha = float(cp.vdot(v, w).real.get())
            w -= alpha * v
            if index:
                w -= betas[-1] * basis[:, index - 1]
            # Two passes keep the entire returned basis orthonormal.
            q = basis[:, : index + 1]
            for _ in range(2):
                w -= q @ (q.conj().T @ w)
            beta = float(cp.linalg.norm(w).get())
            alphas.append(alpha)
            size = index + 1
            exhausted = beta < 1e-14 or size == maximum
            if size in (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024) or exhausted:
                t = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
                values, rotation = np.linalg.eigh(t)  # small Ritz space, never full U
                projected = q @ cp.asarray(rotation)
                overlaps = cp.asnumpy(cp.abs(projected.conj().T @ target_state) ** 2)
                phases = np.arcsin(np.clip(values / scale, -1, 1)) + phase_shift
                try:
                    peak = ground_branch.select_peak(
                        phases,
                        overlaps,
                        n_steps=metrics["n_steps"],
                        target_time=problem["target_time"],
                        identity_coefficient=problem["identity_coefficient"],
                    )
                except RuntimeError:
                    peak = None
                if peak is not None:
                    selected = projected[:, peak["selected_eigen_indices"]]
                    uv = unitary @ selected
                    rays = cp.sum(selected.conj() * uv, axis=0)
                    residual = (
                        float(cp.max(cp.linalg.norm(uv - selected * rays[None, :], axis=0)).get())
                        * scale
                    )
                    final_peak = ground_branch.select_peak(
                        cp.asnumpy(cp.angle(rays)),
                        cp.asnumpy(cp.abs(selected.conj().T @ target_state) ** 2),
                        n_steps=metrics["n_steps"],
                        target_time=problem["target_time"],
                        identity_coefficient=problem["identity_coefficient"],
                    )
                    record = {
                        "attempt": attempt,
                        "krylov_size": size,
                        "residual_hartree": residual,
                        "energy": final_peak["compiled_pf_energy_hartree"],
                        "probability": final_peak["qpe_success_probability_from_ground_input"],
                    }
                    trials.append(record)
                    if residual <= 1e-10:
                        if refine_initial and attempt == 0:
                            stable = (
                                previous is not None
                                and abs(record["energy"] - previous["energy"]) <= 1e-10
                                and abs(record["probability"] - previous["probability"])
                                <= PROBABILITY_ATOL / 10
                            )
                            previous = record
                            if stable or exhausted:
                                break
                        elif (
                            previous is not None
                            and abs(record["energy"] - previous["energy"]) <= 1e-10
                            and abs(record["probability"] - previous["probability"])
                            <= PROBABILITY_ATOL
                        ):
                            return (
                                rays,
                                selected,
                                {
                                    "method": "gpu_matrix_free_ground_input_lanczos",
                                    "hermitian_projection_phase_radians": phase_shift,
                                    "spectrum_complete": False,
                                    "full_diagonalization_permitted": False,
                                    "matvec_calls": calls,
                                    "trials": trials,
                                    "probability_absolute_tolerance": PROBABILITY_ATOL,
                                    "probability_independent_start_difference": abs(
                                        record["probability"] - previous["probability"]
                                    ),
                                },
                            )
                        if not refine_initial and attempt == 0:
                            previous = record
                            break
            if exhausted:
                break
            betas.append(beta)
            v = w / beta
    raise GroundLanczosInconclusive(trials, calls)

"""
GPU mixed-precision dynamics implementations.

All evolution methods inherit from ``MPEvolution`` and share the same
precision infrastructure (``PrecisionCaster``, ``PrecisionConfig``),
ensuring that constraints and metrics work interchangeably across methods.

Public API (evolution methods):
    MPEvolution, MPTrotterEvolution, MPQDriftEvolution,
    MPTaylorEvolution, MPPadeEvolution

Convenience re-exports:
    Precision, PrecisionConfig, ProductFormula, LieTrotter,
    SuzukiTrotter, SuzukiHighOrder, suzuki

Low-level internals (hidden from __all__, usable by advanced users):
    ExpTerm, TrotterSimulator, PrecisionStats, _Exponentiator,
    _ExactExponentiator, _TaylorExponentiator, _KrylovExponentiator,
    PrecisionCaster
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from trovolut.core.base_dynamics import (
    BaseDynamics,
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    SuzukiHighOrder,
    suzuki,
)
from trovolut.core.types import StateVector, DynamicsConfig

from trovolut.backends.gpu_mp_emulator.utils import (
    Precision,
    PrecisionConfig,
    PrecisionCaster,
)


# ============================================================
# ExpTerm
# ============================================================

@dataclass
class ExpTerm:
    """A single exp(-i * coeff * H * dt) term with precision control."""

    name: str
    generator: np.ndarray | Callable[[float], np.ndarray]
    coefficient: float = 1.0
    decomposition: str = "exact"
    taylor_order: int | None = None
    tolerance: float = 1e-10
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)

    def get_generator(self, t: float = 0) -> np.ndarray:
        if callable(self.generator):
            return self.generator(t)
        return self.generator


# ============================================================
# Exponentiators (internal)
# ============================================================

class _Exponentiator(ABC):
    """Internal base class for exponentiator implementations."""

    def __init__(self, caster: PrecisionCaster | None = None):
        self.caster = caster or PrecisionCaster()

    @abstractmethod
    def apply(self, term: ExpTerm, dt: float, state: np.ndarray) -> np.ndarray:
        pass

    def _prepare_generator(self, term: ExpTerm, t: float = 0) -> np.ndarray:
        H = term.get_generator(t) * term.coefficient
        return self.caster.cast(H, term.precision.compute, term.precision)

    def _accumulate(self, state, delta, precision):
        return self.caster.cast(state, precision) + self.caster.cast(delta, precision)


class _ExactExponentiator(_Exponentiator):
    """Compute exp(-iHdt)|psi> via scipy.linalg.expm."""

    def apply(self, term, dt, state):
        from scipy.linalg import expm

        H = self._prepare_generator(term)
        if hasattr(H, "toarray"):
            H = H.toarray()
        U = expm(-1j * H.astype(np.complex128) * dt)
        U_low = self.caster.cast(U, term.precision.compute)
        return self.caster.cast(U_low @ state, term.precision.accumulate)


class _TaylorExponentiator(_Exponentiator):
    """Compute exp(-iHdt)|psi> via Taylor series."""

    def apply(self, term, dt, state):
        H = self._prepare_generator(term)
        order = term.taylor_order or 20
        prec = term.precision
        if hasattr(H, "toarray"):
            H = H.toarray()
        result = self.caster.cast(state, prec.accumulate)
        term_k = self.caster.cast(state, prec.compute)
        for k in range(1, order + 1):
            term_k = self.caster.cast((-1j * dt / k) * (H @ term_k), prec.compute)
            result = self._accumulate(
                result, self.caster.cast(term_k, prec.accumulate), prec.accumulate
            )
            if np.linalg.norm(term_k) < term.tolerance:
                break
        return result


class _KrylovExponentiator(_Exponentiator):
    """Compute exp(-iHdt)|psi> via Krylov subspace method."""

    def __init__(self, krylov_dim: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.krylov_dim = krylov_dim

    def apply(self, term, dt, state):
        H = self._prepare_generator(term)
        prec = term.precision
        if hasattr(H, "toarray"):
            H = H.toarray()
        n = len(state)
        m = min(self.krylov_dim, n)
        V = np.zeros((n, m + 1), dtype=np.complex128)
        H_krylov = np.zeros((m + 1, m), dtype=np.complex128)
        beta = np.linalg.norm(state)
        V[:, 0] = state / beta
        for j in range(m):
            V_j_low = self.caster.cast(V[:, j], prec.compute)
            H_low = self.caster.cast(H, prec.compute)
            w = (H_low @ V_j_low).astype(np.complex128)
            for i in range(j + 1):
                H_krylov[i, j] = np.vdot(V[:, i], w)
                w -= H_krylov[i, j] * V[:, i]
            H_krylov[j + 1, j] = np.linalg.norm(w)
            if np.abs(H_krylov[j + 1, j]) > 1e-12:
                V[:, j + 1] = w / H_krylov[j + 1, j]
            else:
                m = j + 1
                break
        from scipy.linalg import expm

        exp_H = expm(-1j * dt * H_krylov[:m, :m])
        return self.caster.cast(beta * (V[:, :m] @ exp_H[:, 0]), prec.accumulate)


# ============================================================
# TrotterSimulator
# ============================================================

@dataclass
class PrecisionStats:
    """Statistics collected during a Trotter simulation."""

    terms_by_precision: dict[Precision, int] = field(default_factory=dict)
    estimated_memory_bytes: int = 0
    precision_transitions: int = 0


@dataclass
class TrotterSimulator:
    """
    Low-level mixed-precision Trotter simulator.

    The ``formula`` field accepts any ``ProductFormula`` from
    ``trovolut.core.base_dynamics`` (e.g. ``SuzukiTrotter()``,
    ``SuzukiHighOrder(4)``, ``LieTrotter()``).
    """

    terms: list[ExpTerm]
    formula: ProductFormula = field(default_factory=SuzukiTrotter)
    n_steps: int = 100
    global_accumulate_precision: Precision = Precision.FP64
    caster: PrecisionCaster = field(
        default_factory=lambda: PrecisionCaster(use_torch=False)
    )
    exponentiators: dict[str, _Exponentiator] = field(default=None)

    def __post_init__(self):
        if self.exponentiators is None:
            self.exponentiators = {
                "exact": _ExactExponentiator(self.caster),
                "taylor": _TaylorExponentiator(self.caster),
                "krylov": _KrylovExponentiator(caster=self.caster),
            }

    def get_sequence(
        self, terms: list[ExpTerm], dt: float
    ) -> list[tuple[ExpTerm, float]]:
        """
        Map core ``ProductFormula.get_sequence`` index-based output into
        ``(ExpTerm, scaled_dt)`` pairs that exponentiators and
        ``CostAnalyzer`` expect.
        """
        index_seq = self.formula.get_sequence(len(terms), dt)
        return [(terms[j], tau) for j, tau in index_seq]

    def evolve(
        self,
        state: np.ndarray,
        total_time: float,
        track_stats: bool = False,
    ) -> tuple[np.ndarray, PrecisionStats | None]:
        dt = total_time / self.n_steps
        stats = PrecisionStats() if track_stats else None
        if track_stats:
            for term in self.terms:
                p = term.precision.compute
                stats.terms_by_precision[p] = stats.terms_by_precision.get(p, 0) + 1
        state = self.caster.cast(state, self.global_accumulate_precision)
        for _step in range(self.n_steps):
            sequence = self.get_sequence(self.terms, dt)
            for term, scaled_dt in sequence:
                exp_method = self.exponentiators[term.decomposition]
                state = exp_method.apply(term, scaled_dt, state)
                state = self.caster.cast(state, self.global_accumulate_precision)
                if track_stats:
                    stats.precision_transitions += 1
        return state, stats

    def print_precision_summary(self):
        """Print summary of precision configuration."""
        print("=" * 60)
        print("MIXED PRECISION TROTTERIZATION SUMMARY")
        print("=" * 60)
        for term in self.terms:
            p = term.precision
            print(f"\n{term.name}:")
            print(f"  Compute:    {p.compute.value:12s} ({p.compute.bits} bits)")
            print(f"  Storage:    {p.storage.value:12s} ({p.storage.bits} bits)")
            print(f"  Accumulate: {p.accumulate.value:12s} ({p.accumulate.bits} bits)")
            print(f"  Method:     {term.decomposition}")
        print("=" * 60)


# ============================================================
# Hamiltonian helpers (shared by all methods)
# ============================================================

def _to_numpy(a):
    """Convert array to numpy if needed."""
    if hasattr(a, "get"):  # GPU array
        return np.asarray(a.get())
    return np.asarray(a)


def _coerce_hamiltonian_to_term_arrays(hamiltonian: Any) -> List[tuple[str, np.ndarray]]:
    """
    Convert hamiltonian to list of (name, array) term arrays.

    Supports:
    - Hamiltonian object with .terms() -> [{"coef": x, "matrix": M}, ...]
    - List of arrays [H1, H2, ...]
    - Single array H
    """
    if hasattr(hamiltonian, "terms"):
        terms = []
        for i, t in enumerate(hamiltonian.terms()):
            coef = float(t.get("coef", t.get("coeff", 1.0)))
            mat = _to_numpy(t.get("matrix", t.get("matrix")))
            if mat is None:
                raise ValueError("term missing 'matrix' or 'coef'")
            if hasattr(mat, "toarray"):
                mat = mat.toarray()
            name = t.get("name", f"term_{i}")
            terms.append((str(name), coef * mat))
        return terms
    if isinstance(hamiltonian, (list, tuple)):
        out = []
        for i, H in enumerate(hamiltonian):
            arr = _to_numpy(H)
            if hasattr(arr, "toarray"):
                arr = arr.toarray()
            out.append((f"term_{i}", arr))
        return out
    arr = _to_numpy(hamiltonian)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return [("term_0", arr)]


def _resolve_schedule(
    schedule: Union[None, Precision, PrecisionConfig, List, Dict],
    n_terms: int,
    names: List[str],
    default_precision: Precision = Precision.FP32,
) -> List[PrecisionConfig]:
    """Turn precision_schedule into a list of PrecisionConfig of length n_terms."""
    if schedule is None:
        return [PrecisionConfig(compute=default_precision) for _ in range(n_terms)]
    if isinstance(schedule, Precision):
        return [PrecisionConfig(compute=schedule) for _ in range(n_terms)]
    if isinstance(schedule, PrecisionConfig):
        return [schedule] * n_terms
    if isinstance(schedule, (list, tuple)):
        if len(schedule) != n_terms:
            raise ValueError(
                f"precision_schedule length ({len(schedule)}) != number of terms ({n_terms})"
            )
        out = []
        for p in schedule:
            if isinstance(p, Precision):
                out.append(PrecisionConfig(compute=p))
            elif isinstance(p, PrecisionConfig):
                out.append(p)
            else:
                raise TypeError(f"schedule entry must be Precision or PrecisionConfig, got {type(p)}")
        return out
    if isinstance(schedule, dict):
        configs = [PrecisionConfig(compute=default_precision) for _ in range(n_terms)]
        for key, val in schedule.items():
            if isinstance(key, int):
                idx = key
            elif isinstance(key, str):
                try:
                    idx = names.index(key)
                except ValueError:
                    idx = int(key.replace("term_", "")) if key.replace("term_", "").isdigit() else -1
                if idx < 0 or idx >= n_terms:
                    continue
            else:
                continue
            if isinstance(val, Precision):
                configs[idx] = PrecisionConfig(compute=val)
            elif isinstance(val, PrecisionConfig):
                configs[idx] = val
        return configs
    raise TypeError(
        "precision_schedule must be None, Precision, PrecisionConfig, "
        "list of Precision/PrecisionConfig, or dict"
    )


def _resolve_decomposition_schedule(
    schedule: Union[None, str, List, Dict],
    n_terms: int,
    names: List[str],
    default: str = "exact",
) -> List[str]:
    """Turn decomposition_schedule into a list of method names of length n_terms."""
    if schedule is None:
        return [default] * n_terms
    if isinstance(schedule, str):
        return [schedule] * n_terms
    if isinstance(schedule, (list, tuple)):
        if len(schedule) != n_terms:
            raise ValueError(
                f"decomposition_schedule length ({len(schedule)}) != number of terms ({n_terms})"
            )
        return list(schedule)
    if isinstance(schedule, dict):
        methods = [default] * n_terms
        for key, val in schedule.items():
            if isinstance(key, int):
                idx = key
            elif isinstance(key, str):
                try:
                    idx = names.index(key)
                except ValueError:
                    idx = int(key.replace("term_", "")) if key.replace("term_", "").isdigit() else -1
                if idx < 0 or idx >= n_terms:
                    continue
            else:
                continue
            methods[idx] = str(val)
        return methods
    raise TypeError("decomposition_schedule must be None, str, list, or dict")


def _build_exp_terms(
    term_arrays: List[tuple[str, np.ndarray]],
    precision_schedule=None,
    decomposition_schedule=None,
    accumulate_precision: Precision = Precision.FP64,
    default_compute_precision: Precision = Precision.FP32,
) -> List[ExpTerm]:
    """Build list of ExpTerm from term arrays and schedules."""
    from dataclasses import replace

    names = [n for n, _ in term_arrays]
    n_terms = len(term_arrays)
    prec_configs = _resolve_schedule(
        precision_schedule, n_terms, names, default_precision=default_compute_precision
    )
    prec_configs = [replace(pc, accumulate=accumulate_precision) for pc in prec_configs]
    methods = _resolve_decomposition_schedule(decomposition_schedule, n_terms, names)

    terms = []
    for i, (name, arr) in enumerate(term_arrays):
        terms.append(
            ExpTerm(
                name=name,
                generator=arr,
                coefficient=1.0,
                decomposition=methods[i],
                precision=prec_configs[i],
            )
        )
    return terms


# ============================================================
# Base class for all mixed-precision evolution methods
# ============================================================

class MPEvolution(BaseDynamics):
    """
    Abstract base for all mixed-precision evolution methods in gpu_mp_emulator.

    All subclasses share the same precision infrastructure and support
    per-term precision schedules.  The ``cost_profile()`` method returns a
    standardised ``CostProfile`` so that ``MemoryCost`` / ``FlopsCost`` /
    ``SpaceTimeCost`` constraints work interchangeably across methods.

    Subclasses: ``MPTrotterEvolution``, ``MPQDriftEvolution``,
    ``MPTaylorEvolution``, ``MPPadeEvolution``.

    Parameters
    ----------
    precision_schedule : None | Precision | PrecisionConfig | list | dict
        Per-term compute precision.
    accumulate_precision : Precision
        Precision for state accumulation (default FP64).
    use_torch : bool
        Use PyTorch for casting when available (default False).
    """

    def __init__(
        self,
        precision_schedule=None,
        accumulate_precision: Precision = Precision.FP64,
        use_torch: bool = False,
    ):
        self.precision_schedule = precision_schedule
        self.accumulate_precision = accumulate_precision
        self.use_torch = use_torch
        self._caster = PrecisionCaster(use_torch=use_torch)

    @abstractmethod
    def evolve(
        self,
        hamiltonian: Any,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs,
    ) -> StateVector:
        pass

    @abstractmethod
    def cost_profile(self, hamiltonian: Any, **kwargs):
        """Return a ``CostProfile`` for this evolution on the given hamiltonian."""
        pass

    # -- shared helpers -------------------------------------------------------

    def _prepare_terms(self, hamiltonian: Any, **kwargs) -> List[ExpTerm]:
        """Build ExpTerm list from hamiltonian using this evolution's schedules."""
        term_arrays = _coerce_hamiltonian_to_term_arrays(hamiltonian)
        return _build_exp_terms(
            term_arrays,
            precision_schedule=kwargs.get("precision_schedule", self.precision_schedule),
            decomposition_schedule=kwargs.get(
                "decomposition_schedule",
                getattr(self, "decomposition_schedule", None),
            ),
            accumulate_precision=self.accumulate_precision,
        )

    def _build_hamiltonian_mp(self, hamiltonian: Any, **kwargs) -> np.ndarray:
        """Sum precision-cast terms into a single Hamiltonian matrix."""
        terms = self._prepare_terms(hamiltonian, **kwargs)
        H_mp = None
        for term in terms:
            H = term.get_generator(0) * term.coefficient
            H_cast = self._caster.cast(H, term.precision.compute, term.precision)
            H_cast = H_cast.astype(np.complex128)
            if H_mp is None:
                H_mp = H_cast
            else:
                H_mp += H_cast
        return H_mp

    def _infer_dim(self, hamiltonian: Any) -> int:
        term_arrays = _coerce_hamiltonian_to_term_arrays(hamiltonian)
        if term_arrays:
            return term_arrays[0][1].shape[0]
        raise ValueError("Cannot infer state dimension from hamiltonian")

    @staticmethod
    def _generator_memory(terms: List[ExpTerm], n: int) -> tuple[int, dict[str, int]]:
        """Total generator memory (bytes) and breakdown by precision."""
        total = 0
        by_prec: dict[str, int] = {}
        for term in terms:
            bits = term.precision.compute.bits
            bpe = max(1, bits // 8)
            mem = n * n * bpe * 2  # complex
            total += mem
            key = term.precision.compute.value
            by_prec[key] = by_prec.get(key, 0) + mem
        return total, by_prec


# ============================================================
# Trotter evolution
# ============================================================

class MPTrotterEvolution(MPEvolution):
    """
    Mixed-precision Suzuki-Trotter product formula.

    Parameters
    ----------
    trotter_steps : int
        Number of Trotter steps (default 100).
    formula : ProductFormula
        Product formula from ``trovolut.core.base_dynamics`` (default
        ``SuzukiTrotter()``).  Use ``suzuki(order)`` factory for higher
        orders (4, 6, 8, ...) or ``LieTrotter()`` for first-order.
    precision_schedule, decomposition_schedule, accumulate_precision, use_torch
        See ``MPEvolution``.
    """

    def __init__(
        self,
        trotter_steps: int = 100,
        formula: ProductFormula | None = None,
        precision_schedule=None,
        decomposition_schedule=None,
        accumulate_precision: Precision = Precision.FP64,
        use_torch: bool = False,
    ):
        super().__init__(precision_schedule, accumulate_precision, use_torch)
        self.trotter_steps = trotter_steps
        self.formula = formula if formula is not None else SuzukiTrotter()
        self.decomposition_schedule = decomposition_schedule

    def evolve(self, hamiltonian, initial_state, time, config=None, **kwargs):
        config = config or DynamicsConfig()
        extra = getattr(config, "extra", None) or {}
        steps = kwargs.get("trotter_steps", extra.get("trotter_steps", self.trotter_steps))
        prec_schedule = kwargs.get("precision_schedule", extra.get("precision_schedule", self.precision_schedule))
        decomp_schedule = kwargs.get("decomposition_schedule", extra.get("decomposition_schedule", self.decomposition_schedule))

        term_arrays = _coerce_hamiltonian_to_term_arrays(hamiltonian)
        if not term_arrays:
            return np.asarray(initial_state, dtype=np.complex128)

        terms = _build_exp_terms(
            term_arrays,
            precision_schedule=prec_schedule,
            decomposition_schedule=decomp_schedule,
            accumulate_precision=self.accumulate_precision,
        )

        simulator = TrotterSimulator(
            terms=terms,
            formula=self.formula,
            n_steps=steps,
            global_accumulate_precision=self.accumulate_precision,
            caster=self._caster,
        )

        psi = _to_numpy(initial_state).astype(np.complex128)
        state, _ = simulator.evolve(psi, total_time=time, track_stats=False)
        return state

    def cost_profile(self, hamiltonian, **kwargs):
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile, CostAnalyzer

        steps = kwargs.get("trotter_steps", self.trotter_steps)
        term_arrays = _coerce_hamiltonian_to_term_arrays(hamiltonian)
        if not term_arrays:
            return CostProfile(method="trotter")

        terms = _build_exp_terms(
            term_arrays,
            precision_schedule=kwargs.get("precision_schedule", self.precision_schedule),
            decomposition_schedule=kwargs.get("decomposition_schedule", self.decomposition_schedule),
            accumulate_precision=self.accumulate_precision,
        )

        sim = TrotterSimulator(
            terms=terms,
            formula=self.formula,
            n_steps=steps,
            global_accumulate_precision=self.accumulate_precision,
            caster=self._caster,
        )

        cost = CostAnalyzer().analyze(sim)
        return CostProfile(
            method="trotter",
            peak_memory_bytes=cost.peak_memory_bytes,
            generator_memory_bytes=cost.generator_memory_bytes,
            state_memory_bytes=cost.state_memory_bytes,
            auxiliary_memory_bytes=cost.auxiliary_memory_bytes,
            memory_by_precision=dict(cost.memory_by_precision),
            total_flops=cost.total_flops,
            total_depth=cost.total_depth,
            n_terms=cost.n_terms,
            n_steps=cost.n_steps,
        )

    def estimate_memory_bytes(self, hamiltonian: Any) -> int:
        return int(self.cost_profile(hamiltonian).peak_memory_bytes)


# ============================================================
# QDrift evolution
# ============================================================

class MPQDriftEvolution(MPEvolution):
    """
    Mixed-precision qDRIFT randomized product formula.

    Samples terms proportional to their spectral norms and applies
    single-term exponentials using each term's assigned precision.

    Parameters
    ----------
    n_samples : int
        Number of random samples (default 1000).
    decomposition_schedule : None | str | list | dict
        Per-term exponentiation method.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        precision_schedule=None,
        decomposition_schedule=None,
        accumulate_precision: Precision = Precision.FP64,
        use_torch: bool = False,
        seed: int | None = None,
    ):
        super().__init__(precision_schedule, accumulate_precision, use_torch)
        self.n_samples = n_samples
        self.decomposition_schedule = decomposition_schedule
        self.seed = seed

    def evolve(self, hamiltonian, initial_state, time, config=None, **kwargs):
        terms = self._prepare_terms(hamiltonian, **kwargs)
        if not terms:
            return np.asarray(initial_state, dtype=np.complex128)

        n_samples = kwargs.get("n_samples", self.n_samples)

        norms = []
        for term in terms:
            H = term.get_generator(0) * term.coefficient
            norms.append(float(np.linalg.norm(H, ord=2)))
        lambda_total = sum(norms)
        if lambda_total == 0:
            return np.asarray(initial_state, dtype=np.complex128)
        probs = np.array([n / lambda_total for n in norms])

        dt_eff = time * lambda_total / n_samples

        exponentiators = {
            "exact": _ExactExponentiator(self._caster),
            "taylor": _TaylorExponentiator(self._caster),
            "krylov": _KrylovExponentiator(caster=self._caster),
        }

        psi = _to_numpy(initial_state).astype(np.complex128)
        rng = np.random.default_rng(self.seed)

        for _ in range(n_samples):
            j = rng.choice(len(terms), p=probs)
            norm_j = norms[j]
            if norm_j == 0:
                continue
            normalized_gen = terms[j].get_generator(0) * terms[j].coefficient / norm_j
            normalized_term = ExpTerm(
                name=terms[j].name,
                generator=normalized_gen,
                coefficient=1.0,
                decomposition=terms[j].decomposition,
                precision=terms[j].precision,
            )
            psi = exponentiators[terms[j].decomposition].apply(normalized_term, dt_eff, psi)
            psi = self._caster.cast(psi, self.accumulate_precision)

        return psi

    def cost_profile(self, hamiltonian, **kwargs):
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile, DecompositionCostModel

        terms = self._prepare_terms(hamiltonian, **kwargs)
        if not terms:
            return CostProfile(method="qdrift")

        n = terms[0].get_generator(0).shape[0]
        n_samples = kwargs.get("n_samples", self.n_samples)
        cost_model = DecompositionCostModel()

        gen_memory, mem_by_prec = self._generator_memory(terms, n)
        state_bytes = n * 16

        norms = []
        for term in terms:
            H = term.get_generator(0) * term.coefficient
            norms.append(float(np.linalg.norm(H, ord=2)))
        lambda_total = sum(norms)
        probs = [norm / lambda_total for norm in norms] if lambda_total > 0 else [1.0 / len(terms)] * len(terms)

        expected_flops = 0.0
        max_aux = 0
        max_depth = 0
        for j, term in enumerate(terms):
            bits = term.precision.compute.bits
            method = term.decomposition
            if method == "taylor":
                mc = cost_model.taylor_cost(n, term.taylor_order or 20, bits)
            elif method == "krylov":
                mc = cost_model.krylov_cost(n, 30, bits)
            else:
                mc = cost_model.exact_expm_cost(n, bits)
            expected_flops += probs[j] * mc["flops"]
            max_aux = max(max_aux, mc["auxiliary_memory"])
            max_depth = max(max_depth, mc["matvec_count"] + mc["matmat_count"] * 10)

        return CostProfile(
            method="qdrift",
            peak_memory_bytes=gen_memory + state_bytes + max_aux,
            generator_memory_bytes=gen_memory,
            state_memory_bytes=state_bytes,
            auxiliary_memory_bytes=max_aux,
            memory_by_precision=mem_by_prec,
            total_flops=int(n_samples * expected_flops),
            total_depth=n_samples * max_depth,
            n_terms=len(terms),
            n_steps=n_samples,
        )


# ============================================================
# Taylor evolution (direct expansion of full H)
# ============================================================

class MPTaylorEvolution(MPEvolution):
    """
    Direct Taylor expansion of exp(-iHt) with mixed-precision Hamiltonian.

    Each Hamiltonian term is cast to its assigned precision before summing
    to form H_mp.  The Taylor expansion is performed at the accumulate
    precision.

    Parameters
    ----------
    order : int
        Maximum Taylor order (default 20).
    tolerance : float
        Early termination tolerance on ||term_k|| (default 1e-12).
    """

    def __init__(
        self,
        order: int = 20,
        tolerance: float = 1e-12,
        precision_schedule=None,
        accumulate_precision: Precision = Precision.FP64,
        use_torch: bool = False,
    ):
        super().__init__(precision_schedule, accumulate_precision, use_torch)
        self.order = order
        self.tolerance = tolerance

    def evolve(self, hamiltonian, initial_state, time, config=None, **kwargs):
        H_mp = self._build_hamiltonian_mp(hamiltonian, **kwargs)
        order = kwargs.get("order", self.order)
        tolerance = kwargs.get("tolerance", self.tolerance)

        psi = _to_numpy(initial_state).astype(np.complex128)
        result = psi.copy()
        term_k = psi.copy()

        for k in range(1, order + 1):
            term_k = (-1j * time / k) * (H_mp @ term_k)
            result += term_k
            if np.linalg.norm(term_k) < tolerance:
                break

        return self._caster.cast(result, self.accumulate_precision)

    def cost_profile(self, hamiltonian, **kwargs):
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile

        terms = self._prepare_terms(hamiltonian, **kwargs)
        if not terms:
            return CostProfile(method="taylor")

        n = terms[0].get_generator(0).shape[0]
        order = kwargs.get("order", self.order)

        gen_memory, mem_by_prec = self._generator_memory(terms, n)
        state_bytes = n * 16
        h_mp_bytes = n * n * 16
        aux_bytes = h_mp_bytes + 2 * state_bytes

        return CostProfile(
            method="taylor",
            peak_memory_bytes=gen_memory + state_bytes + aux_bytes,
            generator_memory_bytes=gen_memory,
            state_memory_bytes=state_bytes,
            auxiliary_memory_bytes=aux_bytes,
            memory_by_precision=mem_by_prec,
            total_flops=order * (2 * n ** 2),
            total_depth=order,
            n_terms=len(terms),
            n_steps=order,
        )


# ============================================================
# Pade evolution (scipy.expm on mixed-precision H)
# ============================================================

class MPPadeEvolution(MPEvolution):
    """
    Pade approximant of exp(-iHt) with mixed-precision Hamiltonian.

    Each Hamiltonian term is cast to its assigned precision before summing
    to form H_mp.  The matrix exponential is computed via
    ``scipy.linalg.expm`` (Pade(13,13) algorithm).
    """

    def __init__(
        self,
        precision_schedule=None,
        accumulate_precision: Precision = Precision.FP64,
        use_torch: bool = False,
    ):
        super().__init__(precision_schedule, accumulate_precision, use_torch)

    def evolve(self, hamiltonian, initial_state, time, config=None, **kwargs):
        import scipy.linalg

        H_mp = self._build_hamiltonian_mp(hamiltonian, **kwargs)
        psi = _to_numpy(initial_state).astype(np.complex128)
        U = scipy.linalg.expm(-1j * H_mp * time)
        return self._caster.cast(U @ psi, self.accumulate_precision)

    def cost_profile(self, hamiltonian, **kwargs):
        from trovolut.backends.gpu_mp_emulator.measure import CostProfile

        terms = self._prepare_terms(hamiltonian, **kwargs)
        if not terms:
            return CostProfile(method="pade")

        n = terms[0].get_generator(0).shape[0]

        gen_memory, mem_by_prec = self._generator_memory(terms, n)
        state_bytes = n * 16
        h_mp_bytes = n * n * 16
        aux_bytes = h_mp_bytes + 6 * n * n * 16 + state_bytes

        return CostProfile(
            method="pade",
            peak_memory_bytes=gen_memory + state_bytes + aux_bytes,
            generator_memory_bytes=gen_memory,
            state_memory_bytes=state_bytes,
            auxiliary_memory_bytes=aux_bytes,
            memory_by_precision=mem_by_prec,
            total_flops=13 * (2 * n ** 3),
            total_depth=13 * 10,
            n_terms=len(terms),
            n_steps=1,
        )


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Evolution methods (primary public API)
    "MPEvolution",
    "MPTrotterEvolution",
    "MPQDriftEvolution",
    "MPTaylorEvolution",
    "MPPadeEvolution",
    # Precision infrastructure (convenience re-exports)
    "Precision",
    "PrecisionConfig",
    # Core formulas (re-exported for convenience)
    "ProductFormula",
    "LieTrotter",
    "SuzukiTrotter",
    "SuzukiHighOrder",
    "suzuki",
]

"""
GPU mixed-precision dynamics implementations.

Provides the same evolve(hamiltonian, initial_state, time, config, **kwargs) interface
as other backends, with per-term quantization schedules so you can set different
precision levels per Hamiltonian term without building ExpTerms by hand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.types import StateVector, DynamicsConfig

from trovolut.backends.gpu_mp_emulator._mp_utils import (
    Precision,
    PrecisionConfig,
    PrecisionCaster,
    ExpTerm,
    TrotterSimulator,
    SecondOrderTrotter,
    PrecisionStats,
)


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
    # Hamiltonian object
    if hasattr(hamiltonian, "terms"):
        terms = []
        for i, t in enumerate(hamiltonian.terms()):
            coef = float(t.get("coef", t.get("coeff", 1.0)))
            mat = _to_numpy(t.get("matrix", t.get("matrix")))
            if mat is None:
                raise ValueError("term missing 'matrix' or 'coef'")
            name = t.get("name", f"term_{i}")
            terms.append((str(name), coef * mat))
        return terms
    
    # List of arrays
    if isinstance(hamiltonian, (list, tuple)):
        return [(f"term_{i}", _to_numpy(H)) for i, H in enumerate(hamiltonian)]
    
    # Single array
    return [("term_0", _to_numpy(hamiltonian))]


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
                f"precision_schedule length ({len(schedule)}) must equal number of terms ({n_terms})"
            )
        out = []
        for p in schedule:
            if isinstance(p, Precision):
                out.append(PrecisionConfig(compute=p))
            elif isinstance(p, PrecisionConfig):
                out.append(p)
            else:
                raise TypeError(f"precision_schedule entry must be Precision or PrecisionConfig, got {type(p)}")
        return out
    
    if isinstance(schedule, dict):
        # Keys: int (index) or str (name)
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
        "list of Precision/PrecisionConfig, or dict mapping index/name to Precision/PrecisionConfig"
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
                f"decomposition_schedule length ({len(schedule)}) must equal number of terms ({n_terms})"
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
    precision_schedule: Union[None, Precision, PrecisionConfig, List, Dict] = None,
    decomposition_schedule: Union[None, str, List, Dict] = None,
    accumulate_precision: Precision = Precision.FP64,
    default_compute_precision: Precision = Precision.FP32,
) -> List[ExpTerm]:
    """Build list of ExpTerm from term arrays and schedules."""
    names = [n for n, _ in term_arrays]
    n_terms = len(term_arrays)
    
    from dataclasses import replace
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


class MixedPrecisionTrotterEvolution(BaseDynamics):
    """
    Mixed-precision Trotter evolution with per-term quantization schedules.
    
    Same interface as other backends: evolve(hamiltonian, initial_state, time, config=None, **kwargs).
    You do not build ExpTerms or quantize by hand; set precision_schedule and
    optionally decomposition_schedule to assign precisions and methods per term.
    
    Parameters
    ----------
    trotter_steps : int
        Number of Trotter steps (default 100).
    formula : TrotterFormula
        Trotter formula (default SecondOrderTrotter()).
    precision_schedule : None | Precision | PrecisionConfig | list | dict
        Per-term compute precision. None = FP32 for all. Single Precision/PrecisionConfig
        = same for all. List length n_terms = one per term. Dict = index or name -> Precision/PrecisionConfig.
    decomposition_schedule : None | str | list | dict
        Per-term exponentiation method: "exact", "taylor", "krylov". Same shapes as precision_schedule.
    accumulate_precision : Precision
        Precision for state accumulation (default FP64).
    use_torch : bool
        Use PyTorch for casting when available (default False).
    
    Example
    -------
    >>> from trovolut.backends.gpu_mp_emulator.dynamics import MixedPrecisionTrotterEvolution, Precision
    >>> dyn = MixedPrecisionTrotterEvolution(
    ...     trotter_steps=50,
    ...     precision_schedule=[Precision.FP32, Precision.FP16, Precision.FP8_E4M3],  # 3 terms
    ... )
    >>> state = dyn.evolve(hamiltonian, psi0, time=1.0)
    """
    
    def __init__(
        self,
        trotter_steps: int = 100,
        formula: Optional[Any] = None,
        precision_schedule: Union[None, Precision, PrecisionConfig, List, Dict] = None,
        decomposition_schedule: Union[None, str, List, Dict] = None,
        accumulate_precision: Precision = Precision.FP64,
        use_torch: bool = False,
    ):
        self.trotter_steps = trotter_steps
        self.formula = formula if formula is not None else SecondOrderTrotter()
        self.precision_schedule = precision_schedule
        self.decomposition_schedule = decomposition_schedule
        self.accumulate_precision = accumulate_precision
        self.use_torch = use_torch
        self._caster = PrecisionCaster(use_torch=use_torch)
    
    def evolve(
        self,
        hamiltonian: Any,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs,
    ) -> StateVector:
        """
        Evolve initial_state under hamiltonian for time.
        
        hamiltonian can be:
        - Hamiltonian object (with .terms() returning [{"coef", "matrix"}, ...])
        - List of arrays [H1, H2, ...]
        - Single array H
        """
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
    
    def estimate_memory_bytes(self, hamiltonian: Any) -> int:
        """
        Estimate peak memory (bytes) for this dynamics method with the given hamiltonian.
        Used by constraints and metrics to report memory for one dynamics method.
        """
        from trovolut.backends.gpu_mp_emulator.measure import CostAnalyzer
        
        term_arrays = _coerce_hamiltonian_to_term_arrays(hamiltonian)
        if not term_arrays:
            return 0
        terms = _build_exp_terms(
            term_arrays,
            precision_schedule=self.precision_schedule,
            decomposition_schedule=self.decomposition_schedule,
            accumulate_precision=self.accumulate_precision,
        )
        sim = TrotterSimulator(
            terms=terms,
            formula=self.formula,
            n_steps=self.trotter_steps,
            global_accumulate_precision=self.accumulate_precision,
            caster=self._caster,
        )
        cost = CostAnalyzer().analyze(sim)
        return int(cost.peak_memory_bytes)


# Re-export for convenience so users can do:
#   from trovolut.backends.gpu_mp_emulator.dynamics import MixedPrecisionTrotterEvolution, Precision, PrecisionConfig
__all__ = [
    "MixedPrecisionTrotterEvolution",
    "Precision",
    "PrecisionConfig",
    "_coerce_hamiltonian_to_term_arrays",
    "_build_exp_terms",
]

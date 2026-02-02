"""
Emulator dynamics implementation using mixed-precision Trotter methods.

This module provides CPU-based simulation of quantized exponentials for studying
numerical precision effects without requiring GPU hardware.
"""

from typing import Any, Dict, Optional, List
import numpy as np
from trovolut.core.base_dynamics import BaseDynamics
from trovolut.core.types import StateVector, DynamicsConfig

# Import mixed-precision Trotter tools
from trovolut.backends.gpu_emulator.mixed_fps.mp_trotter import (
    Precision,
    PrecisionConfig,
    ExpTerm,
    MixedPrecisionTrotter,
    TrotterConfig,
)
from trovolut.backends.gpu_emulator.dynamics import suzuki_step


class EmulatorDynamics(BaseDynamics):
    """
    Mixed-precision Trotter dynamics for studying quantized exponentials.
    
    This class provides CPU-based emulation of GPU low-precision arithmetic
    using various numerical formats (FP64, FP32, FP16, BF16, FP8, INT8).
    
    Usage:
        >>> dynamics = EmulatorDynamics()
        >>> # For standard Suzuki-Trotter
        >>> psi_final = dynamics.evolve(H_terms, psi0, time=1.0, order=2)
        
        >>> # For mixed-precision simulation
        >>> from trovolut.backends.gpu_emulator.mixed_fps.mp_trotter import Precision
        >>> psi_final = dynamics.evolve_mixed_precision(
        ...     exp_terms, psi0, time=1.0, steps=100,
        ...     precision=Precision.FP16
        ... )
    """
    
    def __init__(self, use_cupynumeric: bool = False):
        """
        Initialize emulator dynamics.
        
        Args:
            use_cupynumeric: If True, try to use cupynumeric for GPU acceleration
        """
        self.use_cupynumeric = use_cupynumeric
        self._np = self._get_numpy_module()
    
    def _get_numpy_module(self):
        """Get NumPy or cupynumeric module."""
        if self.use_cupynumeric:
            try:
                import cupynumeric as np
                return np
            except ImportError:
                import numpy as np
                return np
        else:
            import numpy as np
            return np
    
    def evolve(
        self,
        hamiltonian: List[np.ndarray] | np.ndarray,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        order: int = 2,
        steps: int = 1,
        **kwargs
    ) -> StateVector:
        """
        Evolve quantum state using Suzuki-Trotter decomposition.
        
        Args:
            hamiltonian: Either a list of Hamiltonian terms [H1, H2, ...] or 
                        a single Hamiltonian matrix. If a list, uses Suzuki-Trotter.
                        If a single matrix, uses standard matrix exponential.
            initial_state: Initial quantum state vector
            time: Total evolution time
            config: Optional configuration (not used in standard mode)
            order: Suzuki-Trotter order (1, 2, 4, 6, ...)
            steps: Number of Trotter steps
            **kwargs: Additional options
            
        Returns:
            Final quantum state vector
        """
        dt = time / steps
        psi = self._np.asarray(initial_state, dtype=complex)
        
        # If hamiltonian is a list of terms, use Suzuki-Trotter
        if isinstance(hamiltonian, (list, tuple)):
            H_terms = [self._np.asarray(H, dtype=complex) for H in hamiltonian]
            
            for _ in range(steps):
                U = suzuki_step(H_terms, dt, order=order)
                psi = U @ psi
        
        # If hamiltonian is a single matrix, use standard expm
        else:
            from scipy.linalg import expm
            H = self._np.asarray(hamiltonian, dtype=complex)
            U = expm(-1j * H * time)
            psi = U @ psi
        
        return psi
    
    def evolve_mixed_precision(
        self,
        exp_terms: List[ExpTerm],
        initial_state: StateVector,
        time: float,
        steps: int = 100,
        config: Optional[PrecisionConfig] = None,
        trotter_config: Optional[TrotterConfig] = None,
        **kwargs
    ) -> tuple[StateVector, Dict[str, Any]]:
        """
        Evolve using mixed-precision Trotter with quantized exponentials.
        
        This method allows different Hamiltonian terms to use different precisions,
        enabling study of quantization effects on quantum dynamics.
        
        Args:
            exp_terms: List of ExpTerm objects (from mp_trotter module)
            initial_state: Initial quantum state
            time: Total evolution time
            steps: Number of Trotter steps
            config: Precision configuration
            trotter_config: Trotter configuration
            **kwargs: Additional options
            
        Returns:
            Tuple of (final_state, metrics)
            
        Example:
            >>> from trovolut.backends.gpu_emulator.mixed_fps.mp_trotter import (
            ...     Precision, ExpTerm, TrotterConfig
            ... )
            >>> # Create terms with different precisions
            >>> terms = [
            ...     ExpTerm(H1, Precision.FP16, "exact"),
            ...     ExpTerm(H2, Precision.FP8_E4M3, "exact"),
            ... ]
            >>> psi_final, metrics = dynamics.evolve_mixed_precision(
            ...     terms, psi0, time=1.0, steps=100
            ... )
        """
        # Set default configs
        if config is None:
            config = PrecisionConfig()
        if trotter_config is None:
            trotter_config = TrotterConfig(order=2)
        
        # Create mixed-precision Trotter simulator
        simulator = MixedPrecisionTrotter(
            exp_terms=exp_terms,
            precision_config=config,
            trotter_config=trotter_config,
        )
        
        # Run simulation
        psi = self._np.asarray(initial_state, dtype=complex)
        dt = time / steps
        
        metrics = {
            "steps": steps,
            "dt": dt,
            "order": trotter_config.order,
            "precisions": [term.precision.value for term in exp_terms],
        }
        
        for step_idx in range(steps):
            psi = simulator.step(psi, dt)
        
        return psi, metrics
    
    def evolve_with_metrics(
        self,
        hamiltonian,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs
    ) -> tuple[StateVector, Dict[str, Any]]:
        """
        Evolve and return metrics.
        
        For standard evolution, metrics are minimal.
        For mixed-precision, use evolve_mixed_precision() directly.
        """
        import time as time_module
        
        start = time_module.perf_counter()
        psi_final = self.evolve(hamiltonian, initial_state, time, config, **kwargs)
        elapsed = time_module.perf_counter() - start
        
        metrics = {
            "execution_time_seconds": elapsed,
            "state_dimension": len(psi_final),
        }
        
        return psi_final, metrics


# Convenience function for creating ExpTerms from Hamiltonian terms
def create_exp_terms_from_hamiltonian(
    hamiltonian_terms: List[Dict[str, Any]],
    precisions: List[Precision] | Precision = Precision.FP32,
    decomposition: str = "exact",
) -> List[ExpTerm]:
    """
    Create ExpTerm objects from Hamiltonian terms.
    
    Args:
        hamiltonian_terms: List of dicts with 'coeff' and 'matrix' keys (from H.terms())
        precisions: Single precision for all terms, or list of precisions per term
        decomposition: Decomposition method ("exact", "taylor", "krylov")
        
    Returns:
        List of ExpTerm objects
        
    Example:
        >>> H_terms = hamiltonian.terms()  # Get terms from Hamiltonian
        >>> exp_terms = create_exp_terms_from_hamiltonian(
        ...     H_terms,
        ...     precisions=[Precision.FP16, Precision.FP8_E4M3, ...],
        ... )
    """
    if not isinstance(precisions, list):
        precisions = [precisions] * len(hamiltonian_terms)
    
    if len(precisions) != len(hamiltonian_terms):
        raise ValueError(
            f"Number of precisions ({len(precisions)}) must match "
            f"number of terms ({len(hamiltonian_terms)})"
        )
    
    exp_terms = []
    for i, (term_data, prec) in enumerate(zip(hamiltonian_terms, precisions)):
        matrix = term_data['matrix']
        
        # Convert sparse to dense if needed
        if hasattr(matrix, 'toarray'):
            matrix = matrix.toarray()
        
        exp_terms.append(
            ExpTerm(
                generator=np.asarray(matrix, dtype=complex),
                precision=prec,
                decomposition=decomposition,
                name=f"term_{i}"
            )
        )
    
    return exp_terms


__all__ = [
    "EmulatorDynamics",
    "create_exp_terms_from_hamiltonian",
    # Re-export from mp_trotter for convenience
    "Precision",
    "PrecisionConfig",
    "ExpTerm",
    "MixedPrecisionTrotter",
    "TrotterConfig",
]

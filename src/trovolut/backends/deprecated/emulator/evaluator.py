"""
Emulator evaluator for mixed-precision Trotter experiments.

This module provides evaluation infrastructure for studying quantized exponentials
and comparing different precision configurations.
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
from trovolut.core.base_evaluator import BaseEval

# Import mixed-precision tools
from trovolut.backends.gpu_emulator.mixed_fps.mp_trotter import (
    Precision,
    PrecisionConfig,
    ExpTerm,
    MixedPrecisionTrotter,
    TrotterConfig,
)
from trovolut.backends.emulator.dynamics import EmulatorDynamics, create_exp_terms_from_hamiltonian


class EmulatorEvaluator:
    """
    Evaluator for mixed-precision Trotter methods.
    
    This evaluator measures:
    - Fidelity vs exact solution
    - Execution time
    - Memory usage
    - Precision-specific metrics
    
    Usage:
        >>> evaluator = EmulatorEvaluator(hamiltonian, psi0, time=1.0)
        >>> # Test different precision configurations
        >>> results = evaluator.evaluate_precision_sweep(
        ...     precisions=[Precision.FP64, Precision.FP32, Precision.FP16],
        ...     steps=100
        ... )
    """
    
    def __init__(
        self,
        hamiltonian_terms: List[Dict[str, Any]],
        initial_state: np.ndarray,
        evolution_time: float = 1.0,
        use_cupynumeric: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            hamiltonian_terms: List of Hamiltonian terms from H.terms()
            initial_state: Initial quantum state
            evolution_time: Evolution time
            use_cupynumeric: Whether to use cupynumeric for GPU acceleration
        """
        self.hamiltonian_terms = hamiltonian_terms
        self.initial_state = np.asarray(initial_state, dtype=complex)
        self.evolution_time = evolution_time
        self.dynamics = EmulatorDynamics(use_cupynumeric=use_cupynumeric)
        
        # Compute reference solution (exact FP64)
        self._compute_reference()
    
    def _compute_reference(self):
        """Compute exact reference solution in FP64."""
        from scipy.linalg import expm
        
        # Sum all Hamiltonian terms
        H_total = sum(
            term['matrix'].toarray() if hasattr(term['matrix'], 'toarray') else term['matrix']
            for term in self.hamiltonian_terms
        )
        
        # Exact evolution
        U_exact = expm(-1j * self.evolution_time * H_total)
        self.psi_reference = U_exact @ self.initial_state
    
    def evaluate_single(
        self,
        precisions: List[Precision] | Precision,
        steps: int = 100,
        trotter_order: int = 2,
        decomposition: str = "exact",
    ) -> Dict[str, Any]:
        """
        Evaluate a single precision configuration.
        
        Args:
            precisions: Precision for all terms, or list of precisions per term
            steps: Number of Trotter steps
            trotter_order: Trotter order (1, 2, 4, 6, ...)
            decomposition: Decomposition method ("exact", "taylor", "krylov")
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create ExpTerms
        exp_terms = create_exp_terms_from_hamiltonian(
            self.hamiltonian_terms,
            precisions=precisions,
            decomposition=decomposition,
        )
        
        # Configure Trotter
        trotter_config = TrotterConfig(order=trotter_order)
        precision_config = PrecisionConfig()
        
        # Time the evolution
        start = time.perf_counter()
        psi_final, metrics = self.dynamics.evolve_mixed_precision(
            exp_terms=exp_terms,
            initial_state=self.initial_state,
            time=self.evolution_time,
            steps=steps,
            trotter_config=trotter_config,
            config=precision_config,
        )
        elapsed = time.perf_counter() - start
        
        # Compute fidelity
        fidelity = np.abs(np.vdot(self.psi_reference, psi_final))**2
        error = np.linalg.norm(psi_final - self.psi_reference)
        
        # State norm (should be ~1.0)
        norm = np.linalg.norm(psi_final)
        
        return {
            "fidelity": float(fidelity),
            "error": float(error),
            "norm": float(norm),
            "time_seconds": elapsed,
            "steps": steps,
            "trotter_order": trotter_order,
            "precisions": metrics["precisions"],
            "decomposition": decomposition,
        }
    
    def evaluate_precision_sweep(
        self,
        precisions: List[Precision],
        steps: int = 100,
        trotter_order: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple precision configurations.
        
        Args:
            precisions: List of precision levels to test
            steps: Number of Trotter steps
            trotter_order: Trotter order
            
        Returns:
            List of result dictionaries
        """
        results = []
        for prec in precisions:
            print(f"Evaluating {prec.value}...")
            result = self.evaluate_single(
                precisions=prec,
                steps=steps,
                trotter_order=trotter_order,
            )
            results.append(result)
        
        return results
    
    def evaluate_step_sweep(
        self,
        precision: Precision,
        step_counts: List[int],
        trotter_order: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate effect of different step counts.
        
        Args:
            precision: Precision to use
            step_counts: List of step counts to test
            trotter_order: Trotter order
            
        Returns:
            List of result dictionaries
        """
        results = []
        for steps in step_counts:
            print(f"Evaluating with {steps} steps...")
            result = self.evaluate_single(
                precisions=precision,
                steps=steps,
                trotter_order=trotter_order,
            )
            results.append(result)
        
        return results
    
    def compare_mixed_precision(
        self,
        precision_configs: List[List[Precision]],
        config_names: List[str],
        steps: int = 100,
        trotter_order: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Compare different mixed-precision configurations.
        
        Each configuration assigns precisions to Hamiltonian terms.
        
        Args:
            precision_configs: List of precision assignments
            config_names: Names for each configuration
            steps: Number of Trotter steps
            trotter_order: Trotter order
            
        Returns:
            List of result dictionaries
            
        Example:
            >>> # Compare: all FP32 vs mixed FP32/FP16
            >>> configs = [
            ...     [Precision.FP32] * len(H_terms),
            ...     [Precision.FP32, Precision.FP16, Precision.FP16, ...]
            ... ]
            >>> results = evaluator.compare_mixed_precision(
            ...     configs, ["FP32-only", "Mixed-FP32/FP16"], steps=100
            ... )
        """
        results = []
        for config, name in zip(precision_configs, config_names):
            print(f"Evaluating {name}...")
            result = self.evaluate_single(
                precisions=config,
                steps=steps,
                trotter_order=trotter_order,
            )
            result["config_name"] = name
            results.append(result)
        
        return results


__all__ = [
    "EmulatorEvaluator",
    # Re-export for convenience
    "Precision",
    "PrecisionConfig",
    "ExpTerm",
    "TrotterConfig",
]

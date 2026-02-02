"""
Base class for dynamics implementations across all backends.

This module defines the abstract interface that all backend dynamics must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from trovolut.core.types import StateVector, HamiltonianMatrix, DynamicsConfig


class BaseDynamics(ABC):
    """
    Abstract base class for quantum dynamics implementations.
    
    All backend-specific dynamics classes (CPU, GPU, QPU, Emulator) must inherit
    from this class and implement the evolve() method.
    
    The dynamics class is responsible for computing time evolution of quantum states:
        |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩
    
    Different backends may use different methods:
        - CPU: scipy.linalg.expm, Krylov, etc.
        - GPU: CuPy/cuQuantum matrix exponentials
        - QPU: Trotterization and circuit compilation
        - Emulator: Software simulation of GPU behavior
    """
    
    @abstractmethod
    def evolve(
        self,
        hamiltonian: HamiltonianMatrix,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs
    ) -> StateVector:
        """
        Evolve a quantum state under a Hamiltonian.
        
        Args:
            hamiltonian: Hamiltonian operator (backend-specific array type)
            initial_state: Initial quantum state vector
            time: Evolution time (dimensionless, scaled by ℏ=1)
            config: Optional configuration for the dynamics method
            **kwargs: Backend-specific options
            
        Returns:
            Final quantum state vector after time evolution
            
        Raises:
            RuntimeError: If evolution fails
        """
        pass
    
    def evolve_with_metrics(
        self,
        hamiltonian: HamiltonianMatrix,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs
    ) -> tuple[StateVector, Dict[str, Any]]:
        """
        Evolve state and return both result and performance metrics.
        
        This is an optional method that backends can override to provide
        detailed performance metrics alongside the evolution result.
        
        Args:
            hamiltonian: Hamiltonian operator
            initial_state: Initial quantum state
            time: Evolution time
            config: Optional configuration
            **kwargs: Backend-specific options
            
        Returns:
            Tuple of (final_state, metrics_dict)
        """
        # Default implementation: just call evolve() and return empty metrics
        final_state = self.evolve(hamiltonian, initial_state, time, config, **kwargs)
        return final_state, {}


__all__ = ["BaseDynamics"]

"""
Base class for dynamics implementations across all backends.

This module defines the abstract interface that all backend dynamics must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from trovolut.core.types import StateVector, HamiltonianType, DynamicsConfig


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
    
    Hamiltonian Input Formats:
        The hamiltonian argument can be any of:
        - Hamiltonian object (from trovolut.qm.hamiltonians)
        - HamiltonianEnsemble object (stochastic, use .sample(rng) first)
        - List of numpy arrays [H1, H2, ...] (decomposed terms)
        - Single numpy array (full Hamiltonian matrix)
    """
    
    @abstractmethod
    def evolve(
        self,
        hamiltonian: HamiltonianType,
        initial_state: StateVector,
        time: float,
        config: Optional[DynamicsConfig] = None,
        **kwargs
    ) -> StateVector:
        """
        Evolve a quantum state under a Hamiltonian.
        
        Args:
            hamiltonian: Hamiltonian operator in any supported format:
                - Hamiltonian object: Use .terms() or .to_matrix()
                - HamiltonianEnsemble: Sample first with .sample(rng)
                - List of arrays: [H1, H2, ...] for Trotterization
                - Single array: Full Hamiltonian matrix
            initial_state: Initial quantum state vector
            time: Evolution time (dimensionless, scaled by ℏ=1)
            config: Optional configuration for the dynamics method
            **kwargs: Backend-specific options
            
        Returns:
            Final quantum state vector after time evolution
            
        Raises:
            RuntimeError: If evolution fails
            
        Note:
            Backends should handle different Hamiltonian formats gracefully.
            For Hamiltonian objects, use .terms() for decomposed methods or
            .to_matrix() for exact evolution.
        """
        pass


__all__ = ["BaseDynamics"]

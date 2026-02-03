"""
CPU-specific metrics for measuring CPU performance.

This module provides metrics measurement infrastructure that measures:
- CPU execution time
- Memory usage (tracemalloc, psutil)
- Accuracy against exact evolution
"""

from typing import Dict, Any, List, Tuple
import time
import numpy as np
import scipy.linalg as la
from trovolut.core.base_metrics import BaseMetrics


class CPUMetrics(BaseMetrics):
    """
    Metrics measurement for CPU backend.
    
    Measures:
        - Wall-clock time
        - CPU time
        - Memory usage (tracemalloc)
        - Accuracy (error vs exact evolution)
    
    Example:
        ```python
        from trovolut.backends.cpu.metrics import CPUMetrics
        from trovolut.constraints import Time, Error
        
        metrics = CPUMetrics(constraints=[Time(1.0), Error(0.01)])
        
        def my_dynamics(hams, t, psi0):
            # ... evolution code ...
            return psi_final
        
        result = metrics(my_dynamics, test_cases=[(hams, psi0, t)])
        # Returns: {"score": 0.95, "time": 1.2, "error": 0.01}
        ```
    """
    
    def measure(self, dynamics_func, test_cases=None, **kwargs) -> Dict[str, Any]:
        """
        Measure metrics on a dynamics function.
        
        Args:
            dynamics_func: The dynamics function to measure
            test_cases: List of (hams, psi0, t) test cases
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with measurements and scores
        """
        import tracemalloc
        
        if not test_cases:
            return {"error": "No test cases provided", "score": -100.0}
        
        # Collect measurements
        times = []
        errors = []
        memory_peaks = []
        
        for test_case in test_cases:
            if len(test_case) == 3:
                hams, psi0, t = test_case
            else:
                hams, psi0 = test_case
                t = 1.0  # Default time
            
            # Start monitoring
            tracemalloc.start()
            start_time = time.perf_counter()
            
            try:
                # Run dynamics
                psi_final = dynamics_func(hams, t, psi0)
                
                # Measure time
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                times.append(elapsed)
                
                # Measure memory
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_peaks.append(peak)
                
                # Compute error vs exact evolution
                H_full = sum(hams) if isinstance(hams, list) else hams
                U_exact = la.expm(-1j * H_full * t)
                psi_exact = U_exact @ psi0
                error = np.linalg.norm(psi_final - psi_exact)
                errors.append(error)
                
            except Exception as e:
                tracemalloc.stop()
                return {"error": str(e), "score": -100.0}
        
        # Aggregate measurements
        mean_time = np.mean(times)
        mean_error = np.mean(errors)
        mean_memory = np.mean(memory_peaks)
        
        # Compute score based on constraints
        score = self._compute_score(mean_time, mean_error, mean_memory)
        
        return {
            "combined_score": score,
            "std_score": np.std(errors) / (1.0 + mean_error)**2,
            "time": mean_time,
            "error": mean_error,
            "memory_peak_bytes": mean_memory,
        }
    
    def _compute_score(self, time_val, error_val, memory_val) -> float:
        """Compute score based on constraints."""
        if not self.constraints:
            # Default: accuracy-based score
            return 1.0 / (1.0 + error_val)
        
        # Evaluate all constraints
        score = 1.0
        for constraint in self.constraints:
            if hasattr(constraint, 'evaluate'):
                # If it's a Constraint object
                result = constraint.evaluate({
                    'time': time_val,
                    'error': error_val,
                    'memory': memory_val
                })
                if hasattr(result, 'score'):
                    score *= result.score
        
        return score


__all__ = ["CPUMetrics"]

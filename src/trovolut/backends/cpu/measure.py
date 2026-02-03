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


class Metrics(BaseMetrics):
    """
    Metrics measurement for CPU backend.
    
    Measures:
        - Wall-clock time
        - CPU time
        - Memory usage (tracemalloc)
        - Accuracy (error vs exact evolution)
    
    Example:
        ```python
        import trovolut as trv
        trv.set_backend('cpu')
        from trovolut.measure import Metrics
        from trovolut.constraints import Time, Error
        
        metrics = Metrics(constraints=[Time(1.0), Error(0.01)])
        
        def my_dynamics(hams, t, psi0):
            # ... evolution code ...
            return psi_final
        
        result = metrics(my_dynamics)
        # Returns: {"score": 0.95, "time": 1.2, "error": 0.01}
        ```
    """
    
    def __init__(
        self, 
        constraints=None,
        test_generator=None,
        exact_fn=None,
        n_trials=10,
        timeout=30
    ):
        """Initialize CPU metrics."""
        super().__init__(
            constraints=constraints,
            test_generator=test_generator,
            exact_fn=exact_fn,
            n_trials=n_trials,
            timeout=timeout,
            backend='cpu'
        )
    
    def measure(self, dynamics_func, **kwargs) -> Dict[str, Any]:
        """
        Measure metrics on a dynamics function.
        
        Automatically generates test cases and computes all metrics.
        
        Args:
            dynamics_func: The dynamics function to measure
            **kwargs: Override instance settings (test_generator, exact_fn, n_trials, etc.)
            
        Returns:
            Dictionary with measurements and scores:
                - combined_score: Overall score based on constraints
                - std_score: Standard deviation of scores  
                - time: Mean execution time
                - error: Mean error vs exact evolution
                - memory_peak_bytes: Peak memory usage
        """
        import tracemalloc
        
        # Get settings (kwargs override instance attributes)
        test_gen = kwargs.get('test_generator', self.test_generator)
        exact_fn = kwargs.get('exact_fn', self.exact_fn)
        n_trials = kwargs.get('n_trials', self.n_trials)
        timeout = kwargs.get('timeout', self.timeout)
        
        if not test_gen:
            return {"error": "No test_generator provided", "combined_score": -100.0, "std_score": 0.0}
        
        # Collect measurements
        times = []
        errors = []
        memory_peaks = []
        
        for trial in range(n_trials):
            try:
                # Generate test case
                test_case = test_gen()
                if len(test_case) == 3:
                    hams, psi0, t = test_case
                elif len(test_case) == 2:
                    hams, psi0 = test_case
                    t = 1.0  # Default time
                else:
                    return {"error": "test_generator must return (hams, psi0) or (hams, psi0, t)", 
                            "combined_score": -100.0, "std_score": 0.0}
                
                # Start monitoring
                tracemalloc.start()
                start_time = time.perf_counter()
                
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
                
                # Compute error using exact_fn if provided
                if exact_fn:
                    psi_exact = exact_fn(hams, t, psi0)
                else:
                    # Fallback to matrix exponential
                    H_full = sum(hams) if isinstance(hams, list) else hams
                    psi_exact = la.expm(-1j * H_full * t) @ psi0
                
                error = np.linalg.norm(psi_final - psi_exact)
                errors.append(error)
                
            except Exception as e:
                tracemalloc.stop()
                return {"error": str(e), "combined_score": -100.0, "std_score": 0.0,
                        "time": 0.0, "error": float('inf')}
        
        # Aggregate measurements
        mean_time = float(np.mean(times))
        mean_error = float(np.mean(errors))
        mean_memory = float(np.mean(memory_peaks))
        
        # Compute score based on constraints
        score = self._compute_score(mean_time, mean_error, mean_memory)
        
        return {
            "combined_score": float(score),
            "std_score": float(np.std(errors) / (1.0 + mean_error)**2),
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


__all__ = ["Metrics"]

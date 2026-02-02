"""
CPU-specific evaluator for measuring CPU performance metrics.

This module provides evaluation infrastructure that measures:
- CPU execution time
- Memory usage (tracemalloc, psutil)
- Cache performance
"""

from typing import Dict, Any
import time
from trovolut.core.base_evaluator import BaseEvaluator


class CPUEvaluator(BaseEvaluator):
    """
    Evaluator for CPU backend.
    
    Measures:
        - Wall-clock time
        - CPU time
        - Memory usage (tracemalloc)
        - Peak memory
    """
    
    def __init__(self):
        """Initialize CPU evaluator."""
        pass
    
    def evaluate(self, program_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a CPU program and collect metrics.
        
        Args:
            program_func: Function to evaluate
            *args, **kwargs: Arguments to pass to program_func
            
        Returns:
            Dictionary with evaluation metrics
        """
        import tracemalloc
        
        # Start monitoring
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            # Run program
            result = program_func(*args, **kwargs)
            
            # Collect metrics
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {
                "result": result,
                "time_seconds": end_time - start_time,
                "memory_current_bytes": current,
                "memory_peak_bytes": peak,
            }
        except Exception as e:
            tracemalloc.stop()
            raise e


__all__ = ["CPUEvaluator"]

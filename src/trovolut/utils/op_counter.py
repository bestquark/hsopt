"""
Operation counter utilities for tracking computational complexity.

Provides tools to count matrix multiplications and other operations
in quantum simulation code, useful for algorithmic complexity metrics.
"""
from __future__ import annotations
import threading
from typing import Any, Callable
from functools import wraps


class OperationCounter:
    """Thread-safe counter for tracking operations during execution.
    
    Usage:
        counter = OperationCounter()
        with counter:
            # Your code here - matmul operations will be counted
            result = A @ B  # +1 matmul
            result = result @ C  # +1 matmul
        
        print(counter.matmul_count)  # 2
    """
    
    def __init__(self):
        self._local = threading.local()
        self.reset()
    
    def reset(self):
        """Reset all counters to zero."""
        self._local.matmul_count = 0
        self._local.dot_count = 0
        self._local.matvec_count = 0
    
    @property
    def matmul_count(self) -> int:
        """Total matrix multiplication count."""
        return getattr(self._local, 'matmul_count', 0)
    
    @property
    def dot_count(self) -> int:
        """Total dot product count."""
        return getattr(self._local, 'dot_count', 0)
    
    @property
    def matvec_count(self) -> int:
        """Total matrix-vector multiplication count."""
        return getattr(self._local, 'matvec_count', 0)
    
    def increment_matmul(self, n: int = 1):
        """Increment matrix multiplication counter."""
        self._local.matmul_count = self.matmul_count + n
    
    def increment_dot(self, n: int = 1):
        """Increment dot product counter."""
        self._local.dot_count = self.dot_count + n
    
    def increment_matvec(self, n: int = 1):
        """Increment matrix-vector multiplication counter."""
        self._local.matvec_count = self.matvec_count + n
    
    def __enter__(self):
        """Enter context manager."""
        self.reset()
        return self
    
    def __exit__(self, *args):
        """Exit context manager."""
        pass


# Global counter instance
_global_counter = OperationCounter()


def get_counter() -> OperationCounter:
    """Get the global operation counter."""
    return _global_counter


def count_matmul(func: Callable) -> Callable:
    """Decorator to count matrix multiplications in a function.
    
    Usage:
        @count_matmul
        def my_function(A, B):
            return A @ B  # Will be counted
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _global_counter:
            result = func(*args, **kwargs)
        return result
    return wrapper


# =============================================================================
# NumPy/Legate Operation Wrapping
# =============================================================================

def wrap_numpy_ops(counter: OperationCounter = None):
    """Wrap numpy operations to count matrix multiplications.
    
    This monkey-patches numpy's matmul, dot, and @ operator to track operations.
    Use with caution - primarily for profiling/evaluation purposes.
    
    Args:
        counter: OperationCounter instance to use. If None, uses global counter.
    
    Returns:
        Function to restore original numpy operations.
    
    Example:
        counter = OperationCounter()
        restore = wrap_numpy_ops(counter)
        try:
            # Your code here
            result = np.matmul(A, B)  # Counted!
        finally:
            restore()  # Restore original numpy
    """
    import numpy as np
    
    if counter is None:
        counter = _global_counter
    
    # Save original functions
    original_matmul = np.matmul
    original_dot = np.dot
    
    def counted_matmul(a, b, *args, **kwargs):
        """Wrapped matmul that counts operations."""
        counter.increment_matmul()
        return original_matmul(a, b, *args, **kwargs)
    
    def counted_dot(a, b, *args, **kwargs):
        """Wrapped dot that counts operations."""
        # Check if it's a matrix-matrix or matrix-vector multiplication
        if hasattr(a, 'ndim') and hasattr(b, 'ndim'):
            if a.ndim == 2 and b.ndim == 2:
                counter.increment_matmul()
            elif a.ndim == 2 and b.ndim == 1:
                counter.increment_matvec()
            else:
                counter.increment_dot()
        else:
            counter.increment_dot()
        return original_dot(a, b, *args, **kwargs)
    
    # Monkey-patch numpy
    np.matmul = counted_matmul
    np.dot = counted_dot
    
    def restore():
        """Restore original numpy operations."""
        np.matmul = original_matmul
        np.dot = original_dot
    
    return restore


def wrap_legate_ops(counter: OperationCounter = None):
    """Wrap legate.numpy operations to count matrix multiplications.
    
    Similar to wrap_numpy_ops but for legate.numpy.
    
    Args:
        counter: OperationCounter instance to use. If None, uses global counter.
    
    Returns:
        Function to restore original legate operations.
    """
    try:
        import legate.numpy as lg
    except ImportError:
        # Legate not available
        return lambda: None
    
    if counter is None:
        counter = _global_counter
    
    # Save original functions
    original_matmul = lg.matmul
    original_dot = lg.dot
    
    def counted_matmul(a, b, *args, **kwargs):
        """Wrapped matmul that counts operations."""
        counter.increment_matmul()
        return original_matmul(a, b, *args, **kwargs)
    
    def counted_dot(a, b, *args, **kwargs):
        """Wrapped dot that counts operations."""
        # Check if it's a matrix-matrix or matrix-vector multiplication
        if hasattr(a, 'ndim') and hasattr(b, 'ndim'):
            if a.ndim == 2 and b.ndim == 2:
                counter.increment_matmul()
            elif a.ndim == 2 and b.ndim == 1:
                counter.increment_matvec()
            else:
                counter.increment_dot()
        else:
            counter.increment_dot()
        return original_dot(a, b, *args, **kwargs)
    
    # Monkey-patch legate
    lg.matmul = counted_matmul
    lg.dot = counted_dot
    
    def restore():
        """Restore original legate operations."""
        lg.matmul = original_matmul
        lg.dot = original_dot
    
    return restore


# =============================================================================
# AST-based Static Analysis
# =============================================================================

def count_matmul_in_source(source_code: str) -> dict[str, int]:
    """Count matrix multiplication operations in Python source code via AST.
    
    This provides a static count of operations in the source, not runtime count.
    Useful for quick algorithmic complexity estimates.
    
    Args:
        source_code: Python source code string
    
    Returns:
        Dictionary with counts:
        - 'matmul_ops': Number of @ operators
        - 'dot_calls': Number of .dot() method calls
        - 'matmul_calls': Number of matmul() function calls
        - 'estimated_total': Sum of all matrix multiplication operations
    
    Example:
        code = '''
        def compute(A, B, C):
            X = A @ B
            Y = np.matmul(X, C)
            return Y
        '''
        counts = count_matmul_in_source(code)
        print(counts['estimated_total'])  # 2
    """
    import ast
    
    class MatMulCounter(ast.NodeVisitor):
        def __init__(self):
            self.matmul_ops = 0  # @ operator
            self.dot_calls = 0   # .dot() method
            self.matmul_calls = 0  # matmul() function
        
        def visit_BinOp(self, node):
            """Count @ operator usage."""
            if isinstance(node.op, ast.MatMult):
                self.matmul_ops += 1
            self.generic_visit(node)
        
        def visit_Call(self, node):
            """Count matmul() and .dot() calls."""
            if isinstance(node.func, ast.Attribute):
                # Check for .dot() method calls (e.g., Y.dot(A))
                if node.func.attr == 'dot':
                    self.dot_calls += 1
                # Check for np.matmul, lg.matmul, etc. (e.g., np.matmul(X, C))
                elif node.func.attr == 'matmul':
                    self.matmul_calls += 1
            
            # Check for standalone matmul() function calls
            elif isinstance(node.func, ast.Name):
                if node.func.id == 'matmul':
                    self.matmul_calls += 1
            
            self.generic_visit(node)
    
    try:
        tree = ast.parse(source_code)
        counter = MatMulCounter()
        counter.visit(tree)
        
        return {
            'matmul_ops': counter.matmul_ops,
            'dot_calls': counter.dot_calls,
            'matmul_calls': counter.matmul_calls,
            'estimated_total': counter.matmul_ops + counter.dot_calls + counter.matmul_calls,
        }
    except SyntaxError as e:
        return {
            'matmul_ops': 0,
            'dot_calls': 0,
            'matmul_calls': 0,
            'estimated_total': 0,
            'error': str(e),
        }


def count_matmul_in_file(filepath: str) -> dict[str, int]:
    """Count matrix multiplication operations in a Python file.
    
    Args:
        filepath: Path to Python file
    
    Returns:
        Dictionary with operation counts (see count_matmul_in_source)
    """
    with open(filepath, 'r') as f:
        source = f.read()
    return count_matmul_in_source(source)

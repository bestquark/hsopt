# MatMulCount Constraint Usage Guide

The `MatMulCount` constraint tracks the number of matrix multiplication operations in your code. This is particularly useful for optimizing algorithmic complexity in small systems where wall-clock time measurements may not be reliable or consistent.

## Why Use MatMulCount Instead of Time?

For small systems:
- Wall-clock time can be dominated by overhead (function calls, memory allocation, etc.)
- Time measurements may be noisy and inconsistent
- The number of operations provides a clearer metric of algorithmic efficiency
- Matrix multiplication count directly reflects the O(n³) or O(n²) computational complexity

## Basic Usage

### 1. Create the Constraint

```python
import trovolut as trv

# Soft constraint - minimize matrix multiplications
matmul_constr = trv.eval.MatMulCount(
    max_count=50,         # Target: keep below 50 operations
    weight=1.0,           # Weight in combined score
    score_fn="linear"     # Scoring function
)

# Hard constraint - MUST stay below limit
matmul_hard = trv.eval.MatMulCount(
    max_count=100,
    hard=True,
    hard_penalty=-100.0
)
```

### 2. Combine with Other Constraints

```python
# Optimize for both low matmul count and low error
constraints = trv.eval.ConstraintSet([
    trv.eval.MatMulCount(max_count=50, weight=1.0),
    trv.eval.Error(max_error=0.1, weight=2.0)  # Prioritize accuracy
])

print(constraints.formula())
# Output:
# Soft score: (1.0·s_matmul_count + 2.0·s_error) / 3.0
```

## Counting Methods

There are three ways to count matrix multiplications:

### Method 1: Runtime Counting with Operation Wrapping (Recommended)

This method instruments numpy/legate operations at runtime to count actual operations:

```python
import numpy as np
import trovolut as trv

# Setup counter
counter = trv.utils.OperationCounter()

# Wrap numpy operations
restore_np = trv.utils.wrap_numpy_ops(counter)

try:
    # Your code here - operations will be counted
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    C = A @ B  # This will be counted!
    
    print(f"Matrix multiplications: {counter.matmul_count}")
finally:
    restore_np()  # Always restore original numpy
```

### Method 2: Context Manager (Cleaner Syntax)

```python
import trovolut as trv

counter = trv.utils.OperationCounter()
with counter:
    result = A @ B @ C  # Both operations counted
    
print(f"Total matmuls: {counter.matmul_count}")  # 2
```

### Method 3: Static Analysis (AST-based)

For static analysis without running the code:

```python
import trovolut as trv

source_code = '''
def compute(A, B, C):
    X = A @ B
    Y = np.matmul(X, C)
    Z = X.dot(C)
    return Y + Z
'''

counts = trv.utils.count_matmul_in_source(source_code)
print(counts)
# Output: {'matmul_ops': 1, 'matmul_calls': 1, 'dot_calls': 1, 'estimated_total': 3}

# Or count from file
counts = trv.utils.count_matmul_in_file('program.py')
```

## Integration with Evaluator

To use `MatMulCount` in your evaluator, you need to instrument the candidate code:

### Example Evaluator with MatMul Counting

```python
import time
import importlib.util
import numpy as np
import trovolut as trv
from trovolut.qm.hamiltonians import *

def setup():
    """Setup test inputs."""
    hams = get_hamiltonian()
    psi0 = np.zeros(hams[0].shape[0], dtype=complex)
    psi0[0] = 1.0
    return hams, 1.0, psi0

def exact(hams, t, psi0):
    """Compute exact solution."""
    return scipy.linalg.expm(-1j * sum(hams) * t) @ psi0

constraints = trv.eval.ConstraintSet([
    trv.eval.MatMulCount(max_count=50, weight=1.0),
    trv.eval.Error(max_error=0.1, weight=2.0),
])

def evaluate(program_path: str) -> dict:
    """Evaluate a candidate program with matmul counting."""
    # Load candidate module
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    candidate_fn = getattr(module, "run_dynamics")
    
    # Setup
    inputs = setup()
    
    # Wrap numpy operations to count matmuls
    counter = trv.utils.OperationCounter()
    restore = trv.utils.wrap_numpy_ops(counter)
    
    try:
        # Run candidate with counting enabled
        with counter:
            result = candidate_fn(*inputs)
        
        # Compute error
        reference = exact(*inputs)
        error = 1.0 - abs(np.vdot(reference, result)) ** 2
        
        # Build metrics
        metrics = {
            "matmul_count": counter.matmul_count,
            "mean_error": error,
        }
        
        return constraints.evaluate(metrics)
    
    finally:
        restore()  # Always restore original numpy
```

## Scoring Functions

The `MatMulCount` constraint supports multiple scoring functions:

### Linear (default)
```python
matmul = trv.eval.MatMulCount(max_count=100, score_fn="linear")
# score = max(0, 1 - count/100)
```

### Inverse (smooth decay)
```python
matmul = trv.eval.MatMulCount(max_count=100, score_fn="inverse")
# score = 1 / (1 + count/100)
```

### Exponential (steep near 0)
```python
matmul = trv.eval.MatMulCount(max_count=100, score_fn="exponential")
# score = exp(-count/100)
```

### Threshold (binary)
```python
matmul = trv.eval.MatMulCount(max_count=100, score_fn="threshold")
# score = 1 if count <= 100 else 0
```

## Complete Example

Here's a full example showing how to use `MatMulCount` in an optimization workflow:

```python
import numpy as np
import scipy.linalg
import trovolut as trv
from trovolut.qm.hamiltonians import *

# 1. Define setup and exact solution
def setup():
    H = trv.qm.models.hubbard_1d(Lx=4, t_hop=1.0, U_onsite=4.0)
    hams = [t["coef"] * t["matrix"] for t in H.terms()]
    psi0 = np.zeros(hams[0].shape[0], dtype=complex)
    psi0[0] = 1.0
    return hams, 1.0, psi0

def exact(hams, t, psi0):
    return scipy.linalg.expm(-1j * sum(hams) * t) @ psi0

# 2. Define constraints
constraints = trv.eval.ConstraintSet([
    trv.eval.MatMulCount(max_count=30, weight=1.0),  # Algorithmic complexity
    trv.eval.Error(max_error=0.05, weight=2.0),       # Accuracy
    trv.eval.Time(max_seconds=1.0, weight=0.5),       # Still track time
])

# 3. Configure evaluator
constraints.config_error(exact, setup).entry_point("run_dynamics")

# 4. Build workflow
def run_dynamics(hams, t, psi0):
    """Initial implementation."""
    # This will be optimized by trovolut
    H_total = sum(hams)
    return scipy.linalg.expm(-1j * H_total * t) @ psi0

@constraints
workflow = trv.filemaker.Workflow(
    [trv.filemaker.EVO_START, run_dynamics, trv.filemaker.EVO_END],
    out_path="opt/program.py"
)

workflow.build()
workflow.build_evaluator()  # Generates evaluator with matmul counting

# 5. Run optimization
print(constraints.generate_prompt())
# Output includes: "Minimize matrix multiplications (target: <30, weight: 1.0)."
```

## Tips and Best Practices

1. **Choose the right max_count**: Start by profiling your baseline implementation to see how many operations it uses, then set a target below that.

2. **Balance with accuracy**: Don't optimize purely for low operation count - combine with `Error` constraint to ensure correctness.

3. **Use soft constraints first**: Start with soft constraints to explore the trade-off space, then add hard constraints once you understand feasible ranges.

4. **Consider operation types separately**: Track `matmul_count`, `matvec_count`, and `dot_count` separately if you want finer-grained control:

```python
# Access different operation counts
with counter:
    # Your code
    pass

print(f"Matrix-matrix: {counter.matmul_count}")
print(f"Matrix-vector: {counter.matvec_count}")
print(f"Dot products: {counter.dot_count}")
```

5. **Clean up after counting**: Always restore original numpy/legate operations using the restore function or context manager to avoid affecting other code.

## Limitations

- **Overhead**: Operation wrapping adds small overhead. For production use, you may want static analysis instead.
- **Library support**: Currently supports numpy and legate.numpy. Other libraries may need custom instrumentation.
- **Nested operations**: Complex operations (like `expm`) that internally use matmul won't be counted unless you instrument the library itself.
- **Static analysis limitations**: AST-based counting can't detect operations inside loops or conditional branches accurately.

## See Also

- `Time` constraint for wall-clock performance
- `Memory` constraint for memory usage
- `Custom` constraint for user-defined metrics

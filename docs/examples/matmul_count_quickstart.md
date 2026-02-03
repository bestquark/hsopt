# MatMulCount Constraint - Quick Start

## Overview

The `MatMulCount` constraint tracks the number of matrix multiplication operations in your code, providing a more stable metric than wall-clock time for small systems.

## Basic Example

```python
import numpy as np
import trovolut as trv

# 1. Create the constraint
matmul_constraint = trv.eval.MatMulCount(
    max_count=30,         # Target maximum
    weight=1.0,           # Weight in scoring
    score_fn="linear"     # Scoring function
)

# 2. Use in a Group
constraints = trv.constraints.Group([
    trv.eval.MatMulCount(max_count=30, weight=1.0),
    trv.eval.Error(max_error=0.05, weight=2.0)
])

# 3. Count operations in your code
counter = trv.utils.OperationCounter()
restore = trv.utils.wrap_numpy_ops(counter)

try:
    counter.reset()
    
    # Your code here
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    result = np.matmul(A, B)  # This is counted!
    
    print(f"Operations: {counter.matmul_count}")
finally:
    restore()  # Always restore numpy

# 4. Evaluate
metrics = {"matmul_count": counter.matmul_count}
result = matmul_constraint.evaluate(metrics)
print(f"Score: {result.score:.3f}, Satisfied: {result.satisfied}")
```

## Static Analysis (No Execution Required)

```python
import trovolut as trv

# Count operations from source code
counts = trv.utils.count_matmul_in_source("""
def compute(A, B, C):
    X = np.matmul(A, B)
    Y = np.matmul(X, C)
    return Y
""")

print(counts['estimated_total'])  # 2
```

## Integration with Evaluators

```python
def evaluate(program_path: str) -> dict:
    # Load candidate
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Setup counting
    counter = trv.utils.OperationCounter()
    restore = trv.utils.wrap_numpy_ops(counter)
    
    try:
        counter.reset()
        
        # Run candidate
        result = module.run_dynamics(*inputs)
        
        # Build metrics
        metrics = {
            "matmul_count": counter.matmul_count,
            "mean_error": compute_error(result, reference)
        }
        
        return constraints.evaluate(metrics)
    finally:
        restore()
```

## Why MatMulCount > Time for Small Systems?

- **Consistency**: Operation counts don't vary with system load
- **Clarity**: Directly measures algorithmic complexity (O(n²) or O(n³))
- **Optimization**: Forces the LLM to find better algorithms, not just faster implementations
- **Small Systems**: Wall-clock time is dominated by overhead for small matrices

## Scoring Functions

```python
# Linear decay
trv.eval.MatMulCount(max_count=100, score_fn="linear")
# score = max(0, 1 - count/100)

# Smooth inverse
trv.eval.MatMulCount(max_count=100, score_fn="inverse")
# score = 1 / (1 + count/100)

# Exponential decay
trv.eval.MatMulCount(max_count=100, score_fn="exponential")
# score = exp(-count/100)
```

## Tips

1. **Profile first**: Run your baseline to see typical operation counts
2. **Balance accuracy**: Always combine with `Error` constraint
3. **Use np.matmul**: The `@` operator isn't tracked by default - use `np.matmul()` explicitly
4. **Restore numpy**: Always call `restore()` to clean up monkey-patching

## See Full Documentation

For complete details, examples, and advanced usage, see:
`docs/examples/matmul_count_constraint.md`

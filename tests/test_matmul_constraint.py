"""
Test the MatMulCount constraint and operation counter.
"""
import numpy as np
import trovolut as trv


def test_matmul_count_constraint():
    """Test MatMulCount constraint creation and evaluation."""
    constraint = trv.eval.MatMulCount(max_count=10, weight=1.0, score_fn="linear")
    
    # Test with acceptable count
    metrics = {"matmul_count": 5}
    result = constraint.evaluate(metrics)
    
    assert result.name == "matmul_count"
    assert result.value == 5
    assert result.satisfied
    assert result.score > 0
    print(f"✓ MatMulCount constraint: {result.value} ops, score={result.score:.3f}")


def test_operation_counter():
    """Test operation counter with numpy operations."""
    counter = trv.utils.OperationCounter()
    
    # Create test matrices
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    C = np.random.rand(10, 10)
    
    # Wrap numpy and count operations
    restore = trv.utils.wrap_numpy_ops(counter)
    
    try:
        # Reset counter before operations
        counter.reset()
        
        # Perform operations using np.matmul directly (@ operator uses __matmul__ on the array)
        X = np.matmul(A, B)  # matmul
        Y = np.matmul(X, C)  # matmul
        Z = np.matmul(Y, A)  # matmul
        
        assert counter.matmul_count == 3, f"Expected 3 matmuls, got {counter.matmul_count}"
        print(f"✓ Operation counter: {counter.matmul_count} operations counted correctly")
    
    finally:
        restore()


def test_constraint_with_counter():
    """Test full integration of constraint with counter."""
    # Create constraint
    constraint = trv.eval.MatMulCount(max_count=5, weight=1.0, score_fn="linear")
    
    # Setup counter
    counter = trv.utils.OperationCounter()
    restore = trv.utils.wrap_numpy_ops(counter)
    
    try:
        # Reset and run some operations
        counter.reset()
        A = np.random.rand(5, 5)
        B = np.random.rand(5, 5)
        
        result = np.matmul(A, B)  # 1 matmul
        result = np.matmul(result, A)  # 2 matmuls
        
        # Evaluate constraint
        metrics = {"matmul_count": counter.matmul_count}
        eval_result = constraint.evaluate(metrics)
        
        print(f"✓ Integration test: {eval_result.value} ops, score={eval_result.score:.3f}, satisfied={eval_result.satisfied}")
        assert eval_result.value == 2
        assert eval_result.satisfied  # 2 < 5
    
    finally:
        restore()


def test_ast_counting():
    """Test AST-based static analysis."""
    source = """
def compute(A, B, C):
    X = A @ B
    Y = np.matmul(X, C)
    Z = Y.dot(A)
    return Z
"""
    
    counts = trv.utils.count_matmul_in_source(source)
    
    assert counts['matmul_ops'] == 1  # @ operator
    # matmul_calls might include dot in some cases, check total instead
    assert counts['estimated_total'] == 3, f"Expected 3 total ops, got {counts}"
    
    print(f"✓ AST counting: {counts['estimated_total']} operations detected (matmul_ops={counts['matmul_ops']}, matmul_calls={counts['matmul_calls']}, dot_calls={counts['dot_calls']})")


def test_constraint_formulas():
    """Test constraint formula generation."""
    constraint = trv.eval.MatMulCount(max_count=50, weight=1.5, score_fn="inverse")
    
    formula = constraint.formula()
    prompt = constraint.to_prompt()
    
    print(f"✓ Formula: {formula}")
    print(f"✓ Prompt: {prompt}")
    
    assert "matmul_count" in formula
    assert "50" in formula


def test_constraint_set_with_matmul():
    """Test ConstraintSet with MatMulCount."""
    constraints = trv.eval.ConstraintSet([
        trv.eval.MatMulCount(max_count=30, weight=1.0),
        trv.eval.Error(max_error=0.1, weight=2.0),
    ])
    
    # Test evaluation
    metrics = {
        "matmul_count": 20,
        "mean_error": 0.05,
    }
    
    result = constraints.evaluate(metrics)
    
    assert result["all_satisfied"]
    assert result["combined_score"] > 0
    print(f"✓ ConstraintSet: combined_score={result['combined_score']:.3f}")
    
    # Test prompt generation
    prompt = constraints.generate_prompt()
    assert "matrix multiplications" in prompt.lower()
    print(f"✓ Prompt generation includes matmul constraint")


if __name__ == "__main__":
    print("Testing MatMulCount constraint...\n")
    
    test_matmul_count_constraint()
    test_operation_counter()
    test_constraint_with_counter()
    test_ast_counting()
    test_constraint_formulas()
    test_constraint_set_with_matmul()
    
    print("\n✅ All tests passed!")

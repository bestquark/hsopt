"""
Evaluation module for quantum simulation optimization.

Provides composable constraints for OpenEvolve evaluators.

Example:
    import trovolut as trv
    
    # Create constraints
    constraints = trv.eval.ConstraintSet([
        trv.eval.Time(max_seconds=30, hard=True),
        trv.eval.Accuracy(weight=1.0),
    ])
    
    # Evaluate metrics
    result = constraints.evaluate({"time_seconds": 10.5, "mean_error": 0.01})
    
    # Generate prompts for OpenEvolve
    prompt = constraints.generate_prompt()
    
    # View formulas
    constraints.explain()
    
    # Use with filemaker to generate evaluator files
    evaluator = trv.filemaker.Workflow([
        get_hamiltonian,
        evaluate,
    ], out_path="evaluator.py").build()
"""

# Constraints - new simplified names
from trovolut.eval.constraints import (
    Constraint,
    ConstraintSet,
    ConstraintResult,
    # Core constraints (simplified names)
    Time,
    Error,
    Memory,
    GPUTime,
    Custom,
    MatMulCount,
    # Legate GPU profiling constraints
    LegateGPUSeconds,
    LegateFramebuffer,
    LegateMakespan,
    LegateConcurrency,
    # Backward compatibility aliases
    TimeConstraint,
    ErrorConstraint,
    MemoryConstraint,
    GPUTimeConstraint,
    CustomConstraint,
    MatMulCountConstraint,
    LegateGPUSecondsConstraint,
    LegateFramebufferConstraint,
    LegateMakespanConstraint,
    LegateConcurrencyConstraint,
)

# Legacy (kept for code_evolution/)
from trovolut.eval.gpu_evals import GPUSpaceTimeEval

__all__ = [
    # Base classes
    "Constraint",
    "ConstraintSet",
    "ConstraintResult",
    # Core constraints (new simplified names)
    "Time",
    "Error",
    "Memory",
    "GPUTime",
    "Custom",
    "MatMulCount",
    # Legate GPU constraints (new simplified names)
    "LegateGPUSeconds",
    "LegateFramebuffer",
    "LegateMakespan",
    "LegateConcurrency",
    # Backward compatibility (old names with 'Constraint' suffix)
    "TimeConstraint",
    "ErrorConstraint",
    "MemoryConstraint",
    "GPUTimeConstraint",
    "CustomConstraint",
    "MatMulCountConstraint",
    "LegateGPUSecondsConstraint",
    "LegateFramebufferConstraint",
    "LegateMakespanConstraint",
    "LegateConcurrencyConstraint",
    # Legacy
    "GPUSpaceTimeEval",
]

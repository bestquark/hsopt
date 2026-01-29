import time
import importlib.util
from typing import List, Tuple, Dict, Any, Optional, Sequence, Union
import numpy as np
import scipy.linalg
import trovolut as trv
from trovolut.qm.hamiltonians import *  # noqa: F401,F403

# WARNING: Could not extract source for FLOPConstraint: type object 'FLOPConstraint' has no attribute '__code__'

def get_hamiltonian():
    """Return Hamiltonian terms (cached, sparse matrices)."""
    return [t["coef"] * t["matrix"] for t in H_qubit.terms()]

def setup():
    """Return inputs for dynamics: (hams, t, psi0)."""
    hams = get_hamiltonian()

    # Random initial state
    psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi0 /= np.linalg.norm(psi0)

    return hams, 1.0, psi0

def exact_evolution(hams, t, psi0):
    """Exact time evolution for reference."""
    H_total = sum(hams)
    if sp.issparse(H_total):
        # Use sparse exponential action
        return sp.linalg.expm_multiply(-1j * t * H_total, psi0)
    else:
        return scipy.linalg.expm(-1j * t * H_total) @ psi0

constraints = trv.eval.ConstraintSet([
    FLOPConstraint(target_gflops=50, hard=False, weight=2.0, hard_penalty=-100.0),
    trv.eval.Error(max_error=0.1, weight=1.0, hard=True, hard_penalty=-100.0, score_fn="inverse"),
])


def evaluate(program_path: str) -> dict:
    """Auto-generated evaluator function."""
    # Load candidate module
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the function to evaluate
    candidate_fn = getattr(module, "run_dynamics")

    metrics = {}

    # Setup inputs
    inputs = setup()
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # Run candidate
    result = candidate_fn(*inputs)

    # Compute error
    reference = exact_evolution(*inputs)
    metrics["mean_error"] = 1.0 - abs(np.vdot(reference, result)) ** 2

    return constraints.evaluate(metrics)
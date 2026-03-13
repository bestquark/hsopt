"""Autoresearch module for universal quantum simulation optimization.

Provides an autoresearch loop that can optimize ANY quantum simulation code
(CUDA-Q, PyTorch tensor networks, custom Python) autonomously using
OpenEvolve or an LLM agent.

Quick Start:
    ```python
    import trovolut as trv

    problem = trv.autoresearch.Problem(
        target="my_simulation.py",
        constraints=trv.constraints.Group([
            trv.constraints.Time(60, hard=True),
            trv.constraints.Error(0.01, hard=True),
        ]),
        budget_seconds=300,
    )
    results = trv.autoresearch.run(problem)
    ```
"""

from trovolut.autoresearch.problem import Problem
from trovolut.autoresearch.history import History, ExperimentRecord
from trovolut.autoresearch.loop import Loop, run
from trovolut.autoresearch.resume import (
    run_evolution_with_resume,
    find_latest_checkpoint,
    save_best_policy,
)
from trovolut.autoresearch.generic_evaluator import generate_evaluator

__all__ = [
    "Problem",
    "History",
    "ExperimentRecord",
    "Loop",
    "run",
    "run_evolution_with_resume",
    "find_latest_checkpoint",
    "save_best_policy",
    "generate_evaluator",
]

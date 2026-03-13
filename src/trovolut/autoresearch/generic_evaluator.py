"""Evaluator generation for generic (non-trovolut) scripts.

Like the existing ``trovolut.filemaker.evaluator.Evaluator`` but generates
code that uses ``GenericRunner`` or ``PyTorchRunner`` to evaluate arbitrary
Python scripts by parsing metrics from their stdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from trovolut.autoresearch.problem import Problem


def generate_evaluator(problem: Problem, out_path: Optional[str] = None) -> str:
    """Generate an OpenEvolve-compatible evaluator file for a Problem.

    The generated evaluator:
    1. Instantiates the appropriate runner (GenericRunner or PyTorchRunner)
    2. Runs the candidate script as a subprocess
    3. Parses metrics from stdout
    4. Scores constraints
    5. Returns an OpenEvolve-compatible dict

    Parameters
    ----------
    problem : Problem
        The optimization problem definition.
    out_path : str, optional
        Output path for the evaluator file. Defaults to
        ``{problem.output_dir}/evaluator.py``.

    Returns
    -------
    str
        Path to the generated evaluator file.
    """
    if out_path is None:
        out_path = str(Path(problem.output_dir) / "evaluator.py")

    runner_type = problem._detect_runner()

    # Determine runner class and import
    if runner_type == "pytorch":
        runner_import = "from trovolut.filemaker.runners import PyTorchRunner"
        runner_init = f'runner = PyTorchRunner(timeout={int(problem.budget_seconds)}, metric_parser="{problem.metric_parser}")'
    else:
        runner_import = "from trovolut.filemaker.runners import GenericRunner"
        runner_init = f'runner = GenericRunner(timeout={int(problem.budget_seconds)}, metric_parser="{problem.metric_parser}")'

    # Generate constraints code
    constraints_code = "constraints = None"
    constraints_imports = ""
    if problem.constraints is not None:
        constraints_code = f"constraints = {repr(problem.constraints)}"
        # Auto-detect needed imports from constraint repr
        constraint_repr = repr(problem.constraints)
        needed_base = []
        for cls_name in ("Group", "Time", "Error", "Custom"):
            if f"{cls_name}(" in constraint_repr:
                needed_base.append(cls_name)
        needed_pytorch = []
        for cls_name in ("GPUMemory", "Throughput", "Fidelity"):
            if f"{cls_name}(" in constraint_repr:
                needed_pytorch.append(cls_name)

        if needed_base:
            constraints_imports += f"from trovolut.core.base_constraints import {', '.join(needed_base)}\n"
        if needed_pytorch:
            constraints_imports += f"from trovolut.backends.pytorch.constraints import {', '.join(needed_pytorch)}\n"

    code = f'''\
"""Auto-generated evaluator for autoresearch loop."""

{runner_import}
{constraints_imports}

{runner_init}
{constraints_code}


def evaluate(program_path):
    """OpenEvolve-compatible evaluator entry point."""
    return runner.evaluate(
        program_path,
        constraints=constraints,
        timeout={int(problem.budget_seconds)},
    )


# Alias for direct calling
evaluator = evaluate

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--program_path", type=str, required=True)
        args = parser.parse_args()
        result = evaluate(args.program_path)
        print(f"combined_score: {{result['combined_score']}}")
'''

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)

    return str(path)


__all__ = [
    "generate_evaluator",
]

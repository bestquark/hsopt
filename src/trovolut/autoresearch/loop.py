"""The autoresearch optimization loop.

Orchestrates iterative optimization of quantum simulation code using
either OpenEvolve (evolutionary search) or an LLM agent loop.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from trovolut.autoresearch.problem import Problem
from trovolut.autoresearch.history import History, ExperimentRecord
from trovolut.autoresearch.generic_evaluator import generate_evaluator


class Loop:
    """The autoresearch optimization loop.

    Manages the iterative process of:
    1. Generating candidate programs (mutations of the target)
    2. Evaluating them via the appropriate runner
    3. Tracking results in History
    4. Selecting the best candidates for the next iteration

    Parameters
    ----------
    problem : Problem
        The optimization problem definition.
    mode : str
        Optimization mode: ``"openevolve"`` or ``"agent"``.
    """

    def __init__(self, problem: Problem, mode: str = "openevolve"):
        self.problem = problem
        self.mode = mode
        self._output_dir = Path(problem.output_dir)
        self._history = History()
        self._setup_done = False

    def setup(self) -> None:
        """Prepare the optimization environment.

        Creates output directory, generates evaluator.py and program.py
        (initial candidate), and program.md (agent instructions).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Copy target as initial program
        program_path = self._output_dir / "program.py"
        shutil.copy2(self.problem.target, program_path)

        # Generate evaluator
        evaluator_path = generate_evaluator(
            self.problem,
            out_path=str(self._output_dir / "evaluator.py"),
        )

        # Generate agent instructions
        program_md_path = self._output_dir / "program.md"
        program_md_path.write_text(self.problem.to_program_md())

        self._setup_done = True

    def run(self, max_iterations: Optional[int] = None) -> History:
        """Run the optimization loop.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations. If None, uses OpenEvolve's default.

        Returns
        -------
        History
            The experiment history with all results.
        """
        if not self._setup_done:
            self.setup()

        if self.mode == "openevolve":
            return self._run_openevolve(max_iterations)
        elif self.mode == "agent":
            return self._run_agent(max_iterations)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'openevolve' or 'agent'.")

    def _run_openevolve(self, max_iterations: Optional[int] = None) -> History:
        """Run optimization via OpenEvolve controller."""
        try:
            from openevolve.main import OpenEvolve as OpenEvolveController
        except ImportError:
            raise ImportError(
                "OpenEvolve is required for 'openevolve' mode. "
                "Install with: pip install openevolve"
            )

        config = self.problem.to_openevolve_config()
        if max_iterations is not None:
            config.max_iterations = max_iterations

        program_path = str(self._output_dir / "program.py")
        evaluator_path = str(self._output_dir / "evaluator.py")

        controller = OpenEvolveController(
            initial_program_path=program_path,
            evaluation_file=evaluator_path,
            config=config,
            output_dir=str(self._output_dir),
        )

        import asyncio
        best_program = asyncio.run(controller.run(iterations=max_iterations))

        # Record the best result
        if best_program is not None:
            code = getattr(best_program, "code", "")
            metrics = getattr(best_program, "metrics", {})
            self._history.record(ExperimentRecord(
                iteration=len(self._history),
                combined_score=metrics.get("combined_score", 0.0),
                metrics=metrics,
                all_satisfied=metrics.get("all_satisfied", False),
                code_hash=hashlib.md5(code.encode()).hexdigest() if code else "",
            ))

        self._history.save(self._output_dir / "history.json")
        return self._history

    def _run_agent(self, max_iterations: Optional[int] = None) -> History:
        """Run optimization via iterative LLM agent loop.

        Uses the generated program.md as instructions for the LLM to
        iteratively improve the target script.
        """
        from trovolut.filemaker.runners import runner_for_backend

        runner_type = self.problem._detect_runner()
        runner = runner_for_backend(runner_type)

        max_iter = max_iterations or 10
        program_path = self._output_dir / "program.py"

        for i in range(max_iter):
            # Evaluate current candidate
            result = runner.evaluate(
                str(program_path),
                constraints=self.problem.constraints,
                timeout=int(self.problem.budget_seconds),
            )

            code = program_path.read_text()
            self._history.record(ExperimentRecord(
                iteration=i,
                combined_score=result.get("combined_score", -100.0),
                metrics=result,
                all_satisfied=result.get("all_satisfied", False),
                code_hash=hashlib.md5(code.encode()).hexdigest(),
            ))

            # Check if we've found a satisfactory solution
            if result.get("all_satisfied", False):
                best = self._history.best()
                if best and best.combined_score > 0.95:
                    break

        self._history.save(self._output_dir / "history.json")
        return self._history

    @property
    def history(self) -> History:
        """Access the current history."""
        return self._history


def run(problem: Problem, mode: str = "openevolve", **kwargs) -> History:
    """Convenience function to run an autoresearch optimization.

    Parameters
    ----------
    problem : Problem
        The optimization problem.
    mode : str
        ``"openevolve"`` or ``"agent"``.
    **kwargs
        Additional arguments passed to ``Loop.run()``.

    Returns
    -------
    History
        The experiment history.

    Example
    -------
    >>> import trovolut as trv
    >>> problem = trv.autoresearch.Problem(
    ...     target="my_simulation.py",
    ...     constraints=trv.constraints.Group([
    ...         trv.constraints.Time(60, hard=True),
    ...         trv.constraints.Error(0.01, hard=True),
    ...     ]),
    ... )
    >>> results = trv.autoresearch.run(problem)
    """
    loop = Loop(problem, mode=mode)
    return loop.run(**kwargs)


__all__ = [
    "Loop",
    "run",
]

"""
Evaluator generator for OpenEvolve.

Creates standalone evaluator files that measure program performance using
trovolut's constraint and runner system.

The generated file is minimal: it serializes user-provided functions
(``test_generator``, ``exact_fn``, and their dependencies), instantiates a
backend-specific :class:`~trovolut.filemaker.runners.BaseRunner`, and
delegates all execution logic to the runner.

Example (user code)::

    evaluator = Evaluator(
        test_generator=get_test_cases,
        workflow=workflow,
        constraints=group_constr,
        exact_fn=exact_evolution,
        backend="cpu",
        out_path="opt/evaluator.py",
    )
    evaluator.build()

The generated ``opt/evaluator.py`` looks like::

    from trovolut.filemaker.runners import CPURunner
    ...
    runner = CPURunner()
    constraints = Group([Time(1), Error(0.05, hard=True)])

    def evaluate(program_path):
        return runner.evaluate(program_path, ...)
"""

import ast
import inspect
from pathlib import Path
from typing import Callable, Optional, List, Dict, Set, Any

from trovolut.filemaker.workflow import Workflow


# ============================================================
# AST helpers (unchanged from previous version)
# ============================================================

def _extract_function_calls(func) -> Set[str]:
    """Extract all function calls from a function's source code using AST."""
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)

        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        calls.add(node.func.value.id)
        return calls
    except Exception:
        return set()


def _get_function_dependencies(func, caller_globals: Dict) -> Dict[str, Callable]:
    """Recursively find all function dependencies from a function's globals."""
    dependencies: Dict[str, Callable] = {}
    to_process = [func]
    processed: Set[str] = set()

    while to_process:
        current_func = to_process.pop(0)
        func_name = getattr(current_func, "__name__", None)
        if func_name in processed:
            continue
        processed.add(func_name)

        calls = _extract_function_calls(current_func)
        for call_name in calls:
            if call_name in caller_globals and callable(caller_globals[call_name]):
                potential_dep = caller_globals[call_name]
                if (
                    hasattr(potential_dep, "__code__")
                    and call_name not in dependencies
                    and call_name != func_name
                ):
                    dependencies[call_name] = potential_dep
                    to_process.append(potential_dep)
    return dependencies


# ============================================================
# Evaluator
# ============================================================

class Evaluator:
    """Generates OpenEvolve-compatible evaluator files.

    The generated file delegates all execution logic to a
    :class:`~trovolut.filemaker.runners.BaseRunner`.

    Parameters
    ----------
    test_generator : callable
        ``() -> (hams, psi0)`` or ``() -> (hams, psi0, t)``.
    workflow : Workflow
        Workflow that defines the initial program structure.
    constraints : Constraint, Group, or list, optional
        Constraints to evaluate.
    exact_fn : callable, optional
        ``(hams, t, psi0) -> psi_exact``.
    backend : str, optional
        ``'cpu'``, ``'gpu'``, ``'gpu-mp-emulator'``, etc.
        If *None*, auto-detected from the active backend.
    runner : BaseRunner, optional
        Explicit runner instance.  If *None*, auto-created from *backend*.
    out_path : str
        Output path for the generated evaluator file.
    n_trials : int
        Number of trials.
    timeout : int
        Timeout in seconds.
    run_fn_name : str
        Name of the function to call in candidate modules.
    gpus, cpus, fbmem_mb : int
        GPU runner parameters (only used when ``backend='gpu'``).
    """

    def __init__(
        self,
        test_generator: Callable,
        workflow: Workflow,
        constraints: Optional[Any] = None,
        exact_fn: Optional[Callable] = None,
        backend: Optional[str] = None,
        runner: Optional[Any] = None,
        out_path: str = "evaluator.py",
        n_trials: int = 10,
        timeout: int = 30,
        run_fn_name: str = "run_dynamics",
        # GPU-specific
        gpus: int = 1,
        cpus: int = 4,
        fbmem_mb: int = 256,
        # Legacy (kept for backward compat, unused)
        metric_function: Optional[Callable] = None,
    ):
        self.test_generator = test_generator
        self.workflow = workflow
        self.constraints = constraints
        self.exact_fn = exact_fn
        self.out_path = Path(out_path)
        self.n_trials = n_trials
        self.timeout = timeout
        self.run_fn_name = run_fn_name
        self.metric_function = metric_function

        # GPU params
        self.gpus = gpus
        self.cpus = cpus
        self.fbmem_mb = fbmem_mb

        # Backend detection
        if backend is None:
            try:
                import trovolut as trv
                backend = trv.get_active_backend() or "cpu"
            except Exception:
                backend = "cpu"
        self.backend = backend

        # Runner
        if runner is not None:
            self.runner = runner
        else:
            from trovolut.filemaker.runners import runner_for_backend
            gpu_kwargs = {}
            if self.backend == "gpu":
                gpu_kwargs = dict(gpus=gpus, cpus=cpus, fbmem_mb=fbmem_mb)
            self.runner = runner_for_backend(self.backend, **gpu_kwargs)

        # Collect function dependencies
        self._function_dependencies = self._collect_function_dependencies()

    # -- dependency analysis -----------------------------------------------

    def _collect_function_dependencies(self) -> Dict[str, Callable]:
        """Collect all helper functions needed by test_generator and exact_fn."""
        dependencies: Dict[str, Callable] = {}
        for func in (self.test_generator, self.exact_fn, self.metric_function):
            if func is not None:
                deps = _get_function_dependencies(func, func.__globals__)
                dependencies.update(deps)
        return dependencies

    def _generate_function_source(self, func: Callable) -> str:
        """Extract dedented source for *func*."""
        try:
            source = inspect.getsource(func)
            lines = source.split("\n")
            if lines:
                min_indent = min(
                    (len(line) - len(line.lstrip()) for line in lines if line.strip()),
                    default=0,
                )
                return "\n".join(
                    line[min_indent:] if len(line) > min_indent else line
                    for line in lines
                )
            return source
        except Exception as e:
            return f"# Could not extract source for {func.__name__}: {e}\n"

    # -- code generation ---------------------------------------------------

    def _generate_imports(self) -> str:
        """Generate import block for the evaluator file."""
        lines = [
            "import numpy as np",
            "import scipy",
            "import trovolut as trv",
        ]

        # Runner import
        runner_cls = type(self.runner).__name__
        lines.append(f"from trovolut.filemaker.runners import {runner_cls}")

        # Constraint imports
        if self.constraints is not None:
            constraint_repr = repr(self.constraints)

            needed_base: Set[str] = set()
            needed_mp: Set[str] = set()
            needed_gpu: Set[str] = set()

            # base_constraints classes
            for cls_name in ("Group", "Time", "Error", "Custom"):
                if f"{cls_name}(" in constraint_repr:
                    needed_base.add(cls_name)

            # gpu_mp_emulator constraints
            for cls_name in ("MemoryCost", "FlopsCost", "SpaceTimeCost", "MatMulCount"):
                if f"{cls_name}(" in constraint_repr:
                    needed_mp.add(cls_name)

            # gpu (Legate) constraints
            for cls_name in (
                "LegateGPUSeconds",
                "LegateFramebuffer",
                "LegateMakespan",
                "LegateConcurrency",
            ):
                if f"{cls_name}(" in constraint_repr:
                    needed_gpu.add(cls_name)

            if needed_base:
                lines.append(
                    f"from trovolut.core.base_constraints import {', '.join(sorted(needed_base))}"
                )
            if needed_mp:
                lines.append(
                    f"from trovolut.backends.gpu_mp_emulator.constraints import {', '.join(sorted(needed_mp))}"
                )
            if needed_gpu:
                lines.append(
                    f"from trovolut.backends.gpu.constraints import {', '.join(sorted(needed_gpu))}"
                )

        return "\n".join(lines) + "\n"

    def _generate_runner_instantiation(self) -> str:
        """Generate the ``runner = ...`` line."""
        runner_cls = type(self.runner).__name__
        if runner_cls == "LegateRunner":
            return (
                f"runner = {runner_cls}("
                f"gpus={self.gpus}, cpus={self.cpus}, fbmem_mb={self.fbmem_mb}"
                f")\n"
            )
        return f"runner = {runner_cls}()\n"

    def _generate_constraints_definition(self) -> str:
        """Generate ``constraints = ...`` line."""
        if self.constraints is not None:
            return f"constraints = {repr(self.constraints)}\n"
        return "constraints = None\n"

    def _generate_evaluate_function(self) -> str:
        """Generate the ``evaluate(program_path)`` entry point."""
        test_gen_name = (
            self.test_generator.__name__
            if self.test_generator
            else "None"
        )
        exact_fn_name = (
            self.exact_fn.__name__
            if self.exact_fn
            else "None"
        )

        return f"""
def evaluate(program_path):
    \"\"\"OpenEvolve-compatible evaluator entry point.\"\"\"
    return runner.evaluate(
        program_path,
        test_generator={test_gen_name},
        exact_fn={exact_fn_name},
        constraints=constraints,
        n_trials={self.n_trials},
        timeout={self.timeout},
        run_fn_name="{self.run_fn_name}",
    )

# Alias for direct calling
evaluator = evaluate
"""

    def _generate_main_block(self) -> str:
        """Generate the CLI ``__main__`` block."""
        return '''
if __name__ == "__main__":
    import sys

    try:
        get_ipython()
        _in_notebook = True
    except NameError:
        _in_notebook = False

    if "ipykernel_launcher" in sys.argv[0]:
        _in_notebook = True

    if not _in_notebook and len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--program_path", type=str, required=True)
        args = parser.parse_args()
        result = evaluate(args.program_path)
        print(f"combined_score: {result[\'combined_score\']}")
        print(f"std_score: {result[\'std_score\']}")
'''

    # -- public API --------------------------------------------------------

    def build(self) -> str:
        """Generate the evaluator file and return its path.

        The generated file contains:

        1. Imports (auto-detected)
        2. User helper functions (auto-detected dependencies)
        3. ``test_generator`` and ``exact_fn`` source
        4. Runner instantiation
        5. Constraint definition
        6. ``evaluate(program_path)`` entry point
        7. CLI ``__main__`` block
        """
        parts: List[str] = []

        # 1. Imports
        parts.append(self._generate_imports())

        # 2. Helper function dependencies
        if self._function_dependencies:
            parts.append("\n# Helper functions (auto-detected dependencies)")
            for _name, func in self._function_dependencies.items():
                parts.append(self._generate_function_source(func))
                parts.append("")

        # 3. Test generator
        if self.test_generator:
            parts.append(self._generate_function_source(self.test_generator))
            parts.append("")

        # 4. Exact evolution function
        if self.exact_fn:
            parts.append(self._generate_function_source(self.exact_fn))
            parts.append("")

        # 5. Runner + constraints
        parts.append("# Runner and constraints")
        parts.append(self._generate_runner_instantiation())
        parts.append(self._generate_constraints_definition())

        # 6. evaluate()
        parts.append(self._generate_evaluate_function())

        # 7. CLI main
        parts.append(self._generate_main_block())

        # Write
        content = "\n".join(parts)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(content, encoding="utf-8")

        return str(self.out_path)

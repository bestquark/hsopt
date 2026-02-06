"""
Evaluator generator for OpenEvolve.

Creates evaluator files that measure program performance using trovolut's metric system.
"""

import ast
import inspect
from pathlib import Path
from typing import Callable, Optional, List, Dict, Set, Any

from trovolut.filemaker.workflow import Workflow


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
                    # For calls like obj.method(), we track the base name
                    if isinstance(node.func.value, ast.Name):
                        calls.add(node.func.value.id)
        
        return calls
    except Exception:
        return set()


def _get_function_dependencies(func, caller_globals: Dict) -> Dict[str, Callable]:
    """
    Recursively find all function dependencies from a function's globals.
    
    Args:
        func: The function to analyze
        caller_globals: The globals dict where the function was defined
        
    Returns:
        Dict mapping function names to their callable objects
    """
    dependencies = {}
    to_process = [func]
    processed = set()
    
    while to_process:
        current_func = to_process.pop(0)
        
        # Skip if already processed
        func_name = getattr(current_func, '__name__', None)
        if func_name in processed:
            continue
        processed.add(func_name)
        
        # Extract function calls from this function
        calls = _extract_function_calls(current_func)
        
        # Find callables in globals that match these calls
        for call_name in calls:
            if call_name in caller_globals and callable(caller_globals[call_name]):
                potential_dep = caller_globals[call_name]
                
                # Only include user-defined functions (not builtins or imports)
                if (hasattr(potential_dep, '__code__') and 
                    call_name not in dependencies and
                    call_name != func_name):
                    
                    dependencies[call_name] = potential_dep
                    to_process.append(potential_dep)
    
    return dependencies


class Evaluator:
    """
    Generates OpenEvolve-compatible evaluator files.
    
    The evaluator measures program performance using trovolut's automatic metrics system.
    """
    
    def __init__(
        self,
        test_generator: Callable,
        workflow: Workflow,
        constraints: Optional[Any] = None,
        exact_fn: Optional[Callable] = None,
        backend: Optional[str] = None,
        out_path: str = "evaluator.py",
        n_trials: int = 10,
        timeout: int = 30,
        metric_function: Optional[Callable] = None,
    ):
        """
        Initialize the Evaluator generator.
        
        Args:
            test_generator: Function that returns test cases (hams, psi0) or (hams, psi0, t)
            workflow: Workflow object that defines the program structure
            constraints: Optional constraint or Group of constraints
            exact_fn: Optional function for computing exact solution
            backend: Backend to use ('cpu', 'gpu', 'qpu'). If None, auto-detects.
            out_path: Path where evaluator file will be generated
            n_trials: Number of trials to run
            timeout: Timeout in seconds for evaluation
            metric_function: Optional custom metric function (deprecated, use constraints instead)
        """
        self.test_generator = test_generator
        self.workflow = workflow
        self.constraints = constraints
        self.exact_fn = exact_fn
        self.out_path = Path(out_path)
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric_function = metric_function
        
        # Auto-detect backend if not specified
        if backend is None:
            try:
                import trovolut as trv
                backend = trv.get_active_backend()
                if backend is None:
                    backend = 'cpu'  # Default fallback
            except Exception:
                backend = 'cpu'
        
        self.backend = backend
        
        # Collect function dependencies
        self._function_dependencies = self._collect_function_dependencies()
    
    def _collect_function_dependencies(self) -> Dict[str, Callable]:
        """Collect all helper functions needed by test_generator and exact_fn."""
        dependencies = {}
        
        # Get globals from test_generator
        if self.test_generator:
            test_gen_globals = self.test_generator.__globals__
            deps = _get_function_dependencies(self.test_generator, test_gen_globals)
            dependencies.update(deps)
        
        # Get globals from exact_fn
        if self.exact_fn:
            exact_fn_globals = self.exact_fn.__globals__
            deps = _get_function_dependencies(self.exact_fn, exact_fn_globals)
            dependencies.update(deps)
        
        # Get globals from metric_function if provided
        if self.metric_function:
            metric_globals = self.metric_function.__globals__
            deps = _get_function_dependencies(self.metric_function, metric_globals)
            dependencies.update(deps)
        
        return dependencies
    
    def _generate_function_source(self, func: Callable) -> str:
        """Generate source code for a function."""
        try:
            source = inspect.getsource(func)
            # Remove any leading indentation
            lines = source.split('\n')
            if lines:
                # Find minimum indentation (excluding empty lines)
                min_indent = min(
                    (len(line) - len(line.lstrip()) 
                     for line in lines if line.strip()),
                    default=0
                )
                # Remove that indentation from all lines
                dedented = '\n'.join(
                    line[min_indent:] if len(line) > min_indent else line
                    for line in lines
                )
                return dedented
            return source
        except Exception as e:
            return f"# Could not extract source for {func.__name__}: {e}\n"
    
    def _generate_imports(self) -> str:
        """Generate import statements."""
        imports = """import concurrent.futures
import numpy as np
import scipy
import trovolut as trv
"""
        
        # Add constraint imports based on what's actually used
        if self.constraints is not None:
            # Extract constraint class names from the repr
            constraint_repr = repr(self.constraints)
            
            # Detect which constraint classes are used
            needed_classes = set()
            if 'Group(' in constraint_repr:
                needed_classes.add('Group')
            if 'Time(' in constraint_repr:
                needed_classes.add('Time')
            if 'Error(' in constraint_repr:
                needed_classes.add('Error')
            if 'Memory(' in constraint_repr:
                needed_classes.add('Memory')
            if 'MatMulCount(' in constraint_repr:
                needed_classes.add('MatMulCount')
            if 'EmulatorMemory(' in constraint_repr:
                needed_classes.add('EmulatorMemory')
            
            # Only import what's needed (base_constraints vs gpu_mp_emulator)
            base_only = needed_classes - {'EmulatorMemory'}
            if base_only:
                imports += "\n# Import constraint classes for constraint definitions\n"
                imports += f"from trovolut.core.base_constraints import {', '.join(sorted(base_only))}\n"
            if 'EmulatorMemory' in needed_classes:
                imports += "from trovolut.backends.gpu_mp_emulator.constraints import EmulatorMemory\n"
        
        return imports
    
    def _generate_default_metric(self) -> str:
        """Generate the compute_accuracy function using trovolut.measure.Metrics."""
        constraints_repr = "None"
        if self.constraints is not None:
            # Import constraints in the generated file
            constraints_repr = repr(self.constraints)
        
        code = f"""def compute_accuracy(program_path):
    \"\"\"Load program and measure metrics using backend metrics class.\"\"\"
    import importlib.util
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Set backend and access the injected measure module
    trv.set_backend('{self.backend}')
    Metrics = trv.measure.Metrics
    
    metrics = Metrics(
        constraints={constraints_repr},
        test_generator=get_test_cases,
        exact_fn=exact_evolution,
        n_trials={self.n_trials}
    )
    
    # Measure using the program's function
    result = metrics(getattr(module, "run_dynamics"))
    
    # Spacetime cost (gpu-mp-emulator): if program exposes evolution, add peak_memory_bytes from CostAnalyzer
    if hasattr(module, 'evolution'):
        try:
            from trovolut.backends.gpu_mp_emulator.measure import cost_from_dynamics_config
            tc = get_test_cases()
            if isinstance(tc, (list, tuple)) and len(tc) and isinstance(tc[0], (list, tuple)):
                hams, psi0, t = tc[0]
            else:
                hams, psi0, t = tc
            cost = cost_from_dynamics_config(hams, module.evolution)
            if cost is not None:
                result['peak_memory_bytes'] = cost.peak_memory_bytes
                result['memory_peak_bytes'] = cost.peak_memory_bytes
        except Exception:
            pass
    
    # Let constraints evaluate the results and add details
    if metrics.constraints:
        constraints = metrics.constraints
        
        # Unwrap list if it contains a single Group
        if isinstance(constraints, list) and len(constraints) == 1:
            constraints = constraints[0]
        
        # Use Group's evaluate method if it's a Group
        if isinstance(constraints, Group):
            eval_result = constraints.evaluate(result)
            result.update(eval_result)
        else:
            # Single constraint or list of constraints
            if not isinstance(constraints, list):
                constraints = [constraints]
            
            all_satisfied = True
            details = {{}}
            for constraint in constraints:
                cr = constraint.evaluate(result)
                details[cr.name] = {{
                    'value': cr.value,
                    'score': cr.score,
                    'satisfied': cr.satisfied
                }}
                if not cr.satisfied:
                    all_satisfied = False
            
            result['all_satisfied'] = all_satisfied
            result['details'] = details
    
    return result
"""
        return code
    
    def _generate_timeout_wrapper(self) -> str:
        """Generate timeout wrapper function."""
        return """def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=30):
    \"\"\"Execute function with timeout.\"\"\"
    if kwargs is None:
        kwargs = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
"""
    
    def _generate_evaluate_function(self) -> str:
        """Generate the main evaluate function."""
        code = f"""def evaluate(program_path):
    \"\"\"OpenEvolve evaluator with timeout and error handling.\"\"\"
    try:
        result = run_with_timeout(
            compute_accuracy,
            args=(program_path,),
            timeout_seconds={self.timeout},
        )
        
        combined = float(result.get("combined_score", -100.0))
        std = float(result.get("std_score", 0.0))
        all_satisfied = result.get("all_satisfied", True)
        details = result.get("details", {{}})
        
        return {{
            "combined_score": combined,
            "std_score": std,
            "all_satisfied": all_satisfied,
            "details": details,
            "time": result.get("time", 0.0),
            "error": result.get("error", 0.0),
            "memory_peak_bytes": result.get("memory_peak_bytes", 0),
        }}
    
    except Exception as e:
        print(f"Evaluation failed: {{e}}")
        return {{
            "combined_score": -100.0,
            "std_score": 0.0,
            "all_satisfied": False,
            "details": {{}},
            "time": 0.0,
            "error": float('inf'),
            "memory_peak_bytes": 0,
        }}

# Alias for direct calling: evaluator(path) same as evaluate(path)
evaluator = evaluate
"""
        return code
    
    def _generate_main_block(self) -> str:
        """Generate the CLI main block."""
        return """
if __name__ == "__main__":
    import sys
    
    # Don't run CLI in Jupyter/IPython or when exec'd
    try:
        # Check if we're in IPython/Jupyter
        get_ipython()
        _in_notebook = True
    except NameError:
        _in_notebook = False
    
    # Also check if running via ipykernel
    if 'ipykernel_launcher' in sys.argv[0]:
        _in_notebook = True
    
    # Only run CLI if not in notebook and arguments are provided
    if not _in_notebook and len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--program_path", type=str, required=True)
        args = parser.parse_args()
        result = evaluate(args.program_path)
        print(f"combined_score: {result['combined_score']}")
        print(f"std_score: {result['std_score']}")
"""
    
    def build(self) -> str:
        """
        Generate the evaluator file.
        
        Returns:
            Path to the generated evaluator file
        """
        parts = []
        
        # 1. Imports
        parts.append(self._generate_imports())
        
        # 2. Helper function dependencies (auto-detected)
        if self._function_dependencies:
            parts.append("\n# Helper functions (auto-detected dependencies)")
            for func_name, func in self._function_dependencies.items():
                parts.append(self._generate_function_source(func))
                parts.append("")  # Blank line between functions
        
        # 3. Test generator
        if self.test_generator:
            parts.append(self._generate_function_source(self.test_generator))
            parts.append("")
        
        # 4. Exact evolution function
        if self.exact_fn:
            parts.append(self._generate_function_source(self.exact_fn))
            parts.append("")
        
        # 5. Metric computation
        parts.append(self._generate_default_metric())
        parts.append("")
        
        # 6. Timeout wrapper
        parts.append(self._generate_timeout_wrapper())
        parts.append("")
        
        # 7. Main evaluate function
        parts.append(self._generate_evaluate_function())
        
        # 8. CLI main block
        parts.append(self._generate_main_block())
        
        # Write to file
        content = "\n".join(parts)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(content, encoding="utf-8")
        
        return str(self.out_path)

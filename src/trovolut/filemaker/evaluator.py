"""
Evaluator generator for OpenEvolve.

Creates standalone evaluator.py files that test evolved dynamics programs.
"""

import inspect
import textwrap
from pathlib import Path

from .workflow import _extract_imports_from_function, EVO_START, EVO_END


class Evaluator:
    """Generate evaluator files for OpenEvolve that test dynamics programs.
    
    The Evaluator class creates standalone evaluator.py files that:
    - Load test cases (Hamiltonian, state pairs)
    - Import and run the generated dynamics program
    - Compare results against ground truth
    - Return scores for OpenEvolve
    
    Args:
        test_generator: Function that returns (hamiltonian, state) pairs for testing.
                       Can return a single pair or a list of pairs.
        workflow: Workflow object containing the dynamics function to test
        metric_function: Function(program_path) -> dict with 'combined_score' and 'std_score'
        out_path: Path for generated evaluator.py file
        timeout: Timeout in seconds for evaluation (default: 30)
        n_trials: Number of test trials (default: 10)
    
    Example:
        ```python
        import trovolut as trv
        
        # Define test case generator
        def get_test_cases():
            H = trv.qm.models.hubbard_1d(Lx=4, t_hop=1.0, U_onsite=4.0)
            hams = [t["coef"] * t["matrix"] for t in H.terms()]
            psi0 = np.zeros(2**8, dtype=complex)
            psi0[0] = 1.0
            return hams, psi0
        
        # Define dynamics
        def run_dynamics(Hs, t, psi0):
            return trv.dynamics.product_formula(Hs, t, r=10) @ psi0
        
        # Create workflow
        workflow = trv.filemaker.Workflow(
            [trv.filemaker.EVO_START, run_dynamics, trv.filemaker.EVO_END],
            out_path="opt/program.py"
        )
        
        # Create evaluator
        evaluator = trv.filemaker.Evaluator(
            test_generator=get_test_cases,
            workflow=workflow,
            out_path="opt/evaluator.py"
        )
        
        # Build both files
        workflow.build()
        evaluator.build()
        ```
    """
    
    def __init__(
        self,
        test_generator,
        workflow,
        metric_function=None,
        out_path="evaluator.py",
        timeout=30,
        n_trials=10
    ):
        self.test_generator = test_generator
        self.workflow = workflow
        self.metric_function = metric_function
        self.out_path = Path(out_path)
        self.timeout = timeout
        self.n_trials = n_trials
        
        # Extract dynamics function name from workflow
        self._dynamics_func_name = self._extract_dynamics_function_name()
    
    def _extract_dynamics_function_name(self):
        """Extract the name of the dynamics function from workflow parts."""
        for part in self.workflow.parts:
            if callable(part) and part not in [EVO_START, EVO_END]:
                return part.__name__
        raise ValueError("Could not find dynamics function in workflow.parts")
    
    def build(self):
        """Generate the evaluator.py file."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.out_path.open("w", encoding="utf-8") as f:
            # Write imports
            f.write(self._generate_imports())
            f.write("\n\n")
            
            # Write test generator function
            f.write(self._generate_test_generator())
            f.write("\n\n")
            
            # Write metric function (if custom) or default
            if self.metric_function is not None:
                f.write(self._generate_custom_metric())
            else:
                f.write(self._generate_default_metric())
            f.write("\n\n")
            
            # Write timeout wrapper
            f.write(self._generate_timeout_wrapper())
            f.write("\n\n")
            
            # Write main evaluate function
            f.write(self._generate_evaluate_function())
            f.write("\n\n")
            
            # Write CLI main block
            f.write(self._generate_main_block())
        
        return self.out_path
    
    def _generate_imports(self):
        """Generate import statements for evaluator."""
        imports = [
            "import concurrent.futures",
            "import numpy as np",
            "import scipy",
            "import trovolut as trv",
        ]
        
        # Extract imports from test generator
        detected = _extract_imports_from_function(self.test_generator)
        imports.extend(sorted(detected))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)
        
        return "\n".join(unique_imports)
    
    def _generate_test_generator(self):
        """Generate the test case generator function."""
        src = inspect.getsource(self.test_generator)
        src = textwrap.dedent(src).rstrip()
        return src
    
    def _generate_custom_metric(self):
        """Generate custom metric function."""
        src = inspect.getsource(self.metric_function)
        src = textwrap.dedent(src).rstrip()
        return src
    
    def _generate_default_metric(self):
        """Generate default metrics class (inherits from BaseMetrics)."""
        func_name = self._dynamics_func_name
        test_gen = self.test_generator.__name__
        n_trials = self.n_trials
        
        return f'''class ProgramMetrics(trv.core.BaseMetrics):
    """Metrics for measuring program performance against exact evolution."""
    
    def measure(self, dynamics_func, test_cases=None, **kwargs):
        """Measure accuracy and performance of dynamics function."""
        # Get test cases
        if test_cases is None:
            test_result = {test_gen}()
            if isinstance(test_result, tuple) and len(test_result) == 2:
                hams, psi0 = test_result
                test_cases = [(hams, psi0, 1.0) for _ in range({n_trials})]
            else:
                test_cases = test_result
        
        errors = []
        times = []
        
        for test_case in test_cases:
            if len(test_case) == 3:
                hams, psi0, t = test_case
            else:
                hams, psi0 = test_case
                t = 1.0
            
            # Measure time
            import time as time_module
            start = time_module.perf_counter()
            psi_program = dynamics_func(hams, t, psi0)
            elapsed = time_module.perf_counter() - start
            times.append(elapsed)
            
            # Compute exact evolution
            H_full = sum(hams) if isinstance(hams, list) else hams
            U_exact = scipy.linalg.expm(-1j * H_full * t)
            psi_exact = U_exact @ psi0
            
            # Compute error
            error = np.linalg.norm(psi_program - psi_exact)
            errors.append(error)
        
        # Aggregate measurements
        mean_err = np.mean(errors)
        acc = 1.0 / (1.0 + mean_err)
        var_err = np.var(errors) / len(errors)
        std_acc = np.sqrt(var_err) / (1.0 + mean_err)**2
        
        return {{
            "combined_score": acc,
            "std_score": std_acc,
            "error": mean_err,
            "time": np.mean(times)
        }}

def compute_accuracy(program_path):
    """Load program and measure metrics."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Create metrics instance
    metrics = ProgramMetrics()
    
    # Measure using the program's function
    result = metrics(getattr(module, "{func_name}"))
    return result'''
    
    def _generate_timeout_wrapper(self):
        """Generate timeout wrapper function."""
        timeout = self.timeout
        return f'''def run_with_timeout(func, args=(), kwargs=None, timeout_seconds={timeout}):
    """Execute function with timeout."""
    if kwargs is None:
        kwargs = {{}}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {{timeout_seconds}} seconds")'''
    
    def _generate_evaluate_function(self):
        """Generate main evaluate function for OpenEvolve."""
        metric_name = self.metric_function.__name__ if self.metric_function else "compute_accuracy"
        timeout = self.timeout
        
        return f'''def evaluate(program_path):
    """OpenEvolve evaluator with timeout and error handling."""
    try:
        result = run_with_timeout(
            {metric_name},
            args=(program_path,),
            timeout_seconds={timeout},
        )
        
        combined = float(result.get("combined_score", -100.0))
        std = float(result.get("std_score", 0.0))
        
        return {{
            "combined_score": combined,
            "std_score": std,
        }}
    
    except Exception as e:
        print(f"Evaluation failed: {{e}}")
        return {{
            "combined_score": -100.0,
            "std_score": 0.0,
        }}'''
    
    def _generate_main_block(self):
        """Generate CLI main block."""
        return '''if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, required=True)
    args = parser.parse_args()
    result = evaluate(args.program_path)
    print(f"combined_score: {result['combined_score']}")
    print(f"std_score: {result['std_score']}")'''

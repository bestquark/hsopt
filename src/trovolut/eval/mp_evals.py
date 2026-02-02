from dataclasses import dataclass, field
from typing import Callable
from enum import Enum
import numpy as np

# ============================================================
# 1. Cost data structures
# ============================================================

@dataclass
class TermCost:
    """Cost breakdown for a single ExpTerm."""
    name: str
    
    # Memory (bytes)
    generator_memory: int = 0
    auxiliary_memory: int = 0      # Krylov basis, taylor terms, etc.
    
    # Depth (operations)
    matvec_count: int = 0          # matrix-vector products
    matmat_count: int = 0          # matrix-matrix products (for expm)
    scalar_ops: int = 0            # scalar multiplications/additions
    
    # Effective FLOPs (accounting for precision)
    flops: int = 0
    
    # Precision info
    compute_bits: int = 64
    
    @property
    def total_memory(self) -> int:
        return self.generator_memory + self.auxiliary_memory
    
    @property
    def depth(self) -> int:
        """Sequential operation count (matvecs dominate)."""
        return self.matvec_count + self.matmat_count * 10  # matmat ≈ 10x matvec cost


@dataclass
class StepCost:
    """Cost for one complete Trotter step."""
    term_applications: list[tuple[str, float]]  # (term_name, scaled_dt)
    total_matvecs: int = 0
    total_depth: int = 0
    

@dataclass
class SpacetimeCost:
    """Complete spacetime cost analysis."""
    
    # === SPACE (Memory) ===
    generator_memory_bytes: int = 0      # All H matrices
    state_memory_bytes: int = 0          # |ψ⟩ vector
    auxiliary_memory_bytes: int = 0      # Working memory
    peak_memory_bytes: int = 0           # Max at any point
    
    # Memory by precision tier
    memory_by_precision: dict[str, int] = field(default_factory=dict)
    
    # === TIME (Depth) ===
    total_matvec_products: int = 0       # Total H|ψ⟩ operations
    total_matrix_exponentials: int = 0   # expm() calls
    total_flops: int = 0                 # Floating point operations
    
    # Depth structure
    depth_per_step: int = 0              # Sequential ops in one step
    total_depth: int = 0                 # Total sequential ops
    parallel_depth: int = 0              # If terms could be parallelized
    
    # === BREAKDOWN ===
    cost_by_term: dict[str, TermCost] = field(default_factory=dict)
    n_steps: int = 0
    n_terms: int = 0
    sequence_length: int = 0             # Terms per Trotter step (after formula expansion)
    
    # === DERIVED METRICS ===
    @property
    def space_time_product(self) -> int:
        """Space × Time product (quantum-inspired metric)."""
        return self.peak_memory_bytes * self.total_depth
    
    @property
    def memory_gb(self) -> float:
        return self.peak_memory_bytes / (1024**3)
    
    @property
    def effective_flops_adjusted(self) -> float:
        """FLOPs adjusted for precision (FP16 = 2x throughput on modern GPUs)."""
        # Rough GPU throughput multipliers vs FP32
        multipliers = {64: 0.5, 32: 1.0, 16: 2.0, 8: 4.0}
        adjusted = 0
        for term_name, term_cost in self.cost_by_term.items():
            mult = multipliers.get(term_cost.compute_bits, 1.0)
            adjusted += term_cost.flops / mult
        return adjusted * self.n_steps


# ============================================================
# 2. Cost estimation functions per decomposition method
# ============================================================

class DecompositionCostModel:
    """Estimates cost for different exponentiation methods."""
    
    @staticmethod
    def exact_expm_cost(n: int, precision_bits: int) -> dict:
        """
        Cost of scipy.linalg.expm via Padé approximation.
        Typically uses ~13 matrix multiplications for Padé(13,13).
        """
        bytes_per_element = precision_bits // 8
        
        # expm stores: input matrix, scaled matrix, powers, Padé numerator/denominator
        auxiliary_matrices = 6  # rough estimate
        
        return {
            "matvec_count": 0,
            "matmat_count": 13,  # Padé approximation order
            "auxiliary_memory": auxiliary_matrices * n * n * bytes_per_element * 2,  # complex
            "flops": 13 * (2 * n**3),  # matrix multiply is O(n³)
        }
    
    @staticmethod
    def taylor_cost(n: int, order: int, precision_bits: int) -> dict:
        """Cost of Taylor series: sum_{k=0}^{order} (-iHt)^k / k!"""
        bytes_per_element = precision_bits // 8
        
        return {
            "matvec_count": order,      # One H|ψ⟩ per term
            "matmat_count": 0,
            "auxiliary_memory": 2 * n * bytes_per_element * 2,  # term_k and accumulator
            "flops": order * (2 * n**2),  # matvec is O(n²)
        }
    
    @staticmethod
    def krylov_cost(n: int, krylov_dim: int, precision_bits: int) -> dict:
        """Cost of Krylov subspace method."""
        bytes_per_element = precision_bits // 8
        m = min(krylov_dim, n)
        
        return {
            "matvec_count": m,                    # Arnoldi iterations
            "matmat_count": 1,                    # Small m×m expm
            "auxiliary_memory": (
                n * (m + 1) * bytes_per_element * 2 +  # Krylov basis V
                (m + 1) * m * bytes_per_element * 2     # Hessenberg matrix
            ),
            "flops": m * (2 * n**2) + 13 * (2 * m**3),  # Arnoldi + small expm
        }
    
    @staticmethod 
    def sparse_cost(n: int, nnz: int, order: int, precision_bits: int) -> dict:
        """Cost for sparse matrix methods."""
        bytes_per_element = precision_bits // 8
        
        return {
            "matvec_count": order,
            "matmat_count": 0,
            "auxiliary_memory": 2 * n * bytes_per_element * 2,
            "flops": order * (2 * nnz),  # Sparse matvec is O(nnz)
        }


# ============================================================
# 3. Main Cost Analyzer
# ============================================================

class CostAnalyzer:
    """Analyzes spacetime cost of a TrotterSimulator without running it."""
    
    # Default parameters for methods
    DEFAULT_TAYLOR_ORDER = 20
    DEFAULT_KRYLOV_DIM = 30
    
    def __init__(self):
        self.cost_models = DecompositionCostModel()
    
    def analyze(self, simulator, state_dim: int | None = None) -> SpacetimeCost:
        """
        Analyze the spacetime cost of a TrotterSimulator.
        
        Args:
            simulator: TrotterSimulator instance
            state_dim: Dimension of state vector (inferred from generators if None)
        """
        # Infer dimensions
        n = state_dim or self._infer_dimension(simulator.terms)
        
        cost = SpacetimeCost(
            n_steps=simulator.n_steps,
            n_terms=len(simulator.terms),
        )
        
        # Analyze each term
        for term in simulator.terms:
            term_cost = self._analyze_term(term, n)
            cost.cost_by_term[term.name] = term_cost
            
            # Accumulate generator memory
            cost.generator_memory_bytes += term_cost.generator_memory
            
            # Track by precision
            prec_key = term.precision.compute.value
            cost.memory_by_precision[prec_key] = (
                cost.memory_by_precision.get(prec_key, 0) + 
                term_cost.generator_memory
            )
        
        # Analyze formula expansion (how many term applications per step)
        dummy_dt = 1.0
        sequence = simulator.formula.get_sequence(simulator.terms, dummy_dt)
        cost.sequence_length = len(sequence)
        
        # Calculate per-step costs
        step_matvecs = 0
        step_depth = 0
        step_flops = 0
        step_aux_memory = 0
        
        for term, _ in sequence:
            tc = cost.cost_by_term[term.name]
            step_matvecs += tc.matvec_count
            step_depth += tc.depth
            step_flops += tc.flops
            step_aux_memory = max(step_aux_memory, tc.auxiliary_memory)
        
        cost.depth_per_step = step_depth
        cost.auxiliary_memory_bytes = step_aux_memory
        
        # Total costs across all steps
        cost.total_matvec_products = step_matvecs * simulator.n_steps
        cost.total_depth = step_depth * simulator.n_steps
        cost.total_flops = step_flops * simulator.n_steps
        cost.total_matrix_exponentials = sum(
            1 for term, _ in sequence 
            if cost.cost_by_term[term.name].matmat_count > 0
        ) * simulator.n_steps
        
        # State memory
        state_bytes_per_element = 16  # complex128
        cost.state_memory_bytes = n * state_bytes_per_element
        
        # Peak memory
        cost.peak_memory_bytes = (
            cost.generator_memory_bytes +
            cost.state_memory_bytes +
            cost.auxiliary_memory_bytes
        )
        
        # Parallel depth (if all terms in a step could run in parallel)
        max_term_depth = max(tc.depth for tc in cost.cost_by_term.values())
        cost.parallel_depth = max_term_depth * simulator.n_steps
        
        return cost
    
    def _infer_dimension(self, terms: list) -> int:
        """Infer state dimension from generators."""
        for term in terms:
            gen = term.get_generator(0)
            if isinstance(gen, np.ndarray):
                return gen.shape[0]
        raise ValueError("Cannot infer dimension: no concrete generators found")
    
    def _analyze_term(self, term, n: int) -> TermCost:
        """Analyze cost of a single ExpTerm."""
        prec = term.precision
        bits = prec.compute.bits
        bytes_per_element = bits // 8
        
        # Generator storage
        gen = term.get_generator(0) if not callable(term.generator) else None
        if gen is not None:
            is_sparse = hasattr(gen, 'nnz')
            if is_sparse:
                gen_memory = gen.nnz * bytes_per_element * 2  # complex
            else:
                gen_memory = n * n * bytes_per_element * 2
        else:
            # Callable generator - assume dense
            gen_memory = n * n * bytes_per_element * 2
        
        # Method-specific costs
        method = term.decomposition
        
        if method == "exact":
            method_cost = self.cost_models.exact_expm_cost(n, bits)
        elif method == "taylor":
            order = term.taylor_order or self.DEFAULT_TAYLOR_ORDER
            method_cost = self.cost_models.taylor_cost(n, order, bits)
        elif method == "krylov":
            method_cost = self.cost_models.krylov_cost(n, self.DEFAULT_KRYLOV_DIM, bits)
        elif method == "sparse":
            nnz = gen.nnz if hasattr(gen, 'nnz') else n * n
            order = term.taylor_order or self.DEFAULT_TAYLOR_ORDER
            method_cost = self.cost_models.sparse_cost(n, nnz, order, bits)
        else:
            # Default to exact
            method_cost = self.cost_models.exact_expm_cost(n, bits)
        
        return TermCost(
            name=term.name,
            generator_memory=gen_memory,
            auxiliary_memory=method_cost["auxiliary_memory"],
            matvec_count=method_cost["matvec_count"],
            matmat_count=method_cost["matmat_count"],
            flops=method_cost["flops"],
            compute_bits=bits,
        )
    
    def compare_formulas(self, simulator, formulas: list, state_dim: int = None) -> dict:
        """Compare costs across different Trotter formulas."""
        results = {}
        original_formula = simulator.formula
        
        for formula in formulas:
            simulator.formula = formula
            cost = self.analyze(simulator, state_dim)
            results[formula.__class__.__name__] = cost
        
        simulator.formula = original_formula
        return results
    
    def compare_n_steps(self, simulator, step_counts: list[int], 
                        state_dim: int = None) -> dict:
        """Compare costs across different step counts."""
        results = {}
        original_steps = simulator.n_steps
        
        for n in step_counts:
            simulator.n_steps = n
            cost = self.analyze(simulator, state_dim)
            results[n] = cost
        
        simulator.n_steps = original_steps
        return results


# ============================================================
# 4. Pretty printing and visualization
# ============================================================

def format_bytes(b: int) -> str:
    """Human-readable byte count."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"


def format_flops(f: int) -> str:
    """Human-readable FLOP count."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if f < 1000:
            return f"{f:.2f} {unit}FLOPs"
        f /= 1000
    return f"{f:.2f} EFLOPs"


def print_cost_report(cost: SpacetimeCost, title: str = "SPACETIME COST ANALYSIS"):
    """Print a detailed cost report."""
    
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    
    # Overview
    print("\n┌─ OVERVIEW " + "─" * 58 + "┐")
    print(f"│  Terms: {cost.n_terms:<10}  Steps: {cost.n_steps:<10}  "
          f"Sequence/step: {cost.sequence_length:<10} │")
    print("└" + "─" * 69 + "┘")
    
    # Memory breakdown
    print("\n┌─ SPACE (MEMORY) " + "─" * 52 + "┐")
    print(f"│  Generator storage:    {format_bytes(cost.generator_memory_bytes):>15}          │")
    print(f"│  State vector:         {format_bytes(cost.state_memory_bytes):>15}          │")
    print(f"│  Auxiliary (working):  {format_bytes(cost.auxiliary_memory_bytes):>15}          │")
    print(f"│  ─────────────────────────────────────────────────────────────── │")
    print(f"│  PEAK MEMORY:          {format_bytes(cost.peak_memory_bytes):>15}          │")
    print("└" + "─" * 69 + "┘")
    
    # Memory by precision
    if cost.memory_by_precision:
        print("\n  Memory by precision:")
        for prec, mem in sorted(cost.memory_by_precision.items()):
            bar_len = int(40 * mem / cost.generator_memory_bytes) if cost.generator_memory_bytes > 0 else 0
            print(f"    {prec:12s} │{'█' * bar_len}{' ' * (40 - bar_len)}│ {format_bytes(mem)}")
    
    # Time/Depth breakdown
    print("\n┌─ TIME (DEPTH) " + "─" * 54 + "┐")
    print(f"│  Matrix-vector products:  {cost.total_matvec_products:>15,}              │")
    print(f"│  Matrix exponentials:     {cost.total_matrix_exponentials:>15,}              │")
    print(f"│  Total FLOPs:             {format_flops(cost.total_flops):>15}              │")
    print(f"│  ─────────────────────────────────────────────────────────────── │")
    print(f"│  Depth per step:          {cost.depth_per_step:>15,}              │")
    print(f"│  TOTAL DEPTH:             {cost.total_depth:>15,}              │")
    print(f"│  Parallel depth:          {cost.parallel_depth:>15,}              │")
    print("└" + "─" * 69 + "┘")
    
    # Per-term breakdown
    print("\n┌─ PER-TERM BREAKDOWN " + "─" * 48 + "┐")
    print(f"│  {'Term':<15} {'Precision':>10} {'MatVecs':>10} {'Memory':>12} {'FLOPs':>12}  │")
    print(f"│  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*12}  │")
    for name, tc in cost.cost_by_term.items():
        print(f"│  {name:<15} {tc.compute_bits:>7} bit {tc.matvec_count:>10,} "
              f"{format_bytes(tc.total_memory):>12} {format_flops(tc.flops):>12}  │")
    print("└" + "─" * 69 + "┘")
    
    # Composite metrics
    print("\n┌─ COMPOSITE METRICS " + "─" * 49 + "┐")
    print(f"│  Space × Time product:    {cost.space_time_product:>20,}         │")
    print(f"│  Adjusted FLOPs (GPU):    {format_flops(int(cost.effective_flops_adjusted)):>20}         │")
    print("└" + "─" * 69 + "┘")
    print()


def print_comparison_table(results: dict, metric: str = "total_depth"):
    """Print comparison table for multiple configurations."""
    print("\n" + "=" * 70)
    print(f" COMPARISON: {metric}")
    print("=" * 70)
    
    headers = ["Config", "Peak Memory", "Total Depth", "FLOPs", "Space×Time"]
    print(f"\n  {headers[0]:<20} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12} {headers[4]:>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*14}")
    
    for name, cost in results.items():
        print(f"  {str(name):<20} {format_bytes(cost.peak_memory_bytes):>12} "
              f"{cost.total_depth:>12,} {format_flops(cost.total_flops):>12} "
              f"{cost.space_time_product:>14,}")
    print()

"""
Measurement and cost analysis for the GPU mixed-precision emulator backend.

Public API:
    - CostProfile: Method-agnostic cost dataclass returned by dynamics.cost_profile()
    - Metrics: Unified metrics class for all gpu_mp_emulator dynamics
    - CostAnalyzer: Low-level spacetime cost analyzer for TrotterSimulator
    - print_cost_report / print_comparison_table: Pretty-print helpers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np

from trovolut.core.base_metrics import BaseMetrics


# ============================================================
# 1. CostProfile (unified, method-agnostic)
# ============================================================

@dataclass
class CostProfile:
    """
    Method-agnostic cost profile for any mixed-precision evolution.

    Returned by ``MPEvolution.cost_profile()`` and consumed by
    ``SpaceCost``, ``TimeCost``, ``SpaceTimeCost`` constraints via the
    ``Metrics`` class.
    """

    method: str = ""

    # === SPACE ===
    peak_memory_bytes: int = 0
    generator_memory_bytes: int = 0
    state_memory_bytes: int = 0
    auxiliary_memory_bytes: int = 0
    memory_by_precision: dict[str, int] = field(default_factory=dict)

    # === TIME ===
    total_flops: int = 0
    total_depth: int = 0

    # === PARAMETERS ===
    n_terms: int = 0
    n_steps: int = 0

    @property
    def space_time_product(self) -> int:
        """Space x Time product (quantum-inspired metric)."""
        return self.peak_memory_bytes * self.total_depth

    @property
    def memory_mb(self) -> float:
        return self.peak_memory_bytes / (1024 ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten into a dict suitable for constraint evaluation."""
        return {
            "method": self.method,
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_bytes": self.peak_memory_bytes,
            "generator_memory_bytes": self.generator_memory_bytes,
            "state_memory_bytes": self.state_memory_bytes,
            "auxiliary_memory_bytes": self.auxiliary_memory_bytes,
            "memory_by_precision": dict(self.memory_by_precision),
            "total_flops": self.total_flops,
            "total_depth": self.total_depth,
            "space_time_product": self.space_time_product,
            "memory_mb": self.memory_mb,
            "n_terms": self.n_terms,
            "n_steps": self.n_steps,
        }


# ============================================================
# 2. Unified Metrics class
# ============================================================

def _parse_test_case(test_case):
    """Parse (ham, psi0), (ham, psi0, t), or (ham, psi0, t, config)."""
    if len(test_case) == 4:
        return test_case[0], test_case[1], test_case[2], test_case[3]
    elif len(test_case) == 3:
        return test_case[0], test_case[1], test_case[2], None
    elif len(test_case) == 2:
        return test_case[0], test_case[1], 1.0, None
    raise ValueError("test_generator must return (ham, psi0), (ham, psi0, t), or (ham, psi0, t, config)")


def _default_exact(ham, t, psi0):
    """Fallback exact evolution when no exact_fn is provided."""
    import scipy.linalg as la

    if hasattr(ham, "to_matrix"):
        H = ham.to_matrix()
    elif isinstance(ham, (list, tuple)):
        H = sum(ham) if ham else np.zeros_like(psi0)
    else:
        H = np.asarray(ham)
    if hasattr(H, "toarray"):
        H = H.toarray()
    return la.expm(-1j * H * t) @ psi0


class Metrics(BaseMetrics):
    """
    Unified metrics for all gpu_mp_emulator dynamics methods.

    Validates that the dynamics object is a ``MPEvolution``, computes a
    static cost profile via ``dynamics.cost_profile(ham)``, runs the
    evolution for wall-clock time and error, and returns a combined dict
    suitable for constraint evaluation.

    Example::

        from trovolut.backends.gpu_mp_emulator.dynamics import (
            MPTrotterEvolution, Precision,
        )
        from trovolut.backends.gpu_mp_emulator.measure import Metrics
        from trovolut.backends.gpu_mp_emulator.constraints import SpaceCost, TimeCost

        dyn = MPTrotterEvolution(
            trotter_steps=100,
            precision_schedule=Precision.FP32,
        )

        metrics = Metrics(
            constraints=[SpaceCost(max_memory_bytes=50*1024*1024), TimeCost(max_flops=1e12)],
            test_generator=lambda: (H_qubit, psi0, 1.0),
        )
        result = metrics(dyn)
    """

    def __init__(
        self,
        constraints=None,
        test_generator=None,
        exact_fn=None,
        n_trials: int = 3,
        timeout: int = 30,
    ):
        super().__init__(
            constraints=constraints,
            test_generator=test_generator,
            exact_fn=exact_fn,
            n_trials=n_trials,
            timeout=timeout,
            backend="gpu-mp-emulator",
        )

    def measure(self, dynamics_evolution, **kwargs) -> Dict[str, Any]:
        import time as time_mod
        from trovolut.backends.gpu_mp_emulator.dynamics import MPEvolution

        if not isinstance(dynamics_evolution, MPEvolution):
            raise TypeError(
                f"Metrics expects an MPEvolution instance, "
                f"got {type(dynamics_evolution).__name__}. "
                f"Use dynamics from trovolut.backends.gpu_mp_emulator.dynamics."
            )

        test_gen = kwargs.get("test_generator", self.test_generator)
        exact_fn = kwargs.get("exact_fn", self.exact_fn)
        n_trials = kwargs.get("n_trials", self.n_trials)

        if not test_gen:
            return {"error": "No test_generator provided", "combined_score": -100.0}

        times = []
        errors = []
        cost: CostProfile | None = None

        for _ in range(n_trials):
            test_case = test_gen()
            ham, psi0, t, extra_config = _parse_test_case(test_case)

            # Static cost profile (compute once on first test case)
            if cost is None:
                try:
                    cost = dynamics_evolution.cost_profile(ham, **kwargs)
                except Exception:
                    cost = CostProfile(method="unknown")

            # Run evolution
            start = time_mod.perf_counter()
            psi_final = dynamics_evolution.evolve(ham, psi0, t, config=extra_config, **kwargs)
            times.append(time_mod.perf_counter() - start)

            # Compute error
            if exact_fn:
                psi_exact = exact_fn(ham, t, psi0)
            else:
                psi_exact = _default_exact(ham, t, psi0)
            errors.append(float(np.linalg.norm(psi_final - psi_exact)))

        mean_error = float(np.mean(errors))
        result: Dict[str, Any] = {
            "combined_score": 0.0,
            "time": float(np.mean(times)),
            "time_seconds": float(np.mean(times)),
            "error": mean_error,
            "mean_error": mean_error,
            "std_score": float(np.std(errors)) / (1.0 + mean_error) ** 2,
        }
        result.update(cost.to_dict())
        return result


# ============================================================
# 3. Detailed cost data structures (for CostAnalyzer)
# ============================================================

@dataclass
class TermCost:
    """Cost breakdown for a single ExpTerm."""
    name: str

    # Memory (bytes)
    generator_memory: int = 0
    auxiliary_memory: int = 0

    # Depth (operations)
    matvec_count: int = 0
    matmat_count: int = 0
    scalar_ops: int = 0

    # Effective FLOPs
    flops: int = 0

    # Precision info
    compute_bits: int = 64

    @property
    def total_memory(self) -> int:
        return self.generator_memory + self.auxiliary_memory

    @property
    def depth(self) -> int:
        return self.matvec_count + self.matmat_count * 10


@dataclass
class StepCost:
    """Cost for one complete Trotter step."""
    term_applications: list[tuple[str, float]] = field(default_factory=list)
    total_matvecs: int = 0
    total_depth: int = 0


@dataclass
class SpacetimeCost:
    """Complete spacetime cost analysis (Trotter-specific detail)."""

    # === SPACE (Memory) ===
    generator_memory_bytes: int = 0
    state_memory_bytes: int = 0
    auxiliary_memory_bytes: int = 0
    peak_memory_bytes: int = 0

    memory_by_precision: dict[str, int] = field(default_factory=dict)

    # === TIME (Depth) ===
    total_matvec_products: int = 0
    total_matrix_exponentials: int = 0
    total_flops: int = 0

    depth_per_step: int = 0
    total_depth: int = 0
    parallel_depth: int = 0

    # === BREAKDOWN ===
    cost_by_term: dict[str, TermCost] = field(default_factory=dict)
    n_steps: int = 0
    n_terms: int = 0
    sequence_length: int = 0

    # === DERIVED METRICS ===
    @property
    def space_time_product(self) -> int:
        return self.peak_memory_bytes * self.total_depth

    @property
    def memory_gb(self) -> float:
        return self.peak_memory_bytes / (1024 ** 3)

    @property
    def effective_flops_adjusted(self) -> float:
        multipliers = {64: 0.5, 32: 1.0, 16: 2.0, 8: 4.0, 4: 4.0}
        adjusted = 0
        for _name, tc in self.cost_by_term.items():
            mult = multipliers.get(tc.compute_bits, 1.0)
            adjusted += tc.flops / mult
        return adjusted * self.n_steps


# ============================================================
# 4. Cost estimation per decomposition method
# ============================================================

class DecompositionCostModel:
    """Estimates cost for different exponentiation methods."""

    @staticmethod
    def exact_expm_cost(n: int, precision_bits: int) -> dict:
        bpe = max(1, precision_bits // 8)
        auxiliary_matrices = 6
        return {
            "matvec_count": 0,
            "matmat_count": 13,
            "auxiliary_memory": auxiliary_matrices * n * n * bpe * 2,
            "flops": 13 * (2 * n ** 3),
        }

    @staticmethod
    def taylor_cost(n: int, order: int, precision_bits: int) -> dict:
        bpe = max(1, precision_bits // 8)
        return {
            "matvec_count": order,
            "matmat_count": 0,
            "auxiliary_memory": 2 * n * bpe * 2,
            "flops": order * (2 * n ** 2),
        }

    @staticmethod
    def krylov_cost(n: int, krylov_dim: int, precision_bits: int) -> dict:
        bpe = max(1, precision_bits // 8)
        m = min(krylov_dim, n)
        return {
            "matvec_count": m,
            "matmat_count": 1,
            "auxiliary_memory": (
                n * (m + 1) * bpe * 2
                + (m + 1) * m * bpe * 2
            ),
            "flops": m * (2 * n ** 2) + 13 * (2 * m ** 3),
        }

    @staticmethod
    def sparse_cost(n: int, nnz: int, order: int, precision_bits: int) -> dict:
        bpe = max(1, precision_bits // 8)
        return {
            "matvec_count": order,
            "matmat_count": 0,
            "auxiliary_memory": 2 * n * bpe * 2,
            "flops": order * (2 * nnz),
        }


# ============================================================
# 5. CostAnalyzer (low-level, for TrotterSimulator)
# ============================================================

class CostAnalyzer:
    """Analyzes spacetime cost of a TrotterSimulator without running it."""

    DEFAULT_TAYLOR_ORDER = 20
    DEFAULT_KRYLOV_DIM = 30

    def __init__(self):
        self.cost_models = DecompositionCostModel()

    def analyze(self, simulator, state_dim: int | None = None) -> SpacetimeCost:
        n = state_dim or self._infer_dimension(simulator.terms)

        cost = SpacetimeCost(
            n_steps=simulator.n_steps,
            n_terms=len(simulator.terms),
        )

        for term in simulator.terms:
            tc = self._analyze_term(term, n)
            cost.cost_by_term[term.name] = tc
            cost.generator_memory_bytes += tc.generator_memory
            prec_key = term.precision.compute.value
            cost.memory_by_precision[prec_key] = (
                cost.memory_by_precision.get(prec_key, 0) + tc.generator_memory
            )

        dummy_dt = 1.0
        sequence = simulator.get_sequence(simulator.terms, dummy_dt)
        cost.sequence_length = len(sequence)

        step_matvecs = step_depth = step_flops = 0
        step_aux_memory = 0
        for term, _ in sequence:
            tc = cost.cost_by_term[term.name]
            step_matvecs += tc.matvec_count
            step_depth += tc.depth
            step_flops += tc.flops
            step_aux_memory = max(step_aux_memory, tc.auxiliary_memory)

        cost.depth_per_step = step_depth
        cost.auxiliary_memory_bytes = step_aux_memory
        cost.total_matvec_products = step_matvecs * simulator.n_steps
        cost.total_depth = step_depth * simulator.n_steps
        cost.total_flops = step_flops * simulator.n_steps
        cost.total_matrix_exponentials = sum(
            1 for term, _ in sequence if cost.cost_by_term[term.name].matmat_count > 0
        ) * simulator.n_steps

        cost.state_memory_bytes = n * 16
        cost.peak_memory_bytes = (
            cost.generator_memory_bytes + cost.state_memory_bytes + cost.auxiliary_memory_bytes
        )

        max_term_depth = max(tc.depth for tc in cost.cost_by_term.values())
        cost.parallel_depth = max_term_depth * simulator.n_steps
        return cost

    def _infer_dimension(self, terms: list) -> int:
        for term in terms:
            gen = term.get_generator(0)
            if isinstance(gen, np.ndarray):
                return gen.shape[0]
        raise ValueError("Cannot infer dimension: no concrete generators found")

    def _analyze_term(self, term, n: int) -> TermCost:
        prec = term.precision
        bits = prec.compute.bits
        bpe = max(1, bits // 8)

        gen = term.get_generator(0) if not callable(term.generator) else None
        if gen is not None:
            is_sparse = hasattr(gen, "nnz")
            gen_memory = gen.nnz * bpe * 2 if is_sparse else n * n * bpe * 2
        else:
            gen_memory = n * n * bpe * 2

        method = term.decomposition
        if method == "exact":
            mc = self.cost_models.exact_expm_cost(n, bits)
        elif method == "taylor":
            mc = self.cost_models.taylor_cost(n, term.taylor_order or self.DEFAULT_TAYLOR_ORDER, bits)
        elif method == "krylov":
            mc = self.cost_models.krylov_cost(n, self.DEFAULT_KRYLOV_DIM, bits)
        elif method == "sparse":
            nnz = gen.nnz if hasattr(gen, "nnz") else n * n
            mc = self.cost_models.sparse_cost(n, nnz, term.taylor_order or self.DEFAULT_TAYLOR_ORDER, bits)
        else:
            mc = self.cost_models.exact_expm_cost(n, bits)

        return TermCost(
            name=term.name,
            generator_memory=gen_memory,
            auxiliary_memory=mc["auxiliary_memory"],
            matvec_count=mc["matvec_count"],
            matmat_count=mc["matmat_count"],
            flops=mc["flops"],
            compute_bits=bits,
        )

    def compare_formulas(self, simulator, formulas: list, state_dim: int = None) -> dict:
        results = {}
        original_formula = simulator.formula
        for formula in formulas:
            simulator.formula = formula
            results[formula.__class__.__name__] = self.analyze(simulator, state_dim)
        simulator.formula = original_formula
        return results

    def compare_n_steps(self, simulator, step_counts: list[int], state_dim: int = None) -> dict:
        results = {}
        original_steps = simulator.n_steps
        for n in step_counts:
            simulator.n_steps = n
            results[n] = self.analyze(simulator, state_dim)
        simulator.n_steps = original_steps
        return results


# ============================================================
# 6. Legacy helper
# ============================================================

def cost_from_dynamics_config(
    hamiltonian,
    dynamics_evolution,
    config=None,
    state_dim: int | None = None,
    **kwargs,
) -> SpacetimeCost | None:
    """
    Compute SpacetimeCost for a dynamics method without running evolution.

    Works for any ``MPEvolution`` that has a ``cost_profile()`` method,
    or for a raw ``TrotterSimulator``.
    """
    # Try the unified cost_profile first
    if hasattr(dynamics_evolution, "cost_profile"):
        try:
            profile = dynamics_evolution.cost_profile(hamiltonian, **kwargs)
            # Wrap CostProfile into a SpacetimeCost for backward compat
            return SpacetimeCost(
                generator_memory_bytes=profile.generator_memory_bytes,
                state_memory_bytes=profile.state_memory_bytes,
                auxiliary_memory_bytes=profile.auxiliary_memory_bytes,
                peak_memory_bytes=profile.peak_memory_bytes,
                memory_by_precision=profile.memory_by_precision,
                total_flops=profile.total_flops,
                total_depth=profile.total_depth,
                n_steps=profile.n_steps,
                n_terms=profile.n_terms,
            )
        except Exception:
            pass
    return None


# ============================================================
# 7. Pretty printing and visualization
# ============================================================

def format_bytes(b: int) -> str:
    """Human-readable byte count."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"


def format_flops(f: int) -> str:
    """Human-readable FLOP count."""
    for unit in ["", "K", "M", "G", "T", "P"]:
        if f < 1000:
            return f"{f:.2f} {unit}FLOPs"
        f /= 1000
    return f"{f:.2f} EFLOPs"


def print_cost_report(cost: SpacetimeCost, title: str = "SPACETIME COST ANALYSIS"):
    """Print a detailed cost report."""

    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    print("\n┌─ OVERVIEW " + "─" * 58 + "┐")
    print(
        f"│  Terms: {cost.n_terms:<10}  Steps: {cost.n_steps:<10}  "
        f"Sequence/step: {cost.sequence_length:<10} │"
    )
    print("└" + "─" * 69 + "┘")

    print("\n┌─ SPACE (MEMORY) " + "─" * 52 + "┐")
    print(f"│  Generator storage:    {format_bytes(cost.generator_memory_bytes):>15}          │")
    print(f"│  State vector:         {format_bytes(cost.state_memory_bytes):>15}          │")
    print(f"│  Auxiliary (working):  {format_bytes(cost.auxiliary_memory_bytes):>15}          │")
    print(f"│  ─────────────────────────────────────────────────────────────── │")
    print(f"│  PEAK MEMORY:          {format_bytes(cost.peak_memory_bytes):>15}          │")
    print("└" + "─" * 69 + "┘")

    if cost.memory_by_precision:
        print("\n  Memory by precision:")
        for prec, mem in sorted(cost.memory_by_precision.items()):
            bar_len = int(40 * mem / cost.generator_memory_bytes) if cost.generator_memory_bytes > 0 else 0
            print(f"    {prec:12s} │{'█' * bar_len}{' ' * (40 - bar_len)}│ {format_bytes(mem)}")

    print("\n┌─ TIME (DEPTH) " + "─" * 54 + "┐")
    print(f"│  Matrix-vector products:  {cost.total_matvec_products:>15,}              │")
    print(f"│  Matrix exponentials:     {cost.total_matrix_exponentials:>15,}              │")
    print(f"│  Total FLOPs:             {format_flops(cost.total_flops):>15}              │")
    print(f"│  ─────────────────────────────────────────────────────────────── │")
    print(f"│  Depth per step:          {cost.depth_per_step:>15,}              │")
    print(f"│  TOTAL DEPTH:             {cost.total_depth:>15,}              │")
    print(f"│  Parallel depth:          {cost.parallel_depth:>15,}              │")
    print("└" + "─" * 69 + "┘")

    print("\n┌─ PER-TERM BREAKDOWN " + "─" * 48 + "┐")
    print(f"│  {'Term':<15} {'Precision':>10} {'MatVecs':>10} {'Memory':>12} {'FLOPs':>12}  │")
    print(f"│  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*12}  │")
    for name, tc in cost.cost_by_term.items():
        print(
            f"│  {name:<15} {tc.compute_bits:>7} bit {tc.matvec_count:>10,} "
            f"{format_bytes(tc.total_memory):>12} {format_flops(tc.flops):>12}  │"
        )
    print("└" + "─" * 69 + "┘")

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
        print(
            f"  {str(name):<20} {format_bytes(cost.peak_memory_bytes):>12} "
            f"{cost.total_depth:>12,} {format_flops(cost.total_flops):>12} "
            f"{cost.space_time_product:>14,}"
        )
    print()

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence, Literal
from enum import Enum
import numpy as np

# ============================================================
# 1. Precision definitions and casting infrastructure
# ============================================================

class Precision(Enum):
    """Supported precision levels."""
    FP64 = "float64"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8_E4M3 = "float8_e4m3"   # NVIDIA FP8 (range)
    FP8_E5M2 = "float8_e5m2"   # NVIDIA FP8 (precision)
    INT8 = "int8"              # quantized
    
    @property
    def bits(self) -> int:
        return {
            "float64": 64, "float32": 32, "float16": 16,
            "bfloat16": 16, "float8_e4m3": 8, "float8_e5m2": 8, "int8": 8
        }[self.value]


@dataclass
class PrecisionConfig:
    """Fine-grained control over precision at different stages."""
    compute: Precision = Precision.FP32      # Precision for matrix ops
    storage: Precision = Precision.FP32      # Precision for storing generators
    accumulate: Precision = Precision.FP64   # Precision for state accumulation
    
    # Quantization parameters (for INT8/FP8)
    scale: float | None = None               # quantization scale
    zero_point: float = 0.0
    dynamic_scaling: bool = True             # recompute scale per step


class PrecisionCaster:
    """Handles casting between precisions, including emulated formats."""
    
    # Native numpy dtypes (real and complex)
    NATIVE_DTYPES = {
        Precision.FP64: (np.float64, np.complex128),
        Precision.FP32: (np.float32, np.complex64),
        Precision.FP16: (np.float16, np.complex64),  # no complex32, use complex64
    }
    
    def __init__(self, use_torch: bool = False, use_cuda: bool = False):
        self.use_torch = use_torch
        self.use_cuda = use_cuda
        self._torch = None
        
        if use_torch:
            try:
                import torch
                self._torch = torch
            except ImportError:
                self.use_torch = False
    
    def cast(self, arr: np.ndarray, precision: Precision, 
             config: PrecisionConfig | None = None) -> np.ndarray:
        """Cast array to target precision, handling complex numbers."""
        
        is_complex = np.iscomplexobj(arr)
        
        # Native numpy types
        if precision in self.NATIVE_DTYPES:
            real_dtype, complex_dtype = self.NATIVE_DTYPES[precision]
            target_dtype = complex_dtype if is_complex else real_dtype
            return arr.astype(target_dtype)
        
        # For emulated types, process real/imag separately
        if is_complex:
            real_part = self._cast_real(arr.real, precision, config)
            imag_part = self._cast_real(arr.imag, precision, config)
            # Return as complex64 (minimum complex dtype)
            return real_part.astype(np.float32) + 1j * imag_part.astype(np.float32)
        else:
            return self._cast_real(arr, precision, config)
    
    def _cast_real(self, arr: np.ndarray, precision: Precision,
                   config: PrecisionConfig | None) -> np.ndarray:
        """Cast real-valued array to target precision."""
        
        # Ensure we're working with real array
        arr = np.asarray(arr, dtype=np.float64)
        
        if precision == Precision.BF16:
            return self._to_bfloat16(arr)
        elif precision == Precision.FP8_E4M3:
            return self._to_fp8_e4m3(arr, config)
        elif precision == Precision.FP8_E5M2:
            return self._to_fp8_e5m2(arr, config)
        elif precision == Precision.INT8:
            return self._to_int8(arr, config)
        
        raise ValueError(f"Unsupported precision: {precision}")
    
    def _to_bfloat16(self, arr: np.ndarray) -> np.ndarray:
        """Emulate bfloat16 by truncating fp32 mantissa."""
        if self.use_torch and self._torch:
            t = self._torch.from_numpy(arr.astype(np.float32))
            return t.to(self._torch.bfloat16).float().numpy()
        
        # Pure numpy emulation
        fp32 = arr.astype(np.float32).copy()
        int_view = fp32.view(np.uint32)
        int_view &= 0xFFFF0000
        return fp32
    
    def _to_fp8_e4m3(self, arr: np.ndarray, config: PrecisionConfig | None) -> np.ndarray:
        return self._quantize_to_fp8(arr, e_bits=4, m_bits=3, config=config)
    
    def _to_fp8_e5m2(self, arr: np.ndarray, config: PrecisionConfig | None) -> np.ndarray:
        return self._quantize_to_fp8(arr, e_bits=5, m_bits=2, config=config)
    
    def _quantize_to_fp8(self, arr: np.ndarray, e_bits: int, m_bits: int,
                         config: PrecisionConfig | None) -> np.ndarray:
        """Generic FP8 emulation."""
        if self.use_torch and self._torch and hasattr(self._torch, 'float8_e4m3fn'):
            t = self._torch.from_numpy(arr.astype(np.float32))
            if self.use_cuda and self._torch.cuda.is_available():
                t = t.cuda()
            dtype = (self._torch.float8_e4m3fn if e_bits == 4 
                     else self._torch.float8_e5m2)
            result = t.to(dtype).float()
            if self.use_cuda:
                result = result.cpu()
            return result.numpy()
        
        # Pure numpy emulation
        bias = 2**(e_bits - 1) - 1
        max_exp = 2**e_bits - 1 - bias
        min_exp = 1 - bias
        max_val = 2**max_exp * (2 - 2**(-m_bits))
        
        fp32 = arr.astype(np.float32)
        sign = np.sign(fp32)
        abs_val = np.abs(fp32)
        
        # Handle zeros
        nonzero_mask = abs_val > 0
        result = np.zeros_like(fp32)
        
        if np.any(nonzero_mask):
            abs_nz = abs_val[nonzero_mask]
            abs_nz = np.clip(abs_nz, 2**(min_exp - m_bits), max_val)
            
            log2_val = np.floor(np.log2(abs_nz + 1e-45))
            mantissa_lsb = 2**(log2_val - m_bits)
            quantized = np.round(abs_nz / mantissa_lsb) * mantissa_lsb
            
            result[nonzero_mask] = sign[nonzero_mask] * quantized
        
        return result
    
    def _to_int8(self, arr: np.ndarray, config: PrecisionConfig | None) -> np.ndarray:
        """Symmetric INT8 quantization."""
        config = config or PrecisionConfig()
        
        if config.dynamic_scaling or config.scale is None:
            max_val = np.max(np.abs(arr))
            scale = max_val / 127.0 if max_val > 0 else 1.0
        else:
            scale = config.scale
        
        quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
        return quantized.astype(np.float32) * scale



# ============================================================
# 2. Updated ExpTerm with precision config
# ============================================================

@dataclass
class ExpTerm:
    """A single exp(-i * coeff * H * dt) term with precision control."""
    name: str
    generator: np.ndarray | Callable[[float], np.ndarray]
    coefficient: float = 1.0
    
    # Decomposition method
    decomposition: str = "exact"
    taylor_order: int | None = None
    tolerance: float = 1e-10
    
    # === NEW: Precision configuration ===
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    
    def get_generator(self, t: float = 0) -> np.ndarray:
        if callable(self.generator):
            return self.generator(t)
        return self.generator


# ============================================================
# 3. Precision-aware exponentiators
# ============================================================

class Exponentiator(ABC):
    def __init__(self, caster: PrecisionCaster | None = None):
        self.caster = caster or PrecisionCaster()
    
    @abstractmethod
    def apply(self, term: ExpTerm, dt: float, state: np.ndarray) -> np.ndarray:
        pass
    
    def _prepare_generator(self, term: ExpTerm, t: float = 0) -> np.ndarray:
        """Get generator cast to compute precision."""
        H = term.get_generator(t) * term.coefficient
        return self.caster.cast(H, term.precision.compute, term.precision)
    
    def _accumulate(self, state: np.ndarray, delta: np.ndarray, 
                    precision: Precision) -> np.ndarray:
        """Accumulate in specified precision."""
        acc_state = self.caster.cast(state, precision)
        acc_delta = self.caster.cast(delta, precision)
        return acc_state + acc_delta


class ExactExponentiator(Exponentiator):
    """Matrix exponential with precision control."""
    
    def apply(self, term: ExpTerm, dt: float, state: np.ndarray) -> np.ndarray:
        from scipy.linalg import expm
        
        # Get generator in compute precision
        H = self._prepare_generator(term)
        
        # Convert sparse to dense if needed
        if hasattr(H, 'toarray'):
            H = H.toarray()
        
        # Compute exponential in high precision for stability
        H_hp = H.astype(np.complex128)
        U = expm(-1j * H_hp * dt)
        
        # Cast U to compute precision for matmul
        U_low = self.caster.cast(U, term.precision.compute)
        
        # Apply
        result = U_low @ state
        
        # Return in accumulate precision
        return self.caster.cast(result, term.precision.accumulate)



class TaylorExponentiator(Exponentiator):
    """Taylor series with mixed precision per iteration."""
    
    def apply(self, term: ExpTerm, dt: float, state: np.ndarray) -> np.ndarray:
        H = self._prepare_generator(term)
        order = term.taylor_order or 20
        prec = term.precision
        
        # Convert sparse to dense if needed (sparse matrix-vector is fine, but keep consistent)
        if hasattr(H, 'toarray'):
            H = H.toarray()
        
        # Accumulator in high precision
        result = self.caster.cast(state, prec.accumulate)
        
        # Term computation in compute precision
        term_k = self.caster.cast(state, prec.compute)
        
        for k in range(1, order + 1):
            # Matrix-vector product in compute precision
            Hv = H @ term_k
            term_k = self.caster.cast((-1j * dt / k) * Hv, prec.compute)
            
            # Accumulate in high precision
            result = self._accumulate(
                result, 
                self.caster.cast(term_k, prec.accumulate),
                prec.accumulate
            )
            
            if np.linalg.norm(term_k) < term.tolerance:
                break
        
        return result

class KrylovExponentiator(Exponentiator):
    """Krylov subspace method with precision control."""
    
    def __init__(self, krylov_dim: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.krylov_dim = krylov_dim
    
    def apply(self, term: ExpTerm, dt: float, state: np.ndarray) -> np.ndarray:
        H = self._prepare_generator(term)
        prec = term.precision
        
        # Convert sparse to dense if needed for compatibility
        if hasattr(H, 'toarray'):
            H = H.toarray()
        
        n = len(state)
        m = min(self.krylov_dim, n)
        
        # Always use complex128 for Krylov basis (numerical stability)
        # Precision control happens only in the matvec
        V = np.zeros((n, m + 1), dtype=np.complex128)
        H_krylov = np.zeros((m + 1, m), dtype=np.complex128)
        
        beta = np.linalg.norm(state)
        V[:, 0] = state / beta
        
        for j in range(m):
            # === Low precision matvec ===
            V_j_low = self.caster.cast(V[:, j], prec.compute)
            H_low = self.caster.cast(H, prec.compute)
            w = H_low @ V_j_low
            
            # Cast back to complex128 for stable orthogonalization
            w = w.astype(np.complex128)
            
            # Arnoldi orthogonalization (always high precision)
            for i in range(j + 1):
                H_krylov[i, j] = np.vdot(V[:, i], w)
                w = w - H_krylov[i, j] * V[:, i]  # both complex128
            
            H_krylov[j + 1, j] = np.linalg.norm(w)
            if np.abs(H_krylov[j + 1, j]) > 1e-12:
                V[:, j + 1] = w / H_krylov[j + 1, j]
            else:
                m = j + 1  # early termination
                break
        
        # Small matrix exponential (always high precision)
        from scipy.linalg import expm
        exp_H = expm(-1j * dt * H_krylov[:m, :m])
        
        result = beta * (V[:, :m] @ exp_H[:, 0])
        return self.caster.cast(result, prec.accumulate)



# ============================================================
# 4. Trotter formulas (unchanged interface)
# ============================================================

class TrotterFormula(ABC):
    @abstractmethod
    def get_sequence(self, terms: list[ExpTerm], dt: float) -> list[tuple[ExpTerm, float]]:
        pass


class SecondOrderTrotter(TrotterFormula):
    def get_sequence(self, terms: list[ExpTerm], dt: float):
        seq = []
        for term in terms[:-1]:
            seq.append((term, dt / 2))
        seq.append((terms[-1], dt))
        for term in reversed(terms[:-1]):
            seq.append((term, dt / 2))
        return seq


# ============================================================
# 5. Main simulator with precision tracking
# ============================================================

@dataclass
class PrecisionStats:
    """Track precision-related statistics."""
    terms_by_precision: dict[Precision, int] = field(default_factory=dict)
    estimated_memory_bytes: int = 0
    precision_transitions: int = 0  # number of casts between precisions


@dataclass 
class TrotterSimulator:
    terms: list[ExpTerm]
    formula: TrotterFormula = field(default_factory=SecondOrderTrotter)
    n_steps: int = 100
    
    # Global precision fallback
    global_accumulate_precision: Precision = Precision.FP64
    
    # Infrastructure
    caster: PrecisionCaster = field(default_factory=lambda: PrecisionCaster(use_torch=False))
    exponentiators: dict[str, Exponentiator] = field(default=None)
    
    def __post_init__(self):
        if self.exponentiators is None:
            self.exponentiators = {
                "exact": ExactExponentiator(self.caster),
                "taylor": TaylorExponentiator(self.caster),
                "krylov": KrylovExponentiator(caster=self.caster),
            }
    
    def evolve(self, state: np.ndarray, total_time: float, 
               track_stats: bool = False) -> tuple[np.ndarray, PrecisionStats | None]:
        dt = total_time / self.n_steps
        stats = PrecisionStats() if track_stats else None
        
        # Track precision usage
        if track_stats:
            for term in self.terms:
                p = term.precision.compute
                stats.terms_by_precision[p] = stats.terms_by_precision.get(p, 0) + 1
        
        # Ensure state is in accumulation precision
        state = self.caster.cast(state, self.global_accumulate_precision)
        
        for step in range(self.n_steps):
            sequence = self.formula.get_sequence(self.terms, dt)
            
            for term, scaled_dt in sequence:
                exp_method = self.exponentiators[term.decomposition]
                state = exp_method.apply(term, scaled_dt, state)
                
                # Re-cast to global accumulation precision between terms
                state = self.caster.cast(state, self.global_accumulate_precision)
                
                if track_stats:
                    stats.precision_transitions += 1
        
        return state, stats
    
    def print_precision_summary(self):
        """Print summary of precision configuration."""
        print("=" * 60)
        print("MIXED PRECISION TROTTERIZATION SUMMARY")
        print("=" * 60)
        for term in self.terms:
            p = term.precision
            print(f"\n{term.name}:")
            print(f"  Compute:    {p.compute.value:12s} ({p.compute.bits} bits)")
            print(f"  Storage:    {p.storage.value:12s} ({p.storage.bits} bits)")
            print(f"  Accumulate: {p.accumulate.value:12s} ({p.accumulate.bits} bits)")
            print(f"  Method:     {term.decomposition}")
        print("=" * 60)

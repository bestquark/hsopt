import cupynumeric as np

# ≤8-bit tiny formats + native passthroughs
FMT = {
    "fp4_e2m1": dict(exp_bits=2, man_bits=1, bias=1),
    "fp5_e2m2": dict(exp_bits=2, man_bits=2, bias=1),
    "fp6_e3m2": dict(exp_bits=3, man_bits=2, bias=3),
    "fp7_e3m3": dict(exp_bits=3, man_bits=3, bias=3),
    "fp8_e4m3": dict(exp_bits=4, man_bits=3, bias=7),
    "fp8_e5m2": dict(exp_bits=5, man_bits=2, bias=15),
    # native (passthrough; no quantization)
    "fp16": {"native": np.float16},
    "fp32": {"native": np.float32},
    "fp64": {"native": np.float64},
}
NATIVE = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}

# ----- quant/dequant for REAL arrays -----
def _quantize_fp(x32: np.ndarray, exp_bits: int, man_bits: int, bias: int) -> np.ndarray:
    x = x32.astype(np.float32, copy=False)
    sign = (x < 0).astype(np.uint8)
    ax = np.abs(x)
    m, e = np.frexp(ax)
    m = np.where(ax == 0, 0.5, m)
    e = np.where(ax == 0, 0, e)
    unbiased = e - 1
    exp_stored = unbiased + bias
    frac = m * 2.0 - 1.0
    mant = np.round(frac * (1 << man_bits)).astype(np.int32)
    mant = np.clip(mant, 0, (1 << man_bits) - 1)
    under = exp_stored <= 0
    exp_stored = np.where(under, 0, exp_stored)
    mant = np.where(under, 0, mant)
    exp_max = (1 << exp_bits) - 1
    over = exp_stored >= exp_max
    exp_stored = np.where(over, exp_max - 1, exp_stored)
    mant = np.where(over, (1 << man_bits) - 1, mant)
    code = (sign << (exp_bits + man_bits)) | (exp_stored.astype(np.uint8) << man_bits) | mant.astype(np.uint8)
    return code.astype(np.uint8)

def _dequantize_fp(code: np.ndarray, exp_bits: int, man_bits: int, bias: int) -> np.ndarray:
    c = code.astype(np.uint8, copy=False)
    sign = (c >> (exp_bits + man_bits)) & 0x1
    exp  = (c >> man_bits) & ((1 << exp_bits) - 1)
    mant = c & ((1 << man_bits) - 1)
    is_zero = exp == 0
    frac = mant.astype(np.float32) / (1 << man_bits)
    m = 1.0 + frac
    e_unbias = exp.astype(np.int32) - bias
    val = np.where(is_zero, 0.0, np.ldexp(m, e_unbias)).astype(np.float32)
    val = np.where(sign > 0, -val, val)
    return val

def quantize_real(x: np.ndarray, fmt: str):
    if fmt in NATIVE:  # passthrough
        return x.astype(NATIVE[fmt], copy=False)
    spec = FMT[fmt]
    return _quantize_fp(x, **spec)

def dequantize_real(code: np.ndarray, fmt: str) -> np.ndarray:
    if fmt in NATIVE:  # passthrough
        # return in float32 baseline; compute cast happens later
        return code.astype(np.float32, copy=False)
    spec = FMT[fmt]
    return _dequantize_fp(code, **spec)

# ----- complex-aware wrappers -----
def quantize_any(x: np.ndarray, fmt: str):
    if np.iscomplexobj(x):
        return (quantize_real(x.real, fmt), quantize_real(x.imag, fmt))
    else:
        return quantize_real(x, fmt)

def dequantize_any(code, fmt: str, dtype):
    if isinstance(code, tuple):  # complex path
        re = dequantize_real(code[0], fmt).astype(dtype, copy=False)
        im = dequantize_real(code[1], fmt).astype(dtype, copy=False)
        return re + 1j * im
    else:
        return dequantize_real(code, fmt).astype(dtype, copy=False)

# ----- Mixed-precision matmul (FP16 compute, FP32/complex64 out) -----
def matmul_mixed(A: np.ndarray, fmtA: str, B: np.ndarray, fmtB: str,
                 compute_dtype=np.float16, out_dtype=None):
    A_q = quantize_any(A, fmtA)
    B_q = quantize_any(B, fmtB)
    is_cplx = np.iscomplexobj(A) or np.iscomplexobj(B)
    if out_dtype is None:
        out_dtype = np.complex64 if is_cplx else np.float32
    A_d = dequantize_any(A_q, fmtA, compute_dtype)
    B_d = dequantize_any(B_q, fmtB, compute_dtype)
    return (A_d @ B_d).astype(out_dtype, copy=False)
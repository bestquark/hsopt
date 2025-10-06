import cupynumeric as np

FMT = {
    # "fp4_e2m1": dict(exp_bits=2, man_bits=1, bias=1),
    # "fp5_e2m2": dict(exp_bits=2, man_bits=2, bias=1),
    # "fp6_e3m2": dict(exp_bits=3, man_bits=2, bias=3),
    # "fp7_e3m3": dict(exp_bits=3, man_bits=3, bias=3),
    "fp8_e4m3": dict(exp_bits=4, man_bits=3, bias=7),
    "fp8_e5m2": dict(exp_bits=5, man_bits=2, bias=15),
    "fp8_e3m4": dict(exp_bits=3, man_bits=4, bias=3),
    "fp8_e2m5": dict(exp_bits=2, man_bits=5, bias=1),
}
NATIVE = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}


# --- constants for FP8 dynamic ranges (finite; reserve all-ones exp) ---
def fp8_limits(exp_bits, man_bits, bias):
    exp_max = (1 << exp_bits) - 1
    # max finite exponent field used is exp_max-1
    emax_unbias = (exp_max - 1) - bias
    # min normal exponent field is 1 -> unbias = 1 - bias
    emin_unbias = 1 - bias
    # max finite value: (2 - 2^-m) * 2^emax
    x_max = (2.0 - 2.0 ** (-man_bits)) * (2.0**emax_unbias)
    # min normal value: 1.0 * 2^emin
    x_min_norm = 2.0**emin_unbias
    # min subnormal: (1 / 2^m) * 2^(1-bias)
    x_min_sub = (2.0 ** (-(man_bits))) * (2.0 ** (1 - bias))
    return x_min_sub, x_min_norm, x_max


# --- quantize/dequantize with: scaling, subnormals, mantissa carry, saturation ---
def quantize_fp8(x32, exp_bits, man_bits, bias, scale=None):
    x = x32.astype(np.float32, copy=False)
    sgn = np.signbit(x).astype(np.uint8)
    ax = np.abs(x)

    # choose a per-tensor scale if not provided (simple, robust)
    if scale is None:
        _, _, xmax = fp8_limits(exp_bits, man_bits, bias)
        max_ax = ax.max()
        scale = 1.0 if max_ax == 0 else max_ax / xmax

    y = ax / scale  # scale into FP8 dynamic range

    # frexp: y = m * 2^e, m in [0.5,1) for y>0; handle y==0 separately
    m, e = np.frexp(y)
    m = np.where(y == 0, 0.5, m)
    e = np.where(y == 0, 0, e)

    unbiased = e - 1  # because we’ll form (1.f) * 2^(e-1)
    exp_stored = unbiased + bias

    # fraction in [0,1): frac = m*2 - 1
    frac = m * 2.0 - 1.0
    mant_raw = np.round(frac * (1 << man_bits)).astype(np.int32)

    # --- mantissa carry: if rounding produced 2^m, bump exponent and zero mant ---
    carry = mant_raw >> man_bits  # 1 where mant_raw == 2^m else 0
    mant = mant_raw & ((1 << man_bits) - 1)
    exp_stored = exp_stored + carry

    exp_max = (1 << exp_bits) - 1

    # --- subnormals: exp_stored <= 0 -> encode with exp=0 and subnormal mant ---
    # subnormal value grid: val = (mant / 2^m) * 2^(1-bias)
    # so mant = round( y * 2^(man_bits - (1-bias)) ) when y < 2^(1-bias)
    sub = exp_stored <= 0
    if np.any(sub):
        # compute mant directly from y
        mant_sub = np.round(y * (2.0 ** (man_bits - (1 - bias)))).astype(np.int32)
        mant_sub = np.clip(mant_sub, 0, (1 << man_bits) - 1)
        # zero becomes exact zero
        mant_sub = np.where(y == 0, 0, mant_sub)
        exp_stored = np.where(sub, 0, exp_stored)
        mant = np.where(sub, mant_sub, mant)

    # --- overflow: exp_stored >= exp_max -> saturate to max finite (exp_max-1, mant all ones) ---
    over = exp_stored >= exp_max
    exp_stored = np.where(over, exp_max - 1, exp_stored)
    mant = np.where(over, (1 << man_bits) - 1, mant)

    code = (
        (sgn << (exp_bits + man_bits))
        | ((exp_stored.astype(np.uint8)) << man_bits)
        | mant.astype(np.uint8)
    )
    return code.astype(np.uint8), np.float32(scale)


def dequantize_fp8(code, exp_bits, man_bits, bias, scale):
    c = code.astype(np.uint8, copy=False)
    sgn = (c >> (exp_bits + man_bits)) & 0x1
    exp = (c >> man_bits) & ((1 << exp_bits) - 1)
    mant = c & ((1 << man_bits) - 1)

    is_zero = (exp == 0) & (mant == 0)
    is_sub = (exp == 0) & (mant > 0)

    # normals: m = 1 + mant/2^m, e_unbias = exp - bias
    frac = mant.astype(np.float32) / (1 << man_bits)
    m_norm = 1.0 + frac
    e_unbias = exp.astype(np.int32) - bias
    val_norm = np.ldexp(m_norm, e_unbias)

    # subnormals: val = (mant / 2^m) * 2^(1-bias)
    val_sub = (mant.astype(np.float32) / (1 << man_bits)) * (2.0 ** (1 - bias))

    val = np.where(is_zero, 0.0, np.where(is_sub, val_sub, val_norm)).astype(np.float32)
    val = np.where(sgn > 0, -val, val)
    return (val * scale.astype(np.float32)).astype(np.float32)


def _quantize_fp(
    x32: np.ndarray, exp_bits: int, man_bits: int, bias: int
) -> np.ndarray:
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
    code = (
        (sign << (exp_bits + man_bits))
        | (exp_stored.astype(np.uint8) << man_bits)
        | mant.astype(np.uint8)
    )
    return code.astype(np.uint8)


def _dequantize_fp(
    code: np.ndarray, exp_bits: int, man_bits: int, bias: int
) -> np.ndarray:
    c = code.astype(np.uint8, copy=False)
    sign = (c >> (exp_bits + man_bits)) & 0x1
    exp = (c >> man_bits) & ((1 << exp_bits) - 1)
    mant = c & ((1 << man_bits) - 1)
    is_zero = exp == 0
    frac = mant.astype(np.float32) / (1 << man_bits)
    m = 1.0 + frac
    e_unbias = exp.astype(np.int32) - bias
    val = np.where(is_zero, 0.0, np.ldexp(m, e_unbias)).astype(np.float32)
    val = np.where(sign > 0, -val, val)
    return val


# --- quantize/dequantize: always return/expect (array_or_code, scale) ---


def quantize_real(x: np.ndarray, fmt: str):
    if fmt in NATIVE:  # native path returns (array, scale=1.0)
        return (x.astype(NATIVE[fmt], copy=False), np.float32(1.0))
    spec = FMT[fmt]
    code, scale = quantize_fp8(x, **spec)  # from the FP8 impl I sent
    return (code, scale)


def dequantize_real(code_and_scale, fmt: str, out_dtype=np.float32) -> np.ndarray:
    if fmt in NATIVE:
        arr, _ = code_and_scale  # scale is 1.0
        return arr.astype(out_dtype, copy=False)
    spec = FMT[fmt]
    code, scale = code_and_scale
    return dequantize_fp8(code, **spec, scale=scale).astype(out_dtype, copy=False)


def quantize_any(x: np.ndarray, fmt: str):
    if np.iscomplexobj(x):
        re = quantize_real(x.real, fmt)  # (payload, scale)
        im = quantize_real(x.imag, fmt)
        return (re, im)  # ((re_payload, re_scale), (im_payload, im_scale))
    else:
        return quantize_real(x, fmt)


def dequantize_any(code, fmt: str, out_dtype):
    if isinstance(code, tuple) and len(code) == 2 and isinstance(code[0], tuple):
        re = dequantize_real(code[0], fmt, out_dtype)
        im = dequantize_real(code[1], fmt, out_dtype)
        return re + 1j * im
    return dequantize_real(code, fmt, out_dtype)


def matmul_mixed(A, fmtA, B, fmtB, compute_dtype=np.float32, out_dtype=None):
    Aq = quantize_any(A, fmtA)  # returns (payload, scale) or tuple-of-tuples
    Bq = quantize_any(B, fmtB)
    is_cplx = np.iscomplexobj(A) or np.iscomplexobj(B)
    if out_dtype is None:
        out_dtype = np.complex64 if is_cplx else np.float64
    Ad = dequantize_any(Aq, fmtA, compute_dtype)
    Bd = dequantize_any(Bq, fmtB, compute_dtype)
    return (Ad @ Bd).astype(out_dtype, copy=False)

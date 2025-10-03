import cupynumeric as np
from src.mixed_fps.simulate_quantization import matmul_mixed, quantize_any, dequantize_any, NATIVE


def _choose_target_dtype(fmt_or_dtype, is_complex):
    if isinstance(fmt_or_dtype, str) and fmt_or_dtype in NATIVE:
        base = NATIVE[fmt_or_dtype]
    elif hasattr(fmt_or_dtype, "kind"):
        base = fmt_or_dtype
    else:
        return None

    if is_complex:
        if base == np.float64:
            return np.complex128
        return np.complex64
    else:
        if base in (np.complex64, np.complex128):
            return np.float64 if base == np.complex128 else np.float32
        return base


def cast_matrix_and_vector_mixed(H, psi, fmt_or_dtype, compute_dtype=np.float16):
    """
    If fmt_or_dtype is a numpy/cupynumeric dtype or 'fp16/fp32/fp64' -> cast (respecting complexness).
    If it's a tiny-fp format string -> emulate quant/dequant, then cast to compute_dtype.
    """
    tgt = _choose_target_dtype(fmt_or_dtype, np.iscomplexobj(psi) or np.iscomplexobj(H))
    if tgt is not None:
        return H.astype(tgt, copy=False), psi.astype(tgt, copy=False)

    fmt = str(fmt_or_dtype)
    H_q = quantize_any(H, fmt)
    psi_q = quantize_any(psi, fmt)
    H_d = dequantize_any(H_q, fmt, compute_dtype)
    psi_d = dequantize_any(psi_q, fmt, compute_dtype)
    return H_d, psi_d


def expmv_hermitian(
    H, v, dt, work_dtype=np.float64, out_dtype=np.complex128, normalize=False
):
    """
    Stable expmv for Hermitian H: use eigh. If H is complex, solve in complex dtype.
    """
    H_is_cplx = np.iscomplexobj(H)
    if H_is_cplx and work_dtype in (np.float32, np.float64):
        work_dtype = np.complex128
    if not H_is_cplx and work_dtype in (np.complex64, np.complex128):
        work_dtype = np.float64

    Hh = H.astype(work_dtype, copy=False)
    vh = v.astype(out_dtype, copy=False)

    w, Q = np.linalg.eigh(Hh)
    phase = np.exp((-1j * dt) * w.astype(out_dtype))
    y = Q.conj().T @ vh
    y *= phase
    out = (Q @ y).astype(out_dtype, copy=False)

    if normalize:
        n = np.linalg.norm(out)
        if n != 0:
            out /= n
    return out


def lie_trotter_step(
    H_terms,
    psi,
    dt,
    fmt_list,
    step_idx,
    compute_dtype=np.float16,
    work_dtype=np.float32,
    out_dtype=np.complex64,
    verbose=True,
    normalize=False,
):
    """
    Apply exp(-i H_j dt) in sequence. Each term uses a format selected from fmt_list,
    rotating per (step_idx + j). Each format can be:
      - native dtype object (e.g., np.float16/32/64) or "fp16"/"fp32"/"fp64"
      - tiny emulated string: "fp4_e2m1", "fp8_e4m3", "fp8_e5m2", etc.
    """
    psi_new = psi
    if verbose:
        sel = fmt_list[(step_idx) % len(fmt_list)]
        name = sel.name if hasattr(sel, "name") else str(sel)
        print(f"Step {step_idx}, format start: {name}")

    for j, H in enumerate(H_terms):
        fmt = fmt_list[(step_idx + j) % len(fmt_list)]
        Hc, psic = cast_matrix_and_vector_mixed(
            H, psi_new, fmt, compute_dtype=compute_dtype
        )
        psi_new = expmv_hermitian(
            Hc,
            psic,
            dt,
            work_dtype=work_dtype,
            out_dtype=out_dtype,
            normalize=normalize,
        )
    return psi_new


def simulate_hamiltonian(
    H_terms,
    psi0,
    t,
    r,
    fmt_list,
    compute_dtype=np.float16,
    work_dtype=np.float32,
    out_dtype=np.complex64,
    verbose=True,
    normalize=False,
):
    """
    First-order Lie–Trotter: (∏_j e^{-i H_j Δt})^r ψ0
    """
    dt = t / r
    psi = psi0
    for step in range(r):
        psi = lie_trotter_step(
            H_terms,
            psi,
            dt,
            fmt_list,
            step,
            compute_dtype=compute_dtype,
            work_dtype=work_dtype,
            out_dtype=out_dtype,
            verbose=verbose,
            normalize=normalize,
        )
    return psi

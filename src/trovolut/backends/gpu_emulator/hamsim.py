try:
    import cupynumeric as np
except ImportError:
    import numpy as np

from trovolut.backends.gpu_emulator.mixed_fps.simulate_quantization import quantize_any, dequantize_any, NATIVE

BYTES_PER_ELT = {
    "fp4_e2m1": 0.5,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "fp16": 2.0,
    "fp32": 4.0,
    "fp64": 8.0,
}

# BYTES_PER_ELT = {
#     'fp8_e4m3': 0.375,
#     'fp8_e5m2': 0.25,
#     'fp8_e3m4': 0.5,
#     'fp8_e2m5': 0.625,
#     'fp16': 1.25,
#     'fp32': 2.875,
#     'fp64': 6.5,
# }


def _fmt_bpe(fmt_or_dtype) -> float:
    """
    Return bytes-per-element for a dtype or tiny-fp string.
    Always a numeric float.
    """
    # tiny/emulated formats by name
    if isinstance(fmt_or_dtype, str):
        return BYTES_PER_ELT.get(fmt_or_dtype.lower(), 4.0)

    # anything dtype-ish (np.float32, np.dtype('float32'), np.complex64, etc.)
    try:
        return float(np.dtype(fmt_or_dtype).itemsize)
    except Exception:
        # last resort default
        return 4.0


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


def _complex_of(s):
    return np.complex128 if s == np.float64 else np.complex64


def cast_matrix_and_vector_mixed(H, psi, fmt_or_dtype, compute_dtype=np.float32):
    """
    Return (Hc, psic) both cast to compute_dtype (or its complex counterpart).
    For tiny/emulated formats: quantize->dequantize into compute_dtype.
    For native dtypes / 'fp16/32/64': cast directly to compute_dtype.
    """
    is_cplx = np.iscomplexobj(psi) or np.iscomplexobj(H)
    tgt_compute = _complex_of(compute_dtype) if is_cplx else compute_dtype

    # Native dtype or 'fp16'/'fp32'/'fp64'
    if (isinstance(fmt_or_dtype, str) and fmt_or_dtype in NATIVE) or hasattr(fmt_or_dtype, "kind"):
        Hc = H.astype(tgt_compute, copy=False)
        psic = psi.astype(tgt_compute, copy=False)
        return Hc, psic

    # Emulated tiny formats
    fmt = str(fmt_or_dtype)
    H_q = quantize_any(H, fmt)
    psi_q = quantize_any(psi, fmt)
    H_d = dequantize_any(H_q, fmt, tgt_compute)
    psi_d = dequantize_any(psi_q, fmt, tgt_compute)
    return H_d, psi_d


def expmv_hermitian(H, v, dt, work_dtype=np.float64, out_dtype=np.complex128, normalize=False):
    """
    Stable expmv for Hermitian H using eigh.
    All *internal* math in work_dtype; cast to out_dtype only at the end.
    """
    H_is_cplx = np.iscomplexobj(H)

    # Ensure work_dtype matches (complex if H is complex)
    if H_is_cplx and work_dtype in (np.float32, np.float64):
        work_dtype = np.complex128
    if (not H_is_cplx) and work_dtype in (np.complex64, np.complex128):
        work_dtype = np.float64

    Hd = H.astype(work_dtype, copy=False)
    vd = v.astype(work_dtype if H_is_cplx else work_dtype, copy=False)

    w, Q = np.linalg.eigh(Hd)  # w: real (float), Q: work_dtype
    # compute phase in work precision
    phase = np.exp((-1j * dt) * w.astype(np.float64))  # exponent in high precision
    phase = phase.astype(np.complex128 if work_dtype == np.complex128 else np.complex64, copy=False)

    y = Q.conj().T @ vd
    y = y.astype(phase.dtype, copy=False) * phase
    out = Q @ y

    out = out.astype(out_dtype, copy=False)
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
        Hc, psic = cast_matrix_and_vector_mixed(H, psi_new, fmt, compute_dtype=compute_dtype)
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

    # --- memory accounting (approx bytes touched) ---
    bytes_used = 0

    for step in range(r):
        # count “active precision” for this step’s formats (rough proxy)
        for j, H in enumerate(H_terms):
            fmt = fmt_list[(step + j) % len(fmt_list)]
            bpe = _fmt_bpe(fmt)
            bytes_used += int(H.size * bpe)
        # include vector too (before/after)
        fmt0 = fmt_list[step % len(fmt_list)]
        bpe_vec = _fmt_bpe(fmt0)
        bytes_used += int(psi.size * bpe_vec)

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

        # after step vector in out dtype (very rough)
        bytes_used += psi.size * _fmt_bpe(out_dtype)

    mem_mb = bytes_used / (1024.0 * 1024.0)
    return psi, {"r": r, "fmts": fmt_list, "bytes_used": bytes_used, "mem_mb": mem_mb}


# do simulation without mixed precision, just directly on the full Hamiltonian
def lie_trotter_no_simulated_precision(
    H,
    psi,
    dt,
    normalize=False,
):
    """
    Apply exp(-i H dt) directly without mixed precision.
    All *internal* math in work_dtype; cast to out_dtype only at the end.
    """
    psi_new = psi
    for h in H:
        psi_new = np.linalg.expm(-1j * h * dt) @ psi_new
    return psi_new


def simulate_hamiltonian_no_simulated_precision(
    H,
    psi0,
    t,
    r,
    normalize=False,
):
    """
    Simulate dynamics under H without mixed precision.
    """
    dt = t / r
    psi = psi0

    for _ in range(r):
        # include Hamiltonian
        psi = lie_trotter_no_simulated_precision(
            H,
            psi,
            dt,
            normalize=normalize,
        )

    return psi, {
        "r": r,
    }

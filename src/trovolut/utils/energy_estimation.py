"""
Quantum Energy Level Estimation from Time-Evolved Wavefunctions

Methods:
  - FFT (Welch-style) peak picking on autocorrelation C(t)=<ψ(0)|ψ(t)>
  - ESPRIT subspace method (superresolution)

Uncertainty (optional, slower):
  - FFT: bootstrap across Welch segments (MAD of medians)
  - ESPRIT: block-bootstrap (keeps time order)

CRB (optional, cheap-ish):
  - Single-tone baseline CRB using amplitude estimated by projecting onto exp(-i E t)
    (much more stable than using a single FFT bin)

Conventions:
  C(t) = Σ a_n exp(-i E_n t), ħ=1
  => FFT peaks lie at NEGATIVE frequencies, energies are E = -ω_peak.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal


# =============================================================================
# Shared utilities
# =============================================================================

def _mad(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    return float(np.median(np.abs(a - med)))

def _hann(n: int) -> np.ndarray:
    if n < 2:
        return np.ones(n)
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))

def build_autocorr(psi_t: np.ndarray) -> np.ndarray:
    X = np.asarray(psi_t, dtype=np.complex128)
    if X.ndim == 1:
        return X
    psi0 = X[0]
    return np.einsum("ti,i->t", X, np.conj(psi0))

def estimate_noise_variance(C: np.ndarray) -> float:
    C = np.asarray(C)
    if C.size < 5:
        return 1e-10
    d2 = np.diff(C, n=2)
    sigma_d2 = 1.4826 * _mad(np.abs(d2))
    return max((sigma_d2 / np.sqrt(6)) ** 2, 1e-20)

def _refine_peak_quadratic(P: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(P) - 1:
        return float(k)
    lp = np.log(P[k - 1] + 1e-30)
    lc = np.log(P[k] + 1e-30)
    ln = np.log(P[k + 1] + 1e-30)
    denom = lp - 2 * lc + ln
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (lp - ln) / denom
    return k + np.clip(delta, -0.5, 0.5)

def _robust_peaks(P: np.ndarray,
                  min_prom_frac: float = 0.02,
                  k_fallback: int = 12,
                  low_bin_skip: int = 0,
                  baseline_win: int = 31) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    n = P.size
    if n < 3:
        return np.array([], dtype=int)

    # light smoothing
    w = 5
    Ps = np.convolve(P, np.ones(w) / w, mode="same") if n >= w else P.copy()

    # sliding-min baseline
    bw = max(3, int(baseline_win) | 1)
    half = bw // 2
    baseline = np.empty_like(Ps)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        baseline[i] = np.min(Ps[a:b])

    mids = np.arange(1, n - 1)
    is_max = (Ps[mids] >= Ps[mids - 1]) & (Ps[mids] >= Ps[mids + 1])
    idx = mids[is_max]

    if idx.size:
        prom = Ps[idx] - baseline[idx]
        thr = float(min_prom_frac * np.max(Ps))
        idx = idx[prom >= thr]

    idx = idx[idx >= int(low_bin_skip)]

    if idx.size == 0:
        lo = int(low_bin_skip)
        if lo >= n:
            return np.array([], dtype=int)
        cand = np.arange(lo, n)
        k = min(int(k_fallback), cand.size)
        top = cand[np.argpartition(Ps[cand], -k)[-k:]]
        idx = top[np.argsort(Ps[top])[::-1]]
    else:
        idx = idx[np.argsort(Ps[idx])[::-1]]

    return idx


# =============================================================================
# CRB (single-tone baseline) using projection amplitude (stable)
# =============================================================================

@dataclass
class CramerRaoBound:
    variance: np.ndarray
    std: np.ndarray
    snr: np.ndarray
    n_samples: int
    dt: float
    noise_variance: float

def cramer_rao_bound(
    C: np.ndarray,
    dt: float,
    mode_amplitudes: np.ndarray,
    noise_variance: float | None = None,
) -> CramerRaoBound:
    C = np.asarray(C)
    N = C.size
    mode_amplitudes = np.atleast_1d(mode_amplitudes).astype(float)
    if noise_variance is None:
        noise_variance = estimate_noise_variance(C)

    amp_sq = np.maximum(mode_amplitudes, 1e-20)
    crb_var = 12 * noise_variance / (N ** 3 * amp_sq * dt ** 2)
    snr = amp_sq / noise_variance

    return CramerRaoBound(
        variance=crb_var,
        std=np.sqrt(crb_var),
        snr=snr,
        n_samples=N,
        dt=dt,
        noise_variance=noise_variance,
    )

def crb_from_projection(
    C: np.ndarray,
    dt: float,
    energies: np.ndarray,
    noise_variance: float | None = None,
) -> CramerRaoBound:
    """
    Baseline CRB where |a| is estimated by LS projection onto exp(-i E t).
    Much more stable than using a single FFT bin.
    """
    C = np.asarray(C, dtype=np.complex128)
    energies = np.atleast_1d(energies).astype(float)
    N = C.size
    t = dt * np.arange(N)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(C)

    amps = np.empty(len(energies), dtype=float)
    for i, E in enumerate(energies):
        v = np.exp(-1j * E * t)
        a_hat = np.vdot(v, C) / np.vdot(v, v)   # scalar LS
        amps[i] = float(np.abs(a_hat) ** 2)

    return cramer_rao_bound(C, dt, amps, noise_variance=noise_variance)


# =============================================================================
# FFT (Welch) energies
# =============================================================================

@dataclass
class FFTResult:
    energies: np.ndarray
    uncertainty: np.ndarray | None
    cramer_rao: CramerRaoBound | None
    autocorr: np.ndarray

def fft_energy_levels(
    psi_t: np.ndarray,
    dt: float,
    n_levels: int = 5,
    segment_len: int | None = None,
    overlap: float = 0.5,
    min_prom_frac: float = 0.02,
    zero_pad_factor: int = 4,
    refine_peaks: bool = True,
    compute_errors: bool = False,
    n_bootstrap: int = 200,
    compute_crb: bool = False,
    noise_variance: float | None = None,
    seed: int = 0,
) -> FFTResult:
    """
    Fast by default; set compute_errors / compute_crb True when needed.

    For C(t)=Σ a exp(-iEt), FFT peaks are at negative freqs -> E=-ω.
    """
    C = build_autocorr(psi_t)
    T = C.size
    if T < 8:
        raise ValueError("Need at least 8 time points")

    if segment_len is None:
        segment_len = max(64, min(T, T // 4))
    segment_len = int(np.clip(segment_len, 8, T))

    hop = max(1, int(segment_len * (1.0 - overlap)))
    starts = np.arange(0, T - segment_len + 1, hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    w = _hann(segment_len)
    n_fft = int(segment_len * zero_pad_factor)

    seg_peaks_E: list[np.ndarray] = []

    for s in starts:
        seg = C[s:s + segment_len]
        seg = seg - np.mean(seg)
        X = np.fft.fft(seg * w, n=n_fft)
        P_full = np.abs(X) ** 2

        freqs_full = np.fft.fftfreq(n_fft, d=dt)
        omega_full = 2 * np.pi * freqs_full

        neg = freqs_full < 0
        P = P_full[neg]
        omega = omega_full[neg]  # negative

        # drop the bin closest to 0 (last one on the neg side)
        if P.size > 2:
            P = P[:-1]
            omega = omega[:-1]

        pk = _robust_peaks(
            P,
            min_prom_frac=min_prom_frac,
            k_fallback=max(n_levels, 10),
            low_bin_skip=0,
            baseline_win=31,
        )

        if pk.size == 0:
            seg_peaks_E.append(np.array([], dtype=float))
            continue

        if refine_peaks:
            pk_ref = np.array([_refine_peak_quadratic(P, int(k)) for k in pk])
            omega_pk = np.interp(pk_ref, np.arange(len(omega)), omega)
        else:
            omega_pk = omega[pk]

        E_pk = -omega_pk
        seg_peaks_E.append(E_pk[:max(n_levels, 10)])

    if all(pe.size == 0 for pe in seg_peaks_E):
        raise RuntimeError("No spectral peaks found")

    pooled = np.concatenate([pe for pe in seg_peaks_E if pe.size])
    pooled.sort()

    # physical resolution ~ 2π/(segment_len*dt)
    dE = 2 * np.pi / (segment_len * dt)
    tol = max(2.0 * dE, 1e-12)
    min_support = max(2, int(0.3 * len(starts)))

    modes: list[np.ndarray] = []
    used = []

    for center in pooled:
        if len(modes) >= n_levels:
            break
        if any(abs(center - c) <= 1.5 * tol for c in used):
            continue

        picks = []
        for pe in seg_peaks_E:
            if pe.size == 0:
                continue
            j = int(np.argmin(np.abs(pe - center)))
            if abs(pe[j] - center) <= tol:
                picks.append(pe[j])

        if len(picks) >= min_support:
            picks = np.asarray(picks, dtype=float)
            modes.append(picks)
            used.append(float(np.median(picks)))

    if len(modes) == 0:
        # fallback: strongest peak from full signal
        n_fft_full = int(T * zero_pad_factor)
        X = np.fft.fft((C - np.mean(C)) * _hann(T), n=n_fft_full)
        P_full = np.abs(X) ** 2
        freqs = np.fft.fftfreq(n_fft_full, d=dt)
        omega = 2 * np.pi * freqs
        neg = freqs < 0
        Pn = P_full[neg]
        on = omega[neg]
        if Pn.size > 2:
            Pn = Pn[:-1]
            on = on[:-1]
        k = int(np.argmax(Pn))
        E0 = float(-on[k])
        return FFTResult(
            energies=np.array([E0]),
            uncertainty=None,
            cramer_rao=None,
            autocorr=C,
        )

    E_hat = np.array([np.median(m) for m in modes], dtype=float)
    E_hat.sort()
    E_hat = E_hat[:n_levels]

    unc = None
    if compute_errors:
        rng = np.random.default_rng(seed)
        unc_list = []
        for m in modes[:len(E_hat)]:
            if m.size < 3:
                unc_list.append(np.nan)
                continue
            meds = np.array([np.median(rng.choice(m, size=m.size, replace=True)) for _ in range(n_bootstrap)])
            sigma = 1.4826 * _mad(meds)
            if not np.isfinite(sigma) or sigma == 0.0:
                sigma = float(np.std(meds, ddof=1))
            unc_list.append(sigma)
        unc = np.array(unc_list, dtype=float)

    crb = None
    if compute_crb and E_hat.size:
        crb = crb_from_projection(C, dt, E_hat, noise_variance=noise_variance)

    # If clustering misses weaker peaks, fill using full-length FFT peaks
    if E_hat.size < n_levels:
        n_fft_full = int(T * zero_pad_factor)
        X = np.fft.fft((C - np.mean(C)) * _hann(T), n=n_fft_full)
        P_full = np.abs(X) ** 2
        freqs_full = np.fft.fftfreq(n_fft_full, d=dt)
        omega_full = 2 * np.pi * freqs_full

        neg = freqs_full < 0
        P = P_full[neg]
        omega = omega_full[neg]
        if P.size > 2:
            P = P[:-1]
            omega = omega[:-1]

        pk = _robust_peaks(P, min_prom_frac=min_prom_frac,
                        k_fallback=max(10, 4*n_levels),
                        low_bin_skip=0, baseline_win=31)
        omega_pk = omega[pk]
        E_pk = -omega_pk
        # remove near-duplicates of existing E_hat
        out = list(E_hat)
        for e in E_pk:
            if len(out) >= n_levels:
                break
            if all(abs(e - ee) > 2*(2*np.pi/(T*dt)) for ee in out):
                out.append(float(e))
        E_hat = np.sort(np.array(out, dtype=float))[:n_levels]

    return FFTResult(energies=E_hat, uncertainty=unc, cramer_rao=crb, autocorr=C)


# =============================================================================
# ESPRIT
# =============================================================================

@dataclass
class ESPRITResult:
    energies: np.ndarray
    uncertainty: np.ndarray | None
    singular_values: np.ndarray
    model_order: int
    cramer_rao: CramerRaoBound | None
    autocorr: np.ndarray

def esprit_energy_levels(
    psi_t: np.ndarray,
    dt: float,
    n_levels: int = 5,
    pencil_param: int | None = None,
    model_order: int | None = None,
    sv_threshold: float = 0.01,
    compute_errors: bool = False,
    n_bootstrap: int = 200,
    block_len: int | None = None,
    compute_crb: bool = False,
    noise_variance: float | None = None,
    seed: int = 0,
) -> ESPRITResult:
    """
    Optional uncertainty uses a BLOCK bootstrap to preserve ordering.
    """
    C = build_autocorr(psi_t)
    T = C.size
    if T < 10:
        raise ValueError("Need at least 10 time points for ESPRIT")

    if pencil_param is None:
        pencil_param = max(10, T // 3)
    L = int(np.clip(pencil_param, 5, T - 5))

    H = np.lib.stride_tricks.sliding_window_view(C, L)[:T - L + 1]
    U, s, _ = np.linalg.svd(H, full_matrices=False)

    if model_order is None:
        r = int(np.sum((s / s[0]) > sv_threshold))
        r = max(1, min(r, n_levels, len(s) - 1))
    else:
        r = int(np.clip(model_order, 1, min(n_levels, len(s) - 1)))

    Us = U[:, :r]
    Phi, *_ = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)
    eigs = np.linalg.eigvals(Phi)

    omegas = -np.angle(eigs) / dt
    damp = np.abs(eigs)
    mask = (omegas > 0) & (damp > 0.1) & (damp < 10)
    omegas = np.sort(omegas[mask])[:n_levels]

    if omegas.size == 0:
        return ESPRITResult(
            energies=np.array([]),
            uncertainty=None,
            singular_values=s,
            model_order=r,
            cramer_rao=None,
            autocorr=C,
        )

    unc = None
    if compute_errors:
        rng = np.random.default_rng(seed)
        # moving block bootstrap
        if block_len is None:
            block_len = max(8, T // 10)
        block_len = int(np.clip(block_len, 4, T))
        n_blocks = int(np.ceil(T / block_len))

        E0s = []
        for _ in range(n_bootstrap):
            starts = rng.integers(0, T - block_len + 1, size=n_blocks)
            Cb = np.concatenate([C[s0:s0 + block_len] for s0 in starts], axis=0)[:T]

            Hb = np.lib.stride_tricks.sliding_window_view(Cb, L)[:T - L + 1]
            Ub, sb, _ = np.linalg.svd(Hb, full_matrices=False)
            Usb = Ub[:, :r]
            Phib, *_ = np.linalg.lstsq(Usb[:-1], Usb[1:], rcond=None)
            eigb = np.linalg.eigvals(Phib)

            omb = -np.angle(eigb) / dt
            dab = np.abs(eigb)
            mb = (omb > 0) & (dab > 0.1) & (dab < 10)
            omb = np.sort(omb[mb])
            if omb.size:
                E0s.append(float(omb[0]))

        if len(E0s) >= 20:
            E0s = np.asarray(E0s, dtype=float)
            sigma0 = 1.4826 * _mad(E0s)
        else:
            sigma0 = np.nan

        unc = np.full_like(omegas, np.nan, dtype=float)
        unc[0] = sigma0

    crb = None
    if compute_crb:
        crb = crb_from_projection(C, dt, omegas, noise_variance=noise_variance)

    return ESPRITResult(
        energies=omegas,
        uncertainty=unc,
        singular_values=s,
        model_order=r,
        cramer_rao=crb,
        autocorr=C,
    )


def estimate_spectrum(
    psi_t: np.ndarray,
    dt: float,
    method: Literal["fft", "esprit"] = "esprit",
    n_levels: int = 5,
    compute_uncertainties: bool = False,
    compute_crb: bool = False,
    noise_variance: float | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray | None, CramerRaoBound | None, dict]:
    """
    Estimate multiple energy levels (the spectrum).

    Returns:
        energies (sorted, length <= n_levels),
        sigma (same shape or None),
        crb (dataclass or None),
        info (dict for diagnostics)
    """
    if method == "fft":
        # Filter to FFT-specific kwargs only
        fft_keys = {"segment_len", "overlap", "min_prom_frac", "zero_pad_factor", "refine_peaks", "n_bootstrap", "seed"}
        fft_kwargs = {k: v for k, v in kwargs.items() if k in fft_keys}
        res = fft_energy_levels(
            psi_t, dt,
            n_levels=n_levels,
            compute_errors=compute_uncertainties,
            compute_crb=compute_crb,
            noise_variance=noise_variance,
            **fft_kwargs,
        )
        return res.energies, res.uncertainty, res.cramer_rao, res.__dict__

    if method == "esprit":
        # Filter to ESPRIT-specific kwargs only
        esprit_keys = {"pencil_param", "model_order", "sv_threshold", "n_bootstrap", "block_len", "seed"}
        esprit_kwargs = {k: v for k, v in kwargs.items() if k in esprit_keys}
        res = esprit_energy_levels(
            psi_t, dt,
            n_levels=n_levels,
            compute_errors=compute_uncertainties,
            compute_crb=compute_crb,
            noise_variance=noise_variance,
            **esprit_kwargs,
        )
        return res.energies, res.uncertainty, res.cramer_rao, res.__dict__

    raise ValueError("method must be one of: 'fft', 'esprit'")


def estimate_ground_state_energy_fast(
    psi_t: np.ndarray,
    dt: float,
    method: Literal["esprit", "fft"] = "esprit",
    **kwargs,
) -> float:
    """
    Ultra-fast ground state only. No sigma, no CRB, minimal allocations.
    """
    if method == "esprit":
        res = esprit_energy_levels(
            psi_t, dt,
            n_levels=1,
            pencil_param=kwargs.get("pencil_param", None),
            model_order=kwargs.get("model_order", None),
            sv_threshold=kwargs.get("sv_threshold", 0.01),
            compute_errors=False,
            compute_crb=False,
        )
        return float(res.energies[0]) if res.energies.size else np.nan

    if method == "fft":
        # fastest FFT GS: one full FFT, one peak pick (no Welch/bootstraps)
        C = build_autocorr(psi_t)
        T = C.size
        n_fft = int(kwargs.get("zero_pad_factor", 4) * T)
        X = np.fft.fft((C - np.mean(C)) * _hann(T), n=n_fft)
        P = np.abs(X) ** 2
        freqs = np.fft.fftfreq(n_fft, d=dt)
        omega = 2 * np.pi * freqs
        neg = freqs < 0
        Pn = P[neg]
        on = omega[neg]
        if Pn.size > 2:
            Pn = Pn[:-1]
            on = on[:-1]
        k = int(np.argmax(Pn))
        return float(-on[k])

    raise ValueError("method must be one of: 'esprit' or 'fft'")


def estimate_ground_state_energy(
    psi_t: np.ndarray,
    dt: float,
    method: Literal["fft", "esprit"] = "esprit",
    compute_uncertainty: bool = False,
    compute_crb: bool = False,
    noise_variance: float | None = None,
    **kwargs,
) -> tuple[float, float | None, float | None, dict]:
    """
    Ground-state API that can optionally compute sigma/CRB,
    while internally using the spectrum estimators.

    Returns:
        E0, sigma0 (or None), crb0 (or None), info
    """
    energies, sig, crb, info = estimate_spectrum(
        psi_t, dt,
        method=("fft" if method == "fft" else "esprit"),
        n_levels=max(1, int(kwargs.pop("n_levels", 3))),
        compute_uncertainties=compute_uncertainty,
        compute_crb=compute_crb,
        noise_variance=noise_variance,
        **kwargs,
    )
    E0 = float(energies[0]) if energies.size else np.nan
    sigma0 = float(sig[0]) if (sig is not None and sig.size) else None
    crb0 = float(crb.std[0]) if (crb is not None and crb.std.size) else None
    return E0, sigma0, crb0, info

def main(compute_uncertainties: bool = True):
    """
    Benchmark spectrum estimators + show ultra-fast GS-only calls.

    Args:
        compute_uncertainties:
            False → no bootstrap σ, no CRB (fast)
            True  → compute σ and CRB where supported (slower)
    """
    import time
    np.random.seed(42)

    # --- synthetic test system ---
    E_true = np.array([1.0, 2.5, 4.2])
    amps_true = np.array([0.6, 0.3, 0.1])

    T = 500
    dt = 0.05
    t = dt * np.arange(T)

    C_clean = sum(a * np.exp(-1j * E * t) for a, E in zip(amps_true, E_true))
    noise_level = 0.02
    C = C_clean + noise_level * (np.random.randn(T) + 1j * np.random.randn(T))

    n_levels = 3
    noise_var = (noise_level**2) if compute_uncertainties else None

    print("=" * 110)
    mode = "WITH σ + CRB" if compute_uncertainties else "NO σ / NO CRB (fast)"
    print(f"Spectrum / Ground-State Benchmark — {mode}")
    print("=" * 110)
    print(f"True energies: {E_true}")
    print(f"T={T}, dt={dt}, noise σ={noise_level}")
    print()

    # -------------------------
    # Spectrum estimators
    # -------------------------
    spectrum_methods = ["esprit", "fft"]

    print("-" * 120)
    print(f"{'Spectrum method':<16} {'E0':>12} {'Sigma0':>12} {'CRB0':>12} {'σ/CRB':>10} {'Time (ms)':>12}   Energies")
    print("-" * 120)

    for method in spectrum_methods:
        kwargs = {}
        if method == "fft":
            kwargs.update(dict(
                zero_pad_factor=4,
                min_prom_frac=0.02,
                n_bootstrap=300,   # used only if compute_uncertainties=True
            ))
        elif method == "esprit":
            kwargs.update(dict(
                sv_threshold=0.01,
                n_bootstrap=200,   # used only if compute_uncertainties=True
            ))

        t0 = time.perf_counter()
        energies, sig, crb, info = estimate_spectrum(
            C, dt,
            method=method,
            n_levels=n_levels,
            compute_uncertainties=compute_uncertainties,
            compute_crb=compute_uncertainties,
            noise_variance=noise_var,
            **kwargs,
        )
        ms = (time.perf_counter() - t0) * 1e3

        E0 = float(energies[0]) if energies.size else np.nan
        sigma0 = float(sig[0]) if (sig is not None and sig.size and np.isfinite(sig[0])) else None
        crb0 = float(crb.std[0]) if (crb is not None and crb.std.size and np.isfinite(crb.std[0])) else None

        sig_str = f"{sigma0:.6g}" if sigma0 is not None else "None"
        crb_str = f"{crb0:.6g}" if crb0 is not None else "None"
        eff_str = "None"
        if sigma0 is not None and crb0 is not None and crb0 > 0:
            eff_str = f"{(sigma0 / crb0):.2f}"

        e_show = np.array2string(np.asarray(energies)[:n_levels], precision=6)
        print(f"{method:<16} {E0:>12.6f} {sig_str:>12} {crb_str:>12} {eff_str:>10} {ms:>12.2f}   {e_show}")

    print("-" * 120)
    print(f"{'True E0':<16} {E_true[0]:>12.6f}")
    print()

    if compute_uncertainties:
        crb_known = cramer_rao_bound(C, dt, np.array([amps_true[0]]), noise_variance=noise_level**2)
        print("Known-noise single-tone CRB baseline for E0 (using true a0):")
        print(f"  CRB std(E0) ≈ {crb_known.std[0]:.6g}")
        print()

    # -------------------------
    # Ultra-fast GS-only calls
    # -------------------------
    print("=" * 110)
    print("Ultra-fast GS-only calls (no σ, no CRB)")
    print("=" * 110)
    gs_methods = ["esprit", "fft"]

    print("-" * 80)
    print(f"{'GS method':<12} {'E0':>12} {'Time (ms)':>12}")
    print("-" * 80)

    for m in gs_methods:
        t0 = time.perf_counter()
        E0 = estimate_ground_state_energy_fast(
            C, dt,
            method=m,
            sv_threshold=0.01,
            zero_pad_factor=4,
        )
        ms = (time.perf_counter() - t0) * 1e3
        print(f"{m:<12} {E0:>12.6f} {ms:>12.2f}")

    print("-" * 80)
    print()

    # Also show the "GS API" that can compute σ/CRB if requested
    if compute_uncertainties:
        print("=" * 110)
        print("Ground-state API (optionally σ + CRB)")
        print("=" * 110)
        for m in ["esprit", "fft"]:
            t0 = time.perf_counter()
            E0, s0, crb0, _ = estimate_ground_state_energy(
                C, dt,
                method=m,
                compute_uncertainty=True,
                compute_crb=True,
                noise_variance=noise_level**2,
                n_levels=n_levels,
                sv_threshold=0.01,
                zero_pad_factor=4,
                n_bootstrap=200 if m == "esprit" else 300,
            )
            ms = (time.perf_counter() - t0) * 1e3
            print(f"{m:<8} E0={E0:.6f}  sigma0={s0 if s0 is not None else None}  crb0={crb0 if crb0 is not None else None}  time_ms={ms:.2f}")
        print()


if __name__ == "__main__":
    main(compute_uncertainties=False)   # fast
    # main(compute_uncertainties=True)  # slower, with σ + CRB
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
    """Return autocorrelation C(t)=<psi(0)|psi(t)>.

    If `psi_t` is (T, dim), we compute <psi0|psi(t)> and normalize so C(0)=1
    (when possible). This makes FFT peak picking much more stable when norms drift.

    If `psi_t` is already (T,), it's assumed to be an autocorrelation-like signal
    and we normalize similarly.
    """
    X = np.asarray(psi_t, dtype=np.complex128)

    # If user already passed an autocorrelation-like 1D signal
    if X.ndim == 1:
        C = X
        if C.size:
            c0 = C[0]
            if np.abs(c0) > 0:
                C = C / c0
        return C

    # Otherwise compute from time series of states
    psi0 = X[0]
    n0 = np.vdot(psi0, psi0)
    if np.abs(n0) == 0:
        raise ValueError("psi_t[0] has zero norm; cannot build autocorrelation")

    # <psi(0)|psi(t)>
    C = np.einsum("ti,i->t", X, np.conj(psi0))

    # Normalize by ||psi0||^2 so C(0)=1 (assuming psi(t) is normalized similarly)
    C = C / n0

    # If norms drift significantly, additionally normalize by ||psi(t)||
    # so C remains bounded (helps peak finding when the simulation is noisy).
    norms_t = np.sqrt(np.einsum("ti,ti->t", X.conj(), X).real)
    n0s = float(np.sqrt(n0.real))
    denom = np.maximum(norms_t * n0s, 1e-30)
    C = C / denom

    # Force exact C(0)=1 when possible
    if C.size and np.abs(C[0]) > 0:
        C = C / C[0]

    return C

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
class FFTSpectrum:
    """Raw FFT spectrum data for reuse across functions."""
    energies: np.ndarray      # E = -omega (sorted, all positive values)
    power: np.ndarray         # Power spectrum at those energies
    autocorr: np.ndarray      # Autocorrelation C(t)
    dt: float
    n_fft: int

def _compute_fft_spectrum(
    psi_t: np.ndarray,
    dt: float,
    zero_pad_factor: int = 4,
    detrend: Literal["none", "mean"] = "none",
) -> FFTSpectrum:
    """
    Compute raw FFT spectrum from time-evolved wavefunctions.
    
    Returns energies and power spectrum for visualization or peak extraction.
    For C(t)=Σ a exp(-iEt), FFT peaks are at negative freqs -> E=-ω.
    """
    C = build_autocorr(psi_t)
    T = C.size
    if T < 8:
        raise ValueError("Need at least 8 time points")

    n_fft = int(T * zero_pad_factor)
    w = _hann(T)
    
    if detrend == "mean":
        C_use = C - np.mean(C)
    elif detrend == "none":
        C_use = C
    else:
        raise ValueError("detrend must be 'none' or 'mean'")

    X = np.fft.fft(C_use * w, n=n_fft)
    P_full = np.abs(X) ** 2

    freqs_full = np.fft.fftfreq(n_fft, d=dt)
    omega_full = 2 * np.pi * freqs_full

    # Use negative frequencies (where energies appear for C(t) = Σ a exp(-iEt))
    neg = freqs_full < 0
    P = P_full[neg]
    omega = omega_full[neg]

    # Drop bin closest to 0
    if P.size > 2:
        P = P[:-1]
        omega = omega[:-1]

    # Convert to energies and sort
    E = -omega
    sort_idx = np.argsort(E)
    E = E[sort_idx]
    P = P[sort_idx]

    return FFTSpectrum(energies=E, power=P, autocorr=C, dt=dt, n_fft=n_fft)


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
    min_prom_frac: float = 0.01,
    zero_pad_factor: int = 4,
    refine_peaks: bool = True,
    detrend: Literal["none", "mean"] = "none",
    compute_errors: bool = False,
    n_bootstrap: int = 200,
    compute_crb: bool = False,
    noise_variance: float | None = None,
    seed: int = 0,
) -> FFTResult:
    """
    FFT-based energy level estimation using full-signal FFT.

    For C(t)=Σ a exp(-iEt), FFT peaks are at negative freqs -> E=-ω.
    """
    # Use shared FFT computation
    spec = _compute_fft_spectrum(psi_t, dt, zero_pad_factor=zero_pad_factor, detrend=detrend)
    C = spec.autocorr
    T = C.size
    
    # For peak finding, we need to work in the original (unsorted) order
    # Recompute P and omega for peak finding with normalization
    n_fft = spec.n_fft
    w = _hann(T)
    C_use = C - np.mean(C) if detrend == "mean" else C
    X = np.fft.fft(C_use * w, n=n_fft)
    P_full = np.abs(X) ** 2
    freqs_full = np.fft.fftfreq(n_fft, d=dt)
    omega_full = 2 * np.pi * freqs_full
    
    neg = freqs_full < 0
    P = P_full[neg]
    omega = omega_full[neg]
    
    # Normalize to tame dynamic range and make the peak threshold less brittle
    med = float(np.median(P)) if P.size else 1.0
    if med > 0:
        P = P / med

    if P.size > 2:
        P = P[:-1]
        omega = omega[:-1]

    # Find peaks
    pk = _robust_peaks(
        P,
        min_prom_frac=min_prom_frac,
        k_fallback=max(n_levels * 4, 20),
        low_bin_skip=0,
        baseline_win=31,
    )

    if pk.size == 0:
        raise RuntimeError("No spectral peaks found")

    # Refine peak positions with quadratic interpolation
    if refine_peaks:
        pk_ref = np.array([_refine_peak_quadratic(P, int(k)) for k in pk])
        omega_pk = np.interp(pk_ref, np.arange(len(omega)), omega)
    else:
        omega_pk = omega[pk]

    E_hat = -omega_pk  # E = -ω for negative frequency peaks
    E_hat = np.sort(E_hat[E_hat > 0])[:n_levels]  # keep positive energies, sorted

    if E_hat.size == 0:
        raise RuntimeError("No positive energy peaks found")

    # Uncertainty estimation via bootstrap (resample the signal with noise)
    unc = None
    if compute_errors:
        rng = np.random.default_rng(seed)
        # For each energy level, collect bootstrap estimates
        boot_energies = [[] for _ in range(E_hat.size)]
        
        for _ in range(n_bootstrap):
            # Perturbation scale tied to estimated noise on C (more stable than |C| std)
            noise_var_hat = estimate_noise_variance(C)
            noise_scale = float(np.sqrt(noise_var_hat))
            noise = noise_scale * (rng.standard_normal(T) + 1j * rng.standard_normal(T))
            Cb = C + noise
            if detrend == "mean":
                Cb_use = Cb - np.mean(Cb)
            else:
                Cb_use = Cb
            Xb = np.fft.fft(Cb_use * w, n=n_fft)
            Pb = np.abs(Xb) ** 2
            Pb_neg = Pb[neg]
            if Pb_neg.size > 2:
                Pb_neg = Pb_neg[:-1]

            # Match the same normalization used for the main spectrum
            med_b = float(np.median(Pb_neg)) if Pb_neg.size else 1.0
            if med_b > 0:
                Pb_neg = Pb_neg / med_b
            
            # Find peaks in bootstrap spectrum
            try:
                pk_b = _robust_peaks(Pb_neg, min_prom_frac=min_prom_frac, 
                                     k_fallback=max(n_levels * 4, 20), 
                                     low_bin_skip=0, baseline_win=31)
                if pk_b.size == 0:
                    continue
                    
                # Refine and convert to energies
                if refine_peaks:
                    pk_ref_b = np.array([_refine_peak_quadratic(Pb_neg, int(k)) for k in pk_b])
                    omega_pk_b = np.interp(pk_ref_b, np.arange(len(omega)), omega)
                else:
                    omega_pk_b = omega[pk_b]
                    
                E_boot = -omega_pk_b
                E_boot = np.sort(E_boot[E_boot > 0])[:n_levels]
                
                # Match bootstrap energies to original energies (nearest neighbor)
                for i, E_orig in enumerate(E_hat):
                    if E_boot.size > 0:
                        idx = np.argmin(np.abs(E_boot - E_orig))
                        boot_energies[i].append(E_boot[idx])
            except Exception:
                continue
        
        # Compute robust std for each energy level
        unc = np.zeros(E_hat.size, dtype=float)
        for i in range(E_hat.size):
            if len(boot_energies[i]) > 10:
                unc[i] = 1.4826 * _mad(np.array(boot_energies[i]))
            else:
                unc[i] = np.nan  # Not enough bootstrap samples

    crb = None
    if compute_crb and E_hat.size:
        crb = crb_from_projection(C, dt, E_hat, noise_variance=noise_variance)

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
    compute_crb: bool = False,
    noise_variance: float | None = None,
    seed: int = 0,
) -> ESPRITResult:
    """
    ESPRIT algorithm for superresolution energy estimation.
    
    Optional uncertainty uses noise-perturbation bootstrap.
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
        # Noise-perturbation bootstrap (NOT block bootstrap which destroys phase)
        # Add small noise to the autocorrelation and re-estimate
        noise_scale = np.std(np.abs(C)) * 0.1  # 10% noise perturbation
        
        # Track all energy levels across bootstraps
        boot_energies = [[] for _ in range(omegas.size)]
        
        for _ in range(n_bootstrap):
            # Add complex noise perturbation
            noise = noise_scale * (rng.standard_normal(T) + 1j * rng.standard_normal(T))
            Cb = C + noise

            Hb = np.lib.stride_tricks.sliding_window_view(Cb, L)[:T - L + 1]
            try:
                Ub, sb, _ = np.linalg.svd(Hb, full_matrices=False)
                Usb = Ub[:, :r]
                Phib, *_ = np.linalg.lstsq(Usb[:-1], Usb[1:], rcond=None)
                eigb = np.linalg.eigvals(Phib)

                omb = -np.angle(eigb) / dt
                dab = np.abs(eigb)
                mb = (omb > 0) & (dab > 0.1) & (dab < 10)
                E_boot = np.sort(omb[mb])
                
                if E_boot.size == 0:
                    continue
                    
                # Match bootstrap energies to original energies
                for i, E_orig in enumerate(omegas):
                    idx = np.argmin(np.abs(E_boot - E_orig))
                    boot_energies[i].append(E_boot[idx])
            except Exception:
                continue

        # Compute robust std for each energy level
        unc = np.zeros(omegas.size, dtype=float)
        for i in range(omegas.size):
            if len(boot_energies[i]) >= 20:
                unc[i] = 1.4826 * _mad(np.array(boot_energies[i]))
            else:
                unc[i] = np.nan

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
        fft_keys = {"min_prom_frac", "zero_pad_factor", "refine_peaks", "detrend", "n_bootstrap", "seed"}
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
        esprit_keys = {"pencil_param", "model_order", "sv_threshold", "n_bootstrap", "seed"}
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
    
    Note: For ESPRIT, we extract multiple levels and return the minimum,
    since n_levels=1 would just return the strongest component (by amplitude),
    not necessarily the lowest energy (ground state).
    """
    if method == "esprit":
        # Extract several levels (default ~10), then pick the minimum energy
        n_levels = kwargs.get("n_levels", 10)
        res = esprit_energy_levels(
            psi_t, dt,
            n_levels=n_levels,
            pencil_param=kwargs.get("pencil_param", None),
            model_order=kwargs.get("model_order", None),
            sv_threshold=kwargs.get("sv_threshold", 0.01),
            compute_errors=False,
            compute_crb=False,
        )
        return float(np.min(res.energies)) if res.energies.size else np.nan

    if method == "fft":
        # Fast FFT GS: find peaks and return the minimum energy (ground state)
        C = build_autocorr(psi_t)
        T = C.size
        n_fft = int(kwargs.get("zero_pad_factor", 4) * T)
        w = _hann(T)

        # Match fft_energy_levels() convention: no detrending by default.
        C_use = C

        X = np.fft.fft(C_use * w, n=n_fft)
        P = np.abs(X) ** 2
        freqs = np.fft.fftfreq(n_fft, d=dt)
        omega = 2 * np.pi * freqs
        neg = freqs < 0
        Pn = P[neg]
        on = omega[neg]

        # Normalize to reduce dynamic range brittleness (robust to a few huge lines)
        med = float(np.median(Pn)) if Pn.size else 1.0
        if med > 0:
            Pn = Pn / med

        if Pn.size > 2:
            Pn = Pn[:-1]
            on = on[:-1]

        # Find multiple peaks, then return the minimum energy
        n_levels = kwargs.get("n_levels", 10)
        pk = _robust_peaks(
            Pn,
            min_prom_frac=float(kwargs.get("min_prom_frac", 0.01)),
            k_fallback=max(n_levels * 4, 20),
            low_bin_skip=0,
            baseline_win=31,
        )
        if pk.size == 0:
            # Fallback to argmax
            k = int(np.argmax(Pn))
            return float(-on[k])

        # Convert peaks to energies
        E_hat = -on[pk]
        E_hat = E_hat[E_hat > 0]  # Keep positive energies
        if E_hat.size == 0:
            k = int(np.argmax(Pn))
            return float(-on[k])

        return float(np.min(E_hat))  # Return ground state (lowest energy)

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


# =============================================================================
# Visualization utilities
# =============================================================================

def plot_fft_spectrum(
    psi_t: np.ndarray,
    dt: float,
    energy_offset: float = 0.0,
    max_energy: float | None = None,
    min_energy: float | None = None,
    true_energies: np.ndarray | None = None,
    estimated_energies: np.ndarray | None = None,
    figsize: tuple = (12, 5),
    log_scale: bool = True,
    title: str | None = None,
    zero_pad_factor: int = 4,
):
    """
    Visualize the FFT power spectrum of time-evolved wavefunctions.
    
    Uses the same FFT computation as fft_energy_levels for consistency.
    
    Args:
        psi_t: Time-evolved wavefunction array (T, dim) or autocorrelation (T,)
        dt: Time step
        energy_offset: Constant to add to FFT energies (e.g., identity term coefficient)
        max_energy: Maximum energy to display
        min_energy: Minimum energy to display
        true_energies: If provided, plot vertical lines at these energies (red dashed)
        estimated_energies: If provided, plot vertical lines at estimated energies (green dotted)
        figsize: Figure size
        log_scale: Use log scale for power spectrum
        title: Plot title
        zero_pad_factor: Zero padding factor for FFT resolution
        
    Returns:
        fig, ax, (energies_plot, power_plot)
    """
    import matplotlib.pyplot as plt
    
    # Use shared FFT computation
    spec = _compute_fft_spectrum(psi_t, dt, zero_pad_factor=zero_pad_factor)
    
    # Apply energy offset
    E_plot = spec.energies + energy_offset
    P_plot = spec.power.copy()
    
    # Apply energy range filter
    mask = np.ones(len(E_plot), dtype=bool)
    if min_energy is not None:
        mask &= E_plot >= min_energy
    if max_energy is not None:
        mask &= E_plot <= max_energy
    E_plot = E_plot[mask]
    P_plot = P_plot[mask]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if E_plot.size == 0:
        ax.text(0.5, 0.5, 'No data in energy range', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
    else:
        ax.plot(E_plot, P_plot, 'b-', linewidth=0.8, alpha=0.8)
        
        if log_scale and P_plot.size > 0 and P_plot.max() > 0:
            ax.set_yscale('log')
    
    # Mark true energies if provided
    if true_energies is not None:
        true_energies = np.atleast_1d(true_energies)
        for i, E in enumerate(true_energies):
            if (min_energy is None or E >= min_energy) and (max_energy is None or E <= max_energy):
                ax.axvline(E, color='r', linestyle='--', alpha=0.7, linewidth=1.5,
                          label='True eigenvalues' if i == 0 else None)
    
    # Mark estimated energies if provided
    if estimated_energies is not None:
        estimated_energies = np.atleast_1d(estimated_energies)
        for i, E in enumerate(estimated_energies):
            E_shifted = E + energy_offset
            if (min_energy is None or E_shifted >= min_energy) and (max_energy is None or E_shifted <= max_energy):
                ax.axvline(E_shifted, color='g', linestyle=':', alpha=0.8, linewidth=2,
                          label='Estimated energies' if i == 0 else None)
    
    ax.set_xlabel('Energy (Hartree)', fontsize=12)
    ax.set_ylabel('Power Spectrum', fontsize=12)
    ax.set_title(title or 'FFT Spectrum of Time-Evolved Wavefunction', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if true_energies is not None or estimated_energies is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig, ax, (E_plot, P_plot)


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
    # Create a simple quantum system with known energy levels
    # Using equal amplitudes so FFT can reliably detect all peaks
    E_true = np.array([1.0, 2.5, 4.2, 7.2])
    amps_true = np.array([1.0, 1.0, 1.0, 1.0]) / np.sqrt(4)  # normalized, equal amplitudes

    T = 500
    dt = 0.05
    t = dt * np.arange(T)
    dim = 10  # dimension of Hilbert space (arbitrary for synthetic test)

    # Build psi(t) as a 2D array (T, dim)
    # psi(t) = Σ c_n * exp(-i E_n t) * |n>
    # where |n> are orthonormal basis vectors
    psi_t = np.zeros((T, dim), dtype=np.complex128)
    for n, (E_n, a_n) in enumerate(zip(E_true, amps_true)):
        # |n> is the n-th basis vector
        basis_n = np.zeros(dim, dtype=np.complex128)
        basis_n[n] = 1.0
        # Add contribution: c_n * exp(-i E_n t) * |n>
        psi_t += a_n * np.exp(-1j * E_n * t)[:, None] * basis_n[None, :]

    # Add noise to the wavefunction
    noise_level = 0.01  # lower noise for cleaner spectrum
    psi_t += noise_level * (np.random.randn(T, dim) + 1j * np.random.randn(T, dim))

    n_levels = 4
    noise_var = (noise_level**2) if compute_uncertainties else None

    print("=" * 110)
    mode = "WITH σ + CRB" if compute_uncertainties else "NO σ / NO CRB (fast)"
    print(f"Spectrum / Ground-State Benchmark — {mode}")
    print("=" * 110)
    print(f"True energies: {E_true}")
    print(f"T={T}, dt={dt}, dim={dim}, noise σ={noise_level}")
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
                min_prom_frac=0.005,  # lower threshold to catch weaker peaks
                n_bootstrap=300,   # used only if compute_uncertainties=True
            ))
        elif method == "esprit":
            kwargs.update(dict(
                sv_threshold=0.01,
                n_bootstrap=200,   # used only if compute_uncertainties=True
            ))

        t0 = time.perf_counter()
        energies, sig, crb, info = estimate_spectrum(
            psi_t, dt,
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
        C = build_autocorr(psi_t)  # for CRB comparison
        crb_known = cramer_rao_bound(C, dt, np.array([amps_true[0]**2]), noise_variance=noise_level**2)
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
            psi_t, dt,
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
                psi_t, dt,
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
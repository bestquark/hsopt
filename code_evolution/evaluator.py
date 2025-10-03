import runpy, time, numpy as np

FMT_COSTS = {
    "fp8_e4m3": 0.25,   # very low precision, low memory
    "fp8_e5m2": 0.3,
    "fp16": 0.5,        # half precision
    "bf16": 0.6,
    "fp32": 1.0,        # single precision (baseline)
    "fp64": 2.0,        # double precision
}

def _make_problem(n=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H1 = (A + A.conj().T) / 2.0
    H2 = (B + B.conj().T) / 2.0
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi /= np.linalg.norm(psi)
    return [H1, H2], psi


def _exact_hermitian(H_terms, psi, t):
    H = H_terms[0] + H_terms[1]
    w, Q = np.linalg.eigh(H)  # H = Q diag(w) Q^H
    phase = np.exp((-1j * t) * w.astype(np.complex128))
    y = Q.conj().T @ psi
    y *= phase
    return (Q @ y).astype(np.complex128, copy=False)


def _fid(psi_ref, psi):
    n1 = max(1e-30, np.linalg.norm(psi_ref))
    n2 = max(1e-30, np.linalg.norm(psi))
    z = np.vdot(psi_ref / n1, psi / n2)
    return float((z.conjugate() * z).real)


def _load(program_path):
    return runpy.run_path(program_path)


def evaluate_stage1(program_path, n=8, t=0.2, seed=123):
    mod = _load(program_path)
    if "run" not in mod:
        return {"stage1_passed": 0.0, "error": "no_run_fn"}
    H_terms, psi0 = _make_problem(n=n, seed=seed)
    try:
        psi_final, meta = mod["run"](H_terms, psi0, t)
    except Exception as e:
        return {"stage1_passed": 0.0, "error": f"runtime_error:{type(e).__name__}"}
    psi_ref = _exact_hermitian(H_terms, psi0, t)
    fid = _fid(psi_ref, psi_final)
    nrm = float(np.linalg.norm(psi_final))
    passed = 1.0 if (fid >= 0.90 and abs(nrm - 1.0) < 5e-3) else 0.0
    return {"stage1_passed": passed, "combined_score": fid, "norm_s1": nrm}


def evaluate_stage2(program_path, **kw):
    return evaluate(program_path, **kw)


def evaluate(program_path, trials=5, n=8, t=0.5, seeds=None):
    mod = _load(program_path)
    if "run" not in mod:
        return {"combined_score": -1e9, "error": "no_run_fn"}

    seeds = seeds or [1337 + i for i in range(trials)]
    fids, times_ms, norms, precision_costs = [], [], [], []

    for sd in seeds:
        H_terms, psi0 = _make_problem(n=n, seed=sd)
        t0 = time.perf_counter()
        try:
            psi_final, meta = mod["run"](H_terms, psi0, t)
        except Exception as e:
            return {
                "combined_score": -1e9,
                "error": f"runtime_error:{type(e).__name__}",
            }
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        psi_ref = _exact_hermitian(H_terms, psi0, t)
        fids.append(_fid(psi_ref, psi_final))
        norms.append(float(np.linalg.norm(psi_final)))
        
        meta_cost = 0.0
        if "fmts" in meta and isinstance(meta["fmts"], list):
            costs = [FMT_COSTS.get(fmt, 1.0) for fmt in meta["fmts"]]
            precision_costs.append(np.mean(costs))
            
    fid_mean = float(np.mean(fids))
    fid_min = float(np.min(fids))
    t_mean = float(np.mean(times_ms))
    norm_dev = float(abs(np.mean(norms) - 1.0))
    mem_mb_mean = float(np.mean(precision_costs)) if precision_costs else 0.0
    
    time_penalty = 0.001 * t_mean
    mem_penalty  = 0.005 * mem_mb_mean
    norm_penalty = 5.0 * norm_dev
    precision_penalty = 0.1 * meta_cost   # weight this however you like

    combined = fid_mean - time_penalty - mem_penalty - norm_penalty - precision_penalty

    return {
    "combined_score": float(combined),
    "fid_mean": fid_mean,
    "fid_min": fid_min,
    "time_ms_mean": t_mean,
    "mem_mb_mean": mem_mb_mean,
    "precision_cost": float(meta_cost),
    "norm_dev": norm_dev,
    "trials": len(seeds),
}


EVALUATOR = evaluate

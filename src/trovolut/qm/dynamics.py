import cupynumeric as np
import scipy.linalg

def suzuki_step(H_terms, dt, order=1):
    """Single-step unitary for e^{-i H dt} with Suzuki formula of given order.

    order = 1: first-order (Lie-Trotter)
    order = 2: standard symmetric 2nd-order
    order = 2k (k>=2): recursive Suzuki construction
    """
    if order == 1:
        U = np.eye(H_terms[0].shape[0], dtype=complex)
        for H_j in H_terms:
            U_j = scipy.linalg.expm(-1j * H_j * dt)
            U = U_j @ U
        return U

    if order % 2 == 1:
        raise ValueError(f"Suzuki formulas are defined for order 1 or even orders >= 2, got {order}.")

    if order == 2:
        # Symmetric 2nd-order: ∏_j e^{-i H_j dt/2} ∏_j^rev e^{-i H_j dt/2}
        half = dt / 2.0
        U = np.eye(H_terms[0].shape[0], dtype=complex)
        for H_j in H_terms:
            U_j = scipy.linalg.expm(-1j * H_j * half)
            U = U_j @ U
        for H_j in reversed(H_terms):
            U_j = scipy.linalg.expm(-1j * H_j * half)
            U = U_j @ U
        return U

    # Recursive Suzuki: S_{2k}(dt) = S_{2k-2}(p_k dt)^2 S_{2k-2}((1-4 p_k) dt) S_{2k-2}(p_k dt)^2
    k = order // 2
    p_k = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))

    U_p = suzuki_step(H_terms, p_k * dt, order=order - 2)
    U_m = suzuki_step(H_terms, (1.0 - 4.0 * p_k) * dt, order=order - 2)

    return U_p @ U_p @ U_m @ U_p @ U_p


def product_formula(H_terms, t, r, order=1):
    """
    Product-formula approximation to e^{-i H t}, H = sum_j H_j.

    H_terms: list of np.ndarray (Hermitian terms H_j)
    t: total evolution time
    r: number of Trotter steps
    order: 1, 2, 4, 6, ... (even orders via Suzuki recursion)
    """
    dt = t / r
    U = np.eye(H_terms[0].shape[0], dtype=complex)
    for _ in range(r):
        U_step = suzuki_step(H_terms, dt, order=order)
        U = U_step @ U
    return U

def qdrift(H_terms, t, N):
    """
    qDRIFT approximation to e^{-i H t}, H = sum_j H_j.

    H_terms: list of np.ndarray (Hermitian terms H_j)
    t: total evolution time
    N: number of sampled exponentials
    """
    d = H_terms[0].shape[0]
    norms = np.array([np.linalg.norm(H_j, ord=2) for H_j in H_terms], dtype=float)
    lam = norms.sum()
    if lam == 0:
        return np.eye(d, dtype=complex)

    probs = norms / lam
    dt_eff = t * lam / N  # effective time per sampled normalized term

    U = np.eye(d, dtype=complex)
    for _ in range(N):
        j = np.random.choice(len(H_terms), p=probs)
        G_j = H_terms[j] / norms[j]  # normalized term so that H = sum a_j G_j
        U_step = scipy.linalg.expm(-1j * G_j * dt_eff)
        U = U_step @ U
    return U
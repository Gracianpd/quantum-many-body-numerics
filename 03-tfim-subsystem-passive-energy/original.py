"""
TFIM (Ising in transverse field) — subsystem energies + passive energy via Gaussian method
OBC, N even, block A = left half (L = N/2)

Guarda figuras en:
"""

import os
import numpy as np
import heapq
import matplotlib.pyplot as plt


# -----------------------------
# 1) TFIM JW quadratic matrices
# -----------------------------
def build_AB_tfim_obc(N: int, Gamma: float):
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, N), dtype=float)

    np.fill_diagonal(A, -Gamma)

    for i in range(N - 1):
        A[i, i + 1] = -0.25
        A[i + 1, i] = -0.25

    # FIX: pairing coefficient must be -0.25 / +0.25 in this convention
    for i in range(N - 1):
        B[i, i + 1] = -0.25
        B[i + 1, i] = +0.25

    return A, B


def build_BdG(A: np.ndarray, B: np.ndarray):
    return np.block([[A, -B],
                     [B, -A.T]])


# -----------------------------------------
# 2) Ground-state correlators from BdG eigs
# -----------------------------------------
def gs_correlators_from_bdg(N: int, Gamma: float):
    A, B = build_AB_tfim_obc(N, Gamma)
    H = build_BdG(A, B)
    evals, evecs = np.linalg.eigh(H)

    neg = np.argsort(evals)[:N]
    VEC = evecs[:, neg]
    U = VEC[:N, :]
    V = VEC[N:, :]

    C = V @ V.T
    F = U @ V.T

    pos = np.argsort(evals)[N:]
    eps_pos = np.sort(np.abs(evals[pos]))
    return C, F, eps_pos


# ---------------------------------------------------------
# 2b) Approx bulk energy density e_inf(Gamma) from a large chain
# ---------------------------------------------------------
def bulk_energy_density_from_bdg(Gamma: float, Mbulk: int = 2000):
    """
    Estimate e_inf(Gamma) ~ E0(M,Gamma)/M using same conventions as block_energy_from_CF.
    We compute the *full-chain* GS energy with L=M, then divide by M.
    """
    if Mbulk % 2 != 0:
        Mbulk += 1
    C, F, _ = gs_correlators_from_bdg(Mbulk, Gamma)
    C = 0.5 * (C + C.T)
    F = 0.5 * (F - F.T)
    E0 = block_energy_from_CF(C, F, Mbulk, Gamma)
    return E0 / Mbulk


# ---------------------------------------------------------
# 3) Build Majorana covariance Γ from (C,F) (block A only)
# ---------------------------------------------------------
def majorana_covariance_from_CF(CA: np.ndarray, FA: np.ndarray):
    L = CA.shape[0]
    I = np.eye(L)

    AA = FA
    DA = CA
    AD = I - CA.T
    DD = FA.conj().T

    Alpha = np.zeros((2 * L, L), dtype=complex)
    Beta  = np.zeros((2 * L, L), dtype=complex)
    for j in range(L):
        Alpha[2 * j,     j] = 1.0
        Beta[2 * j,      j] = 1.0
        Alpha[2 * j + 1, j] = 1.0j
        Beta[2 * j + 1,  j] = -1.0j

    G = (Alpha @ AA @ Alpha.T +
         Alpha @ AD @ Beta.T  +
         Beta  @ DA @ Alpha.T +
         Beta  @ DD @ Beta.T)

    Gamma = 1.0j * (G - np.eye(2 * L))
    Gamma = np.real_if_close(Gamma, tol=1e-10).astype(float)
    Gamma = 0.5 * (Gamma - Gamma.T)
    return Gamma


def entanglement_nu_from_GammaA(GammaA: np.ndarray):
    vals = np.linalg.eigvalsh(1.0j * GammaA)
    vals = np.real_if_close(vals, tol=1e-10).astype(float)
    nu = np.sort(vals[vals > 1e-12])
    return nu


# -----------------------------------
# 4) Energies from correlators (block)
# -----------------------------------
def block_energy_from_CF(C: np.ndarray, F: np.ndarray, L: int, Gamma: float):
    s_bond = 0.0
    for i in range(L - 1):
        s_bond += (C[i, i + 1] + C[i + 1, i] + 2.0 * np.real(F[i + 1, i]))

    onsite = np.trace(C[:L, :L])
    EA = -0.25 * s_bond - Gamma * onsite + 0.5 * Gamma * L
    return float(np.real_if_close(EA))


# -------------------------------------------
# 5) K-smallest subset sums (heap / best-first)
# -------------------------------------------
def k_smallest_subset_sums(weights, K):
    w = np.array(weights, dtype=float)
    w = w[w > 0]
    w.sort()
    n = len(w)
    if K <= 0:
        return []
    if n == 0:
        return [0.0] * K

    res = [0.0]
    heap = [(w[0], 0)]
    while heap and len(res) < K:
        s, i = heapq.heappop(heap)
        res.append(s)
        if i + 1 < n:
            heapq.heappush(heap, (s + w[i + 1], i + 1))
            heapq.heappush(heap, (s - w[i] + w[i + 1], i + 1))
    if len(res) < K:
        res += [res[-1]] * (K - len(res))
    return res[:K]


# -----------------------------------
# 6) Passive energy via top-K + bounds
# -----------------------------------
def passive_energy_from_nu_and_eps(nu, eps_exc, EA0, tol=1e-10, K0=200, Kmax=200000):
    nu = np.array(nu, dtype=float)

    p0 = (1.0 - nu) / 2.0
    p1 = (1.0 + nu) / 2.0
    pmax = np.maximum(p0, p1)
    pmin = np.minimum(p0, p1)

    lam_max = float(np.prod(pmax))

    r = np.divide(pmin, pmax, out=np.zeros_like(pmin), where=(pmax > 0))
    penalties = []
    for rk in r:
        if rk <= 0:
            penalties.append(np.inf)
        else:
            penalties.append(-np.log(rk))
    penalties = np.array(penalties, dtype=float)
    penalties = penalties[np.isfinite(penalties)]

    eps_exc = np.array(eps_exc, dtype=float)
    Emax = float(EA0 + np.sum(eps_exc))

    K = K0
    while True:
        if K > Kmax:
            raise RuntimeError("Kmax alcanzado antes de cumplir tolerancia. Sube Kmax o relaja tol.")

        pen_sums = k_smallest_subset_sums(penalties, K)
        lambdas = lam_max * np.exp(-np.array(pen_sums))
        lambdas = np.sort(lambdas)[::-1]

        exc_sums = k_smallest_subset_sums(eps_exc, K + 1)
        energies = EA0 + np.array(exc_sums)
        EKp1 = float(energies[min(K, len(energies) - 1)])

        K_eff = min(K, len(lambdas), len(energies) - 1)
        S_K = float(np.sum(lambdas[:K_eff] * energies[:K_eff]))
        R_K = float(1.0 - np.sum(lambdas[:K_eff]))

        width = R_K * (Emax - EKp1)
        if width < tol:
            lower = S_K + R_K * EKp1
            upper = S_K + R_K * Emax
            Et = 0.5 * (lower + upper)
            return Et, dict(K=K_eff, R=R_K, lower=lower, upper=upper, width=width)

        K *= 2


# -------------------------
# 7) Full pipeline per (N,Γ)
# -------------------------
def compute_all_for_N_Gamma(N: int, Gamma: float, tol=1e-10):
    if N % 2 != 0:
        raise ValueError("N debe ser par.")
    L = N // 2

    C, F, _ = gs_correlators_from_bdg(N, Gamma)
    C = 0.5 * (C + C.T)
    F = 0.5 * (F - F.T)

    EA = block_energy_from_CF(C, F, L, Gamma)

    CA = C[:L, :L]
    FA = F[:L, :L]

    GammaA = majorana_covariance_from_CF(CA, FA)
    nu = entanglement_nu_from_GammaA(GammaA)
    nu = np.clip(nu, 0.0, 1.0)

    if len(nu) != L:
        if len(nu) < L:
            nu = np.pad(nu, (0, L - len(nu)), mode="constant", constant_values=1.0)
        else:
            nu = nu[-L:]

    Cb, Fb, epsA = gs_correlators_from_bdg(L, Gamma)
    # FIX: also symmetrize block correlators
    Cb = 0.5 * (Cb + Cb.T)
    Fb = 0.5 * (Fb - Fb.T)

    EA0 = block_energy_from_CF(Cb, Fb, L, Gamma)

    Et, info = passive_energy_from_nu_and_eps(nu, epsA, EA0, tol=tol)

    DeltaE = EA - EA0
    W = EA - Et
    Q = Et - EA0

    if not (EA + 1e-8 >= Et >= EA0 - 1e-8):
        print("WARNING: orden EA >= Et >= EA0 violado:", N, Gamma, EA, Et, EA0)

    return dict(N=N, Gamma=Gamma, EA=EA, EA0=EA0, Et=Et,
                DeltaE=DeltaE, W=W, Q=Q, passive_info=info)


# -------------------------
# 8) Scan + plot (split regimes) + save
# -------------------------
def make_N_grid(N_min=20, N_max=400, n_points=30):
    raw = np.geomspace(N_min, N_max, n_points)
    Ns = (2 * np.round(raw / 2)).astype(int)
    Ns = np.unique(np.clip(Ns, N_min, N_max))
    Ns = Ns[Ns % 2 == 0]
    return Ns


def save_figure(fig, out_dir, filename, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Guardada figura: {path}")


def plot_regime(results, gammas, title_suffix, out_dir, tag,
                bulk_density, show=False):
    """
    bulk_density: dict {Gamma: e_inf(Gamma)}
    We subtract L * e_inf(Gamma) from EA, EA0, Et for plotting.
    """
    fig1 = plt.figure()
    for g in gammas:
        Ns_plot = np.array([d["N"] for d in results[g]])
        Ls = Ns_plot // 2
        e_inf = bulk_density[g]

        EA0 = np.array([d["EA0"] for d in results[g]]) - Ls * e_inf
        EA  = np.array([d["EA"] for d in results[g]])  - Ls * e_inf
        Et  = np.array([d["Et"] for d in results[g]])  - Ls * e_inf

        plt.plot(Ns_plot, EA0, "o", markersize=3, label=fr"$E_{{A,0}}-Le_\infty$ (Γ={g})")
        plt.plot(Ns_plot, EA,  "o", markersize=3, label=fr"$E_A-Le_\infty$ (Γ={g})")
        plt.plot(Ns_plot, Et,  "o", markersize=3, label=fr"$\tilde E_A-Le_\infty$ (Γ={g})")

    plt.xlabel("N")
    plt.ylabel("Energy (bulk-subtracted)")
    plt.title(f"Subsystem energies (bulk-subtracted) — {title_suffix}")
    plt.legend(ncols=2)
    plt.tight_layout()
    save_figure(fig1, out_dir, f"energies_{tag}_bulksub.png")
    if not show:
        plt.close(fig1)

    # Differences (no subtraction needed)
    fig2 = plt.figure()
    for g in gammas:
        Ns_plot = np.array([d["N"] for d in results[g]])
        dE = np.array([d["DeltaE"] for d in results[g]])
        W  = np.array([d["W"] for d in results[g]])
        Q  = np.array([d["Q"] for d in results[g]])

        plt.plot(Ns_plot, dE, "o", markersize=3, label=fr"$\Delta E_A$ (Γ={g})")
        plt.plot(Ns_plot, W,  "o", markersize=3, label=fr"$W_A$ (Γ={g})")
        plt.plot(Ns_plot, Q,  "o", markersize=3, label=fr"$Q_A$ (Γ={g})")

    plt.xlabel("N")
    plt.ylabel("Energy difference")
    plt.title(f"Energy differences — {title_suffix}")
    plt.legend(ncols=2)
    plt.tight_layout()
    save_figure(fig2, out_dir, f"differences_{tag}.png")
    if not show:
        plt.close(fig2)


def main():
    # ----- CONFIG -----
    OUT_DIR = "figures_original"
    SHOW_PLOTS = False
    DPI = 300

    Gammas_ferro = [0.5, 0.6, 0.8]   # Γ < 1
    Gammas_para  = [1.2, 1.5, 2.0]   # Γ > 1
    Ns = make_N_grid(N_min=20, N_max=400, n_points=30)

    tol = 1e-10

    # Bulk estimation config
    Mbulk = 2000   # increase to 4000 if you want smoother e_inf
    # -------------------

    all_gammas = Gammas_ferro + Gammas_para

    # Precompute bulk energy density per Gamma
    bulk_density = {}
    print("\n=== Estimating bulk energy densities e_inf(Gamma) ===")
    for g in all_gammas:
        e_inf = bulk_energy_density_from_bdg(float(g), Mbulk=Mbulk)
        bulk_density[g] = e_inf
        print(f"Gamma={g}: e_inf ~ {e_inf:.10f}  (Mbulk={Mbulk})")

    results = {g: [] for g in all_gammas}

    for g in all_gammas:
        print(f"\n=== Gamma = {g} ===")
        for N in Ns:
            out = compute_all_for_N_Gamma(int(N), float(g), tol=tol)
            results[g].append(out)
            info = out["passive_info"]
            print(f"N={N:4d}  EA={out['EA']:.6f}  Et={out['Et']:.6f}  EA0={out['EA0']:.6f} "
                  f"| W={out['W']:.3e} Q={out['Q']:.3e}  (K={info['K']}, width={info['width']:.2e})")

    # Guardar figuras
    plot_regime(results, Gammas_ferro, "Ferromagnetic regime (Γ < 1)", OUT_DIR, "ferro",
                bulk_density=bulk_density, show=SHOW_PLOTS)
    plot_regime(results, Gammas_para,  "Paramagnetic regime (Γ > 1)", OUT_DIR, "para",
                bulk_density=bulk_density, show=SHOW_PLOTS)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()

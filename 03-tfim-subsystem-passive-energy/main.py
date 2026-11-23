import numpy as np
import matplotlib.pyplot as plt

# =============================
# Constants (paper)
# =============================
c0 = 2 / np.pi
cB = 4 / np.pi - 1
vF = 2.0
c = 1.0
gamma = float(np.exp(2.3))  # paper: ln(gamma) ~ 2.3

# =============================
# Utilities
# =============================
def make_N_grid(N_min=40, N_max=4000, n_points=55):
    """
    sparse grid: log-spaced values, rounded to multiples of 4
    Returns sorted unique N values.
    """
    raw = np.geomspace(N_min, N_max, n_points)
    # round to nearest multiple of 4
    Ns = (4 * np.round(raw / 4)).astype(int)
    Ns = np.clip(Ns, N_min, N_max)
    Ns = np.unique(Ns)
    # ensure even and multiple of 4
    Ns = Ns[Ns % 4 == 0]
    return Ns

def S_cos_sum(M: int, x: np.ndarray) -> np.ndarray:
    """
    Vectorized sum:
      S_M(x) = sum_{k=1}^M cos(k x)
            = sin(M x/2) * cos((M+1) x/2) / sin(x/2)
    Handles x~0 via limit S_M(0) = M.
    """
    x = np.asarray(x, dtype=float)
    denom = np.sin(x / 2.0)
    out = np.empty_like(x)

    mask = np.abs(denom) > 1e-14
    out[mask] = (np.sin(M * x[mask] / 2.0) * np.cos((M + 1) * x[mask] / 2.0)) / denom[mask]
    out[~mask] = float(M)
    return out

def correlation_block_CA(N: int) -> np.ndarray:
    """
    Exact correlation submatrix C_A for open chain at half filling, block A = left half.

    For OBC eigenmodes: U_{k,j} = sqrt(2/(N+1)) sin(k j pi/(N+1))
    Half filling: k=1..M with M=N/2 occupied.

    Exact identity:
      C_{ij} = (1/(N+1)) [ S_M((i-j)α) - S_M((i+j)α) ]
    with α = π/(N+1) and i,j counted from 1.
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")
    L = N // 2
    M = N // 2
    alpha = np.pi / (N + 1)

    i = np.arange(1, L + 1)[:, None]
    j = np.arange(1, L + 1)[None, :]

    x1 = (i - j) * alpha
    x2 = (i + j) * alpha

    CA = (S_cos_sum(M, x1) - S_cos_sum(M, x2)) / (N + 1.0)
    return CA

# =============================
# Exact numerics (fast)
# =============================
def exact_energies(N: int) -> dict:
    """
    Exact brute-force values without diagonalizing NxN:
      - use analytic spectra for E0 and E_A0,
      - build C_A via closed-form sums,
      - diagonalize C_A (size N/2) to get occupations,
      - compute E_A, E_A0, E_A_tilde and differences.
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")
    L = N // 2


    # --- Isolated block A GS energy: length L, half filling
    p = np.arange(1, L + 1)
    epsA = -2.0 * np.cos(p * np.pi / (L + 1))
    EA0 = float(np.sum(epsA[: L // 2]))

    # --- Build C_A and compute E_A from nearest-neighbor correlator
    CA = correlation_block_CA(N)
    EA = float(-2.0 * np.sum(np.diag(CA, k=1)))  # -2 sum_{i=1}^{L-1} C_{i,i+1}

    # --- Passive energy:
    # occupations = eigenvalues of C_A (descending)
    nu = np.linalg.eigvalsh(CA)[::-1]
    # align largest nu with lowest epsA (epsA is ascending)
    EA_tilde = float(np.sum(nu * epsA))

    DeltaE = EA - EA0
    W = EA - EA_tilde
    Q = EA_tilde - EA0

    return dict(N=N, EA=EA, EA0=EA0, EA_tilde=EA_tilde,
                DeltaE=DeltaE, W=W, Q=Q)

# =============================
# Asymptotic (paper) formulas
# =============================
def energies_asymptotic(Ns):
    Ns = np.asarray(Ns, dtype=float)

    # Eq (27): E_{A,0}
    EA0 = -c0 * (Ns / 2.0 - 1.0) - cB - (np.pi / (6.0 * Ns))

    # Eq (32): E_A
    EA = -c0 * (Ns / 2.0 - 1.0) - (cB / 2.0) - ((np.pi / 24.0 + 0.5) / Ns)

    # Eq (39): E~_A
    EA_tilde = (
        -c0 * (Ns / 2.0 - 1.0)
        - cB
        - (np.pi / (6.0 * Ns))
        + (np.log(gamma * Ns) ** 2) / (6.0 * np.pi * Ns)
    )

    return dict(
        EA=EA, EA0=EA0, EA_tilde=EA_tilde,
        DeltaE=EA - EA0,          # Eq (33)
        W=EA - EA_tilde,          # Eq (40)
        Q=EA_tilde - EA0          # Eq (41)
    )

# =============================
# Main plotting routine
# =============================
def main():
    Ns = make_N_grid(N_min=20, N_max=4000, n_points=200)
    print(f"Using {len(Ns)} exact points up to N={Ns.max()}:")
    print(Ns)

    exact = [exact_energies(int(N)) for N in Ns]

    EA_ex  = np.array([d["EA"] for d in exact])
    Et_ex  = np.array([d["EA_tilde"] for d in exact])
    EA0_ex = np.array([d["EA0"] for d in exact])

    dE_ex = np.array([d["DeltaE"] for d in exact])
    W_ex  = np.array([d["W"] for d in exact])
    Q_ex  = np.array([d["Q"] for d in exact])

    asym = energies_asymptotic(Ns)

    # ---- Top: energies (paper plots residuals: subtract Bulk + shift boundary) ----
    bulk = -c0 * (Ns / 2.0 - 1.0)

    # Exact residuals
    EA_ex_plot = EA_ex - bulk + (cB / 2.0)
    Et_ex_plot = Et_ex - bulk + cB
    EA0_ex_plot = EA0_ex - bulk + cB

    # Asymptotic residuals
    EA_as_plot = asym["EA"] - bulk + (cB / 2.0)
    Et_as_plot = asym["EA_tilde"] - bulk + cB
    EA0_as_plot = asym["EA0"] - bulk + cB

    plt.figure()
    plt.plot(Ns, EA0_ex_plot, "o", markersize=3, label=r"Exact $E_{A,0}-\mathrm{Bulk}+c_B$")
    plt.plot(Ns, EA_ex_plot, "o", markersize=3, label=r"Exact $E_A-\mathrm{Bulk}+c_B/2$")
    plt.plot(Ns, Et_ex_plot, "o", markersize=3, label=r"Exact $\tilde E_A-\mathrm{Bulk}+c_B$")

    plt.plot(Ns, EA0_as_plot, "-", label=r"Asymptotic $E_{A,0}-\mathrm{Bulk}+c_B$")
    plt.plot(Ns, EA_as_plot, "-", label=r"Asymptotic $E_A-\mathrm{Bulk}+c_B/2$")
    plt.plot(Ns, Et_as_plot, "-", label=r"Asymptotic $\tilde E_A-\mathrm{Bulk}+c_B$")

    plt.xlabel(r"$N$")
    plt.ylabel("Energy (bulk-subtracted)")
    plt.title("Bulk-subtracted energies — A = left half")
    plt.legend()
    plt.tight_layout()

    # ---- Bottom: differences ----
    plt.figure()
    plt.plot(Ns, dE_ex, "o", markersize=3, label=r"Exact $\Delta E_A$")
    plt.plot(Ns, W_ex,  "o", markersize=3, label=r"Exact $W_A$")
    plt.plot(Ns, Q_ex,  "o", markersize=3, label=r"Exact $Q_A$")

    plt.plot(Ns, asym["DeltaE"], "-", label=r"Asymptotic $\Delta E_A$")
    plt.plot(Ns, asym["W"],      "-", label=r"Asymptotic $W_A$")
    plt.plot(Ns, asym["Q"],      "-", label=r"Asymptotic $Q_A$")

    plt.xlabel(r"$N$")
    plt.ylabel("Energy difference")
    plt.title("Differences")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

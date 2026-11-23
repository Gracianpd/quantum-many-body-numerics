import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# -------------------------
# 1) Full Hilbert space Heisenberg Hamiltonian (brute force)
# -------------------------
def heisenberg_hamiltonian_full(N: int, J: float = 1.0, periodic: bool = True):
    """
    Build the full 2^N x 2^N sparse Hamiltonian for the spin-1/2 Heisenberg chain.

    Basis: computational Sz basis, states labeled by integers s=0..2^N-1
    with bit i (0/1) representing down/up at site i.

    Local bond action:
      - if spins equal: diagonal +J/4
      - if spins different: diagonal -J/4 and off-diagonal swap with +J/2
    """
    dim = 1 << N
    rows, cols, data = [], [], []
    diag = np.zeros(dim, dtype=np.float64)

    n_bonds = N if periodic else (N - 1)

    for s in range(dim):
        for i in range(n_bonds):
            j = (i + 1) % N
            bi = (s >> i) & 1
            bj = (s >> j) & 1

            if bi == bj:
                diag[s] += 0.25 * J
            else:
                diag[s] += -0.25 * J
                # swap 01 <-> 10 by flipping both bits
                s2 = s ^ ((1 << i) | (1 << j))
                rows.append(s2)
                cols.append(s)
                data.append(0.5 * J)

    # add diagonal terms
    rows.extend(range(dim))
    cols.extend(range(dim))
    data.extend(diag.tolist())

    H = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    return H


# -------------------------
# 2) Ground state + gap from full spectrum (lowest two eigenvalues)
# -------------------------
def ground_and_gap_full(H: sp.csr_matrix, tol=1e-12):
    """
    Compute E0, E1 and ground state vector psi0 from the full Hamiltonian.
    Gap = E1 - E0.
    """
    w, v = spla.eigsh(H, k=2, which="SA", tol=tol)
    order = np.argsort(w)
    E0 = float(w[order[0]])
    E1 = float(w[order[1]])
    psi0 = v[:, order[0]]
    psi0 = psi0 / np.linalg.norm(psi0)
    gap = E1 - E0
    return E0, gap, psi0


# -------------------------
# 3) Entanglement entropy (now psi is already full)
# -------------------------
def entanglement_entropy_contiguous_block_full(N: int, psi_full: np.ndarray, ell: int):
    """
    Von Neumann entropy of block A = sites 0..ell-1, using Schmidt/SVD.

    psi_full has length 2^N in computational basis.
    """
    dimA = 1 << ell
    dimB = 1 << (N - ell)

    # Arrange amplitudes into C[a,b] where a labels A (ell bits), b labels B (N-ell bits)
    C = psi_full.reshape((dimB, dimA)).T  # shape (2^ell, 2^(N-ell))

    svals = np.linalg.svd(C, compute_uv=False)
    p = svals**2
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


# -------------------------
# 4) Run + plot ED points vs CFT trend (anchored)
# -------------------------
def run_study_bruteforce(N_list=(8, 10, 12, 14, 16), J=1.0, do_plots=True):
    e_inf = J * (0.25 - math.log(2.0))  # your chosen bulk energy density

    results = []
    for N in N_list:
        H = heisenberg_hamiltonian_full(N, J=J, periodic=True)
        E0, gap, psi0 = ground_and_gap_full(H)

        deltaE0 = E0 - e_inf * N  # finite-size correction (Casimir piece)
        ell = N // 2
        S_half = entanglement_entropy_contiguous_block_full(N, psi0, ell)

        results.append((N, E0, gap, deltaE0, S_half))

        print(
            f"N={N:2d} | E0={E0: .12f} | gap={gap: .12f} | gap*N={(gap*N): .12f} | "
            f"deltaE0={deltaE0: .12f} | S(N/2)={S_half: .6f}"
        )

    if not do_plots:
        return results

    Ns = np.array([r[0] for r in results], dtype=float)
    E0s = np.array([r[1] for r in results], dtype=float)
    gaps = np.array([r[2] for r in results], dtype=float)
    deltaE0 = np.array([r[3] for r in results], dtype=float)
    S_half = np.array([r[4] for r in results], dtype=float)

    # -------------------------
    # NEW Plot 0: Ground-state energy E0 vs N (no trendline)
    # -------------------------
    plt.figure()
    plt.plot(Ns, E0s, "o")
    plt.xlabel("N")
    plt.ylabel(r"$E_0(N)$")
    plt.title("Ground-state energy vs N")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # anchor trends at largest N
    Nref = Ns[-1]
    gap_ref = gaps[-1]
    cas_ref = deltaE0[-1]
    Sh_ref = S_half[-1]

    Ngrid = np.linspace(Ns.min(), Ns.max(), 500)

    # ---- Plot 1: GAP trend ~ 1/N (anchored) ----
    gap_trend = gap_ref * (Nref / Ngrid)

    plt.figure()
    plt.plot(Ns, gaps, "o", label="ED (full space)")
    plt.plot(Ngrid, gap_trend, "--", label=r"CFT trend $\propto 1/N$")
    plt.xlabel("N")
    plt.ylabel(r"$\Delta(N)=E_1-E_0$")
    plt.title("Gap vs N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: Casimir piece trend ~ -1/N (anchored) ----
    cas_trend = cas_ref * (Nref / Ngrid)

    plt.figure()
    plt.plot(Ns, deltaE0, "o", label=r"ED: $\delta E_0=E_0-e_\infty N$")
    plt.plot(Ngrid, cas_trend, "--", label=r"CFT trend $\propto -1/N$")
    plt.xlabel("N")
    plt.ylabel(r"$\delta E_0(N)$")
    plt.title("Casimir finite-size correction vs N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional (often clearer): plot |deltaE0| vs N to see 1/N decay
    plt.figure()
    plt.plot(Ns, np.abs(deltaE0), "o", label=r"ED: $|\delta E_0|$")
    plt.plot(Ngrid, abs(cas_ref) * (Nref / Ngrid), "--", label=r"CFT trend $|\delta E_0|\propto 1/N$")
    plt.xlabel("N")
    plt.ylabel(r"$|\delta E_0(N)|$")
    plt.title("Casimir magnitude vs N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Half-chain EE trend ~ (1/3) log(N/pi) + const (anchored) ----
    s0 = Sh_ref - (1.0/3.0) * math.log(Nref / math.pi)
    ee_trend = (1.0/3.0) * np.log(Ngrid / math.pi) + s0

    plt.figure()
    plt.plot(Ns, S_half, "o", label=r"ED: $S(N/2)$")
    plt.plot(Ngrid, ee_trend, "--", label=r"CFT trend $\frac{1}{3}\log(N/\pi)+\mathrm{cte}$")
    plt.xlabel("N")
    plt.ylabel(r"$S(N/2)$")
    plt.title("Half-chain entanglement entropy vs N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    run_study_bruteforce()

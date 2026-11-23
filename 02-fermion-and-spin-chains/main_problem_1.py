import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import csv
import os
import matplotlib.pyplot as plt


# ----------------------------
# Utilities: basis + bit ops
# ----------------------------

def make_basis_fixed_P(N: int, P: int):
    """All possible bitstrings with P ones"""
    basis = []
    for occ_sites in combinations(range(N), P):
        s = 0
        for i in occ_sites:
            s |= (1 << i)
        basis.append(s)
    index = {s: k for k, s in enumerate(basis)}
    return basis, index

def occ(s: int, i: int) -> int:
    return (s >> i) & 1

def popcount(x: int) -> int:
    return x.bit_count()

def jw_sign_cdagger_c(s: int, i: int, j: int) -> int:
    """
    Jordan-Wigner sign for applying c_i^† c_j on occupation bitstring s

    Assumes: site j occupied and site i empty.
    sign = (-1)^(num of occupied sites between i and j).
    """
    if i == j:
        return 1
    a, b = (i, j) if i < j else (j, i)
    mask = ((1 << b) - 1) ^ ((1 << (a + 1)) - 1)  # bits a+1 ... b-1
    return -1 if (popcount(s & mask) % 2) else 1


# ----------------------------
# Build local model matrices J and V (PBC)
# ----------------------------

def build_local_matrices(N: int, Jhop: float, Vnn: float):
    """
      J_{i,i+1} = J_{i+1,i} = -Jhop
      V_{i,i+1} = V_{i+1,i} = Vnn
    """
    J = np.zeros((N, N), dtype=np.complex128)
    V = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        j = (i + 1) % N
        J[i, j] = -Jhop
        J[j, i] = -Jhop
        V[i, j] = Vnn
        V[j, i] = Vnn

    np.fill_diagonal(V, 0.0)
    return J, V


# ----------------------------
# Exact Diagonalization (ED)
# ----------------------------

def build_hamiltonian_ED(N: int, Jhop: float, Vnn: float, P: int):
    basis, index = make_basis_fixed_P(N, P)
    dim = len(basis)
    H = lil_matrix((dim, dim), dtype=np.float64)

    for a, s in enumerate(basis):
        # diagonal interaction energy: V * sum_i n_i n_{i+1}
        e = 0.0
        for i in range(N):
            j = (i + 1) % N
            e += Vnn * occ(s, i) * occ(s, j)
        H[a, a] = e

        # hopping: -J (c_i^† c_{i+1} + h.c.)
        for i in range(N):
            j = (i + 1) % N

            # move particle j -> i  (c_i^† c_j)
            if occ(s, j) == 1 and occ(s, i) == 0:
                s2 = (s & ~(1 << j)) | (1 << i)
                b = index[s2]
                sign = jw_sign_cdagger_c(s, i, j)
                H[b, a] += -Jhop * sign

            # move particle i -> j  (c_j^† c_i)
            if occ(s, i) == 1 and occ(s, j) == 0:
                s2 = (s & ~(1 << i)) | (1 << j)
                b = index[s2]
                sign = jw_sign_cdagger_c(s, j, i)
                H[b, a] += -Jhop * sign

    return csr_matrix(H), basis, index

def ed_ground_and_gap(N: int, Jhop: float, Vnn: float):
    P = N // 2
    H, basis, index = build_hamiltonian_ED(N, Jhop, Vnn, P)

    # Ground + first excited
    evals, evecs = eigsh(H, k=2, which="SA")
    order = np.argsort(evals)
    E0, E1 = float(evals[order[0]]), float(evals[order[1]])
    psi0 = evecs[:, order[0]]
    return E0, (E1 - E0), psi0, basis, index

def ed_density(psi0, basis, N: int):
    prob = np.abs(psi0) ** 2
    n = np.zeros(N, dtype=np.float64)
    for a, s in enumerate(basis):
        for i in range(N):
            n[i] += prob[a] * occ(s, i)
    return n


# ----------------------------
# Hartree–Fock (HF)
# ----------------------------

def slater_C_from_hopping(J_eff: np.ndarray, P: int):
    evals, evecs = np.linalg.eigh(J_eff)
    idx = np.argsort(evals.real)
    U_occ = evecs[:, idx[:P]]
    C = U_occ @ U_occ.conj().T
    return C, evals[idx].real

def build_J_tilde(J: np.ndarray, V: np.ndarray, C: np.ndarray):
    N = J.shape[0]
    Jtilde = J.astype(np.complex128).copy()

    Cdiag = np.real(np.diag(C))        # C_jj = <n_j>
    hartree = V @ Cdiag                # hartree[i] = sum_j V_ij C_jj

    for i in range(N):
        Jtilde[i, i] = J[i, i] + hartree[i]

    for i in range(N):
        for j in range(N):
            if i != j:
                Jtilde[i, j] = J[i, j] - V[i, j] * np.conjugate(C[i, j])

    Jtilde = 0.5 * (Jtilde + Jtilde.conj().T)
    return Jtilde

def hf_energy(J: np.ndarray, V: np.ndarray, C: np.ndarray):
    term1 = np.sum(J * C).real

    Cii = np.real(np.diag(C))
    CiiCjj = np.outer(Cii, Cii)
    absC2 = np.abs(C) ** 2

    mask = np.ones_like(V, dtype=bool)
    np.fill_diagonal(mask, False)

    term2 = np.sum(V[mask] * (CiiCjj[mask] - absC2[mask]))
    return float(term1 + term2)

def hf_scf(J: np.ndarray, V: np.ndarray, P: int, tol=1e-10, max_iter=500):
    C, _ = slater_C_from_hopping(J, P)
    E_old = hf_energy(J, V, C)

    eps_sp = None
    errC = np.inf
    Jtilde = None

    for it in range(1, max_iter + 1):
        Jtilde = build_J_tilde(J, V, C)
        C_new, eps_sp = slater_C_from_hopping(Jtilde, P)

        errC = float(np.max(np.abs(C_new - C)))
        C = C_new

        E = hf_energy(J, V, C)
        if abs(E - E_old) < tol and errC < tol:
            E_old = E
            break
        E_old = E

    gap_sp = float(eps_sp[P] - eps_sp[P - 1]) if (eps_sp is not None and P < len(eps_sp)) else 0.0
    return C, Jtilde, E_old, gap_sp, it, errC


# ----------------------------
# Grid run
# ----------------------------

def run_grid_to_csv(N_list, v_list, filename="ed_hf_grid.csv"):
    Jhop = 1.0  # adimensionalize: J = 1
    rows = []

    for N in N_list:
        if N % 2 != 0:
            raise ValueError(f"N must be even for half-filling P=N/2. Got N={N}.")
        P = N // 2

        for v in v_list:
            Vnn = float(v) * Jhop

            # ED
            E0, gap_mb, psi0, basis, index = ed_ground_and_gap(N, Jhop, Vnn)
            n_ed = ed_density(psi0, basis, N)
            ed_order = float(np.max(np.abs(n_ed - 0.5)))

            # HF
            Jmat, Vmat = build_local_matrices(N, Jhop, Vnn)
            C_hf, Jtilde, E_hf, gap_sp, iters, errC = hf_scf(Jmat, Vmat, P)
            n_hf = np.real(np.diag(C_hf))
            hf_order = float(np.max(np.abs(n_hf - 0.5)))

            rows.append({
                "N": int(N),
                "P": int(P),
                "v": float(v),
                "E0_ED": float(E0),
                "E0_ED_per_site": float(E0 / N),
                "gap_ED": float(gap_mb),
                "E_HF": float(E_hf),
                "E_HF_per_site": float(E_hf / N),
                "gap_HF_sp": float(gap_sp),
                "order_ED_max_abs_n_minus_half": ed_order,
                "order_HF_max_abs_n_minus_half": hf_order,
                "HF_iters": int(iters),
                "HF_errC": float(errC),
            })

    out_path = os.path.join(os.getcwd(), filename)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ----------------------------
# Plot errors vs v for each N
# ----------------------------

def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert needed fields to float/int
            r2 = dict(r)
            r2["N"] = int(r2["N"])
            r2["v"] = float(r2["v"])
            r2["E0_ED_per_site"] = float(r2["E0_ED_per_site"])
            r2["E_HF_per_site"] = float(r2["E_HF_per_site"])
            r2["gap_ED"] = float(r2["gap_ED"])
            r2["gap_HF_sp"] = float(r2["gap_HF_sp"])
            rows.append(r2)
    return rows

def plot_errors(csv_path, save_prefix="plots"):
    rows = read_csv_rows(csv_path)

    # group by N
    Ns = sorted(set(r["N"] for r in rows))

    # ---- Plot 1: E error vs v ----
    plt.figure()
    for N in Ns:
        data = sorted([r for r in rows if r["N"] == N], key=lambda x: x["v"])
        v = [r["v"] for r in data]
        errE = [abs(r["E_HF_per_site"] - r["E0_ED_per_site"]) for r in data]
        plt.plot(v, errE, marker="o", label=f"N={N}")

    plt.xlabel("v = V/J")
    plt.ylabel(r"|E_HF/N - E_ED/N|")
    plt.title("Error in GS energy per site")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_error_energy.png", dpi=200)
    plt.show()

    # ---- Plot 2: gap error vs v ----
    plt.figure()
    for N in Ns:
        data = sorted([r for r in rows if r["N"] == N], key=lambda x: x["v"])
        v = [r["v"] for r in data]
        errGap = [abs(r["gap_HF_sp"] - r["gap_ED"]) for r in data]
        plt.plot(v, errGap, marker="o", label=f"N={N}")

    plt.xlabel("v = V/J")
    plt.ylabel(r"|gap_HF - gap_exact|")
    plt.title("Gap difference (HF one-particle vs ED many body)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_error_gap.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    N_list = [8, 10, 12]
    v_list = [0.0, 1.0, 2.0, 3.0]

    csv_path = run_grid_to_csv(N_list, v_list, filename="ed_hf_grid.csv")
    print("Wrote CSV:", csv_path)

    plot_errors(csv_path, save_prefix="ed_hf")

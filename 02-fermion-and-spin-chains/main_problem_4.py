
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Sampling disorder (mu_i)
# -------------------------
def sample_mu(
    N: int,
    W: float = 1.0,
    mu0: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(N)
    return mu0 + W * (u - 0.5)


# -------------------------
# Build Hamiltonian (PBC)
# -------------------------
def build_hamiltonian_pbc(mu: np.ndarray, t: float = 1.0) -> np.ndarray:
    N = mu.size
    H = np.zeros((N, N), dtype=float)
    np.fill_diagonal(H, mu)

    idx = np.arange(N - 1)
    H[idx, idx + 1] = -t
    H[idx + 1, idx] = -t

    # PBC wrap
    H[0, N - 1] = -t
    H[N - 1, 0] = -t
    return H


# -------------------------
# Diagonalize and compute P_k
# -------------------------
def diagonalize_and_ipr(mu: np.ndarray, t: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    H = build_hamiltonian_pbc(mu, t=t)
    E, V = np.linalg.eigh(H)     # columns are eigenvectors
    P = np.sum(np.abs(V) ** 4, axis=0)  # P_k = sum_i |psi_{i,k}|^4
    return E, P


# -------------------------
# Results container
# -------------------------
@dataclass
class PRResults:
    N: int
    t: float
    W: float
    mu0: float
    R: int

    mean_Ek: np.ndarray
    mean_Pk: np.ndarray
    stderr_Pk: np.ndarray

    E_bins: np.ndarray | None = None
    mean_P_of_E: np.ndarray | None = None
    stderr_P_of_E: np.ndarray | None = None


# -------------------------
# Disorder averaging
# -------------------------
def compute_pr_statistics(
    N: int,
    W: float,
    R: int = 500,
    t: float = 1.0,
    mu0: float = 0.5,
    seed: int | None = 0,
    energy_bins: int | None = 60,
) -> PRResults:
    rng = np.random.default_rng(seed)

    Es = np.empty((R, N), dtype=float)
    Ps = np.empty((R, N), dtype=float)

    for r in range(R):
        mu = sample_mu(N, W=W, mu0=mu0, rng=rng)
        E, P = diagonalize_and_ipr(mu, t=t)
        Es[r] = E
        Ps[r] = P

    mean_Ek = Es.mean(axis=0)
    mean_Pk = Ps.mean(axis=0)
    stderr_Pk = Ps.std(axis=0, ddof=1) / np.sqrt(R)

    res = PRResults(
        N=N, t=t, W=W, mu0=mu0, R=R,
        mean_Ek=mean_Ek,
        mean_Pk=mean_Pk,
        stderr_Pk=stderr_Pk,
    )

    # Energy-binned averages
    if energy_bins is not None and energy_bins > 5:
        E_all = Es.ravel()
        P_all = Ps.ravel()

        E_min, E_max = float(E_all.min()), float(E_all.max())
        bins = np.linspace(E_min, E_max, energy_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        b = np.digitize(E_all, bins) - 1
        valid = (b >= 0) & (b < energy_bins)
        b = b[valid]
        P_all = P_all[valid]

        mean_P_of_E = np.full(energy_bins, np.nan)
        stderr_P_of_E = np.full(energy_bins, np.nan)

        for j in range(energy_bins):
            mask = (b == j)
            if np.any(mask):
                pj = P_all[mask]
                mean_P_of_E[j] = pj.mean()
                stderr_P_of_E[j] = pj.std(ddof=1) / np.sqrt(pj.size) if pj.size > 1 else 0.0

        res.E_bins = bin_centers
        res.mean_P_of_E = mean_P_of_E
        res.stderr_P_of_E = stderr_P_of_E

    return res


# -------------------------
# Localization length proxy
# -------------------------
def xi_eff_from_P(P: np.ndarray, cap: float | None = None) -> np.ndarray:
    xi = 1.0 / np.maximum(P, 1e-300)
    if cap is not None:
        xi = np.minimum(xi, cap)
    return xi


# -------------------------
# Plot + save helpers
# -------------------------
def ensure_outdir(outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_fig(outpath: Path, dpi: int = 300) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


# -------------------------
# Plotting functions (results pack)
# -------------------------
def plot_Pk_vs_k(results_list: list[PRResults], show_errorbars: bool = False) -> None:
    plt.figure()
    for res in results_list:
        k = np.arange(1, res.N + 1)
        if show_errorbars:
            plt.errorbar(k, res.mean_Pk, yerr=res.stderr_Pk, capsize=2, label=f"N={res.N}, W={res.W}")
        else:
            plt.plot(k, res.mean_Pk, label=f"N={res.N}, W={res.W}")
    plt.xlabel("k (energy-ordered index)")
    plt.ylabel(r"$\langle P_k\rangle$  (log scale)")
    plt.yscale("log")
    plt.title(r"$\langle P_k\rangle$ vs $k$ (log y-axis)")
    plt.legend()
    plt.tight_layout()



def plot_NPk_vs_k(results_list: list[PRResults], show_errorbars: bool = False) -> None:
    plt.figure()
    for res in results_list:
        k = np.arange(1, res.N + 1)
        y = res.N * res.mean_Pk
        if show_errorbars:
            plt.errorbar(k, y, yerr=res.N * res.stderr_Pk, capsize=2, label=f"N={res.N}, W={res.W}")
        else:
            plt.plot(k, y, label=f"N={res.N}, W={res.W}")
    plt.xlabel("k (energy-ordered index)")
    plt.ylabel(r"$N\,\langle P_k\rangle$")
    plt.legend()
    plt.tight_layout()


def plot_xi_eff_k(results_list: list[PRResults], cap_to_N: bool = True) -> None:
    plt.figure()
    for res in results_list:
        k = np.arange(1, res.N + 1)
        cap = float(res.N) if cap_to_N else None
        xi = xi_eff_from_P(res.mean_Pk, cap=cap)
        plt.plot(k, xi, label=f"N={res.N}, W={res.W}")
    plt.xlabel("k (energy-ordered index)")
    plt.ylabel(r"$\xi_{\mathrm{eff}}(k)\approx 1/\langle P_k\rangle$  (log scale)")
    plt.yscale("log")
    plt.title(r"$\xi_{\mathrm{eff}}(k)$ vs $k$ (log y-axis)")
    plt.legend()
    plt.tight_layout()


def plot_P_of_E(results_list: list[PRResults]) -> None:
    plt.figure()
    for res in results_list:
        if res.E_bins is None or res.mean_P_of_E is None:
            continue
        plt.plot(res.E_bins, res.mean_P_of_E, label=f"N={res.N}, W={res.W}")
    plt.xlabel("Energy E")
    plt.ylabel(r"$\langle P(E)\rangle$  (log scale)")
    plt.yscale("log")
    plt.title(r"$\langle P(E)\rangle$ (log y-axis)")
    plt.legend()
    plt.tight_layout()


def plot_xi_eff_E(results_list: list[PRResults], cap_to_N: bool = True) -> None:
    plt.figure()
    for res in results_list:
        if res.E_bins is None or res.mean_P_of_E is None:
            continue
        cap = float(res.N) if cap_to_N else None
        xiE = xi_eff_from_P(res.mean_P_of_E, cap=cap)
        plt.plot(res.E_bins, xiE, label=f"N={res.N}, W={res.W}")
    plt.xlabel("Energy E")
    plt.ylabel(r"$\xi_{\mathrm{eff}}(E)\approx 1/\langle P(E)\rangle$  (log scale)")
    plt.yscale("log")
    plt.title(r"$\xi_{\mathrm{eff}}(E)$ (log y-axis)")
    plt.legend()
    plt.tight_layout()


# -------------------------
# Main: compute + save all plots
# -------------------------
if __name__ == "__main__":
    outdir = ensure_outdir("figures")

    # Parameters
    Ns = [100, 200, 400]
    Ws = [0.25, 0.5, 1.0]
    R = 300
    t = 1.0
    mu0 = 0.5
    seed = 123
    energy_bins = 60

    # -------------------------
    # A) Fixed W, varying N
    # -------------------------
    fixed_W = 1.0
    res_fixedW = [
        compute_pr_statistics(N=N, W=fixed_W, R=R, t=t, mu0=mu0, seed=seed, energy_bins=energy_bins)
        for N in Ns
    ]

    plot_Pk_vs_k(res_fixedW, show_errorbars=False)
    save_fig(outdir / f"Pk_vs_k_fixedW_{fixed_W}_Ns_{'-'.join(map(str, Ns))}_R{R}.png")

    plot_NPk_vs_k(res_fixedW, show_errorbars=False)
    save_fig(outdir / f"NPK_vs_k_fixedW_{fixed_W}_Ns_{'-'.join(map(str, Ns))}_R{R}.png")

    plot_xi_eff_k(res_fixedW, cap_to_N=True)
    save_fig(outdir / f"xi_eff_k_fixedW_{fixed_W}_Ns_{'-'.join(map(str, Ns))}_R{R}.png")

    plot_P_of_E(res_fixedW)
    save_fig(outdir / f"P_of_E_fixedW_{fixed_W}_Ns_{'-'.join(map(str, Ns))}_R{R}.png")

    plot_xi_eff_E(res_fixedW, cap_to_N=True)
    save_fig(outdir / f"xi_eff_E_fixedW_{fixed_W}_Ns_{'-'.join(map(str, Ns))}_R{R}.png")

    # -------------------------
    # B) Fixed N, varying W
    # -------------------------
    fixed_N = 200
    res_fixedN = [
        compute_pr_statistics(N=fixed_N, W=W, R=R, t=t, mu0=mu0, seed=seed, energy_bins=energy_bins)
        for W in Ws
    ]

    plot_Pk_vs_k(res_fixedN, show_errorbars=False)
    save_fig(outdir / f"Pk_vs_k_fixedN_{fixed_N}_Ws_{'-'.join(map(str, Ws))}_R{R}.png")

    plot_NPk_vs_k(res_fixedN, show_errorbars=False)
    save_fig(outdir / f"NPK_vs_k_fixedN_{fixed_N}_Ws_{'-'.join(map(str, Ws))}_R{R}.png")

    plot_xi_eff_k(res_fixedN, cap_to_N=True)
    save_fig(outdir / f"xi_eff_k_fixedN_{fixed_N}_Ws_{'-'.join(map(str, Ws))}_R{R}.png")

    plot_P_of_E(res_fixedN)
    save_fig(outdir / f"P_of_E_fixedN_{fixed_N}_Ws_{'-'.join(map(str, Ws))}_R{R}.png")

    plot_xi_eff_E(res_fixedN, cap_to_N=True)
    save_fig(outdir / f"xi_eff_E_fixedN_{fixed_N}_Ws_{'-'.join(map(str, Ws))}_R{R}.png")

    print(f"Saved all result plots to: {outdir.resolve()}")

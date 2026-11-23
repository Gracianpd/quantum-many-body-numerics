import numpy as np
import matplotlib.pyplot as plt

def page_mean_approx(dA, dB):

    if dA > dB:
        dA, dB = dB, dA
    return np.log(dA) - dA/(2*dB)

def haar_random_state(dim, rng):
    psi = rng.normal(size=dim) + 1j*rng.normal(size=dim)
    return psi / np.linalg.norm(psi)

def entanglement_entropy_first_k_qubits(psi, N, k):
    dA = 2**k
    dB = 2**(N-k)
    M = psi.reshape(dA, dB)
    rhoA = M @ M.conj().T
    evals = np.linalg.eigvalsh(rhoA)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))  # nats

def run(N=8, Ns=50000, ks=(1,2,3,4), bins=60, seed=0):
    rng = np.random.default_rng(seed)
    dim = 2**N

    S = {k: np.empty(Ns, dtype=float) for k in ks}

    for s in range(Ns):
        psi = haar_random_state(dim, rng)
        for k in ks:
            S[k][s] = entanglement_entropy_first_k_qubits(psi, N, k)

    # Print summary + Page approx check + deltas
    print("k | mean(S)   std(S)    Page_approx  Δ_num=mean-Page  Δ_max=log(d*)-Page   dA  dB")
    for k in ks:
        dA, dB = 2**k, 2**(N-k)
        d_small = min(dA, dB)
        d_large = max(dA, dB)

        page = page_mean_approx(dA, dB)
        meanS = S[k].mean()
        stdS = S[k].std()

        delta_num = meanS - page
        delta_max = np.log(d_small) - page

        print(f"{k} | {meanS:.6f}  {stdS:.6f}  {page:.6f}     {delta_num:+.2e}        {delta_max:.6f}         {dA:3d} {dB:3d}")

    # Histograms + vertical lines
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, k in zip(axes, ks):
        ax.hist(S[k], bins=bins, density=True)

        dA, dB = 2**k, 2**(N-k)
        d_small = min(dA, dB)

        page = page_mean_approx(dA, dB)
        meanS = S[k].mean()

        Smax = np.log(d_small)
        ax.axvline(meanS, linestyle="-", linewidth=2,
                   label=f"Numerical = {meanS:.4f}", color="red")
        ax.axvline(page, linestyle="--", linewidth=2,
                   label=f"Page approx = {page:.4f}", color="green")
        ax.axvline(Smax, linestyle=":", linewidth=2,
                   label=f"Smax = log(d*) = {Smax:.4f}", color="blue")

        ax.set_title(f"N={N}, block size k={k}")
        ax.set_xlabel("S (nats)")
        ax.set_ylabel("pdf")
        ax.legend()

    plt.show()

    return S, hists

if __name__ == "__main__":
    S, hists = run()

import os
os.makedirs("Outputs", exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
def plot_ground_state(Gamma_values, E0_list, N):
    #plot ground states for fun
    plt.figure(figsize=(7,5))
    plt.plot(Gamma_values, E0_list, color='purple', lw=2)
    plt.xlabel("Γ", fontsize=12)
    plt.ylabel("Energía del estado fundamental $E_0$", fontsize=12)
    plt.title(f"Ising en campo transversal (N={N})", fontsize=13)
    plt.axvline(1.0, color='k', ls=':', label='Γc ≈ 1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Outputs/ground_state.png")
    #plt.show()


def plot_magnetization(Gamma_values, mx_list, mz_list, N):
    #plot magnetizations for excercise
    plt.figure(figsize=(7,5))
    plt.plot(Gamma_values, mx_list, label="⟨σx⟩", color='tab:blue', lw=2)
    plt.plot(Gamma_values, mz_list, label="⟨σz⟩", color='tab:red', ls='--', lw=2)
    plt.axvline(1.0, color='k', ls=':', label='Γc ≈ 1')
    plt.xlabel("Γ", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title(f"Average Magnetizations (N={N})", fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Outputs/magnetizations_{N}.png")
    #plt.show()

def plot_correlation_heat(C, Gamma, N, op_string):
    plt.figure(figsize=(4, 4))
    plt.imshow(np.real(C), cmap="OrRd", origin="lower", vmin=0, vmax=1)
    plt.colorbar(label=rf"$\langle\sigma_i^{op_string}\sigma_j^{op_string}\rangle$")
    plt.title(rf"$C^{{{op_string}}}$ (Γ={Gamma:.2f}, N={N})")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    #plt.show()


def plot_correlation_grid(Cz_list, Cx_list, Gammas, N):

    fig, axes = plt.subplots(2, len(Gammas), figsize=(3*len(Gammas), 6))

    for idx, Gamma in enumerate(Gammas):
        Cz = Cz_list[idx]
        Cx = Cx_list[idx]
        # Top row: C^z
        axes[0, idx].imshow(np.real(Cz), cmap="Reds", origin="lower", vmin=0, vmax=1)
        axes[0, idx].set_title(rf"$C^z$ (Γ={Gamma:.1f})")
        axes[0, idx].set_xticks([]); axes[0, idx].set_yticks([])
        # Bottom row: C^x
        im = axes[1, idx].imshow(np.real(Cx), cmap="Reds", origin="lower", vmin=0, vmax=1)
        axes[1, idx].set_title(rf"$C^x$ (Γ={Gamma:.1f})")
        axes[1, idx].set_xticks([]); axes[1, idx].set_yticks([])
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax, label=r"$\langle\sigma_i^\alpha \sigma_j^\alpha\rangle$")
    fig.suptitle(rf"Correlations $\langle\sigma^z\sigma^z\rangle$ and $\langle\sigma^x\sigma^x\rangle$ for N={N}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(f"Outputs/correlations_heat_{N}.png")
    #plt.show()


def plot_energy_gap(Gamma_values, gap_dict):
    plt.figure(figsize=(6, 4))

    for N, gap_list in gap_dict.items():
        plt.plot(Gamma_values, gap_list, lw=2, label=rf"N={N}")

    plt.axvline(1.0, color='black', linestyle='--', label=r"Γ$_c$ ≈ 1")

    plt.xlabel(r"Γ")
    plt.ylabel("Energy gap ΔE")
    plt.title("Energy gap vs Γ")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("Outputs/energy_gap_all.png")
    plt.show()

def plot_entanglement_all(Gamma_values, S_dict):
    plt.figure(figsize=(8,5))

    for N, S_list in S_dict.items():
        plt.plot(Gamma_values, S_list, lw=2, label=f"N = {N}")

    plt.axvline(1.0, color='k', ls=':', lw=1.5, label=r'Critical point $\Gamma_c \approx 1$')
    plt.xlabel(r"Transverse field $\Gamma$")
    plt.ylabel(r"Entanglement entropy [bits]")
    plt.title("Entanglement entropy for different system sizes")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("Outputs/entanglement.png")
    plt.show()
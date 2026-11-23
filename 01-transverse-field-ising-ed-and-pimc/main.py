from buildhamiltonian import *
from plotter import *
from entanglement import *
import numpy as np
N = [4,6,8]
energy_gaps = {}
S_dict = {}
for N in N:
    E0_list, gap_list, mx_list, mz_list, S_list = [], [], [], [], []
    Hsz, Hsx = build_hamiltonian(N, periodic=True)
    Gamma_range = np.linspace(0,2,101)
    Gamma_heat = [0, 0.5, 1, 1.5, 2.0, 4.0]
    prev_ground = None
    for Gamma in Gamma_range:
        H = Hsz + Gamma * Hsx
        evals, evecs = calculate_eigenv(H)
        ground = evecs[:,0]

        prev_ground = ground
        E0 = evals[0]
        E0_list.append(E0)
        gap_list.append(evals[1]-evals[0])

        mx = 0
        mz = 0
        sx = np.array([[0, 1], [1, 0]])
        sz = np.array([[1, 0], [0, -1]])
        for i in range(N):
            # magnetizaztions
            sxi = calcular_operadores_pos_i(sx, i, N)
            szi = calcular_operadores_pos_i(sz, i, N)
            mx += valor_esperado(sxi, ground)
            mz += valor_esperado(szi, ground)


        mx_list.append(mx / N)
        mz_list.append(mz / N)

        S = entanglement_entropy_half_explicit(ground, N)
        S_list.append(S)

    S_dict[N] = S_list
    energy_gaps[N] = gap_list
    #plot_ground_state(Gamma_range, E0_list, N)
    plot_magnetization(Gamma_range, mx_list, mz_list, N)

    #plot gaps
    #plot correlation
    Cz_list, Cx_list = [], []
    for Gamma in Gamma_heat:
        H = Hsz + Gamma * Hsx
        evals, evecs = calculate_eigenv(H)
        ground = evecs[:, 0]

        Cz = correlation_matrix(N, sz, ground)
        Cx = correlation_matrix(N, sx, ground)
        Cz_list.append(Cz)
        Cx_list.append(Cx)
    #plot_correlation_grid(Cz_list, Cx_list, Gamma_heat, N)
plot_entanglement_all(Gamma_range, S_dict)
plot_energy_gap(Gamma_range, energy_gaps)

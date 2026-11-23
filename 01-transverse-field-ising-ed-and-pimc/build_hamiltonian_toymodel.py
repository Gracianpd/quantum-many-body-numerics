import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

a = np.array([[0, 1], [0, 0]])
at = np.array([[0, 0], [1, 0]])
I = np.eye(2)

def calcular_operadores_pos_i(op, i, N):
    factors = [I] * N
    factors[i] = op
    producto_kronecker = reduce(np.kron, factors)
    return producto_kronecker

def build_hamiltonian(N, periodic):
    dim = 2 ** N
    H = np.zeros((dim, dim))
    #construir
    for i in range(0, N-1):
        ai = calcular_operadores_pos_i(a, i, N)
        ai_next = calcular_operadores_pos_i(a, i + 1, N)
        ait = calcular_operadores_pos_i(at, i, N)
        ait_next = calcular_operadores_pos_i(at, i + 1, N)

        H -= ait @ ai_next + ait_next @ ai
    #condiciones periodicas
    if periodic:
        a0 = calcular_operadores_pos_i(a, 0, N)
        aN = calcular_operadores_pos_i(a, N - 1, N)
        at0 = calcular_operadores_pos_i(at, 0, N)
        atN = calcular_operadores_pos_i(at, N - 1, N)
        H -= atN @ a0 + at0 @ aN
    return H

def compute_spectrum(N, periodic=False):
    H = build_hamiltonian(N, periodic=periodic)
    eigvals, _ = np.linalg.eigh(H)
    return np.sort(np.real_if_close(eigvals))

sizes = [4, 6, 8]
spectra = {}

for N in sizes:
    spectra[N] = compute_spectrum(N, periodic=False)


plt.figure(figsize=(7, 4))

for N in sizes:
    eigvals = spectra[N]
    dim = len(eigvals)
    x = np.arange(dim) / dim
    plt.plot(x, eigvals, marker='o', markersize=3, linestyle='-', label=f'N={N}')

plt.xlabel('Normalized eigenvalue index, i / 2^N')
plt.ylabel('Energy [level], E_i')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
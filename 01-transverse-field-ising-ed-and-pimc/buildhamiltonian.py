import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

#pauli matrices
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
I = np.eye(2)

def calcular_operadores_pos_i(op, i, N):
    factors = [I] * N
    factors[i] = op
    producto_kronecker = reduce(np.kron, factors)
    return producto_kronecker

def build_hamiltonian(N, periodic=False):
    #this function calculates the "pieces" of the hamiltonian Hsz and Hsx
    dim = 2 ** N
    H_sz = np.zeros((dim, dim))
    H_sx = np.zeros((dim, dim))
    #contruir parte de acomplamiento
    for i in range(N - 1):
        szi = calcular_operadores_pos_i(sz, i, N)
        szi_next = calcular_operadores_pos_i(sz, i + 1, N)
        H_sz -= szi @ szi_next

    #condicion periodica: los ultimos seran los primeros
    if periodic:
        sz_last = calcular_operadores_pos_i(sz, N - 1, N)
        sz_first = calcular_operadores_pos_i(sz, 0, N)
        H_sz -= sz_last @ sz_first

    #construir parte de interaccion
    for i in range(N):
        sxi = calcular_operadores_pos_i(sx, i, N)
        H_sx -= sxi
    return H_sz, H_sx

def calculate_eigenv(H):
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs

def valor_esperado(op, vector):
    return np.real(np.vdot(vector, op @ vector))

def correlation_matrix(N, op, ground):
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            oi = calcular_operadores_pos_i(op, i, N)
            oj = calcular_operadores_pos_i(op, j, N)
            C[i,j] = np.real(np.vdot(ground,(oi@oj) @ ground))
    return C


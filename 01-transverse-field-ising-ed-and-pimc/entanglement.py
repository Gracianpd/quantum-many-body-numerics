import numpy as np
from itertools import product

def reduced_density_matrix_half_explicit(ground, N):

    m = N // 2  # size of left half
    dimA, dimB = 2**m, 2**m

    # Global density matrix ρ = |ψ><ψ|
    rho_global = np.outer(ground, ground.conj())

    rhoA = np.zeros((dimA, dimA), dtype=complex)

    for IA_bits in product([0, 1], repeat=m):
        IA = sum(b << (m - 1 - i) for i, b in enumerate(IA_bits))
        for JA_bits in product([0, 1], repeat=m):
            JA = sum(b << (m - 1 - i) for i, b in enumerate(JA_bits))
            val = 0.0 + 0.0j
            for KB_bits in product([0, 1], repeat=m):  # right-half states
                KB = sum(b << (m - 1 - i) for i, b in enumerate(KB_bits))
                # Compose indices: I = (IA_bits : KB_bits)
                I = (IA << m) + KB
                J = (JA << m) + KB
                val += rho_global[I, J]
            rhoA[IA, JA] = val

    # Normalize (Tr ρA = 1)
    rhoA /= np.trace(rhoA)
    return rhoA


def entanglement_entropy_half_explicit(psi, N):
    rhoA = reduced_density_matrix_half_explicit(psi, N)
    eigs = np.linalg.eigvalsh(rhoA)
    eigs = np.clip(np.real(eigs), 0.0, 1.0)
    nonzero = eigs[eigs > 1e-15]
    S = -np.sum(nonzero * np.log2(nonzero))
    return S.real
import os
os.makedirs("Outputs", exist_ok=True)
import numpy as np
import math
import matplotlib.pyplot as plt


# ============================================================
#  PARAMETERS
# ============================================================

T_MINUS_1 = 1000.0          # inverse temperature
N = 2000         # number of beads
eps = T_MINUS_1 / N      # imaginary-time step
sweeps = 20000       # production sweeps
stride = 10          # record every 'stride' sweeps

rng = np.random.default_rng(1234)

# Proposal window A (adaptive)
A = 1.0


# ============================================================
#  POTENTIAL AND LOCAL ACTION
# ============================================================

def V(x):
    #POTENTIAL
    return 0.5 * x * x

def kinetic_term(k, xk, x, eps):
    #KINETIC
    km1 = (k - 1) % N
    kp1 = (k + 1) % N
    kin = ((xk - x[km1])**2 + (x[kp1] - xk)**2) / (2 * eps)
    pot = eps * V(xk)
    return kin + pot

# ============================================================
#  METROPOLIS SWEEP
# ============================================================

def sweep(x, A):
    accepted = 0
    for _ in range(N):
        k = rng.integers(0, N)
        km1 = (k - 1) % N
        kp1 = (k + 1) % N

        x_old = x[k]
        S_old = kinetic_term(k, x_old, x, eps)

        # Propose x'
        x_new = x_old + rng.uniform(-A, A)
        S_new = kinetic_term(k, x_new, x, eps)

        dS = S_new - S_old
        if dS <= 0.0 or rng.random() < math.exp(-dS):
            x[k] = x_new
            accepted += 1

    return accepted, N


# ============================================================
#  ADAPT PROPOSAL A
# ============================================================

def adapt_step(A, acc, prop, low=0.3, high=0.6, factor=1.1):
    #adapts the acceptance window to be ~0.5
    if prop == 0:
        return A
    rate = acc / prop
    if rate > high:
        A *= factor
    elif rate < low:
        A /= factor
    return A


# ============================================================
#  MAIN PIMC
# ============================================================

# Initialize ring polymer near zero
x = rng.normal(0.0, 0.05, size=N)

positions = []   # beads to build histogram

acc_sum = 0 #number of accepted
prop_sum = 0 #number of porposed

# --------------------------
# Production sweeps
# --------------------------
accept_rates = []

for s in range(sweeps):
    acc, prop = sweep(x, A)
    acc_sum += acc
    prop_sum += prop

    # mild adaptation
    if (s % 50) == 0 and prop_sum > 0:
        A = adapt_step(A, acc_sum, prop_sum)
        accept_rates.append(acc_sum / prop_sum)
        acc_sum = 0
        prop_sum = 0

    # record beads
    if (s % stride) == 0:
        positions.append(x.copy())

positions = np.concatenate(positions)
avg_accept = np.mean(accept_rates)


# ============================================================
#  PLOT RESULTS
# ============================================================

# exact GS probability density
X = np.linspace(-3.5, 3.5, 400)
psi2 = (1 / np.sqrt(np.pi)) * np.exp(-X**2)

plt.figure(figsize=(8,5))
plt.hist(positions, bins=80, density=True, alpha=0.6, label="PIMC")
plt.plot(X, psi2, lw=2, label=r"Exact $|\psi_0(x)|^2$")
plt.xlabel("x")
plt.ylabel("Probability density")
T= 1 / T_MINUS_1
plt.title(f"PIMC 1D HO (T={T}, P={N})\nAcceptance ≈ {avg_accept:.2f}")
plt.legend()
plt.savefig("Outputs/PIMC_0.001deg.png")
plt.show()


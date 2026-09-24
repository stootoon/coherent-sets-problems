"""Problem 13: past-future CCA on simulated irreversible OU data recovers Eq. (18)."""
import numpy as np
from scipy.linalg import expm, solve_sylvester
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

c = 1.0
A = np.array([[-1.0, 0.0], [c, -2.0]]); D = np.eye(2)
Sigma = solve_sylvester(A, A.T, -D)
print("Sigma (analytic) =\n", Sigma, "\nA Sigma =\n", A @ Sigma, "  (not symmetric -> irreversible)")

# Euler-Maruyama simulation
dt, T, tau = 0.01, 4000.0, 0.5
n = int(T/dt); k = int(round(tau/dt))
x = np.zeros((n, 2)); x[0] = 0
noise = rng.standard_normal((n, 2)) * np.sqrt(dt)
for t in range(n-1):
    x[t+1] = x[t] + A @ x[t] * dt + noise[t]
x = x[n//10:]                                    # discard transient

# empirical covariance matrices (Eq. 23):  Sigma_tau[i,j] = E[x_i(t) x_j(t+tau)]
X0, X1 = x[:-k], x[k:]
S0 = (X0.T @ X0 + X1.T @ X1) / (2*len(X0))
St = X0.T @ X1 / len(X0)
Smt = St.T
F_hat = np.linalg.solve(S0, St) @ np.linalg.solve(S0, Smt)   # Eq. (24)
B_hat = np.linalg.solve(S0, Smt) @ np.linalg.solve(S0, St)

# analytic Eq. (18)
E = expm(A*tau); Si = np.linalg.inv(Sigma)
MT = E.T @ Si @ E @ Sigma; NT = Si @ E @ Sigma @ E.T

def sorted_eig(M):
    w, V = np.linalg.eig(M); o = np.argsort(-w.real)
    V = V[:, o].real; V /= np.sign(V[0]) * np.linalg.norm(V, axis=0); return w[o].real, V

for name, Hhat, Han in [("F", F_hat, MT), ("B", B_hat, NT)]:
    wh, Vh = sorted_eig(Hhat); wa, Va = sorted_eig(Han)
    print(f"\n{name}: eigenvalues data {wh}  analytic {wa}")
    print(f"   eigenvectors data\n{Vh}\n   analytic\n{Va}")

# CCA check: canonical correlations between x_t and x_{t+tau}
Lh = np.linalg.cholesky(S0)
C = np.linalg.solve(Lh, np.linalg.solve(Lh, St).T).T    # S0^{-1/2} St S0^{-1/2} (via Cholesky whitening)
rho = np.linalg.svd(C, compute_uv=False)
print("\ncanonical correlations^2 =", rho**2, " vs eig(F_hat) =", sorted_eig(F_hat)[0])

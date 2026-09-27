"""Problem 13: Galerkin projection of the Koopman operator for a 1-D OU process.

dx = -x dt + sqrt(2) dW  (stationary N(0,1)), lag tau.
Exact action on monomials: K 1 = 1, K x = e^{-tau} x, K x^2 = (1-e^{-2tau}) + e^{-2tau} x^2.
Feature set (i)  {x, x^2}      : K x^2 leaves the span (constant dropped) -> projection error.
Feature set (ii) {1, x, x^2}   : span is invariant -> Galerkin matrix is exact.
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

dt, T, tau = 0.01, 20000.0, 0.5
n = int(T/dt); k = int(round(tau/dt))
x = np.zeros(n); noise = rng.standard_normal(n)*np.sqrt(2*dt)
for t in range(n-1):
    x[t+1] = x[t] - x[t]*dt + noise[t]
x = x[n//10:]
X0, X1 = x[:-k], x[k:]

def galerkin(feats):
    """Eq. (22)-(23): Sigma_0^{-1} Sigma_tau from lagged feature covariances."""
    P0 = np.stack([f(X0) for f in feats], 1); P1 = np.stack([f(X1) for f in feats], 1)
    S0 = (P0.T@P0 + P1.T@P1)/(2*len(P0)); St = P0.T@P1/len(P0)
    return np.linalg.solve(S0, St)

e1, e2 = np.exp(-tau), np.exp(-2*tau)
print("feature set (i) {x, x^2}")
K_i = galerkin([lambda z: z, lambda z: z**2])
print("  Galerkin K (data) =\n", K_i)
print("  analytic          =\n", np.diag([e1, (1+2*e2)/3]))
print(f"  true coefficient of x^2 in K x^2 = e^(-2 tau) = {e2:.4f}; Galerkin gives (1+2e^(-2tau))/3 = {(1+2*e2)/3:.4f}")

print("\nfeature set (ii) {1, x, x^2}")
K_ii = galerkin([lambda z: np.ones_like(z), lambda z: z, lambda z: z**2])
print("  Galerkin K (data) =\n", K_ii)
print("  analytic (exact)  =\n", np.array([[1, 0, 1-e2], [0, e1, 0], [0, 0, e2]]))
print("  eigenvalues data", np.sort(np.linalg.eigvals(K_ii).real)[::-1], " analytic", np.array([1, e1, e2]))

"""Problem 12: singular vectors of Eq. (18) at a 2-D saddle, as a function of the horizon tau."""
import numpy as np
from scipy.linalg import expm, solve_sylvester
np.set_printoptions(precision=4, suppress=True)

a, b = 1.0, 0.5                                   # stable rate a, unstable rate b (a != b)
e_s = np.array([1.0, 0.0]); e_u = np.array([1.0, 1.0]) / np.sqrt(2)   # oblique eigenvectors
S = np.column_stack([e_s, e_u])
A = S @ np.diag([-a, b]) @ np.linalg.inv(S)
D = np.eye(2)
Sigma = solve_sylvester(A, A.T, -D)              # A Sigma + Sigma A^T = -D
print("A =\n", A, "\nSigma =\n", Sigma, "\neig(Sigma) =", np.linalg.eigvalsh(Sigma), " (indefinite)")
print("residual of Sylvester eq:", np.abs(A @ Sigma + Sigma @ A.T + D).max())
Si = np.linalg.inv(Sigma)
l_u = np.linalg.inv(S)[1]; l_u /= np.linalg.norm(l_u)   # left eigenvector for +b (orthogonal to e_s)

def angle(x, y):
    return np.degrees(np.arccos(abs(x @ y) / np.linalg.norm(x) / np.linalg.norm(y)))

print("\n tau   eig(M^T)                 ang(v1, e_s)  ang(v1, l_u)  ang(u2, e_u)")
for tau in [0.25, 0.5, 1, 2, 4, 8]:
    E = expm(A * tau)
    MT = E.T @ Si @ E @ Sigma                    # Eq. (18a):  MT v = lambda^2 v
    NT = Si @ E @ Sigma @ E.T                    # Eq. (18b):  NT u = lambda^2 u
    wv, Vv = np.linalg.eig(MT); wu, Vu = np.linalg.eig(NT)
    v1 = Vv[:, np.argmax(wv.real)].real           # expanding direction (largest lambda^2)
    u2 = Vu[:, np.argmin(wu.real)].real           # contracting direction (smallest lambda^2)
    print(f"{tau:5.2f}  {np.sort(wv.real)}   {angle(v1, e_s):7.2f}     {angle(v1, l_u):7.2f}      {angle(u2, e_u):7.2f}")
print("\ncomplex parts of eigenvalues (max):", max(abs(np.linalg.eigvals(expm(A*t).T @ Si @ expm(A*t) @ Sigma).imag).max() for t in np.linspace(0.05, 8, 200)))

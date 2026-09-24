"""Problem 11(d): spectrum of the 1-D OU backward generator L^dag g = -a x g' + g''/2 on a grid."""
import numpy as np
from numpy.polynomial.hermite_e import hermeval
a = 1.0; sigma = np.sqrt(1/(2*a))
N = 801; L = 7.0; x = np.linspace(-L, L, N); h = x[1] - x[0]
D1 = (np.eye(N, k=1) - np.eye(N, k=-1)) / (2*h)
D2 = (np.eye(N, k=1) - 2*np.eye(N) + np.eye(N, k=-1)) / h**2
Ldag = -a * np.diag(x) @ D1 + 0.5 * D2
w, V = np.linalg.eig(Ldag)
idx = np.argsort(-w.real)[:5]
print("top eigenvalues of L^dag:", np.round(w[idx].real, 3), " expected -n a =", [-n*a for n in range(5)])
for n, i in enumerate(idx):
    g = V[:, i].real; He = hermeval(x/sigma, [0]*n + [1])
    mask = abs(x) < 3
    c = 1.0 if n == 0 else np.corrcoef(g[mask], He[mask])[0, 1]
    print(f"n={n}: |corr(eigvec, He_n(x/sigma))| = {abs(c):.4f}")

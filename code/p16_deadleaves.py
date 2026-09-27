"""Problem 16: why the transient (derivative) filter of the paper's Fig. 2 is predictive.

Piecewise-constant 'dead-leaves' signal: independent N(0,1) levels on plateaus whose lengths are
  (a) exponential (memoryless)            -> flat hazard, only the level is linearly predictable;
  (b) Pareto, alpha = 1.5 (heavy-tailed)  -> decreasing hazard: young plateaus end sooner.
Past-future CCA on lag vectors of length d with horizon tau = d (adjacent windows).
"""
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=160)
rng = np.random.default_rng(0)

def signal(n, kind, mean_len=20.0):
    y = np.empty(n); age = np.empty(n); i = 0
    while i < n:
        L = rng.exponential(mean_len) if kind == "exp" else 4.0*(1-rng.random())**(-1/1.5)
        L = min(max(1, int(round(L))), n-i); y[i:i+L] = rng.standard_normal(); age[i:i+L] = np.arange(L); i += L
    return y, age

def cca(X, Y):
    """Eq. (24) via the whitened cross-covariance; returns correlations, past/future directions (lag coords) and whitened ones."""
    S0 = (X.T@X + Y.T@Y)/(2*len(X)); St = X.T@Y/len(X)
    w, U = np.linalg.eigh(S0); W = U@np.diag(w**-0.5)@U.T            # S0^{-1/2}
    C = W@St@W                                                       # rows: past, columns: future
    Ub, s, Vt = np.linalg.svd(C)
    return s, W@Ub, W@Vt.T, Ub, Vt.T, C

d = tau = 12
c = lambda u, v: np.corrcoef(u, v)[0, 1]
for kind in ["exp", "pareto"]:
    y, age = signal(800_000, kind)
    n = len(y); idx = np.arange(d-1, n-tau)
    X = np.stack([y[idx-k] for k in range(d)], 1)          # lag vector at t     : (y(t), y(t-1), ..., y(t-d+1))
    Y = np.stack([y[idx+tau-k] for k in range(d)], 1)      # lag vector at t+tau : (y(t+tau), ..., y(t+1))
    s, A, B, At, Bt, C = cca(X, Y)
    J = np.eye(d)[::-1]                                    # lag-reversal permutation
    print(f"\n=== {kind} plateaus: canonical correlations {s[:4]}")
    print(f"    ||C - C^T|| = {np.abs(C-C.T).max():.3f}   ||C - J C^T J|| = {np.abs(C - J@C.T@J).max():.3f}   (reversal permutes lags)")
    for i in range(2):
        za, zb = X@A[:, i], Y@B[:, i]; sgn = np.sign(c(za, zb)); zb, bt = sgn*zb, sgn*Bt[:, i]
        level, jump, young = y[idx], y[idx]-y[idx-3], (age[idx] < 3).astype(float)
        print(f"  pair {i+1}  rho = {s[i]:.3f}")
        print(f"    past filter  (whitened, lag 0..{d-1}):", At[:, i]/np.abs(At[:, i]).max())
        print(f"    future filter(whitened, lag 0..{d-1}):", bt/np.abs(bt).max())
        print(f"    past variate  ~ level {c(za, level):+.2f}, recent jump y(t)-y(t-3) {c(za, jump):+.2f}, young plateau {c(za, young):+.2f}")
        print(f"    future variate ~ y(t+tau) {c(zb, y[idx+tau]):+.2f}, y(t+tau)-y(t) {c(zb, y[idx+tau]-y[idx]):+.2f}")
    for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 10**9)]:
        m = (age[idx] >= lo) & (age[idx] < hi)
        print(f"    P(level changes within tau | plateau age in [{lo},{hi})) = {np.mean(y[idx[m]+tau] != y[idx[m]]):.2f}")

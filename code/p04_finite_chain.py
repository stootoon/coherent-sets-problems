"""Problem 4: coherent sets of a 4-state chain via the SVD of the whitened transition matrix."""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def stationary(P):
    w, V = np.linalg.eig(P.T)
    pi = np.real(V[:, np.argmin(abs(w - 1))]); return pi / pi.sum()

def coherent_svd(P):
    pi = stationary(P)
    Dh = np.diag(np.sqrt(pi)); Dhi = np.diag(1/np.sqrt(pi))
    Pt = Dh @ P @ Dhi                      # whitened Koopman matrix
    U, s, Vh = np.linalg.svd(Pt)           # Pt = U diag(s) Vh
    # Pt (Vh[i]) = s_i U[:, i]  ->  paper's  K u_i = lambda_i v_i  with
    #   u_i (future, time t+1)  = Dhi @ Vh[i]     (right singular vector)
    #   v_i (present, time t)   = Dhi @ U[:, i]   (left singular vector)
    u = Dhi @ Vh.T; v = Dhi @ U
    return pi, s, u, v

# (a)-(c): reversible chain with two weakly coupled pairs {1,2},{3,4}
P = np.array([[0.90, 0.09, 0.01, 0.00],
              [0.09, 0.90, 0.00, 0.01],
              [0.01, 0.00, 0.90, 0.09],
              [0.00, 0.01, 0.09, 0.90]])
pi, s, u, v = coherent_svd(P)
print("reversible chain: pi =", pi, "\nsingular values =", s)
print("v_1 (present) =", v[:, 1] / np.abs(v[:, 1]).max())
print("u_1 (future)  =", u[:, 1] / np.abs(u[:, 1]).max())
A = P[:2, :2].sum(1); print("leak per step out of {1,2} =", 1 - A)
print("objective (4) on A=B={1,2}:", 2 * (P[0, :2].sum() * pi[0] + P[1, :2].sum() * pi[1]) / pi[:2].sum(), " vs 1+sigma_1 =", 1 + s[1])

# (d): an irreversible chain -- a directed ring 1->2->3->4->1 with unequal "stay" probabilities
def ring(stay):
    P = np.diag(np.array(stay, float))
    for i in range(4): P[i, (i+1) % 4] = 1 - stay[i]
    return P

def objective(P, pi, A, B):
    """Eq. (4): P[X' in B | X in A] + P[X' in B^c | X in A^c]."""
    Ac = [i for i in range(4) if i not in A]; Bc = [i for i in range(4) if i not in B]
    f = lambda A, B: sum(pi[i]*P[i, j] for i in A for j in B) / sum(pi[i] for i in A)
    return f(A, B) + f(Ac, Bc)

Pr = ring([0.1, 0.3, 0.1, 0.3])
pi_r, s_r, u_r, v_r = coherent_svd(Pr)
print("\nirreversible ring: pi =", pi_r, "\nsingular values =", s_r)
print("detailed-balance residual max|pi_i P_ij - pi_j P_ji| =", np.abs(pi_r[:, None]*Pr - (pi_r[:, None]*Pr).T).max())
print("v_1 (present, common future) =", v_r[:, 1] / np.abs(v_r[:, 1]).max(), " signs", np.sign(v_r[:, 1]))
print("u_1 (future,  common past)   =", u_r[:, 1] / np.abs(u_r[:, 1]).max(), " signs", np.sign(u_r[:, 1]))
print("objective on the rounded pair ({1,2},{2,3}) =", objective(Pr, pi_r, (0, 1), (1, 2)), "  1+sigma_1 =", 1 + s_r[1])
print("objective on the same-set pair ({1,2},{1,2}) =", objective(Pr, pi_r, (0, 1), (0, 1)))

# (e): weak flow -- sign-rounding is only a heuristic
Pw = ring([0.7, 0.9, 0.7, 0.9])
pi_w, s_w, u_w, v_w = coherent_svd(Pw)
print("\nweak-flow ring: v_1 =", v_w[:, 1]/np.abs(v_w[:, 1]).max(), " u_1 =", u_w[:, 1]/np.abs(u_w[:, 1]).max())
print("objective rounded pair ({1,2},{2,3}) =", objective(Pw, pi_w, (0, 1), (1, 2)),
      "  same-set pair ({1,2},{1,2}) =", objective(Pw, pi_w, (0, 1), (0, 1)), "  1+sigma_1 =", 1 + s_w[1])

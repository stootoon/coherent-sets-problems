"""Problem 15: compressed vs exact singular pairs, and why a neuron cannot compute the residual.

1-D OU dx = -x dt + sqrt(2) dW (stationary N(0,1)), lag tau, feature f(x) = x^2 (no constant).
  compressed (CCA) value   c   = <f, K f> / <f, f>            = (1 + 2 e^{-2tau}) / 3
  Ritz value for K K^dag       = ||K f||^2 / <f, f>           = (1 + 2 e^{-4tau}) / 3
  residual of compressed pair  r^2 = ||K f||^2/<f,f> - c^2
||K f||^2 needs E[ f(X_{t+tau}) f(X'_{t+tau}) ] over SIBLING futures X, X' sharing X_t: not available
from a single trajectory, whose proxy E[f(X_{t+tau})^2] = ||f||^2 overestimates it by the mean
conditional variance (law of total variance).
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
dt, T, tau = 0.01, 20000.0, 0.5
n = int(T/dt); k = int(round(tau/dt))
e2, e4 = np.exp(-2*tau), np.exp(-4*tau)
f = lambda z: z**2

# --- single long trajectory: what lagged second moments give --------------------
x = np.zeros(n); noise = rng.standard_normal(n)*np.sqrt(2*dt)
for t in range(n-1):
    x[t+1] = x[t] - x[t]*dt + noise[t]
x = x[n//10:]; X0, X1 = x[:-k], x[k:]
ff   = np.mean(f(X0)**2)                 # <f,f>
fKf  = np.mean(f(X0)*f(X1))              # <f, K f>  (tower property)
c    = fKf/ff                            # compressed singular value on span{f}
proxy = np.mean(f(X1)**2)/ff             # E[f(X_{t+tau})^2]/<f,f> = 1 by stationarity: NOT ||Kf||^2/<f,f>
print(f"compressed value c        data {c:.4f}   analytic {(1+2*e2)/3:.4f}")
print(f"single-trajectory proxy   data {proxy:.4f}   (= ||f||^2/<f,f> = 1; overestimates ||Kf||^2/<f,f>)")

# --- sibling futures: what ||K f||^2 actually needs ------------------------------
m = 200_000
x0 = rng.choice(x, m)                    # stationary initial points
xa, xb = x0.copy(), x0.copy()            # two independent futures from the same present
for t in range(k):
    xa += -xa*dt + rng.standard_normal(m)*np.sqrt(2*dt)
    xb += -xb*dt + rng.standard_normal(m)*np.sqrt(2*dt)
KfKf = np.mean(f(xa)*f(xb))/ff           # ||K f||^2 / <f,f>  = E[f(X) f(X')]/<f,f>
print(f"Ritz value ||Kf||^2/<f,f> data {KfKf:.4f}   analytic {(1+2*e4)/3:.4f}")
print(f"mean conditional variance  data {1-KfKf:.4f}   analytic {1-(1+2*e4)/3:.4f}   (law of total variance)")
print(f"residual^2 of compressed pair: sibling estimate {KfKf-c**2:.4f}   analytic {(1+2*e4)/3-((1+2*e2)/3)**2:.4f}"
      f"   single-trajectory proxy {proxy-c**2:.4f}")
print(f"for comparison: true eigenpair (He_2, e^-2tau): residual 0;  e^-2tau = {e2:.4f}, c = {(1+2*e2)/3:.4f}")

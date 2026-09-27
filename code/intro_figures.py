"""Generate the schematic panels for the Introduction block (img/intro_*.svg)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.dpi": 100})
rng = np.random.default_rng(1)
RED, BLUE, GREY = "#c0392b", "#2c6fbb", "#8a857d"

# ---------------------------------------------------------------- panel 1: double well
def wells():
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [1, 1.4, 1.2]})
    x = np.linspace(-1.9, 1.9, 300); U = x**4/4 - x**2/2
    ax[0].plot(x, U, "k"); ax[0].set_xlabel("state $x$"); ax[0].set_ylabel("potential $U(x)$")
    ax[0].fill_between(x, U, U.max()+.05, where=x < 0, color=BLUE, alpha=.12)
    ax[0].fill_between(x, U, U.max()+.05, where=x > 0, color=RED, alpha=.12)
    ax[0].axvline(0, color="k", ls=":", lw=.8); ax[0].set_ylim(-.3, .6); ax[0].set_title("two basins, one ridge", fontsize=10)
    dt, T = .01, 40; n = int(T/dt); t = np.arange(n)*dt
    for D, a in [(0.25, ax[1])]:
        for x0, c in [(-1, BLUE), (-1, BLUE), (1, RED), (1, RED)]:
            xs = np.empty(n); xs[0] = x0
            for i in range(n-1): xs[i+1] = xs[i] + (xs[i]-xs[i]**3)*dt + np.sqrt(D*dt)*rng.standard_normal()
            a.plot(t, xs, color=c, lw=.7, alpha=.8)
        a.axhline(0, color="k", ls=":", lw=.8); a.set_xlabel("time"); a.set_ylabel("$x(t)$"); a.set_ylim(-2, 2)
        a.set_title("noisy trajectories ($D=0.25$)", fontsize=10)
    # retention curves
    taus = np.linspace(0, 40, 81); m = 3000
    for D, c, lab in [(0.0, "k", "no noise: invariant"), (0.12, "#555", "$D=0.12$"), (0.25, GREY, "$D=0.25$"), (0.6, "#bbb", "$D=0.6$")]:
        xs = -1 + 0.15*rng.standard_normal(m); keep = [1.0]
        for i in range(1, len(taus)):
            for _ in range(int((taus[i]-taus[i-1])/dt)):
                xs = xs + (xs-xs**3)*dt + np.sqrt(D*dt)*rng.standard_normal(m)
            keep.append(np.mean(xs < 0))
        ax[2].plot(taus, keep, color=c); ax[2].text(41, keep[-1], lab, fontsize=7.5, va="center", color=c)
    ax[2].axhline(.5, color="k", ls=":", lw=.8); ax[2].set_ylim(.4, 1.02); ax[2].set_xlim(0, 62)
    ax[2].set_xlabel(r"horizon $\tau$"); ax[2].set_ylabel(r"P[still left $\mid$ started left]")
    ax[2].set_title("coherence decays with horizon", fontsize=10)
    fig.tight_layout(); fig.savefig("img/intro_wells.svg"); fig.savefig("/private/tmp/claude-501/-Users-sinatootoonian-git-coherent-sets-problems/3c9328fd-68ae-4de3-8d5f-67e1712bd324/scratchpad/intro_wells.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- panel 2: saddle, two colourings
def saddle():
    es, eu = np.array([1., .25]), np.array([.35, 1.])            # stable / unstable eigenvectors
    V = np.stack([es, eu], 1); A = V @ np.diag([-1., 1.]) @ np.linalg.inv(V)
    W = np.linalg.inv(V)                                          # rows: left eigenvectors (stable coord, unstable coord)
    ws, wu = W[0], W[1]
    dt, tau, D, m = .01, 1.0, .06, 2500
    x0 = 0.9*rng.standard_normal((m, 2)); x = x0.copy()
    for _ in range(int(tau/dt)): x = x + x @ A.T * dt + np.sqrt(D*dt)*rng.standard_normal((m, 2))
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.0))
    g = np.linspace(-3, 3, 24); X, Y = np.meshgrid(g, g); U_, V_ = A[0,0]*X + A[0,1]*Y, A[1,0]*X + A[1,1]*Y
    for a, pts, col, ttl in [(ax[0], x0, x @ wu > 0, "predictive set: coloured by the FUTURE\n(sign of unstable coordinate at $t+\\tau$)"),
                             (ax[1], x,  x0 @ ws > 0, "retrospective set: coloured by the PAST\n(sign of stable coordinate at $t-\\tau$)")]:
        a.streamplot(X, Y, U_, V_, color="#d8d3ca", density=.55, linewidth=.6, arrowsize=.7)
        a.scatter(pts[col, 0], pts[col, 1], s=3, color=RED, alpha=.55, lw=0)
        a.scatter(pts[~col, 0], pts[~col, 1], s=3, color=BLUE, alpha=.55, lw=0)
        for e, lab in [(es, "stable manifold"), (eu, "unstable manifold")]:
            a.plot([-3.2*e[0], 3.2*e[0]], [-3.2*e[1], 3.2*e[1]], "k", lw=1.2)
        a.text(2.45, .35, "stable\nmanifold", fontsize=8, ha="center"); a.text(1.55, 2.55, "unstable\nmanifold", fontsize=8, ha="center")
        a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(ttl, fontsize=9.5); a.plot(0, 0, "ko", ms=4)
    for a, w, lab, c in [(ax[0], wu, "$\\mathbf{v}_1\\perp$ stable manifold", "#1b7f3b"), (ax[1], ws, "$\\mathbf{u}_2\\perp$ unstable manifold", "#1b7f3b")]:
        w = 1.6*w/np.linalg.norm(w); a.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2, color=c))
        a.text(w[0]*1.15, w[1]*1.15, lab, color=c, fontsize=8.5, ha="center")
    ax[0].set_xlabel("present state $x(t)$: boundary is the stable manifold", fontsize=9)
    ax[1].set_xlabel("present state $x(t)$: boundary is the unstable manifold", fontsize=9)
    fig.tight_layout(); fig.savefig("img/intro_saddle.svg"); fig.savefig("/private/tmp/claude-501/-Users-sinatootoonian-git-coherent-sets-problems/3c9328fd-68ae-4de3-8d5f-67e1712bd324/scratchpad/intro_saddle.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- panel 3: neuron pipeline
def neuron():
    n, d = 400, 30
    y = np.zeros(n); e = rng.standard_normal(n)
    for i in range(2, n): y[i] = 1.6*y[i-1] - 0.7*y[i-2] + e[i]          # AR(2): damped oscillation
    y /= y.std()
    k = np.arange(d); w = np.exp(-k/6) - np.exp(-k/2); w /= np.abs(w).sum()  # generic temporal filter
    out = np.array([w @ y[i-d+1:i+1][::-1] if i >= d-1 else 0 for i in range(n)])
    fig = plt.figure(figsize=(9.6, 4.2)); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[3, .15, 1, .0])
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 2]); a3 = fig.add_subplot(gs[1, 0]); a4 = fig.add_subplot(gs[2, 0])
    t = np.arange(n); i0 = 300
    a1.plot(t, y, color="#333", lw=.8); a1.axvspan(i0-d+1, i0, color="#f0c36d", alpha=.45)
    a1.text(i0-d/2, 2.6, "lag vector $\\hat{\\mathbf{x}}(t)$\n(last $d$ samples)", ha="center", fontsize=8.5)
    a1.set_ylabel("input $y(t)$"); a1.set_xticks([]); a1.set_ylim(-3.2, 3.8); a1.set_xlim(0, n)
    a2.stem(k, w, linefmt="#7a4a1c", markerfmt=" ", basefmt="k:"); a2.set_title("filter $\\mathbf{w}$ (temporal receptive field)", fontsize=9)
    a2.set_xlabel("lag"); a2.set_yticks([])
    a3.plot(t, out, color="#7a4a1c", lw=.9); a3.axhline(0, color="k", ls=":", lw=.8)
    a3.fill_between(t, out, 0, where=out > 0, color=RED, alpha=.25); a3.fill_between(t, out, 0, where=out < 0, color=BLUE, alpha=.25)
    a3.set_ylabel("$\\mathbf{w}^\\top\\hat{\\mathbf{x}}(t)$"); a3.set_xticks([]); a3.set_xlim(0, n)
    on, off = out > 0, out < 0
    a4.fill_between(t, 0, 1, where=on, color=RED, alpha=.7, step="mid"); a4.fill_between(t, 1.2, 2.2, where=off, color=BLUE, alpha=.7, step="mid")
    a4.set_yticks([.5, 1.7]); a4.set_yticklabels(["ON: $H(+\\mathbf{w}^\\top\\hat{\\mathbf{x}})$", "OFF: $H(-\\mathbf{w}^\\top\\hat{\\mathbf{x}})$"], fontsize=8.5)
    a4.set_xlabel("time"); a4.set_xlim(0, n); a4.set_ylim(-.1, 2.3); a4.spines["left"].set_visible(False)
    fig.text(.75, .55, "membership index\n= sign of a linear\nprojection of the\nlag vector", fontsize=9.5, ha="center", va="center")
    fig.text(.75, .25, "two rectified units\nshare one filter and\nreport the two\ncoherent sets", fontsize=9.5, ha="center", va="center")
    fig.tight_layout(); fig.savefig("img/intro_neuron.svg"); fig.savefig("/private/tmp/claude-501/-Users-sinatootoonian-git-coherent-sets-problems/3c9328fd-68ae-4de3-8d5f-67e1712bd324/scratchpad/intro_neuron.png", dpi=110); plt.close(fig)

wells(); saddle(); neuron(); print("done")

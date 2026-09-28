# Session summary — coherent-sets problem set

Hand-off notes for continuing work on this repository with a fresh Claude session.

## What this repo is

A static, MathJax-rendered problem set (click-to-reveal solutions) for reading
Pughe-Sanford et al., *Neurons as Detectors of Coherent Sets in Sensory Dynamics* (NeurIPS 2025).
The paper and supplement are checked in as `paper.pdf` and `supp.pdf`. The site is published via
GitHub Pages from `main` (remote `github.com:stootoon/coherent-sets-problems`).

The user (Sina) is reading the paper with the set. Their role in this conversation has been to ask
conceptual and step-level questions; my role has been to explain, then fold the explanations into the
set when asked. **They ask to commit and push explicitly** ("add it and push"); do not push unasked.

## Repo layout and workflow

- `src/*.html` — one fragment per block. **Edit these, never the top-level `*.html`.**
- `python3 build.py` regenerates the top-level pages (`index.html`, `blockI.html`, `block0.html`, `blockA.html` … `blockD.html`) from `src/` plus the template in `build.py`. Commit both `src/` and the generated files.
- `style.css` — shared stylesheet (light/dark; `.setting`, `.goal`, `.part`, `.q`, `details.solution`, `.remark`, `.tier`, `figure.fig`, `figure.fig.row`).
- `code/` — NumPy scripts behind numeric problems: `p04_finite_chain.py`, `p11_hermite.py`, `p12_saddle.py`, `p13_galerkin.py`, `p14_cca.py`, `p15_residual.py`, `p16_deadleaves.py`, `intro_figures.py` (makes `img/intro_*.svg`).
- `img/` — `fig1a/b/c.png` (panels extracted from the paper's Fig. 1, credited in caption) and `intro_wells.svg`, `intro_saddle.svg`, `intro_neuron.svg`.
- `README.md` — publishing/editing instructions.
- `prob4.ipynb` — untracked notebook of the user's; leave it alone.

Markup conventions inside a block: `<h3 id="pN">Problem N — title</h3>`, a `<div class="setting">` with `<span class="label">Setting</span>`, `<p class="goal">`, then parts as
`<div class="part"><p><span class="q">(a)</span> …</p><details class="solution"><summary>Solution</summary>…</details></div>`.
Background prose goes in `<div class="remark">`. Confidence tags: `<span class="tier">High|Medium|Speculative</span>`.
MathJax macros defined in `build.py`: `\E \R \tr \diag \ip{..} \K \P \X`. Use `\mathrm{He}_n` for Hermite polynomials (Problem 11 convention).

Environment quirks: Python 3.9 (`build.py` was fixed for it: no backslashes inside f-strings), **no SciPy** (so `p14_cca.py` fails at import; every script I added is NumPy-only), matplotlib 3.9 available, `pdftotext`/`pdfimages`/`pdftoppm` available. To preview, run `python3 -m http.server 8765` and open `http://localhost:8765/blockI.html` in Chrome (file:// URLs are blocked for the browser tool). Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Structure of the set (current)

- **Introduction** (`blockI.html`, nav "I · Why coherent sets") — conceptual, light on math. Sections: sensory stream as trajectory; invariant sets and their noisy replacement; why a neuron wants a membership index; prediction vs retrospection at a saddle; what a neuron actually does (lag vector → filter → threshold), incl. partial observation; road map. Questions Q1–Q9 with answers.
- **Block 0** — Koopman/Perron–Frobenius from scratch (Problems 0–1).
- **Block A** — finite chains: weighted adjoints, reversibility, SVD, coherent sets, spectral clustering (Problems 2–5). Now opens with a remark box "The main result in three lines" (±1-indicator argument for why λ≈1 ⇔ coherent pair; law of total variance).
- **Block B** — generators, contraction, ρ₀-adjoint (Problems 6–7).
- **Block C** — OU: Lyapunov, reverse-time drift, Eq. (18), linear observables suffice, saddle asymptotics (Problems 8–12).
- **Block D** — coherent sets from data:
  - **13** Galerkin projection: derives Eq. (22) from the orthogonality condition; Eq. (23) via tower property; indicator features = Block A / Ulam's method; linear features exact for OU (K = e^{Aᵀτ}); coefficient-space adjoint and Eq. (24); numeric part (f) on {x,x²} vs {1,x,x²} for 1-D OU (projection error leaks the dropped constant into the x² coefficient).
  - **14** Past–future CCA: background box (Hotelling, TICA/VAMP, predictive information, Gaussian IB, Lipshutz et al. circuit); (a) CCA via Lagrange multipliers, whitened SVD form, invariance; (b) whitened features are orthonormal, CCA matrix = compressed Koopman operator, reversibility ⇒ TICA (with the lag-vector caveat), Gaussian MI = −½Σlog(1−ρᵢ²); (c)–(e) the original numeric OU check.
  - **15** "Is the approximation applied too early?": compression cannot inflate singular values (Courant–Fischer); compressed pair = two-sided restricted variational problem, VAMP-2 bound; hidden one-sided (Ritz for KK†) vs two-sided (CCA) choice worked on 1-D OU with feature x²; two-sided invariance ⇒ exact; residual needs ‖Kf‖², a sibling-trajectory quantity a single time series cannot supply (law of total variance). Script `p15_residual.py`.
  - **16** Dead-leaves stimulus (paper's Fig. 2): exponential plateau lengths ⇒ only the level is linearly predictable (signal is second-order an OU); heavy-tailed lengths ⇒ second pair via age-dependent hazard; the transient (derivative) pair encodes mean reversion ("just stepped up → will be lower"), not "edge → flat"; whitening explains the low-pass look of the first filter; lag vectors of a reversible signal give Σ_τᵀ = J Σ_τ J (J = lag reversal), not symmetric Σ_τ, so future filters are lag-reversed past filters. Script `p16_deadleaves.py`.

## Edits made in this conversation (all pushed)

1. 6(b): replaced circular "because K_t = e^{L†t}" with the generator-as-K_dt / semigroup argument; martingale route added.
2. 6(c): integration by parts spelled out (single divergence-theorem rule, applied once/twice; boundary terms; fixed loose "anti-self-adjoint" claim — true only if ∇·b = 0).
3. 7(d): unpacked the two silent steps (definition of P_τ applied to fρ₀; expectation against ρ₀ + taking-out-what-is-known + tower).
4. 8(c): reminder of what L is.
5. Block D Problems 13, 15, 16 added; Problem 14 expanded (old numeric parts became (c)–(e)); old p13_cca.py renamed p14_cca.py.
6. Introduction block added with figures; saddle panel reworked so one shared set of present states is coloured by future and by past.
7. Block A opening remark added.
8. `build.py` fixed for Python 3.9; figure CSS added.

## User preferences (important)

- Solutions were originally too "slick"; the user gets stuck when a step silently invokes a definition. **Name the rule or definition at each equality**, state a general rule once then apply it, never cite a later result circularly. (Saved in memory as `explicit-solution-steps`.)
- Conceptual explanations in chat have been well received; when asked to "add it", fold them into the set as background boxes + problems/questions with click-to-reveal answers.
- Keep problem numbering stable where possible; when renumbering, update `src/index.md.html`, `README.md`, cross-references (`grep -n "Problem N"` across `src/`), and script names.
- Validate HTML tag balance after big edits (a small `html.parser` check was used), rebuild, then commit + push in one go when asked.

## Conceptual conclusions reached (useful context for further questions)

- The whole set is one idea at three levels of generality: coherent sets = subdominant singular vectors of the Koopman operator in the ρ₀ geometry; the adjoint is the spine (reversal = self-adjointness; SVD needs an adjoint; densities/observables dual).
- Isometry (not mere similarity) is what makes whitening legitimate: inner-product objects (adjoint, SVD, orthogonality, contraction) survive only isometries.
- Galerkin/EDMD: every layer of ignorance (unobserved state, finite lags, linear features) is a projection; Problem 15 says what you get (best representable pair) and that the neuron cannot check the residual.
- Partial observation (olfaction): observability means one channel's lag vector can in principle see everything; in practice the "slice" is set by SNR and memory; the population carries the information but not the global coordinates without a pooling stage; prediction: filters cluster by predictive/retrospective type, not by glomerulus.
- Applicability to real senses: the method needs only what is predictable at horizon τ within the representable class; short horizons ⇒ physics not semantics; weakest for intermittent non-Gaussian plume signals.
- Population code: other neurons take further singular vectors (orthogonal partitions ordered by predictability); ON/OFF = two halves of one partition; for n-dim OU, n predictive + n retrospective units span the whitened state.
- Paper naming trap: Section 2 calls predictive v_i "left"; matrix notation in Fig. 1/Eq. 18 makes v a right singular vector. Track u (future/retrospective) vs v (present/predictive).

## Open threads / possible next steps

- A full pass expanding *all* solutions in the "name every rule" style was offered; the user said "for now just fix that one" — do it only if asked.
- Licence of the reproduced Fig. 1 panels was not verified (NeurIPS papers are usually CC BY 4.0).
- `p14_cca.py` needs SciPy (`expm`, `solve_sylvester`); could be rewritten NumPy-only.
- Possible additions: a problem on hierarchical composition / nonlinear singular functions; a check of Problem 16's heavy-tail claim on an actual 1-D dead-leaves scan; a Block C remark connecting the saddle figure of the introduction to Problem 12.

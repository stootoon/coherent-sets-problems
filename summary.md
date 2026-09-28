# Session summary — coherent-sets problem set

Hand-off notes for continuing work on this repository with a fresh Claude session.

## What this repo is

A static, MathJax-rendered problem set (click-to-reveal solutions) for reading
Pughe-Sanford et al., *Neurons as Detectors of Coherent Sets in Sensory Dynamics* (NeurIPS 2025).
The paper and supplement are checked in as `paper.pdf` and `supp.pdf`. The site is published via
GitHub Pages from `main` (remote `github.com:stootoon/coherent-sets-problems`).

The user (Sina) is reading the paper with the set. Their role has been to ask conceptual and
step-level questions; my role has been to explain, then fold the explanations into the set when
asked. **They ask to commit and push explicitly** ("add it and push"); do not push unasked.

## Repo layout and workflow

- `src/*.html` — one fragment per block. **Edit these, never the top-level `*.html`.**
- `python3 build.py` regenerates the top-level pages (`index.html`, `blockI.html`, `block0.html`, `blockA.html` … `blockD.html`) from `src/` plus the template in `build.py`. Commit both `src/` and the generated files.
- `style.css` — shared stylesheet (light/dark; `.setting`, `.goal`, `.part`, `.q`, `details.solution`, `.remark`, `.tier`, `figure.fig`, `figure.fig.row`).
- `code/` — NumPy scripts behind numeric problems: `p04_finite_chain.py`, `p11_hermite.py`, `p12_saddle.py`, `p13_galerkin.py`, `p14_cca.py`, `p15_residual.py`, `p16_deadleaves.py`, `intro_figures.py` (makes `img/intro_*.svg`).
- `img/` — `fig1a/b/c.png` (panels from the paper's Fig. 1, credited in caption) and `intro_wells.svg`, `intro_saddle.svg`, `intro_neuron.svg`.
- `README.md` — publishing/editing instructions.
- `.gitignore` — ignores `scratch-*.html` (see below) and the user's `prob4.ipynb`.

Markup conventions inside a block: `<h3 id="pN">Problem N — title</h3>`, a `<div class="setting">` with `<span class="label">Setting</span>`, `<p class="goal">`, then parts as
`<div class="part"><p><span class="q">(a)</span> …</p><details class="solution"><summary>Solution</summary>…</details></div>`.
Standalone question sets between/after problems use the same part markup with labels like `Q1(i)`, `Q2(iv)` and `<summary>Answer</summary>`. **One foldout per sub-part** — the user asked for (i)/(ii)/(iii) to be separate `details`, never mashed into one.
Background prose goes in `<div class="remark">`. Confidence tags: `<span class="tier">High|Medium|Speculative</span>`.
MathJax macros defined in `build.py`: `\E \R \tr \diag \ip{..} \K \P \X`. Use `\mathrm{He}_n` for Hermite polynomials (Problem 11 convention).

Environment quirks: Python 3.9 (`build.py` fixed for it: no backslashes inside f-strings), **no SciPy** (`p14_cca.py` fails at import; all scripts I added are NumPy-only), matplotlib 3.9, `pdftotext`/`pdfimages`/`pdftoppm` available. Preview: `python3 -m http.server 8765` in the repo root, then `http://localhost:8765/blockB.html` etc. (file:// is blocked for the browser tool). Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

**Math rendering for chat answers**: the user cannot read raw TeX in the terminal. When an answer
gets math-heavy, write it as a standalone MathJax HTML page `scratch-<topic>.html` in the repo root
(untracked, gitignored), make sure the :8765 server is running, and give them the localhost link.
Existing examples: `scratch-roundtrip.html` (the K†K derivation), `scratch-nonstationary.html`.
Note: this CLI session cannot be continued in Claude desktop/web (per-surface session history), and
those surfaces don't render inline `$…$` anyway — the scratch-page workflow is the solution.

## Structure of the set (current)

- **Introduction** (`blockI.html`, nav "I · Why coherent sets") — conceptual. Questions Q1–Q9 with answers (trajectory view; invariant sets under noise; membership index; prediction vs retrospection at a saddle; lag vector → filter → threshold; partial observation/olfaction; road map).
- **Block 0** — Koopman/Perron–Frobenius from scratch (Problems 0–1), followed by **Interlude — conditional expectation in three rules** (h3 id="tower", added 28 Sep 2026): definition via densities; Rule 1 taking out what is known; Rule 2 tower property (basic and general form, outer conditioning must be coarser) with the trap criterion (E[E[f(Z)|Y]|X] = E[f(Z)|X] iff X→Y→Z Markov; semigroup law passes, K†K round trip fails, sibling copy restores it); Rule 3 law of total variance; each rule in expectation and integral form; "how the set uses them" dictionary. Q1(i)–(iii) separate foldouts: (i) Rules 1–2 from densities, combined into the pairing identity; (ii) general tower and the conditional-independence criterion with both instances; (iii) law of total variance, integral form, contraction bound. Block B 7(d), Q2(vi) and Block D Q1(i) link to it.
- **Block A** — finite chains (Problems 2–5). Opens with the remark "The main result in three lines" (±1-indicator argument; law of total variance).
- **Block B** — generators, contraction, ρ₀-adjoint (Problems 6–7), **plus (added this conversation)**:
  - Remark "In what sense are observables dual to densities?" (between Problems 6 and 7): three levels — the bilinear nondegenerate pairing ⟨f,ρ⟩=E_ρ[f] (duality really with signed measures); Banach asymmetry ((L¹)*=L∞ but not conversely; K is *the* adjoint of P, P is the pre-adjoint); the ρ₀-geometry collapse via the dictionary f ↔ fρ₀ (self-duality, 1 and ρ₀ merge into the trivial pair). Q1(i)–(iii), separate foldouts.
  - Remark "The two round trips and the sibling trajectory" (after Problem 7): K between slices has no usable spectrum for non-reversible dynamics; K†K and KK† are self-adjoint round trips; SVD = their matched eigendecompositions; sibling trajectory = shares one state, independent noise; K†K interrogates the future about its past (u_i, retrospective), KK† the present about its future (v_i, predictive); 1−λ₁² = sibling disagreement; Bayes-flip phrasing caveat; type-bookkeeping paragraph (K's input slot = later slice, output = earlier; K†'s the reverse; the composite returns to f's own slice). Q2(i)–(vi), separate foldouts: (i) K†K1_B = sibling probability (cond. independence + tower); (ii) ⟨1_B,K†K1_B⟩ = two-sibling agreement, normalized = self-coherence; (iii) KK†1_A mirror; (iv) reversible ⇒ both round trips = K², u=v, metastable not transported; (v) kernel-composition derivation of the sibling kernel q_τ(y′|y) with sanity checks (symmetry = self-adjointness; K†K1=1); (vi) why the tower property does NOT collapse K†K to I (outer conditioning not coarser), deterministic ⇒ K unitary, noise creates the coherence hierarchy.
  - **Coda — does any of this survive without stationarity?** (unnumbered h3 id="coda" at end of block): remark (skeleton survives — Froyland's non-autonomous origin; choose μ, ν=Pμ, K: L²(ν)→L²(μ); what dies: canonical ρ₀, semigroup → propagator cocycle, eigenvalues not well-typed, reversibility/TICA unstatable; statistical casualty = ergodicity; neuron ⇒ local stationarity, fast non-stationarity = partial observation in disguise, slow = adaptation reshaping filters; Problem 12 read as a fixed-weight-on-both-slices violation of ν=Pμ; hierarchy reversible ⊂ stationary ⊂ non-stationary). Q3(i)–(iv), separate foldouts: (i) contraction L²(ν)→L²(μ) with ν=Pμ replacing stationarity; (ii) why Kf=λf and K†=K fail typing, why SVD survives; (iii) why one realisation can't estimate C₀₀(t); remedies: ensembles / quasi-stationarity (τ_mix ≪ T_window ≪ T_drift) / cyclostationarity; 15(e) compounds; (iv) time-dependent OU: differential Lyapunov equation, linear observables suffice slice-wise, doubly-whitened propagator Σ_{t+τ}^{-1/2}Φ Σ_t^{1/2} = slice-specific CCA.
- **Block C** — OU (Problems 8–12): 8 Lyapunov, 9 reverse-time OU, 10 Eq. (18) from the sibling kernel (its Setting already defines F_τ=KK†, B_τ=K†K with the sibling kernel — Block B Q2(v) cross-references it), 11 linear observables suffice, 12 saddle asymptotics (indefinite Σ, eigenvalues above 1, "closest to unity" only meaningful at short horizons).
- **Block D** — coherent sets from data. Opens with **Prelude — learning versus approximation** (h3 id="learning", added 28 Sep 2026): remark separating the *learning* problem (variational characterisation σ₁ = max E[g(X_t)f(X_{t+τ})] over unit, mean-zero f,g; g present/predictive = paper's v₁, f later/retrospective = paper's u₁; time-lagged Hebbian rule with anti-Hebbian normaliser, two neurons teach each other; Galerkin = same problem in coordinates, Eqs. 22–24 are its fixed point) from the *approximation* problem (finite basis = compressed problem of Problem 15; small dictionary is also a regulariser; hierarchy enlarges the basis). Q1(i)–(iv), separate foldouts: (i) ⟨g,Kf⟩ = lagged second moment, max = σ₁ via singular-system expansion + Cauchy–Schwarz; (ii) Lagrangian, gradients as expectations, one-sample rule, multipliers = objective, fixed point = CCA of 14(a); (iii) complete basis ⇒ exact; finite ⇒ σ₁(C); He₂ example where the in-subspace maximiser is the third singular function and the projection of the second is zero; (iv) interpolating classes drive the empirical objective to 1 (finite basis regularises); rectified ON/OFF outputs strictly enlarge the downstream function class. Then: 13 Galerkin (Eq. 22–24), 14 past–future CCA (background box; TICA/VAMP/Gaussian IB), 15 "approximation applied too early?" (compression can't inflate singular values; sibling-quantity residual obstruction), 16 dead-leaves stimulus.

## Edits made across sessions (all pushed)

1. 6(b) generator/semigroup argument; 6(c) integration by parts spelled out; 7(d) silent steps unpacked; 8(c) reminder of L.
2. Block D Problems 13, 15, 16 added; 14 expanded; Introduction block with figures; Block A opening remark; build.py fixed for Python 3.9.
3. **This conversation**: Block B duality remark + Q1; round-trips remark + Q2 (incl. type bookkeeping, kernel derivation, tower-trap); non-stationarity coda + Q3; Q1/Q2 split into per-sub-part foldouts; index.md.html Block B line and Block B lead updated; `.gitignore` added.

4. **28 Sep 2026 (later)**: Block D prelude on learning vs approximation (see above). Then, at the user's request, every solution that used the tower property got a companion paragraph "By direct integration" (kernel integrals, Fubini, joint density = ρ₀(x)p_τ(y|x)): Block B Problem 6 martingale remark (Chapman–Kolmogorov), 7(d) adjoint check, Q2(i)–(iii) via the sibling kernel q_τ, Q2(vi) (why q_τ ≠ δ; forward∘forward composes, forward∘backward doesn't); Block C 10(c) second equation of (18) by integrating linear observables against the two Gaussian kernels; Block D Q1(i) step 1, 13(b), 15(e) (sibling identity as triple integral; law of total variance in integral form); 14(b)(i) reference reworded. The user wants this pattern kept: whenever a solution invokes the tower property, also give the direct-integration version. Then a mini tutorial on the tower property was added as the Block 0 interlude (see structure above).

## User preferences (important)

- Solutions must **name the rule or definition at each equality** (memory `explicit-solution-steps`): the user gets stuck when a step silently invokes a definition; state a general rule once, then apply it; never cite a later result circularly.
- One foldout (`details`) per sub-part; don't merge (i)(ii)(iii) into one Answer.
- Whenever a solution uses the tower property, add a "By direct integration" version alongside it (kernel integrals; name Fubini and the joint-density factorisation).
- Chat explanations first; fold into the set only when asked ("add it and push").
- Math-heavy chat answers: render to `scratch-*.html` and give the localhost:8765 link (see workflow above).
- Keep problem numbering stable; standalone questions (Q1, Q2, Q3 in Block B) avoid renumbering problems. When renumbering, update `src/index.md.html`, `README.md`, cross-references (`grep -n "Problem N"` across `src/`), and script names.
- Validate HTML tag balance after big edits (small `html.parser` check), rebuild, then commit + push in one go when asked.

## Conceptual conclusions reached (context for further questions)

- One idea at three levels: coherent sets = subdominant singular vectors of K in the ρ₀ geometry; the adjoint is the spine.
- Observables/densities duality: honest nondegenerate bilinear pairing; asymmetric at the Banach level (L¹ pre-dual of L∞); collapsed to self-duality in L²(ρ₀) via f ↔ fρ₀ — and the collapse is load-bearing (SVD needs adjoint needs inner product; ρ₀ is the weight making reversibility = self-adjointness and 1 ≡ ρ₀).
- K†K / KK† round trips: the sibling-trajectory reading (share a state, independent noise); K alone has no good spectrum between slices; SVD = matched eigendecompositions of the round trips; 1−λ₁² = sibling disagreement; deterministic ⇒ unitary ⇒ no coherence hierarchy — noise creates it.
- The tower-property trap: E[E[f(X_{t+τ})|X_t]|X_{t+τ}] does NOT collapse (outer conditioning not coarser); the sibling substitution is the legal move.
- Non-stationarity: the two-slice, two-measure formalism (μ chosen, ν=Pμ) preserves the whole SVD/coherence/variational skeleton (it's Froyland's original setting; VAMP covers it); what dies is slice identification (ρ₀, semigroup, eigenvalues, reversibility) and single-trajectory estimation (needs ensembles / quasi-stationarity / cyclostationarity); for neurons: local stationarity on the learning timescale, adaptation = tracking μ_t, fast drift often = partial observation of a larger stationary system.
- Isometry (not similarity) is what whitening legitimacy rests on; every layer of ignorance is a projection (13/15); paper naming trap: track u (future/retrospective) vs v (present/predictive), Section 2's "left" vs matrix-notation "right".

## Journal club (29 Sep 2026)

The user gave a one-hour chalk-talk journal club on the paper on 29 Sep 2026 to theoretical neuroscientists and ML people. In the 28 Sep session I quizzed them one question at a time (elevator pitch; why invariant sets fail under noise; defining a coherent pair and the role of τ; why Koopman / the relaxation from sets to functions; why SVD not eigenfunctions; where the neuron is). They paused before the data/estimation questions (Section 4, Galerkin, whitening, CCA) to read up. Their preferred SVD convention in discussion: K = USVᵀ, so u lives on the present slice (marks A, eigenfunction of KK†, predictive) and v on the later slice (marks B, eigenfunction of K†K, retrospective); that is the opposite letter assignment to the paper's Eq. (8). Remaining quiz questions not yet asked: what statistics a neuron needs and why whitening is legitimate (isometry, Problems 13–15); what the retinal and olfactory data show; what would falsify the claim.

## Open threads / possible next steps

- Full pass expanding *all* solutions in "name every rule" style — offered, user said "for now just fix that one"; only if asked.
- Licence of the reproduced Fig. 1 panels unverified (NeurIPS usually CC BY 4.0).
- `p14_cca.py` needs SciPy (`expm`, `solve_sylvester`); could be rewritten NumPy-only.
- Possible additions: hierarchical composition / nonlinear singular functions problem; empirical check of Problem 16's heavy-tail claim on a real 1-D dead-leaves scan; Block C remark connecting the intro's saddle figure to Problem 12; a numeric problem for the non-stationarity coda (time-dependent OU, slice-whitened CCA vs stationary CCA on a drifting process) — natural companion to Q3(iv); redoing Problem 12 in the two-measure formalism (Gaussian μ evolved, singular values back ≤ 1) to substantiate the coda's claim.

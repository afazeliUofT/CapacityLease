# Source review notes for `CpctLeaseFinal.tex`

This package transcribes the literal model equations and simulation tables from the TeX source into Python. The default preset is `configs/paper_literal.json`.

Key literal inputs copied from the simulation section:

- User groups: `(N_g, alpha_g, beta_g)` =
  - Group 1: `(200, 3, 0.2)`
  - Group 2: `(200, 3, 0.6)`
  - Group 3: `(900, 2, 0.6)`
  - Group 4: `(700, 1, 1.0)`
- Common constants: `N=2000`, `C=1000 Mbps`, `delta=0.1`, `zeta=0.01`, `pi_V=500`, `lambda=0.1`
- Monopoly noises:
  - Group 1: `N(3, 0.1)`
  - Group 2: `N(2, 1)`
  - Group 3: `N(2, 0.8)`
  - Group 4: `N(0.5, 0.1)`
- Competitive noises:
  - MNO: `[(2.5,0.1), (1.5,1), (1.5,0.8), (0.2,0.1)]`
  - MVNO: `[(0.5,0.1), (2.5,1), (1.5,0.8), (3,0.1)]`

Implementation choices that prioritize stability:

1. Every scalar root is solved with explicit sign bracketing and a Brent solver, with bisection-style fallback.
2. Profitability-set boundaries are computed analytically with Lambert-W rather than by repeated numeric root finding.
3. Flexible-participation acceptance probabilities are evaluated with a bivariate-normal tail routine. When SciPy's fast internal helper is unavailable, the code falls back to the public multivariate normal CDF.
4. All optimization over `n_V` and `n_M` uses coarse-to-fine global refinement rather than local-only optimization.

Important consistency note:

The literal equations and tables in the TeX source do not line up perfectly with some narrative figure descriptions. For example, under the literal table values and equations, several reported narrative quantities (such as the monopoly optimum near `p ≈ 110` and the market-clearing fixed-capacity narrative around `C_V = 100`, `n_M ≈ 200`) are not reproduced exactly. To avoid silently altering the model, the package keeps the equations and table values exactly as written and exposes all numerical search settings separately in the JSON configs.

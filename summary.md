# Single-Asset Path-Dependent Options: PINNs, History Encoding, and BSDEs

Updated: 2026-10-22. This research summary presents results, methodology and derivations, investigations, and conclusions.

## 1. Pricing Problem, Information, and Evaluation

The risk-neutral model and pricing representation are

$$
dS_t=rS_tdt+\sigma S_tdW_t,\qquad
V_t=\mathbb E[e^{-r(T-t)}G\mid\mathcal F_t].
$$

The common market configuration is S0=K=1, r=0.03, sigma=0.4, and T=1, with an upper barrier B=1.2 or a lower barrier B=0.8 and zero rebate. All experiments discussed here concern a single asset.

The project studies ten option types: floating-strike lookback calls and puts, and up/down, knock-in/knock-out barrier calls and puts. Direct knock-in training produces zero terminal payoffs on paths that never knock in. When such paths are prevalent, nonzero payoff signals can become sparse and make learning harder. The four knock-in prices are therefore obtained through exact in-out parity:

$$
V_{\rm in}=V_{\rm vanilla}-V_{\rm out}
$$

The corresponding knock-out is priced first and subtracted from the vanilla price; no separate knock-in network is trained. Parity requires matching models, maturities, strikes, monitoring rules, and historical states. With zero rebate, each knock-in or knock-out price is bounded above by its corresponding vanilla price.

| Method | Historical information available to the network | Sequential history encoding required? | Reference used |
|---|---|---|---|
| Sufficient-state PINN without LSTM | Explicit running extremum or survival state | No; valid states can be sampled directly | Continuous-monitoring analytical solution |
| Raw-history LSTM-PINN | Observed prices and times up to the query, encoded by an LSTM | Yes; differentiation in current time and spot is performed within intervals | Numerical reference with 128 discrete monitoring steps |
| Signature + LSTM-BSDE | Truncated signatures of successive path segments | Yes; signatures are processed chronologically | Discrete reference matching each experiment's grid |
| Raw-history LSTM-BSDE | Fine-grid prices, times, and learnable price-level features | Yes; history encoding is followed by backward datewise training | Numerical reference with 128 discrete monitoring steps |

Removing the LSTM does not remove historical information: it assumes a known sufficient-state reduction. Adding an LSTM does not make the mathematical price function inherently discrete; the implementation uses discrete observations and a finite-dimensional recurrent representation. Continuous and discrete monitoring define different contracts, so their reference prices must not be mixed in a common ranking.

Evaluation separates initial pricing bias, relative L2 errors across states or conditional path prices, Delta/Gamma errors, hedging correlation, and stability across training initializations. An accurate initial price does not imply accurate intermediate prices. The ability to differentiate a model does not establish derivative accuracy. Quadrature references have spatial discretization, tail truncation, and interpolation errors; they are not closed-form analytical solutions.

Three-seed results are reported as **mean [minimum, maximum]** across seeds 1, 2, and 3. These ranges describe variation between training seeds, not standard errors or confidence intervals. For percentage metrics, range endpoints are also percentages. Historical single-seed comparisons are not assigned three-seed ranges.

## 2. PINN without LSTM: Known Sufficient States and Direct State-Space Sampling

### 2.1 Results

Three training seeds have been completed. The table uses barrier soft constraints with an MC anchor, barrier hard constraints with an MC anchor, and H-based models before and after refinement. Lookback models use no MC anchor. Each cell reports **initial absolute relative error: mean [range] / path-price relative L2: mean [range]**, in percent.

| Product | Soft constraints | Hard constraints without H | H before refinement | H + L-BFGS |
|---|---:|---:|---:|---:|
| Lookback call | 0.858[0.651, 1.116] / 1.549[1.212, 2.115] | 0.895[0.499, 1.518] / 0.639[0.527, 0.825] | 0.646[0.164, 1.122] / 0.363[0.154, 0.579] | 0.062[0.012, 0.135] / 0.082[0.069, 0.100] |
| Lookback put | 1.056[0.046, 2.564] / 2.898[2.416, 3.583] | 0.625[0.033, 1.086] / 0.434[0.211, 0.644] | 0.673[0.135, 0.970] / 0.479[0.154, 0.716] | 0.131[0.045, 0.182] / 0.109[0.081, 0.140] |
| Up-and-out call | 2.748[1.555, 4.991] / 24.891[23.364, 27.935] | 52.943[1.331, 96.159] / 68.820[12.881, 99.401] | 1.927[1.379, 2.588] / 0.431[0.346, 0.581] | 0.622[0.132, 1.540] / 0.323[0.175, 0.561] |
| Up-and-out put | 0.288[0.201, 0.334] / 0.612[0.583, 0.639] | 0.302[0.281, 0.336] / 0.120[0.114, 0.132] | 0.003[0.001, 0.006] / 0.003[0.001, 0.005] | 0.004[0.002, 0.005] / 0.003[0.001, 0.004] |
| Down-and-out call | 0.213[0.165, 0.267] / 0.364[0.343, 0.391] | 0.208[0.186, 0.221] / 0.065[0.058, 0.071] | 0.000[0.000, 0.001] / 0.001[0.001, 0.002] | 0.003[0.001, 0.006] / 0.001[0.001, 0.002] |
| Down-and-out put | 1.615[0.648, 2.647] / 18.189[15.847, 20.501] | 18.946[15.772, 24.161] / 6.785[5.617, 8.079] | 0.914[0.798, 1.059] / 0.332[0.228, 0.389] | 0.288[0.093, 0.617] / 0.166[0.081, 0.218] |
| Up-and-in call (parity) | 0.032[0.018, 0.058] / 0.455[0.428, 0.511] | 0.616[0.015, 1.119] / 1.259[0.236, 1.819] | 0.022[0.016, 0.030] / 0.008[0.006, 0.011] | 0.007[0.002, 0.018] / 0.006[0.003, 0.010] |
| Up-and-in put (parity) | 0.858[0.600, 0.997] / 1.411[1.343, 1.473] | 0.900[0.838, 1.003] / 0.276[0.262, 0.305] | 0.010[0.004, 0.019] / 0.007[0.003, 0.011] | 0.012[0.006, 0.016] / 0.007[0.003, 0.010] |
| Down-and-in call (parity) | 1.109[0.863, 1.393] / 1.244[1.171, 1.335] | 1.088[0.970, 1.151] / 0.221[0.198, 0.241] | 0.001[0.000, 0.004] / 0.004[0.002, 0.005] | 0.016[0.005, 0.033] / 0.005[0.003, 0.007] |
| Down-and-in put (parity) | 0.047[0.019, 0.077] / 0.848[0.739, 0.956] | 0.549[0.457, 0.700] / 0.316[0.262, 0.377] | 0.026[0.023, 0.031] / 0.015[0.011, 0.018] | 0.008[0.003, 0.018] / 0.008[0.004, 0.010] |

Values displayed as zero are rounded and do not imply exactly zero error. For the anchored hard-constraint up-and-out call, path-price L2 ranges from approximately 12.88% to 99.40%; its mean alone would conceal substantial instability.

Pathwise Greek relative L2 errors after H-based modeling and L-BFGS refinement, reported as mean [minimum, maximum]:

| Separately trained product | Delta | Gamma |
|---|---:|---:|
| Lookback call | 0.182%[0.127, 0.221] | 0.719%[0.626, 0.779] |
| Lookback put | 0.494%[0.450, 0.519] | 1.415%[1.373, 1.488] |
| Up-and-out call | 0.194%[0.116, 0.312] | 0.127%[0.094, 0.157] |
| Up-and-out put | 0.007%[0.004, 0.010] | 0.061%[0.026, 0.080] |
| Down-and-out call | 0.004%[0.002, 0.007] | 0.048%[0.033, 0.072] |
| Down-and-out put | 0.100%[0.076, 0.121] | 0.083%[0.072, 0.089] |

Source: three-seed aggregate data (`outputs/anchored_transition_seeds/summary.json`). Full state-space, path-price, and Greek metrics are stored in each run's `stages/*/evaluation/comparison.csv`; see, for example, seed 2 after refinement (`outputs/anchored_transition_15000_seed2/stages/heat_refined/evaluation/comparison.csv`). Corresponding models receive 15,000 Adam updates; the refined H models receive additional L-BFGS optimization. These are not equal-total-compute comparisons.

### 2.2 Sufficient States and the PDE

The floating-strike lookback call pays $S_T-m_T$, where $m_t=\min_{u\leq t}S_u$. The put pays $M_T-S_T$, where $M_t=\max_{u\leq t}S_u$. Their sufficient-state prices are $v(t,s,m)$ and $v(t,s,M)$.

Inside the domain $s>m$ for the call, or $s<M$ for the put, the historical extremum does not change locally. Consequently,

$$
\mathcal R[v]:=v_t+rsv_s+\tfrac12\sigma^2s^2v_{ss}-rv=0.
$$

For the call, the terminal and extremum-boundary conditions are

$$
v(T,s,m)=s-m,\qquad v_m(t,s,m)|_{s=m}=0.
$$

For the put, the corresponding conditions are $v(T,s,M)=M-s$ and $v_M|_{s=M}=0$. The extremum-boundary condition removes the $v_m\,dm$ or $v_M\,dM$ term from Itô's formula.

A barrier option satisfies the same PDE inside its surviving domain, with vanilla terminal payoff $g(s)$ and absorbing boundary $v(t,B)=0$. A previously knocked-out contract remains worth zero. Pathwise evaluation must preserve this historical state: a price that has returned inside the barrier must not be treated as a history that never crossed it.

Training can independently sample valid $(t,s,m)$ or $(t,s,M)$ states; barrier models sample $(t,s)$ within the surviving domain. Other state variables are held fixed when differentiating with respect to time and spot. Neither chronological training from t0 to t1 nor a continuous simulated path through every collocation point is required. Sampling support and density still affect accuracy.

### 2.3 From Soft Constraints to Hard Constraints without H

A general soft-constraint objective is

$$
L=L_{\rm PDE}+\lambda_TL_T+\lambda_BL_B+\lambda_+L_+
+\lambda_A[v_\theta(0,S_0)-\widehat V_{\rm MC}]^2.
$$

The last term is used only in the designated barrier-anchor experiments. The implemented lookback objective is PDE + 2 times terminal + 0.1 times extremum boundary + 0.01 times nonnegativity. The soft barrier objective is PDE + 2 times terminal + 5 times boundary, plus the anchor term. The individual loss coefficients are not all one.

The PDE is linear and homogeneous, so zero satisfies its interior equation; if f solves that equation, cf also does. The correct nonzero terminal payoff excludes the zero solution. Thus the full boundary-value problem is not inherently nonunique. Training may fail to enforce its boundary data adequately, or may fit the interior equation poorly despite satisfying structural conditions.

Hard constraints without H embed known conditions in the output. The current barrier parameterization uses

$$
x=\begin{cases}\log(s/B),&\text{down barrier},\\\log(B/s),&\text{up barrier},\end{cases}
\quad a=\sigma\sqrt{\tau},\quad\tau=T-t,
$$

$$
v_\theta=C_{\rm vanilla}(t,s)\tanh\!\left(\frac{x}{a}\,q_\theta\right),
\qquad q_\theta=\frac{\operatorname{softplus}(N_\theta)}{\log2}>0.
$$

These expressions apply when $\tau>0$ and $x>0$; expiry and previously knocked-out branches are assigned their known values directly. The structure enforces zero barrier value, the vanilla upper bound, and the terminal limit at a fixed interior state. It does not guarantee successful interior fitting.

The lookback hard structure uses $x=\log(s/m)$ or $\log(M/s)$ and $\rho=\sqrt{x^2+\sigma^2\tau}$ to construct an even algebraic baseline, then adds $s\sqrt{\tau}N(t,\log s,x^2/(x^2+\sigma^2\tau))$. At expiry, the payoff is assigned directly. Evenness enforces the extremum-boundary derivative. No auxiliary heat equation is solved in this baseline, so it is not an H kernel.

### 2.4 Lookback H: Homogeneity and a Simplified Diffusion

Payoff homogeneity motivates $v=s\,u(\tau,x)$. For a lookback call, $x=\log(s/m)$; for a put, $x=\log(M/s)$; both are nonnegative. Substituting into the true PDE gives

$$
u_\tau=\tfrac12\sigma^2u_{xx}+\varepsilon(r+\tfrac12\sigma^2)u_x,
\quad u_x(\tau,0)=0,
$$

Here epsilon is +1 for the call and -1 for the put, with initial functions $f(x)=1-e^{-x}$ and $f(x)=e^x-1$, respectively.

Dropping the first-order drift term gives the Neumann heat equation on the half-line. Its reflected heat kernel yields

$$
H_{\rm LB}(t,s,y)=s\int_0^\infty
\big[\phi_{\sigma^2\tau}(z-x)+\phi_{\sigma^2\tau}(z+x)\big]f(z)\,dz,
$$

The function $\phi_v$ is the centered Gaussian density with variance v, and y denotes m or M. This auxiliary solution has the correct payoff limit and extremum-boundary condition, but is not the full true price. The current output is

$$
v_\theta=H_{\rm LB}+s\frac{\tau}{T}N_\theta
\left(t/T,\log(s/S_0),\frac{x^2}{x^2+\sigma^2\tau}\right).
$$

The correction vanishes at expiry and is even in x. The true pricing PDE determines the remaining correction, including drift effects. H explicitly uses the true extremum, making this a sufficient-state method.

### 2.5 Barrier H: Absorbing Heat Kernels and Bounded Outputs

In the signed log-distance coordinate $x>0$, the process has dynamics $dx=\mu\,dt+\sigma\,d\widetilde W$, with $\mu=\pm(r-\sigma^2/2)$. Its absorbing transition density is

$$
p_\mu(\tau;x,z)=\phi_{\sigma^2\tau}(z-x-\mu\tau)
-e^{-2\mu x/\sigma^2}\phi_{\sigma^2\tau}(z+x-\mu\tau),\quad z>0.
$$

Integrating the payoff against this density and discounting gives the true continuously monitored price. The auxiliary H used here instead takes the simplified kernel with zero log drift. J is the knock-in contribution under the same auxiliary model. H+J is that model's vanilla value, which need not equal the true vanilla value C.

The current `bounded_odds` structure is

$$
p_0=\frac{H}{H+J},\qquad
v_\theta=C\,\sigma\!\left(\log\frac HJ+\sigma\sqrt{\tau}N_\theta\right).
$$

The outer sigma denotes the sigmoid; the sigma multiplying the square root of time is volatility. Zero values, expiry, previously hit histories, and numerical underflow are handled separately in code. The quantity p0 is the knock-out share of auxiliary vanilla value, not a plain survival probability. This mapping transfers the auxiliary shape to the true vanilla-price scale, enforcing $0\leq v\leq C$ while preserving terminal and absorbing conditions.

The PDE acts on the final output v-theta, rather than checking H alone. Constructing H therefore does not eliminate PDE learning, but supplies substantial prior knowledge about an auxiliary equation, price scales, and boundary-layer shapes. H is not a universally available formula for arbitrary complex options.

### 2.6 Investigations and Conclusions

The investigations covered insufficient fitting of soft terminal and boundary constraints, missing interior coverage near running extrema, algebraic hard structures, reflected and absorbing heat kernels, the true vanilla upper bound, and L-BFGS refinement after Adam. Original and revised experiments are retained separately, including hard barrier models with and without anchors.

For the up-and-out call, mean path L2 is 57.54% without the hard-model MC anchor and 68.82% with it. For the down-and-out put, the corresponding values are 53.61% and 6.79%. Anchoring helps some products but does not guarantee pathwise accuracy; not every failure can be attributed to a missing anchor.

The H structures substantially improve prices and derivatives on the tested domain. These gains combine known sufficient states, appropriate structure, and optimization; they do not establish equivalent performance without prior information. The three seeds share anchors, refinement collocation rules, and evaluation protocols. They assess part of training stability, not independent blind testing or generalization across market parameters.

### 2.7 PINN Advantage: Prices and Derivatives from the Same Function

A trained PINN provides a differentiable price function. Holding historical sufficient states fixed, automatic differentiation gives

$$
\Delta_\theta=\frac{\partial v_\theta}{\partial s},\qquad
\Gamma_\theta=\frac{\partial^2v_\theta}{\partial s^2},\qquad
\Theta_\theta=\frac{\partial v_\theta}{\partial t}.
$$

For example, lookback-call Delta and Gamma hold the historical minimum m fixed. Barrier Greeks are evaluated in the surviving interior. Boundaries, expiry corners, and already-hit states require the appropriate one-sided or piecewise interpretation.

The model can therefore query prices and sensitivities throughout its training domain without solving a new pricing grid for each query. Section 2.1 gives measured derivative errors: after refinement, mean pathwise Delta and Gamma relative L2 are 0.182% and 0.719% for the lookback call, and 0.494% and 1.415% for the lookback put, with three-seed ranges shown in the table.

The advantage is access to risk sensitivities from a differentiable price function, not automatic correctness of the Greeks. Delta and Gamma are evaluated separately against analytical derivatives. The networks are trained at fixed market parameters; merely changing an external parameter does not provide validated Vega or Rho.

## 3. LSTM-PINN: Sequential History Encoding and Interval PDE Constraints

### 3.1 Results

Each separately trained product has three seeds and 3,000 terminal-initialization updates followed by 47,000 joint updates, with cosine learning-rate decay. Evaluation uses the same 128-step monitoring grid and 8,192 held-out histories. Initial bias is signed; other errors are relative L2. All entries report three-seed mean [minimum, maximum].

| Product | Initial bias | Price L2 at t=0.5 | Average price L2 over four dates |
|---|---:|---:|---:|
| Lookback call | -0.40%[-0.59, -0.30] | 1.58%[1.50, 1.64] | 1.62%[1.52, 1.69] |
| Lookback put | -1.17%[-1.42, -0.78] | 2.63%[2.55, 2.75] | 3.20%[3.15, 3.25] |
| Up-and-out call | -5.33%[-8.39, -3.68] | 7.78%[6.53, 8.62] | 8.95%[8.60, 9.47] |
| Up-and-out put | +3.47%[+3.12, +3.66] | 1.13%[1.05, 1.23] | 1.74%[1.26, 2.06] |
| Down-and-out call | +2.11%[+2.09, +2.15] | 1.08%[0.98, 1.21] | 0.89%[0.84, 0.92] |
| Down-and-out put | -7.37%[-8.35, -5.53] | 10.18%[8.79, 11.16] | 12.17%[11.44, 13.40] |
| Up-and-in call (parity) | +0.09%[+0.06, +0.14] | 0.12%[0.10, 0.14] | 0.18%[0.16, 0.20] |
| Up-and-in put (parity) | -13.28%[-13.99, -11.95] | 3.35%[3.11, 3.64] | 4.99%[3.72, 5.87] |
| Down-and-in call (parity) | -14.09%[-14.37, -13.94] | 5.23%[4.72, 5.83] | 4.32%[4.13, 4.45] |
| Down-and-in put (parity) | +0.29%[+0.22, +0.33] | 0.43%[0.37, 0.47] | 0.66%[0.60, 0.73] |

The four dates are t=0.25, 0.5, 0.75, and 0.9375. Their reported average is the arithmetic mean of the four datewise L2 errors, not a pooled L2 over all observations. Source: per-seed aggregate scores (`outputs/lstm_ten_50000/summary.json`).

### 3.2 From Paths to Historical Representations

For observation times t0 through tN,

$$
h_k=\operatorname{LSTM}_\theta(h_{k-1},\psi_\theta(t_k,S_{t_k})),\qquad
v_\theta(t,s;h_k),\quad t_k<t<t_{k+1}.
$$

The feature map psi contains time, log price, and learnable price-level features of the form $\operatorname{sigmoid}(\gamma_j(\log(S/S_0)-c_j))$. It supplies no true extremum or hit flag, but introduces an inductive bias suited to remembering price levels. Historical states must be computed chronologically; this does not require optimizer updates to proceed date by date.

For a discretely monitored contract, the historical observation record does not update between two observations. Conditional on a fixed historical encoding h-k, the interval PDE remains

$$
v_t+rsv_s+\tfrac12\sigma^2s^2v_{ss}-rv=0.
$$

At an observation, the historical update requires the interface condition

$$
v^-(t_{k+1},s;h_k)
=v^+(t_{k+1},s;\operatorname{Update}_\theta(h_k,t_{k+1},s)).
$$

An exact representation would retain any new extremum or hit event. Without explicit state inputs, the recurrent representation and training targets must learn that distinction. The interval PDE does not ignore history; it treats the existing encoding as a conditioning variable.

### 3.3 Output Structure and Implemented Loss

The current output is

$$
v_\theta(t,s;h_k)=\widetilde H(t,s)\operatorname{sigmoid}(A_\theta(t,s,h_k)).
$$

The notation H-tilde distinguishes this lift from the sufficient-state heat kernels of Section 2:

- For barriers, it is the discounted expected payoff restricted to the surviving side at maturity only. It does not check historical or future intermediate hits.
- For lookback calls it is s; for puts it is 6s. The latter is an imposed envelope, not a global bound for arbitrarily large historical-maximum-to-current-price ratios.
- It therefore does not hard-code the complete historical terminal payoff or enforce zero current value for every already-hit history. The LSTM still has to learn historical distinctions.

Ordinary squared regression through this output has an additional chain-rule factor $\widetilde H\operatorname{sigmoid}(A)(1-\operatorname{sigmoid}(A))$. The current canonical regression instead uses

$$
\ell(A;b)=\widetilde H\operatorname{softplus}(A)-bA,
\quad \partial_A\ell=v-b.
$$

Conditional on the available information, the expected-loss minimizer still represents the conditional target mean when that mean lies within the output's representable interval. Removing the extra output-layer attenuation does not eliminate all gradient problems inside the encoder or guarantee convergence of nonconvex optimization.

The implemented joint loss is

$$
L=\mathbb E[(\mathcal R[v]/D)^2]
+\sum_{k=0}^{N-1}\mathbb E[\ell(A_k^-;\operatorname{stopgrad}(v_{k+1}^+))]
+\mathbb E[\ell(A_T;G)].
$$

D is a chosen auxiliary scale with a numerical floor. Interface losses are summed across dates, whereas PDE and terminal terms are averaged over their samples. Unit outer coefficients do not imply balanced gradients. The canonical loss scale, regional normalization, and date summation all affect optimization.

### 3.4 Investigations and Conclusions

Earlier variants tested stopped versus full interface gradients, frozen encoders, continuous interfaces, terminal pretraining, and MC-supervised diagnostic fits. Continuous interfaces removed a particular interface defect but did not by themselves ensure correct pricing. Good terminal-only fitting likewise did not ensure accurate prices after joint training.

The current gains combine price-level features, memory initialization, the canonical loss, PDE scaling, and longer training. The lookback-put residual was initially normalized by 6s, reducing its squared PDE contribution by a factor of 36. Rerunning with normalization by s changed initial bias from approximately -12.3% to -1.2%, and t=0.5 L2 from 17.9% to 2.6%. Original results remain in their directories; the final table uses `pinn_put_uncapped`.

Lookbacks and some barriers are now priced relatively accurately, while low-value knock-outs and some parity-derived knock-ins remain less accurate. It is no longer appropriate to describe all LSTM-PINN experiments as failures. Nor do the results establish a fixed multiplicative loss at each of 128 monitoring dates: measured decompositions include positive and negative cancellation. Comprehensive LSTM Greek validation has not been established.

### 3.5 Derivatives of a History-Conditioned PINN

The LSTM-PINN still provides a differentiable price function $v_\theta(t,s;h_k)$. Between monitoring dates, holding the encoding fixed allows automatic differentiation of current-price sensitivity and curvature. The same model learns how history affects value and computes how that value changes with current state.

At a monitoring date, current spot may also enter the historical update. One must distinguish a derivative holding the previous history fixed from the derivative of the composite function that includes this update. These are not interchangeable definitions of Delta. The current LSTM tables validate prices and do not establish Delta/Gamma accuracy comparable to the sufficient-state PINNs in Section 2.

## 4. BSDE: Signature Encoding and Raw-History Encoding

### 4.1 Current Raw-History Results

Each separately trained product again has three seeds and 50,000 total updates. After 3,000 terminal-initialization updates, 47,000 updates are allocated backward across 16 date intervals. Price and Z use independent networks; next-date targets are frozen, and the final interval uses the true full-path payoff. No MC price anchors or default absorbing-state gates are used. Price and hedging tables report three-seed mean [minimum, maximum].

| Product | Initial bias | Price L2 at t=0.5 | Average price L2 over four dates |
|---|---:|---:|---:|
| Lookback call | +0.07%[-0.01, +0.12] | 2.20%[2.12, 2.29] | 2.27%[2.18, 2.32] |
| Lookback put | +0.69%[-0.37, +2.53] | 3.61%[3.18, 4.24] | 4.37%[4.05, 4.79] |
| Up-and-out call | +0.17%[-2.33, +3.30] | 9.52%[8.61, 10.19] | 11.63%[11.05, 12.09] |
| Up-and-out put | +0.03%[-0.06, +0.20] | 2.48%[2.29, 2.72] | 3.16%[2.98, 3.36] |
| Down-and-out call | -0.40%[-0.56, -0.27] | 5.00%[4.44, 5.83] | 4.69%[4.29, 5.35] |
| Down-and-out put | +1.22%[-1.81, +3.43] | 13.43%[12.82, 14.36] | 15.68%[14.89, 16.78] |
| Up-and-in call (parity) | +0.00%[-0.06, +0.04] | 0.15%[0.14, 0.16] | 0.25%[0.24, 0.26] |
| Up-and-in put (parity) | -0.10%[-0.77, +0.24] | 7.36%[6.80, 8.06] | 8.71%[8.20, 9.26] |
| Down-and-in call (parity) | +2.69%[+1.79, +3.77] | 24.40%[21.65, 28.43] | 20.94%[19.10, 23.92] |
| Down-and-in put (parity) | -0.05%[-0.14, +0.07] | 0.56%[0.54, 0.60] | 0.89%[0.84, 0.95] |

| Separately trained product | Correlation: payoff vs. native stochastic integral |
|---|---:|
| Lookback call | 0.982[0.982, 0.982] |
| Lookback put | 0.943[0.943, 0.944] |
| Up-and-out call | 0.567[0.566, 0.569] |
| Up-and-out put | 0.988[0.987, 0.988] |
| Down-and-out call | 0.986[0.985, 0.986] |
| Down-and-out put | 0.602[0.599, 0.605] |

Initial prices are generally close, but difficult barriers retain intermediate pricing and hedging errors. Correlation is not Delta/Gamma accuracy. The PINN batch was trained on GPU and this BSDE batch on CPU, so these runs do not establish a fair speed ranking. Source: per-seed scores (`outputs/lstm_ten_50000/summary.json`).

### 4.2 Why Use Signatures?

Order-n signatures contain iterated path integrals such as

$$
S^{i_1\dots i_n}_{a,b}(X)=
\int_{a<u_1<\cdots<u_n<b}dX^{i_1}_{u_1}\cdots dX^{i_n}_{u_n}.
$$

The original approach computes truncated logsignatures of fine-grid path segments augmented with time and lead-lag coordinates, then feeds them chronologically to an LSTM. This uses within-segment information with a shorter recurrent sequence instead of simply subsampling and discarding intermediate observations. It does not guarantee better accuracy for every learning task.

Theoretical expressiveness of a full signature does not guarantee that a fixed low-order truncation preserves the exact running maximum or barrier-hit event. A raw-history encoder sees every observation directly, but requires more recurrent steps and must still learn memory. The two encodings trade information representation against computation; neither is automatically more accurate. The paper's motivation is discussed in [Section 4.2](https://arxiv.org/pdf/2011.10630).

### 4.3 Signature Experiments and Investigations

The author's public single-asset lookback-put example was run for 10,000 updates, with 500 Euler steps and 25 BSDE intervals, taking approximately 24 minutes. Its initial prediction was 0.22226225 versus a matched-model MC estimate of 0.22365595, a point-estimate relative error of 0.623%. Hedging correlation was approximately 0.947, although intermediate consistency defects remained under this project's checks. That example uses r=0.05 and sigma=0.3, so it is not included in the common-market ten-product ranking.

For the project's up-and-out call, the author-architecture baseline used 128 monitoring steps, 16 intervals, independent Y/Z networks, stopped next-target gradients, and a soft terminal loss. After 15,000 updates it predicted approximately 0.000004343 versus matched-grid MC of approximately 0.002824654: a near-zero collapse.

Later Signature comparisons used the H-tilde output, frozen next-date targets, and the exact realized payoff as the final-interval target. The following single-case up-and-out-call comparison uses the same 8,192 evaluation histories, seed 994, and numerical quadrature reference:

| Encoding and configuration | Total updates | L2 at t=0.5 | L2 at t=0.75 | L2 at t=0.9375 |
|---|---:|---:|---:|---:|
| Signature, exact-terminal backward recursion | 15000 | 46.8% | 55.8% | 71.7% |
| Signature, exact-terminal backward recursion | 50000 | 40.0% | 50.1% | 62.6% |
| Raw history + level features + canonical loss, seed 1 | 50000 | 10.4% | 11.7% | 18.3% |

The final row is an earlier single-case raw-history run, not the three-seed ten-product aggregate in Section 4.1. More than encoding changes between rows, and compute is not matched. The results demonstrate improvement in the overall configuration, not that signatures caused the failure. Sources: 15,000-update Signature scores (`outputs/sharp_readout_comparison/exact_terminal_bsde_gpu_exact_reference_scores.json`), 50,000-update Signature scores (`outputs/sharp_readout_comparison/exact_terminal_bsde_total50000_gpu_exact_reference_scores.json`), and raw-history single-case scores (`outputs/sharp_bsde_total50000/exact_reference_scores.json`).

Further investigations included additional final-interval training, frozen versus trainable encoders, two-interval true-payoff targets, and zero-value supervision on already-hit prefixes. An exact terminal target removes terminal-network approximation, not intermediate errors. Absorbing-prefix supervision reduced dead-path values but also reduced surviving-path values, with no reliable overall improvement in the completed comparison. A diagnostic probe's ability to recover hit information does not establish that the pricing head uses it correctly.

### 4.4 Implemented BSDE Loss and Conclusions

The continuous BSDE is $dY=rY\,dt+Z\,dW$. A piecewise-constant interval approximation gives

$$
Y_i+Z_i\Delta W_i\approx e^{-r\Delta t_i}Y_{i+1}.
$$

The exact discounted interval integral is $\int_{t_i}^{t_{i+1}}e^{-r(s-t_i)}Z_s\,dW_s$, generally not a constant times a Brownian increment. Piecewise-constant coefficients therefore leave representation error. Z is the Brownian-integral coefficient and must not be multiplied by volatility times spot a second time.

The current Y loss uses canonical regression with target $b=e^{-r\Delta t}\operatorname{stopgrad}(Y_{\rm next})-\operatorname{stopgrad}(Z)\Delta W$. Z uses a squared residual with current Y detached. These stop-gradient choices are explicit implementation decisions, not an inherent mathematical conflict between the two outputs.

The original Signature barrier configurations remain insufficiently successful; the revised raw-history configurations improve substantially. Exact payoffs, independent networks, and stopped gradients do not individually guarantee success. Without a matched experiment that changes only encoding, these results establish neither superior signature accuracy nor signatures as the cause of failure.

### 4.5 BSDE Advantage: A Hedging Sequence Learned Together with Prices

The BSDE learns prices Y and Brownian-integral coefficients Z together. Along each observed history it directly produces

$$
(Y_0,Z_0),\ (Y_1,Z_1),\ldots,\ (Y_{15},Z_{15}),
$$

Each Z-i depends only on information available by its date. The model therefore supplies a sequence of hedge coefficients that updates with new observations, without first differentiating its price network to obtain Z.

In the single-asset Black-Scholes model, holding $\Delta_t$ shares gives diffusion coefficient $\Delta_t\sigma S_t$. Matching this coefficient to Z gives the stock holding and cash-account value:

$$
\Delta_t^{\rm hedge}=\frac{Z_t}{\sigma S_t},\qquad
\text{cash account value}=Y_t-\Delta_t^{\rm hedge}S_t.
$$

The subscript t denotes time, not a time increment. For a smooth exact solution, the holding equals the price function's Delta. Here an independent Z network learns the coefficient directly; it is not automatically the price network's automatic derivative. Positions can be updated over the model's 16 intervals, but discrete rebalancing and constant-coefficient approximations leave hedging error.

The discounted payoff's stochastic-integral representation also makes the learned sequence useful as a Monte Carlo control variate:

$$
I_\theta=\sum_{i=0}^{15}e^{-rt_i}Z_i\Delta W_i,\qquad
\widehat V_{\rm CV}=\frac1M\sum_{j=1}^M
\left[e^{-rT}G^{(j)}-I_\theta^{(j)}\right].
$$

Under the required adaptedness, integrability, and simulated-increment conditions, the integral term has zero mean. Subtracting it can reduce payoff-sampling variance, without establishing that the Y network is unbiased. The measured variance-reduction factor is $\operatorname{Var}(e^{-rT}G)/\operatorname{Var}(e^{-rT}G-I_\theta)$:

| Product | Variance-reduction factor: three-seed mean [min, max] |
|---|---:|
| Lookback call | 28.156[27.949, 28.273] |
| Lookback put | 9.085[8.960, 9.176] |
| Up-and-out call | 1.474[1.471, 1.478] |
| Up-and-out put | 40.499[40.018, 41.135] |
| Down-and-out call | 35.063[34.321, 35.800] |
| Down-and-out put | 1.569[1.557, 1.576] |

These results use native Z evaluations in `outputs/lstm_ten_50000/bsde/*/seed*/scores.json`. Together with the correlations in Section 4.1, they demonstrate the practical contribution of the hedge sequence. Lookbacks and some barriers show substantial variance reduction; gains for difficult knock-outs remain limited. These are stochastic-integral/control-variate measures, not realized rebalancing P&L tests with transaction costs.

## 5. Mathematical Conversion between PINN and BSDE Formulations

### 5.1 PDE to BSDE: Itô's Formula and Discounted Martingales

First consider the sufficient-state price $v(t,S_t,E_t)$, where E is a continuously monitored historical extremum. Itô's formula gives

$$
dv=\left(v_t+rS_tv_s+\tfrac12\sigma^2S_t^2v_{ss}\right)dt
+\sigma S_tv_s\,dW_t+v_E\,dE_t.
$$

E changes only at new extrema. The extremum condition $v_E=0$ removes the final term. For knock-outs, apply the PDE before the hit and the zero absorbing condition at the hit, after which the contract remains worthless. When the interior PDE holds, define

$$
Y_t=v(t,S_t,E_t),\qquad Z_t=\sigma S_tv_s(t,S_t,E_t),
$$

Then

$$
dY_t=rY_tdt+Z_tdW_t,\qquad Y_T=G,
$$

$$
\boxed{Y_t=G-\int_t^T rY_sds-\int_t^T Z_sdW_s.}
$$

Multiplying by the discount factor gives

$$
d(e^{-rt}Y_t)=e^{-rt}Z_tdW_t,
\quad
Y_t=\mathbb E[e^{-r(T-t)}G\mid\mathcal F_t].
$$

Under square-integrability and related conditions, the stochastic integral has zero conditional mean. The PINN constrains local derivatives; the BSDE constrains stochastic increments and the terminal value of the same price process.

### 5.2 BSDE to PDE: Conditional Expectation, Martingale Representation, and Drift Matching

Starting from the pricing definition, set $M_t=\mathbb E[e^{-rT}G\mid\mathcal F_t]$. In a Brownian filtration, subject to the required integrability conditions, the martingale representation theorem yields

$$
M_t=M_0+\int_0^t\widetilde Z_sdW_s.
$$

Taking $Y_t=e^{rt}M_t$ and $Z_t=e^{rt}\widetilde Z_t$ recovers the preceding BSDE. If a sufficiently smooth sufficient-state representation $Y_t=v(t,S_t,E_t)$ exists, applying Itô's formula and matching drift and diffusion coefficients yields

$$
v_t+rsv_s+\tfrac12\sigma^2s^2v_{ss}=rv,
\qquad Z=\sigma s v_s.
$$

Together with the terminal payoff and extremum or absorbing conditions, this recovers the PDE boundary-value problem. If no explicit sufficient-state reduction is available, the price can instead be expressed as a history functional $F(t,\omega_{\leq t})$. Suitable horizontal and vertical path derivatives yield a PPDE under appropriate regularity. Nonsmooth payoffs and barrier hits require appropriate piecewise, weak, or viscosity interpretations rather than an unrestricted assumption of classical second-order differentiability.

### 5.3 Discrete Observations: Interval PDEs and Interface Conditions

Write the exact observed history as $H_k=(S_0,\ldots,S_k)$; here H-k denotes a historical record, not the auxiliary kernel H. Within an interval, its conditional price $f_k(t,s;H_k)$ solves the ordinary PDE. At the next monitoring date,

$$
f_k(t_{k+1}^-,s;H_k)=f_{k+1}(t_{k+1}^+,s;H_k\oplus s).
$$

This equality holds for the true price with full information. After updating the history, newly hit branches are worth zero and surviving branches retain their conditional value. At expiry the full-path payoff applies. The interval Feynman-Kac representation gives

$$
Y_k=\mathbb E[e^{-r\Delta t_k}Y_{k+1}\mid\mathcal F_{t_k}].
$$

Thus interval PDEs with correct historical-update interfaces and correct conditional-expectation backward recursion describe the same discretely monitored pricing problem. An LSTM state only approximates the full history. If it is not a sufficient representation, exact mathematical equivalence does not guarantee accurate network prices.

Even when the BSDE steps backward only once per eight fine observations, its terminal payoff can check every observation. Coarse recursion must still learn the effect of new within-interval information, and constant Z remains an approximation. Fewer backward stages alone do not establish that every source of error has been resolved.

### 5.4 How PDE Residuals Enter the BSDE Error

For an approximate function, define $R_\theta=(v_\theta)_t+rs(v_\theta)_s+\tfrac12\sigma^2s^2(v_\theta)_{ss}-rv_\theta$. On a continuous interval,

$$
d(e^{-rt}v_\theta)=e^{-rt}R_\theta dt
+e^{-rt}\sigma S_t(v_\theta)_s dW_t.
$$

Integrating from t to T and taking conditional expectations gives

$$
v_\theta(t)-V_t=
\mathbb E\!\left[e^{-r(T-t)}(v_\theta(T)-G)
-\int_t^T e^{-r(s-t)}R_\theta(s)ds\mid\mathcal F_t\right].
$$

This identity assumes correct boundary and historical-interface conditions. Otherwise, extremum finite-variation terms, absorbing-boundary defects, or mismatches between the left and right monitoring-date values must also be included. Terminal errors, interior residuals, and interface errors can therefore all influence initial and intermediate prices. Cancellation in their means may make an initial price look accurate while concealing conditional errors.

A finite-sample training residual is not a uniform bound on this conditional integrated error. Small training loss therefore does not establish accurate prices everywhere, and initial bias alone does not identify a unique failure mechanism.

### 5.5 Why the Neural Implementations Are Not Automatically Equivalent

| Mathematical connection | Difference in implementation |
|---|---|
| Same conditional-expectation price | State/history encodings may differ and need not preserve the same sufficient information |
| Matching PDE and BSDE drifts | Collocation points, sampled paths, date grids, and loss scales differ |
| For a smooth solution, Z equals volatility times spot times the spot derivative | An independent Z network need not equal the price network's automatic derivative |
| Matching terminal and boundary conditions | Soft penalties, hard constraints, and exact payoff targets introduce different errors |
| Continuous stochastic-integral representation | Coarse constant Z and frozen approximate next-date targets add approximation errors |

The conversion is a mathematical relation between a pricing equation and its stochastic representation. It is not a line-by-line equivalence between training programs, Lagrangian optimization duality, or an error-free conversion of a trained PINN into a trained BSDE network.

## 6. Overall Conclusions and Scope

### 6.1 Established Findings

The two approaches provide different direct outputs. A PINN supplies a differentiable price function from which Delta, Gamma, and other sensitivities can be computed. A BSDE learns Y and Z together, producing a history-adapted hedge-coefficient sequence that can also reduce Monte Carlo variance. The former is supported by derivative errors in Section 2.1; the latter by correlations in Section 4.1 and variance reduction in Section 4.5. Both advantages are stated within their measured validation scope.

1. With known sufficient states, PINNs can be trained by sampling valid states directly, without recurrent history encoding. H-based structure and refinement deliver accurate prices and Greeks on the tested domain, with explicit mathematical priors contributing to the gains.
2. Without supplied sufficient states, an LSTM must learn task-relevant memory from observed histories. Joint training is possible. Current results are relatively accurate for lookbacks and several barriers, while difficult knock-outs, some knock-ins, and hedging remain less accurate.
3. In the completed barrier experiments, the original Signature configurations did not meet pricing requirements, whereas the revised raw-history configurations performed better. Encoding, loss, and training procedures changed together. This is a comparison of configurations, not evidence that raw-history encoding alone outperforms Signature encoding, nor a successful reproduction of the paper's original method.
4. Exact terminal values, hard boundaries, and single-point anchors address different issues and do not individually guarantee accurate interior conditional prices. Failures of hard structures without H are informative comparisons and are not omitted to favor H-based models.

### 6.2 Scope of the Conclusions

The project concludes with implementations, investigations, and pricing comparisons under the specified single-asset market, monitoring rules, and test sets. Means and ranges over three training seeds describe stability of the current configurations. They do not establish generalization across market parameters, accuracy for arbitrary extreme states, comprehensive dynamic hedging performance, or efficiency at matched computational cost.

No unique cause of the Signature failures has been identified. This report records the original configurations' shortcomings and the improved configurations' results without assigning the change solely to history encoding. The project ends within this scope; no encoding ablation or additional extension experiments are scheduled.

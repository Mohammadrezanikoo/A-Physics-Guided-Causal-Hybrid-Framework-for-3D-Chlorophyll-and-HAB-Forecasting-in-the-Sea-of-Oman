# A Physics-Guided Causal Hybrid Framework for 3D Chlorophyll and Harmful Algal Bloom Forecasting in the Sea of Oman

**Redesign specification — target venue: *Water Research***

---

## 0. Summary of the Redesign

The original pipeline (Physics-Informed ConvLSTM3D + CBAM Attention + Bayesian output layer + Monte-Carlo uncertainty) is retained as the *sequential backbone* and as the source of calibrated epistemic/aleatoric uncertainty. Four additions elevate it from "physics-informed deep learning" to a **Physics-Guided Causal Hybrid Forecasting** system:

1. A **mechanistic ADR (advection–diffusion–reaction) residual** replaces the heuristic vertical-curvature penalty as the physics loss.
2. A **Fourier Neural Operator (FNO) residual block** is inserted after the ConvLSTM3D latent state, plus **regime-specific multi-scale attention** (surface mixed layer / thermocline / DCM) and an **Event-Conditioning Module** (FiLM / cross-attention) for rapid heatwave, stratification-collapse, and nutrient-pulse signals.
3. A **Physics-Guided Dynamic Causal Graph Module**, computed at coarse temporal granularity (not per-batch), whose adjacency and node embeddings condition the hybrid model.
4. A **calibrated, multi-horizon, cross-regime evaluation protocol** with a mandatory ablation ladder, domain-generalization tests, and perturbation-based process attribution tying model internals to known oceanographic mechanisms.

The 3D regular grid (time × depth × lat × lon), the Bayesian convolutional output head, and Monte-Carlo predictive sampling from the original design are preserved unchanged, so the redesign is a drop-in extension of the existing data pipeline rather than a rebuild.

---

## 1. Physics Foundation: Advection–Diffusion–Reaction (ADR) Residual

### 1.1 Governing equation

The heuristic vertical-curvature penalty is replaced by a differentiable residual of the 3D chlorophyll transport equation, evaluated on the model's predicted field Ĉ(t, z, x, y):

```
R_ADR = ∂Ĉ/∂t
        + u ∂Ĉ/∂x + v ∂Ĉ/∂y + w ∂Ĉ/∂z          (advection)
        − ∂/∂z ( K_v ∂Ĉ/∂z ) − ∇_h·(K_h ∇_h Ĉ)   (diffusion)
        − R_bio(Ĉ, N, P, T, L)                    (source/sink)
```

The physics loss is `L_phys = E[ w_mask · R_ADR^2 ]`, evaluated on the predicted field using automatic differentiation (finite-difference stencils implemented as fixed, non-trainable Conv3d kernels along t, z, x, y, applied to the network's mean output Ĉ). `w_mask` down-weights grid points with high data-fill uncertainty (§1.5) so the residual is not enforced where the underlying observation was itself interpolated.

### 1.2 Term-by-term parameterization

| Term | Source / parameterization |
|---|---|
| `u, v` (horizontal velocity) | `Current_U`, `Current_V` from Copernicus, used directly as input channels |
| `w` (vertical velocity) | Diagnosed, **not measured**, via depth-integrated continuity: `w(z) = w(0) − ∫₀^z (∂u/∂x + ∂v/∂y) dz'`, with `w(0)` estimated from the time-derivative of SSH (`w(0) ≈ ∂SSH/∂t` as a linearized surface boundary condition). Both derivatives are computed with the same fixed Conv3d stencils used for `R_ADR`. |
| `K_v` (vertical diffusivity) | Parameterized as a smooth, monotonically decreasing function of stratification: `K_v = K_min + (K_max − K_min) · σ(−α·N² + β·𝟙[z < MLD])`, i.e. large inside the mixed layer, suppressed below it in proportion to N². `α, β, K_min, K_max` are learnable scalars (4 extra parameters), keeping the parameterization physically bounded (`K_v ∈ [K_min, K_max]`) while letting the model fit local turbulence closure from data. |
| `K_h` (horizontal diffusivity) | Fixed to a literature-typical value for shelf/open regions (e.g. O(10–100) m²/s), optionally trainable as a single global scalar. |
| `R_bio` (biological source/sink) | Michaelis–Menten nutrient- and temperature-limited net growth minus a linear loss (grazing + mortality + photo-oxidation): `R_bio = μ_max · f(T) · min( N/(N+k_N), P/(P+k_P) ) · g(L) · Ĉ − m·Ĉ`, with `f(T)` an Eppley-type exponential temperature response, `μ_max, k_N, k_P, m` learnable within physically plausible bounds (enforced via softplus reparameterization), and `g(L)` a Beer–Lambert light-limitation term (§1.3, optional). |

### 1.3 Optional light-attenuation term

When an irradiance proxy is available (satellite PAR, or a day-length/`season_sin`,`season_cos`-derived clear-sky estimate as a fallback), self-shading is modeled with Beer–Lambert attenuation:

```
I(z) = I(0) · exp( −(k_w + k_c · C̄(0→z)) · z )
g(L) = I(z) / (I(z) + k_I)
```

where `k_w` is background (water) attenuation, `k_c` is a chlorophyll-specific self-shading coefficient (learnable), and `C̄(0→z)` is the depth-integrated predicted chlorophyll above `z`. If no irradiance proxy is available, `g(L)` is fixed to 1 (light non-limiting) and this term is flagged as **not active** in the reported configuration (see §10).

### 1.4 Non-negativity and approximate mass conservation: a differentiable projection layer

Rather than a hard clip (`ReLU`), non-negativity and approximate column-mass conservation are enforced by a **differentiable proximal projection layer** appended after the Bayesian output head, applied to the predicted mean field per forward pass:

1. **Non-negativity** via a smooth proximal step: `C⁺ = softplus_β(Ĉ)` with a temperature parameter `β` annealed from soft (training-friendly gradients) to near-hard (≈ ReLU) over training, rather than a fixed activation choice.
2. **Approximate mass conservation** across the forecast step via a single differentiable Dykstra-like alternating projection (2–3 inner iterations, unrolled and fully differentiable) that adjusts `C⁺` toward satisfying:
   `∫∫∫ C⁺ dV(t+Δt) ≈ ∫∫∫ C(t) dV + ∫∫∫ R_bio dV·Δt − Flux_boundary`
   i.e. the predicted total water-column chlorophyll mass should not drift beyond what advection, boundary flux, and net biological production/loss can explain. Boundary flux is estimated from the advective terms at the domain edges.
3. The projection is implemented as a small number of unrolled proximal-gradient steps (not an external convex solver), so it remains trainable end-to-end with standard autograd — this is what distinguishes it from a simple post-hoc clip/renormalize.

### 1.5 Optional stoichiometric coupling residual (Chl–O₂–NO₃)

When co-located O₂ and nitrate profiles are of sufficient quality (§10), an additional **soft** residual penalizes stoichiometric inconsistency using Redfield-ratio-informed relationships between net community production (inferred from the predicted ∂Ĉ/∂t and `R_bio`), oxygen production, and nitrate drawdown:

```
L_stoich = E[ w_mask · ( λ_O2 · (ΔO2_pred − r_O2:C · R_bio·Δt)²
                       + λ_NO3 · (ΔNO3_pred − r_N:C · R_bio·Δt)² ) ]
```

with `r_O2:C`, `r_N:C` fixed at literature Redfield values (or gently learnable within a narrow prior). This term is **masked off** wherever O₂/NO₃ data density in a grid cell over the training window is below a coverage threshold, so it degrades gracefully to a purely data-driven treatment of those channels rather than injecting bias from sparse data (see §10 for the explicit optionality statement).

---

## 2. Hybrid Architecture

### 2.1 Backbone (unchanged)

`ConvLSTM3D` (stacked 3D-convolutional LSTM cells over depth×lat×lon, evolved over the input time window) remains the sequential encoder, now feeding into the additions below rather than directly into the Bayesian head.

### 2.2 Multi-scale CBAM → three physically meaningful regimes

Instead of one global CBAM3D block, the depth axis of the ConvLSTM3D hidden state is **soft-partitioned** into three overlapping depth bands using smooth, learnable Gaussian membership weights centered near the surface, the thermocline (steepest N² gradient per column, computed per grid column, per time step), and the DCM depth range (estimated online as the argmax of a smoothed vertical Ĉ profile from the previous iterate). A CBAM3D block is applied **within each band** (band-specific channel + spatial-depth attention), and the three attended fields are recombined with the same membership weights (soft blending at band boundaries, avoiding discontinuities). This lets the network learn distinct attention patterns for surface bloom dynamics, thermocline shear/mixing effects, and DCM formation/erosion, which single global attention cannot separate.

### 2.3 Fourier Neural Operator (FNO) residual block

An FNO block (chosen over a Neural-ODE alternative because the spatial grid is regular and fixed-resolution) is inserted as a residual branch operating on the multi-scale-attended latent state:

```
Z_fno = Z + FNO_block(Z)
FNO_block(Z) = IFFT3D( W_spectral · FFT3D(Z), modes=(m_z, m_x, m_y) ) + Conv3d_1x1x1(Z)
```

The FFT is taken over (depth, lat, lon); the spectral weight `W_spectral` is truncated to the lowest `(m_z, m_x, m_y)` Fourier modes (kept small, e.g. 4–8 per axis, to control parameter count on a modest regional grid) and a local 1×1×1 convolution supplies the high-frequency complement. The FNO branch is intended to capture domain-scale, quasi-stationary spatial structure (basin-scale gyres, persistent upwelling cells) that local ConvLSTM/CBAM receptive fields under-represent, while the residual (identity + branch) connection keeps training stable and lets the branch contribute zero at initialization.

### 2.4 Event-Conditioning Module (in-path, not a late branch)

Rapid-change indicators — `depth_of_heatwave`, `surface_RHI_frequency`, `subsurface_RHI_frequency`, a stratification-collapse indicator (`−∂N²/∂t` thresholded), and a nutrient-pulse indicator (`∂Nitrate/∂t`, `∂Phosphate/∂t` thresholded) — are embedded through a small MLP into a conditioning vector `e_t` per time step. This vector modulates the FNO-residual output via **Feature-wise Linear Modulation (FiLM)**:

```
Z_cond = γ(e_t) ⊙ Z_fno + β(e_t)
```

with an alternative **cross-attention** variant available (event tokens as keys/values, spatial-depth locations as queries) when the event indicators are themselves spatially resolved (e.g., a gridded heatwave-depth field rather than a scalar). FiLM is the default for its lower cost; cross-attention is offered as an ablation variant (§7).

### 2.5 Bayesian output head and Monte-Carlo uncertainty (unchanged)

`Z_cond` is passed to the existing `BayesianConv3d` head producing `(mean, log_var)`; Monte-Carlo sampling (Bayesian-weight resampling + MC-Dropout retained in the backbone) produces the predictive ensemble used for calibrated HAB probability, exactly as in the original design.

---

## 3. Physics-Guided Dynamic Causal Graph Module

### 3.1 Graph definition

Nodes are aggregated spatial cells (a coarsened version of the lat/lon grid, e.g. block-averaged to keep node count tractable — order tens to low hundreds of nodes, not one node per raw grid cell) or fixed monitoring stations if available. At each node, the feature vector is the local (or column-averaged) state: `{Temp, Sal, Current_U, Current_V, SSH, O2, Nitrate, Phosphate, N2, MLD, Chl}`. Edges are directed and represent a hypothesized causal influence of one node/variable pair on another (e.g., upstream current cell → downstream nutrient cell; SST anomaly node → local N² node).

### 3.2 Causal discovery method and cost control

Given the modest node count and moderate series length, **PCMCI+** is the primary choice (handles autocorrelated, lagged, multivariate time series with reasonable compute; produces both a graph structure and lagged effect strengths). **DYNOTEARS** is offered as a faster, purely linear-Gaussian fallback when compute or series length is limiting; a simple **lagged Granger-causality screen** is the minimal fallback if neither is feasible within the project's compute budget. All three share the same downstream interface (a directed, weighted adjacency `A_t` plus node embeddings), so the choice is a swappable implementation detail, not an architectural one.

**Cost control (critical for feasibility):** the causal graph is **not** recomputed every batch or epoch. Two supported schedules:

- **Periodic update:** recompute `A_t` once per sliding window (i.e., once per `time_steps`-length input sequence) or every *k* windows, using the trailing history available at that point, then hold `A_t` fixed for all training steps until the next scheduled update.
- **Regime-specific precomputed graphs:** cluster the training period into a small number of regimes (monsoon vs. inter-monsoon, using `season_sin`/`season_cos` and wind/current proxies; normal vs. heatwave, using `surface_RHI_frequency`) and precompute one causal graph per regime **once**, offline, before training. At training/inference time the appropriate precomputed graph is selected by the current regime label — zero additional discovery cost during training.

The regime-specific precomputed approach is recommended as the default for reproducibility and cost; the periodic-update approach is offered as a more adaptive but more expensive ablation variant.

### 3.3 Injection into the hybrid model

The adjacency `A_t` (or the regime-selected `A_regime`) and per-node causal embeddings `h_causal` (obtained from a lightweight graph encoder, e.g. a 2-layer graph attention network run once per graph update, not per batch) are injected in two complementary ways:

1. **Conditioning feature:** `h_causal`, broadcast/upsampled from the coarse causal-graph resolution back to the full grid resolution (bilinear/nearest upsampling in lat/lon, repeated across depth within a column, or a learned small deconvolution), is concatenated as extra input channels alongside the existing physics feature channels (§1) before the ConvLSTM3D.
2. **Soft regularization:** a causal-consistency penalty discourages the model's *learned* effective spatial dependency (approximated, cheaply, by the spatial-attention footprint of the multi-scale CBAM module in §2.2) from strongly contradicting high-confidence causal edges in `A_t`:
   `L_causal = E[ Σ_{(i,j): A_t(i,j) high-confidence} max(0, |A_t(i,j)| − attn_footprint(i,j))² ]`
   applied only to edges with discovery confidence above a threshold, so low-confidence edges do not over-constrain the network.

This is the component that most directly elevates the framework from "physics-informed" to **physics-guided *causal* hybrid**: the causal graph supplies directionality and mechanistic hypotheses (e.g., "this upwelling cell causally precedes that nutrient pulse") that a purely correlational spatial-attention or convolutional receptive field cannot represent on its own.

---

## 4. Full Loss Function

```
L_total = L_data                                  (Gaussian NLL on Chl-a, existing Bayesian head)
        + λ_phys   · L_phys                        (ADR residual, §1.1)
        + λ_mass   · L_mass                         (residual mass-conservation penalty from the
                                                       unrolled projection layer, §1.4 — reported even
                                                       though the projection is mostly enforced structurally)
        + λ_stoich · L_stoich                        (optional, §1.5, zero if data coverage insufficient)
        + λ_causal · L_causal                        (causal-consistency regularizer, §3.3)
        + λ_KL     · KL(q(w) ‖ p(w)) / n_batches      (Bayesian weight KL, existing)
```

All `λ_*` are treated as hyperparameters subject to tuning (Optuna, as in the existing pipeline) with physically motivated search ranges (e.g. `λ_phys`, `λ_mass` small enough not to dominate early training before the data term has shaped a reasonable field). A staged/curriculum weighting schedule is recommended: `λ_phys`, `λ_mass`, `λ_causal` ramped from 0 to their target value over the first ~20% of training, so the ADR/mass/causal residuals regularize an already data-plausible field rather than fighting a randomly initialized one.

### 4.1 Pseudocode — forward pass and loss

```
function forward(x_seq, coord_channels, event_features, causal_graph):
    h_seq        = ConvLSTM3D(concat(x_seq, coord_channels))         # existing backbone
    h_multiscale = MultiScaleCBAM(h_seq[-1])                          # §2.2, 3 depth-regime bands
    h_causal_up  = upsample(GraphEncoder(causal_graph))               # §3.3, injected as extra channels
    h_fno        = h_multiscale + FNOBlock(concat(h_multiscale, h_causal_up))   # §2.3
    h_cond       = FiLM(h_fno, EventEncoder(event_features))          # §2.4
    mean, logvar = BayesianConv3d(h_cond)                             # existing Bayesian head
    C_pred       = MassConservingNonnegProjection(mean, C_prev)       # §1.4, unrolled proximal steps
    return C_pred, logvar

function loss(C_pred, logvar, C_true, x_last, causal_graph, attn_footprint):
    L_data    = gaussian_nll(C_pred, logvar, C_true)
    L_phys    = mean(mask * ADR_residual(C_pred, x_last)**2)          # §1.1, fixed-stencil Conv3d derivatives
    L_mass    = mass_conservation_penalty(C_pred, C_prev, x_last)     # residual after projection
    L_stoich  = stoich_residual(C_pred, x_last) if coverage_ok else 0  # §1.5
    L_causal  = causal_consistency_penalty(causal_graph, attn_footprint)  # §3.3
    L_kl      = model.kl_loss() / n_batches
    return L_data + λ_phys*L_phys + λ_mass*L_mass + λ_stoich*L_stoich \
           + λ_causal*L_causal + λ_KL*L_kl
```

---

## 5. Training and Evaluation Protocol

### 5.1 Data splits

- **Primary temporal split** (as in the existing pipeline): chronological train/val/test.
- **Leave-one-year-out (LOYO):** for each available year with sufficient coverage, train on all other years and test on the held-out year, to assess generalization across interannual variability (§5.4).
- **Extreme-event holdout:** identify the top-*k* highest `surface_RHI_frequency` / documented HAB periods; hold at least one such episode entirely out of training (never seen in train or val) to test genuine extrapolation to extreme conditions rather than interpolation within a similar regime.

### 5.2 Multi-horizon evaluation

The model is trained/evaluated at three forecast horizons — **1 day, 3 days, 7 days** — either via three horizon-specific `forecast_horizon` configurations of the existing sliding-window dataset, or a single multi-head model sharing the backbone with horizon-specific Bayesian output heads (recommended, to amortize training cost and allow horizon-consistency analysis).

### 5.3 Metrics

| Category | Metrics |
|---|---|
| Point accuracy | RMSE, MAE, R² (as in the existing pipeline, per horizon, per regime, per depth band) |
| Probabilistic calibration | CRPS, Continuous Ranked Probability Skill Score (CRPSS vs. climatology/persistence), reliability diagrams (predicted vs. observed exceedance frequency), Prediction Interval Coverage Probability (PICP) at 50/80/95% |
| HAB event skill | ROC-AUC and Brier score for `P(Chl-a > threshold)` classification, Precision/Recall/F1 at the operational threshold, lead-time-to-detection distribution |
| Vertical/3D skill | Depth-resolved RMSE profile, DCM-depth error (predicted vs. observed argmax-Chl depth), total water-column mass error (ties to §1.4) |
| Mechanistic diagnostics | ADR residual magnitude maps, causal-consistency loss trend, process-attribution scores (§8) |

### 5.4 Cross-regime evaluation

Held-out test performance is reported **stratified** by: (i) monsoon vs. inter-monsoon (via `season_sin`/`season_cos`-derived phase), (ii) normal vs. heatwave years (via `surface_RHI_frequency`/`subsurface_RHI_frequency`/`depth_of_heatwave` thresholds), and (iii) LOYO folds. This directly tests whether the causal-graph and event-conditioning additions earn their complexity by improving performance specifically in the regimes they were designed for (heatwave/nutrient-pulse periods), not just on average.

### 5.5 Baselines

- **Persistence** (last observed value carried forward).
- **Climatology** (regime/day-of-year mean).
- **Pure Transformer** (spatio-temporal transformer over the same flattened grid tokens, no physics/causal components) — isolates the value of the physics+causal additions versus a strong purely data-driven sequence model.
- **A recent physics-informed neural network for ocean biogeochemistry** (representative published architecture, re-implemented at matching capacity) — positions the contribution against the closest prior art rather than only against non-physics baselines.
- **Copernicus Marine Service product** (e.g., the reanalysis/forecast chlorophyll field itself) is reported as an **operational reference**, not a peer baseline, since it is a data-assimilative product with a different design purpose; framed explicitly as such in reporting to avoid an apples-to-oranges claim.

---

## 6. Mandatory Ablation Ladder

| Variant | Components active | Purpose |
|---|---|---|
| (a) | ConvLSTM3D + CBAM + Bayesian head only (pure data-driven) | Lower bound / existing-pipeline baseline |
| (b) | (a) + ADR physics residual (§1) | Isolates value of mechanistic physics loss alone |
| (c) | (b) + FNO residual + multi-scale attention + event-conditioning (§2) | Isolates value of the hybrid operator/attention/event stack |
| (d) | (c) + Physics-Guided Dynamic Causal Graph (§3) — **full model** | Isolates the marginal contribution of causal guidance |

Each variant is evaluated under the identical protocol in §5 (all horizons, all regimes, all metrics), so the ablation table doubles as the primary evidence for the "elevates from physics-informed to physics-guided causal hybrid" claim. A secondary ablation additionally compares FiLM vs. cross-attention event-conditioning (§2.4) and periodic-update vs. regime-precomputed causal graphs (§3.2), reported as supplementary rather than primary results.

---

## 7. Process Attribution and Interpretability

Three internal signals are mapped onto known regional mechanisms:

1. **ADR residual magnitude fields** — spatial/temporal hotspots of residual magnitude flag where the mechanistic equation under-explains the observed/predicted field, which is interpreted against known coastal upwelling zones, filament regions, and eddy corridors in the Sea of Oman.
2. **Multi-scale attention weights** (§2.2) — band-specific attention maps are compared against independently estimated thermocline depth and DCM depth to check whether the network's learned "thermocline band" and "DCM band" attention actually track the physically diagnosed depths, as a sanity/interpretability check rather than a performance metric.
3. **Causal edge strengths** (§3) — high-confidence edges are qualitatively cross-referenced against known process pathways (e.g., OMZ shoaling → subsurface O2 minimum → altered nutrient recycling → DCM intensification) to assess whether discovered edges are mechanistically plausible, not merely statistically significant.

In addition, a **perturbation-based process attribution** quantifies each physical driver's marginal contribution: each physics input channel (currents, N², MLD, nutrients, temperature) is perturbed (zeroed, shuffled in time, or regime-swapped) one at a time at inference, and the resulting degradation in forecast skill (§5.3 metrics) is reported as that driver's attributed contribution — a model-agnostic complement to the attention/causal-edge diagnostics above.

---

## 8. Suggested Title and Abstract Skeleton (*Water Research*)

**Suggested title:**
*"A Physics-Guided Causal Hybrid Deep Learning Framework for Three-Dimensional Chlorophyll and Harmful Algal Bloom Forecasting in an Oxygen-Deficient Marginal Sea"*

*(Alternative: "Coupling Mechanistic Transport Physics and Dynamic Causal Discovery for Calibrated 3D Harmful Algal Bloom Forecasting: A Case Study of the Sea of Oman")*

**Abstract skeleton:**

1. **Context/problem** — The Sea of Oman is an oxygen-deficient marginal sea with recurrent harmful algal blooms; short-term, depth-resolved, uncertainty-aware forecasts are needed but existing approaches are either purely statistical (limited mechanistic insight, poor extrapolation to extreme regimes) or purely physical (computationally heavy, not easily calibrated against sparse observations).
2. **Gap** — Prior physics-informed deep learning for coastal biogeochemistry typically encodes physics as a soft penalty on a purely correlational spatio-temporal network, without explicit causal structure among the driving variables, limiting both extrapolation and mechanistic interpretability.
3. **Approach** — We introduce a physics-guided causal hybrid framework combining (i) a differentiable advection–diffusion–reaction residual constrained by a mass-conserving, non-negativity-preserving projection layer, (ii) a ConvLSTM–Fourier Neural Operator backbone with regime-specific multi-scale attention and event-conditioned modulation for rapid heatwave/nutrient-pulse dynamics, and (iii) a physics-guided dynamic causal graph module that injects directional, mechanism-consistent structure among key oceanographic drivers, all coupled to a Bayesian output layer for calibrated Monte-Carlo uncertainty.
4. **Data/methods** — CTD and Copernicus Marine Service 3D fields (temperature, salinity, currents, SSH, oxygen, nutrients, mixed-layer depth, stratification, chlorophyll) over [period]; a four-stage ablation, multi-horizon (1/3/7-day) and cross-regime (monsoon/inter-monsoon, normal/heatwave, leave-one-year-out, extreme-event holdout) evaluation against persistence, climatology, transformer, and physics-informed neural network baselines.
5. **Results (skeleton, to be completed after experiments)** — Report calibration (CRPS/CRPSS, reliability), point-accuracy gains per ablation stage, and where the causal-graph/event-conditioning additions specifically improve skill (expected: heatwave and nutrient-pulse regimes) versus where they add limited value on average conditions — framed as an honest, regime-conditional contribution rather than a uniform improvement claim.
6. **Interpretability/attribution** — Perturbation-based process attribution and causal-edge/attention diagnostics link model behavior to coastal upwelling, filament dynamics, eddy pumping, OMZ shoaling, and DCM formation/erosion.
7. **Significance** — Positions the framework as both an operationally relevant, uncertainty-calibrated forecasting tool and a mechanistic-diagnosis tool for HAB drivers in oxygen-deficient marginal seas, with a modular design transferable to other regions given comparable 3D reanalysis/CTD coverage.

---

## 9. Explicit Statement of Optional Components (Data-Coverage Contingencies)

To keep the framework honestly scoped to what the available CTD + Copernicus dataset can support, the following components are **explicitly optional** and the paper/implementation should state, per experiment, which were active:

- **Light attenuation / self-shading term (§1.3):** active only if an irradiance proxy (satellite PAR or a clear-sky day-length estimate from `season_sin`/`season_cos`) is available at adequate resolution; otherwise `g(L) ≡ 1` and this is reported as a limitation, not silently omitted.
- **Stoichiometric Chl–O₂–NO₃ coupling residual (§1.5):** active only where co-located O₂/nitrate data density exceeds a stated coverage threshold in a given grid cell/time window; falls back to purely data-driven treatment of O₂/nitrate elsewhere, with the coverage threshold and resulting spatial mask reported.
- **Causal discovery method (§3.2):** PCMCI+ is the target method; DYNOTEARS or a lagged-Granger screen are acceptable, disclosed fallbacks if compute/series-length constraints make PCMCI+ infeasible at the achieved resolution — the paper should report which was actually used and why.
- **Event-conditioning mechanism (§2.4):** FiLM is the default; cross-attention is used only if event indicators are available as spatially resolved fields rather than scalars/regime labels.
- **Vertical velocity diagnosis (§1.2):** the continuity-based diagnosis of `w` from SSH and horizontal divergence is an approximation; if independent vertical-velocity estimates become available (e.g., from a higher-resolution regional model), they should be substituted and the change reported as a sensitivity experiment rather than assumed equivalent.
- **Causal graph node resolution:** the coarsened node grid (or fixed stations) is a computational necessity, not a claim that causal structure is genuinely absent at finer scales; this should be stated as a resolution limitation.

---

## 10. Compatibility With the Existing Implementation

All additions are designed to be inserted into the current codebase without altering the 3D grid construction, scaling, or dataset/dataloader logic:

- The ADR residual (§1) reuses the existing `x_last_step` physics-feature channels and replaces `physics_informed_loss(...)` with a new function operating on the same tensors, using fixed (non-trainable) `Conv3d` stencils for the required derivatives — no change to `GridSequenceDataset`.
- The multi-scale CBAM, FNO block, and Event-Conditioning Module (§2) are inserted between the existing `ConvLSTM3D` and `BayesianConv3d` in `PhysicsInformedConvLSTMAttentionNet.forward`, preserving the existing Bayesian head and MC-sampling code (`mc_predict`, `forecast_full_grid`) unchanged.
- The causal graph module (§3) is an **offline/periodic preprocessing step** producing a small adjacency + embedding artifact consumed as an extra conditioning input — it does not require changes to the training loop's per-batch cost profile beyond an upsample-and-concatenate.
- The evaluation protocol (§5–7) extends, rather than replaces, the existing `evaluate_model`/metrics code with additional CRPS/reliability/regime-stratified computations over the same `PRED_MEAN`, `PRED_STD`, `HAB_PROB`, `TRUTH` arrays already produced by the pipeline.

This keeps the redesign implementable within the existing dataset size and computational budget, consistent with the constraint stated in the request.

# Denoising as Implicit Regularization: Why Generative Training Makes One-Step Robot Policies Robust

**Paper draft v0.1 — 2026-07-01.**
Scope: clean scripted demonstrations, state observations, mm-precision manipulation (robomimic ToolHang init→insertion). Pending results are marked ⏳.

---

## Abstract (draft)

Behavior cloning with MSE regression fails on precision manipulation long before it fails on
coarse tasks. On ToolHang insertion (~1–2 mm tolerance), an MSE policy trained on 2,000 clean
scripted demonstrations succeeds 69% of the time, while a two-step flow-map policy (MIP)
trained on the *same data* succeeds 96%. We give a quantitative anatomy of this gap and
localize its cause. (1) Failure is a support-geometry event: success rate equals
100 − P(crossing an empirically calibrated point-of-no-return in state space), and the fatal
deviations are *mild* — riding the boundary of the data support during the alignment phase
degrades achievable precision by 20×, after which insertion attempts destroy the episode.
(2) The gap is created *during training, not at inference*: executing only the first step of
MIP — a forward pass byte-identical to the MSE regressor — retains nearly the entire benefit
(94% vs 96%). The auxiliary denoising head, trained with shared weights, is the only
difference. (3) A small-noise expansion shows this head is exactly ridge-type Jacobian
regularization *in the action-input slot*; measured input-output Jacobians confirm the
deployed map is flatter precisely in the off-support band where the MSE policy's error is a
low-rank, systematic bias. (4) The same behavioral robustness is reproduced through data
instead of loss (DART-style recovery coverage), and the two mechanisms are redundant rather
than additive. We correct a common account of diffusion-policy robustness based on
schedule-coefficient shrinkage: it predicts *no* regularization for x0-parameterized policies,
yet our x0-parameterized one-step policy carries the entire effect. The benefit of generative
training for low-variance conditional control is a training-time property of the learned
function — *generative training, discriminative deployment*.

---

## 1. Introduction

- Precision manipulation = low-variance conditional learning: expert action is a deterministic
  function a = f\*(s) plus negligible noise; tolerance τ_task is millimetric.
- Puzzle: same data, same network, same single forward pass at deployment; only the training
  objective differs → 69% vs 94%. Neither capacity, nor convergence, nor inference-time
  iteration explains it (Sec. 5).
- Contributions:
  1. A measurement toolkit (support distance, effective boundary, point-of-no-return) and a
     quantitative failure anatomy for BC on precision tasks.
  2. Causal localization of the flow/diffusion-style benefit to the *training objective*
     (one-step deployment retains it), with capacity/convergence/iteration controls.
  3. A corrected theoretical account: the auxiliary denoising objective is (to second order)
     an explicit Frobenius penalty on the action-input Jacobian; transmission to the deployed
     obs→action map is established by ablation (⏳), not by reconstruction arithmetic —
     which provably does not apply to x0-parameterizations.
  4. A data-side counterpart (recovery coverage) that reproduces the same behavioral
     signature, and evidence the two axes are redundant, with disjoint residual failures.

## 2. Setup

Task: robomimic ToolHang, init→insertion segment; OSC_POSE delta control (7-dim actions,
pos full-scale 5 cm/step); state obs s ∈ R⁵³ (object pose, EEF pose, gripper). Data: N clean
scripted demonstrations (N ∈ {200, 2000, 20000}), near-deterministic expert.
Policies: ChiUNet (~20M params), action chunks H=16, execute 8.
- **MSE**: â = f_θ(0, 0, s) trained with ‖â − a\*‖².
- **MIP**: same network; two heads with *shared weights*:
  head1 f_θ(0, 0, s) → a\* (identical target to MSE) and
  head2 f_θ(τ, a\* + (1−τ)ε, s) → a\*, τ = 0.9, ε ∼ N(0, I).
  Deployment uses head1 (one step; "MIP-step1") or head1∘head2 ("MIP-full").
Evaluation: official-equivalent harness; **held-out** initial-state seeds (21000+; training
seeds are 0–20580); n=100 per cell; a second disjoint seed set (22000+) bounds eval variance
at ±3 pts.

### 2.1 Support and distance

Support = the set of states visited by the training data. Distance
d(s) = min_i ‖(s − s_i)/σ‖₂ (per-dim z-scored 1-NN). Calibrated landmarks:
| d | meaning | source |
|---|---|---|
| ≲1 | inside support | cloud self-NN p95 |
| 1.79 | in-domain boundary | held-out same-distribution trajectories, p95 |
| ≈1.4 | **effective boundary** | where achievable alignment-precision tail crosses the 2 mm insertion tolerance |
| 4.0 | **point of no return (PNR)** | min θ with P(fail \| trajectory ever exceeds θ) = 100% |

## 3. Anatomy: where the 27 points go

### 3.1 SR ≈ 100 − P(crossing PNR)
(n=40/model closed-loop; table)
| model | SR | reach d>2.45 | reach PNR | escalate P(PNR\|off) | 100−reachPNR |
|---|---|---|---|---|---|
| MSE-2k | 68% | 75% | 32% | 43% | 68 ✓ |
| MSE-20k | 88% | 62% | 12% | 20% | 88 ✓ |
| MIP-2k | 92% | 58% | 8% | 13% | 92 ✓ |
| MIP-20k | 100% | 45% | 0% | 0% | 100 ✓ |
| DART-6k | 98% | 78% | 12% | 16% | 88 (+10)† |

†the exception proves the ruler: measured against **its own** (recovery-augmented) support,
DART crosses its PNR on 2% of seeds and the identity is restored. The wall is defined by the
training distribution, not by physics.

Marginal attribution (counterfactual decomposition of reach-PNR = reach-off × escalate):
improving only the escalation channel captures 91% of the MSE→MIP gain; only the escape
channel captures 30%. **But** a threshold-triggered rescue (hand control to MIP/DART only
when off-support) recovers only 70–80% (θ-sweep: benefit appears only as θ→0), so the
channels are not separable: where a policy lands off-support is endogenous to how it drove
in-support.

### 3.2 Failure is precision death at the boundary, not blow-up
Autopsy of all 24 MSE failures (n=80): 6 grasp failures; 18 follow one script — carry/align
while riding the boundary (mean d 1.82 vs 1.30 for successes; in-domain p95 = 1.79) → best
achievable alignment 9.7 mm vs 0.5 mm (successes) → repeated insertion attempts knock the
part → the distance blow-up is a *consequence*. Same-policy control: handed a clean
align_done state, MSE-2k inserts 100/100. The skill exists; the policy cannot *manufacture*
its own clean initial condition over 150 steps of closed loop (train-seed SR 70% ≈ held-out
69% — memorization does not help compounding).

### 3.3 Action error vs distance (the keystone)
Ground-truth = scripted recovery actions on ~41k held-out perturbed states. Systematic bias
‖E[â − a\*]‖ (executed action):
| d | MSE-2k | MSE-20k | MIP-2k | MIP-20k | DART-6k |
|---|---|---|---|---|---|
| [0,1) | 0.077 | 0.066 | 0.061 | 0.060 | 0.002 |
| [1,2) | 0.035 | 0.032 | 0.028 | 0.030 | 0.001 |
| [2,3) | 0.472 | **0.658** | 0.147 | 0.091 | 0.008 |
| [3,4) | 0.529 | **0.641** | 0.096 | 0.061 | 0.020 |
- MSE's off-support error is a *bias* (systematic), low-rank (top direction carries 81% of
  error variance, cos(bias, top-eigvec)=0.95): the network extrapolates along one mode.
- 10× clean data does not fix it (worsens it) while SR still rises 69→80 — scale buys fewer
  excursions, not better off-support actions.
- Precision exchange: within the align phase, when mean d crosses ≈1.4, MSE's alignment-error
  p90 crosses the 2 mm tolerance (0.9→1.5→10.7→18.8 mm); MIP and DART stay ≈1 mm at the same
  deviations. **Task tolerance converts bias(d) into an effective support boundary
  d_eff = bias⁻¹(τ_task); for mm-tasks d_eff ≈ 1.4 < geometric p95 1.79.**

## 4. Localization: training, not inference; not capacity; not convergence

| control | result |
|---|---|
| **MIP-step1** (inference byte-identical to MSE) | **94%** vs MSE 69% (held-out, n=100; second seed set: 93 vs 71) |
| MIP-full (2 steps) | 96% — the second step adds ≈2 pts |
| static bias: step1 already removes 70–80% of the off-support bias | Δ from step2 ≈ 0.02–0.03 |
| capacity: MSE with 3.5× (69M) and 10× (193M) params | ⏳ (trained, eval running) |
| convergence: MSE train loss 3.8e-6 at 300k, memorizes train set | not underfit |
| train-set fit: MIP fits *better* (RMS 8e-4 vs 1.8e-3 at 2k) | no classic fit/generalize tradeoff at 2k; at 20k MSE fits better and generalizes worse (MIP-20k ckpt undertrained: caveat) |

Conclusion: the entire effect lives in the weights; the only training difference is the
shared-weight denoising head.

## 5. Theory (corrected)

### 5.1 What the reconstruction-coefficient argument does and does not cover

[Retained from the note, with corrected scope.] For *field-parameterized* conditional
diffusion/flow (score s_θ, noise ε_θ, velocity v_θ), the clean map is reconstructed as an
affine function of the field, so with the field L-Lipschitz in the conditioning c, the
implied clean map is κ_t L-Lipschitz, with κ_t = σ_t²/α_t (score), σ_t/α_t (ε), |b_t/(a_t ḃ_t
− ȧ_t b_t)| (flow interpolants); κ=1−t for rectified flow, (2/π)cos(πt/2) for the
trigonometric path. All these derivations check out (we re-derived each).

**Scope correction (this is the crucial fix):** for x0-parameterization the coefficient is
κ ≡ 1 (the note's own §5.3): no schedule-induced shrinkage exists. Our MIP uses
x0-prediction in *both* heads and both sampler steps — no reconstruction division appears
anywhere at inference. Since MIP-step1 (x0, one step) carries the benefit, **the κ_t
mechanism cannot explain the observed effect.** The note's Theorem 1/2 remain true but apply
to a different family (ε/score/velocity-parameterized policies); they yield a testable
contrast (§6.4), not an explanation of our data.

Two further gaps of the κ account even where it applies: (i) the equal-Lipschitz-budget
premise is unjustified — to represent the same clean map the field must carry κ⁻¹-larger
c-sensitivity, so shrinkage is real only under an implicit budget cap (the note's own
Limitations 2–3); (ii) the theorem holds at fixed y_t; through a sampling chain y_t depends
on c, and the end-to-end Jacobian is not bounded by κ_t L.

### 5.2 What the denoising head provably does: ridge regularization in the action slot

Write g(a) := f_θ(τ, a, s), σ := 1 − τ (= 0.1). The head-2 loss at a data pair (s, a\*):

L₂(s) = E_ε ‖(g(a\* + σε) − a\*)/σ‖²,  ε ∼ N(0, I).

Taylor-expanding g around a\* (g(a\*+σε) = g(a\*) + σ J_a g·ε + O(σ²)) and using E[ε]=0,
E‖Jε‖² = ‖J‖_F²:

**L₂(s) = (1/σ²)‖g(a\*) − a\*‖² + ‖J_a g(a\*)‖_F² + O(σ).**

I.e., the denoising head is *exactly* (to second order): a fit term at the clean action with
weight 1/σ² = 100, **plus a unit-weight Frobenius penalty on the action-input Jacobian**
∂f/∂a at (τ, a\*, s). This is the conditional version of the classical result that
small-noise denoising ≈ contractive regularization [Bishop 1995; Vincent 2011; Alain &
Bengio 2014]. It also explains the raw loss magnitudes we observe (the logged MIP loss is
dominated by the 1/σ² = 100× term).

**What this does *not* yet give:** the deployed map is h(s) := f_θ(0, 0, s), and the
penalized Jacobian is ∂f/∂a at a different (t, a) slice — not ∂h/∂s. There is no
architecture-free derivation from one to the other. We therefore state the transmission as
an explicit hypothesis:

> **(T) Weight-sharing transmission.** The action-slot contraction penalty at (τ, a\*, s),
> imposed across the data distribution, biases the shared parameters toward functions whose
> obs-Jacobian ∂h/∂s is flat in the near-support region.

(T) is falsifiable and we test it three ways (§6). Empirically its signature is already
measured: ‖∂h/∂s‖ is flat for MIP-step1 (~3.3 across d) and spikes 2× for MSE exactly in the
band d ∈ [2,3) where its bias fires.

### 5.3 What smoothness alone cannot explain

Any account reducible to "flatter ∂h/∂s is better" must survive two of our results:
(i) blunt input-noise smoothing (σ_obs = 0.1/0.2, which both flattens *and convolves the
target*) collapses SR to 18%/3%; (ii) MIP's flatness costs no on-support precision (0.8 mm
alignment; train fit *better* than MSE). The viable form of the hypothesis is
*interpolating smoothness* — min-norm-like selection among solutions that still interpolate
the data — not target smoothing. The discriminating tests are §6.1–6.3.

## 6. Mechanism experiments

| # | intervention (all clean-2k, same net, same eval) | tests | result |
|---|---|---|---|
| 6.1 | **explicit ∂h/∂s penalty** (double-backprop Hutchinson, λ ∈ {1e-3, 1e-4}) | is c-smoothness *sufficient*? | ⏳ training |
| 6.2 | **small-σ obs noise** (σ=0.02, Tikhonov regime — fair version of input-noise) | same, via classical equivalence | ⏳ training |
| 6.3 | **no-weight-sharing** (separate denoiser net, composed with frozen MSE at inference) | is sharing the transmission channel (T)? | ⏳ training |
| 6.4 | **τ sweep** (0.99 / 0.9 / 0.5 ⇒ σ = 0.01/0.1/0.5) | dose-response of the §5.2 penalty | ⏳ training |
| 6.5 | **equal-weight heads** (drop 1/τ², 1/σ² scalings) | is the 100× weight essential? | ⏳ training |
| 6.6 | inference-time projection onto a 20k-action manifold | is a separable *output* prior enough? | **No: destroys everyone** (MSE 68→0–35; even MIP 92→20–38) |
| 6.7 | frozen unconditional manifold *in the loss* (co-adapted, grad through frozen denoiser) | separable prior, training-time | **No: 49%** (< MSE) |
| 6.8 | large obs-noise (σ=0.1/0.2) | blunt smoothing | **No: 18% / 3%** |

Interpretation matrix (pre-registered):
- 6.1 or 6.2 ≈ MIP ⇒ smoothness-in-c is sufficient (advisor's strong claim confirmed;
  mechanism separable).
- 6.1/6.2 fail but 6.3 shows composition ≪ shared ⇒ transmission (T) through shared weights
  is essential; the effective ingredient is the *denoising structure*, not generic smoothness.
- 6.4 monotone in σ ⇒ consistent with the §5.2 penalty scaling; non-monotone ⇒ fit-weight
  (1/σ²) interplay matters (check with 6.5).

## 7. The data-side counterpart, redundancy, and residual failures

- DART-style recovery collection *widens the support itself*: the band that is off-support
  for clean policies (d_clean ∈ [2,4)) is *inside* DART's support (d_dart median 1.4–1.55);
  its off-clean bias is ≈0 everywhere.
- 2×2 (loss × data): clean/MSE 69, clean/MIP 96, DART-2k/MSE 88–91, DART-2k/MIP 93–95:
  **redundant, not additive** — whichever axis solves the boundary first leaves nothing for
  the other.
- Residual failures are disjoint (MIP: {21008, 21029, 21036}; DART: {21010, 21050, 21067});
  neither is a "hard-seed" set — headroom for combination.
- Scale asymmetry: clean 2k→20k improves MSE only via fewer excursions (bias worsens);
  improves MIP little (already at ceiling). Data scale is the wrong axis for precision
  robustness; coverage and objective are the right ones.
- Data-efficiency: MIP-200 = 52% vs MSE-2k = 69% vs MSE-20k = 80%: the objective is worth
  roughly an order of magnitude of clean data. DART-200 ⏳.

## 8. Main SR table (held-out; two disjoint seed sets)

| data | N | MSE | MIP |
|---|---|---|---|
| clean | 200 | 19 / 18 | 52 / 55 |
| clean | 2000 | 69 / 71 | 96 / 93 (step1: 94 / 93) |
| clean | 20000 | 80 / 80 | 97 / 97 |
| DART | 200 | ⏳ | — |
| DART | 2000 | 88\* / 91 | 93 / 95 |
| DART | 4000 | 96.5\* / 96 | — |
| DART | 6000 | 97 / 98 | — |
| MSE 69M / 193M params (clean 2k) | | ⏳ / ⏳ | |

(cells: seeds 21000+ / seeds 22000+; \* = seed set [0,100), held-out for DART models.)

## 9. Limitations

Simulation only; scripted expert; state observations; one embodiment; single training seed
per cell (eval variance bounded at ±3 by disjoint seed sets; training-seed variance
unquantified — priority for camera-ready); one architecture; MIP-20k checkpoint undertrained
(133k); the κ-contrast (ε-pred vs x0-pred policy) not yet run; transmission hypothesis (T)
tested behaviorally, not explained at the function level.

## 10. Relation to the theory note (change log)

1. **Kept**: all §3–§7 derivations (verified: score/ε/FM coefficients, RF κ=1−t, trig
   κ=(2/π)cos(πt/2), VP threshold α>(√5−1)/2); the low-variance setup; the honest
   limitations.
2. **Scope-corrected**: κ_t lemmas apply only to field parameterizations; x0-param ⇒ κ=1 ⇒
   they cannot explain MIP/our data (the note's own §5.3, now made load-bearing).
3. **Replaced as the operative mechanism**: reconstruction arithmetic → training-time
   action-slot ridge penalty (§5.2, small-noise expansion — new derivation, checks out) +
   transmission hypothesis (T) with pre-registered ablations.
4. **Added constraints any mechanism must satisfy**: the 18%/3% obs-noise collapse, the
   projection destruction, the preserved on-support precision, and the fact that the benefit
   survives in a single x0 forward pass.
5. **Fixed**: eq. (29)–(30) broken \frac; "average κ" statements demoted to heuristics
   (deployment uses specific t).

## Figures (existing)
- fig1: 4-panel representative trajectories (escape / pull-back / own-support).
- fig2: bias(d) five-way curves.
- fig3: precision-vs-support (alignment p90 vs d, per model).
- fig4: specialist supports & handoff gap.
- fig5: Jacobian flatness (MSE vs MIP-step1).

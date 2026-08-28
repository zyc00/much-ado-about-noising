# The Extrapolation-Operator Finding: Full Experimental Protocol and Data

**Handoff document for external analysis (2026-07-01).** Self-contained: setup, math
definitions, every measured number, hypotheses, caveats, and open questions. Raw arrays in
`analysis/recovery/operator_fit_b24.npz` (fields DZ (N,53), GT/DM/DP (N,6), d (N),
G_gt/G_m/G_p (6,53), G_on).

---

## 1. Physical setup

- Task: robomimic **ToolHang**, init→insertion segment (grasp a frame, carry, align, insert
  into a stand; insertion tolerance ~1–2 mm).
- Control: **OSC_POSE delta** controller at ~20 Hz. Action a ∈ R⁷:
  a[0:3] = Δposition command (full scale 0.05 m per step, normalized to [−1,1]),
  a[3:6] = Δrotation command (axis–angle, full scale 0.5 rad), a[6] = gripper.
  All analyses below use the executed first action's first 6 dims, in normalized units.
- Observation s ∈ R⁵³: object-state (44) = frame/stand poses (+relative quantities),
  eef_pos (3, dims 44:47 of s), eef_quat (4, dims 47:51), gripper_qpos (2).
- **Expert / GT**: a deterministic scripted policy. On perturbed states it is a *reactive
  corrective controller* (waypoint re-planning + pose servoing toward the nominal
  trajectory). GT actions at off-support states come from DART-style collection: perturb the
  executed action during scripted rollouts, record the *scripted controller's corrective
  response* at the resulting states. These states/actions are held out (never trained on by
  the policies analyzed).

- Policies compared (both trained ONLY on 2,000 clean unperturbed demos, identical ChiUNet
  ~20M params, 300k steps, action chunks H=16):
  - **MSE**: â = f_θ(t=0, a_in=0, s), loss ‖â − a\*‖².
  - **MIP**: same network, two heads with shared weights:
    head1 f_θ(0, 0, s) → a\* (identical target to MSE) and
    head2 f_θ(0.9, a\* + 0.1·ε, s) → a\*, ε∼N(0,I) (denoising head; loss weights 1/0.9²,
    1/0.1² respectively). **Deployed here as “MIP-step1”: only head1, a forward pass
    byte-identical to MSE.** (Closed-loop SR held-out: MSE 69%, MIP-step1 94%.)

## 2. Support distance and the analysis band

Support cloud = all states of 40 clean training demos (~35k states). Distance
d(s) = min_i ‖(s − s_i)/σ‖₂ with per-dimension z-scoring (σ from the cloud).
Calibrated landmarks: cloud self-NN p95 ≈ 1; in-domain p95 (held-out same-distribution
trajectories) = 1.79; **point-of-no-return (PNR) = 4.0** (empirical: P(episode fails |
trajectory ever exceeds 4) = 100% for clean-trained policies; SR ≈ 100 − P(cross PNR)
holds pointwise across models).

The analysis band is d ∈ [2,4): off-support but recoverable; this is where closed-loop
success is decided and where MSE's action bias fires (bias ‖E[â − a\*]‖: MSE 0.47–0.53,
MIP 0.10–0.15, in this band; both ≤0.08 in-support).

## 3. Pair construction (the core protocol)

For every held-out perturbed state s (dart_test_huge_full2ins.hdf5, 460 episodes, stride 4)
with d(s) in the band:
1. Find its clean 1-NN s₀ (demo i, time t₀) in the support cloud.
2. **GT change**: δa\* = a\*(s) − a\*(s₀), where a\*(s) is the recorded scripted corrective
   action at s and a\*(s₀) the clean demo action at (i, t₀).
3. **Policy drift**: δâ_M = â_MSE(s) − â_MSE(s₀); δâ_P = â_MIP1(s) − â_MIP1(s₀)
   (each policy evaluated with its own 2-frame observation window; neighbor windows use the
   clean demo's own previous frame).
4. **State deviation**: δz = (s − s₀)/σ ∈ R⁵³ (z-scored).
N = 2,214 pairs in [2,4) (1,662 in [2,3), 552 in [3,4)).

Prior per-pair facts (same pipeline): ‖δâ_M‖ ≈ ‖δâ_P‖ (0.81 vs 0.73 at [2,3)) — the models
drift *equally far*; cos(δâ, δa\*) = 0.86/0.81 (MIP) vs 0.66/0.59 (MSE) at [2,3)/[3,4) —
MIP drifts in the right direction. At d ∈ [4,6) both collapse to cos ≈ 0.48 (matches PNR).

## 4. Operator fits (ridge regression)

For each target Y ∈ {δa\* (GT), δâ_M (MSE), δâ_P (MIP)} fit a linear operator
G = argmin Σ‖δa − G δz‖² + λ N ‖G‖_F², λ = 10⁻² (results stable for λ ∈ [10⁻³, 10⁻¹]).
Additionally, a **tangent (“script-advance”) operator** G_on is fitted from consecutive clean
frames: pairs (z_{t+1} − z_t → a\*_{t+1} − a\*_t) over the clean demos.

### 4.1 Headline results (band [2,4), N=2,214)

| quantity | GT | MIP | MSE |
|---|---|---|---|
| linearity, 5-fold CV R² | 0.592 | 0.688 | 0.587 |
| eff. rank (90% energy) | 3 | 3 | 2 |
| top-5 singular values | 1.02, .55, .43, .24, .11 | 0.94, .53, .35, .21, .08 | **1.32, .86**, .36, .30, .07 |
| eef-pos block sym-eigs | −0.202, −0.075, −0.022 (**neg-def**) | −0.220, −0.106, −0.101 (**neg-def**) | −0.188, −0.140, **+0.021** (**indefinite**) |
| rot-block (a_rot ← quat dev) Frobenius | 0.102 | 0.118 | **0.478** (top sv 0.478, next 0.017: rank-1 amplifier, ~5× GT) |
| mean per-pair pred cosine vs G_GT | — | **0.73** [0.69, 0.78]₉₀ | 0.40 [0.37, 0.44]₉₀ |
| cos vs tangent G_ON | — | 0.49 | 0.29 |
| cos((G_MSE − G_GT)δz, G_ON δz) | | | −0.14 (≈0: MSE's error is NOT the tangent rule) |

eef-pos 3×3 blocks (input de-z-scored; action in normalized units):
```
G_GT  pos-block            G_MIP pos-block            G_MSE pos-block
[-0.080  0.042  0.044]     [-0.151  0.019  0.071]     [-0.161  0.029  0.083]
[-0.145 -0.172  0.023]     [-0.119 -0.152  0.041]     [ 0.029 -0.050  0.201]
[ 0.029  0.014 -0.047]     [-0.129 -0.101 -0.124]     [ 0.014 -0.050 -0.098]
```

### 4.2 Robustness checks
- Split-half operator consistency (cos of predictions of two independently fitted halves):
  GT 0.92, MIP 0.91, MSE 0.94 → all three are stable laws, not noise.
- Cross-band generalization: fit on [2,3), test on [3,4): R² GT 0.55→0.57, MIP 0.65→0.72,
  MSE 0.56→0.60 → single global laws across the band (MSE's law is *consistently wrong*).
- Bootstrap 90% CIs above; non-overlapping for MIP-vs-GT vs MSE-vs-GT.
- Band profile (all bands measured; N = 12,291 / 2,214 / 373):
  | band | GT R² | cos(MIP,GT) | cos(MSE,GT) | MIP pos-block | MSE pos-block |
  |---|---|---|---|---|---|
  | [1,2) in-domain | 0.12 (|δa\*| near noise floor) | 0.92 | **0.84** | neg-def | **neg-def** |
  | [2,4) recoverable | 0.59 | **0.73** | 0.40 | neg-def | **indefinite (+0.021)** |
  | [4,6) beyond PNR | 0.63 (GT still corrective) | 0.41 | 0.34 | neg-def (weak) | indefinite (+0.003) |
  Reading: in-domain, ALL models carry the correct feedback structure (MSE included);
  MSE's operator becomes indefinite exactly upon crossing the support boundary; beyond the
  PNR even MIP's operator decouples from the GT law (0.73→0.41) although the GT law itself
  is still there (R² 0.63) — the learned negative feedback fades with distance, and the PNR
  coincides with where it no longer matches. MIP's drift remains highly lawful everywhere
  (R² 0.69/0.76) — it follows *a* consistent law; the law is only correct within the band.
- GT residual: median ‖δa\* − G_GT δz‖/‖δa\*‖ = 0.50 → linear law is the dominant but not
  complete structure (R²≈0.6; remainder = saturation / phase dependence / rotation geometry).

### 4.3 The instability link
MSE's positive-feedback eigendirection (eigvec of the +0.021 eigenvalue of its symmetrized
pos-block): (x,y,z) = (0.28, **0.75**, 0.60). The *observed* closed-loop escape/bias
direction (position part of the mean off-support action error, measured independently):
(−0.06, **0.995**, −0.08). |cos| = **0.68**. I.e. the direction in which MSE's fitted
extrapolation operator amplifies deviations is substantially the direction along which its
rollouts actually escape (the long-known "+Δy + rotation" rank-1 bias; the 5× rotation-gain
block is the rotational part of the same pathology).

### 4.4 Is the corrective law contained in the clean dataset itself? — NO (key negative result)

The only place the clean dataset could encode an extrapolation law is its *within-support
conditional variation*: cross-demo, same-phase state/action differences (2,449 pairs from
200 demos, nearest-neighbor across demos, median ‖δz‖ = 0.81, median ‖δa‖ = 0.029).
Fitting G_within on these pairs:
- **in-domain 5-fold R² = 0.045** — the clean data's own cross-trajectory variation contains
  essentially no predictable linear action-response at all;
- extrapolated to the off-support pairs: **R² = −0.32** (worse than predicting the mean),
  prediction-cos vs G_GT = 0.23; gains 10–20× smaller than G_GT
  (pos-block diag −0.004/−0.004/−0.011 vs G_GT −0.08/−0.17/−0.05).
Mechanistic reason: cross-demo state differences are "both trajectories are on their own
plan" differences — no correction is required, so δa ≈ 0; the servo gain K acts only on
*deviation-from-plan* directions, which have (by construction) zero variance in clean data.
**The corrective law is first-order unidentifiable from the clean dataset: every function
agreeing with the data on the manifold — including MSE's indefinite one — is equally
data-consistent.** Consequently MIP's cos-0.73 match to G_GT cannot be data-recovered; it
must be objective-injected. (For contrast, the temporal law G_on has cos 0.51 with G_GT —
directionally closer than G_within but with wrong gain, off-support R² = −0.10.)

### 4.7 Anchor-density sweep and the exact form of the GT law

Hypothesis tested: the R² ≈ 0.6 ceiling of the linear servo fit is anchor sparsity
(1-NN phase mismatch contaminating δz with tangential components). **Refuted**: growing the
support cloud 40 → 20,000 demos (mean anchor distance 2.63 → 1.61) leaves R² flat
(0.549 → 0.572); tangential contamination is small anyway (|δz_t|/|δz| = 0.14) and the
normal-restricted law is no stricter. The ceiling is intrinsic. Its decomposition:
- adding the VELOCITY deviation (2-frame input; the PD controller's D-term): R² 0.57 → 0.65;
- **49% of recovery actions saturate the action clip (|a|≥1 in ≥1 dim)**; excluding them:
  R² → 0.70 (pos+vel, unsaturated);
- residual ~0.3: waypoint switching / replanning discreteness + rotation geometry.
With D-term and saturation handled, the pos-block is negative-definite up to one ≈0
eigenvalue (−0.032, −0.015, +0.002 — a near-null direction, not positive feedback).
**Correct model of the GT: δa\* = clip(−K_p δx − K_d δv + waypoint terms) — a clipped
piecewise-PD servo; the linear negative-(semi)definite law measured throughout is its
correct first-order form, and the fit residual is structured (saturation, switching), not
noise.**

### 4.8 Which ingredient of the denoising objective creates the operator? (variant-training probes)

Each row: a model trained from scratch on clean-2k with one ingredient of the MIP/denoising
recipe altered; its off-support operator fitted by the standard probe (band [2,4),
N=8,564 pairs, test = held-out puredart states); SR = held-out seeds 21000–21100.

| variant | what changes | cos vs G_GT | negdef | SR |
|---|---|---|---|---|
| MIP (reference) | τ=0.9, weight 100× | 0.73 | ✓ | 94–99 (s0/s1/s2) |
| standalone denoiser | no head-1, no sharing | 0.70 | ✓ | 98 (composed w/ frozen MSE) |
| ZEROIN | action input ≡ 0 (no noisy action ever seen) | 0.65 | ✓ | 75 (deployed alone) |
| ZEROIN-FLAT | zeros input AND weight 1× | 0.46 | ✗ (+0.022, sv 1.5) | — |
| DUP100 | same t=0 view duplicated ×101 (weight WITHOUT aux view) | 0.67 | ✗ (+0.010) | **54** |
| RANDIN | action input ≡ N(0,I) (input carries no action info) | 0.69 | ✗ (+0.013) | 78 (deployed alone) |
| SCRAMBLE | action input in fixed orthogonally-scrambled coords | 0.69 | ✓ | — |
| EQW | two heads, weight 1× (τ-weighting removed) | 0.42 | ✗ (+0.015) | 83 |
| TAU50 | τ=0.5 → weight 4× | 0.68 | ✗ (+0.011) | 90 |
| TAU99 | τ=0.99 → weight 10⁴× | 0.84 | ✗ (+0.018) | 95 |
| FLOW | velocity-field parameterization | — | — | 93 |
| MSE (reference) | single head | ~0.5, indefinite (+0.021) | ✗ | 71–79 (s0/s1/s2) |

Readings:
1. **The noisy action input is NOT the load-bearing ingredient**: ZEROIN/RANDIN/SCRAMBLE
   (which respectively remove, randomize, or scramble the action slot) all still produce a
   corrective, ≈negative-definite operator (cos 0.65–0.69). The corrective extension is
   carried by the **s-pathway of the denoising view**, not by "seeing perturbed actions".
2. **The τ-implied loss weight IS load-bearing and dose-monotone**: weight 1× → cos 0.42,
   4× → 0.68, 100× → 0.73, 10⁴× → 0.84; SR moves 83 → 90 → 94–99 → 95 in step. Adam absorbs
   *global* loss scale, so this is a *relative* two-view weighting effect, not a lr effect.
   **DUP100 control (the weight-vs-view separation): duplicating the SAME t=0 view ×101 —
   identical total loss scale to MIP but no auxiliary slice — gives an indefinite operator
   (+0.010) and SR 54, WORSE than plain MSE (71–79).** High weight alone is refuted as the
   cause; the 2×2 factorial is now complete:
   {t=0, 1×} = MSE → indefinite, 71–79; {t=0, 101×} = DUP100 → indefinite, 54;
   {τ, 1×} = ZEROIN-FLAT → indefinite (0.46); {τ, 100×} = ZEROIN → **neg-def 0.65**.
   Only the (τ-slice × high-weight) cell produces the corrective operator: the weight works
   *through* the auxiliary view, and the τ-slice representation regime (the t-embedding,
   since ZEROIN's action input is zeros — the SAME input as the t=0 view) is what the weight
   amplifies. (Caveat: with grad-norm clipping active, DUP100 is not a perfectly clean
   Adam-absorption control, but its SR ≤ MSE settles the direction.)
3. Generic-smoothness alternatives do NOT reproduce it at ANY tested dose: obs-noise σ=0.02
   → SR 68 (worse than MSE); explicit input-Jacobian penalty λ=1e-3 → SR 1, λ=1e-4 → SR 24
   (vs MSE 71–79; monotonically harmful). Sharpest data point: jacreg-1e-4's operator has
   cos 0.92 with G_GT yet is **still indefinite** (+0.023, MSE's escape signature) and SR
   collapses — **directional agreement without negative-definiteness does not stabilize the
   loop; the SR-relevant property of the operator is its (semi)definiteness (stability), not
   its cosine**. This closes the Jacobian-shrinkage / generic-smoothing account.

### 4.9 The two-view kernel test: the operator lives in the FEATURE GEOMETRY

Hypothesis under test (advisor round-4): MIP ≈ weighted two-view kernel regression — the
denoising view reshapes the network's effective kernel/features φ(s) so that even a *linear*
readout extrapolates correctively; and because weights are shared, the main (t=0) view
inherits the reshaped features.

Probe: freeze the trained network, hook the penultimate features (input of the last 1×1 conv
of final_conv, 128×16 = 2048-d) at a chosen view x = (t, a-input, s); fit a ridge readout
(λ=1e-3) from φ to the executed 6-d action on clean states only; then run the standard pairs
protocol through this frozen-feature linear readout.

| features | operator R² | cos vs G_GT | negdef |
|---|---|---|---|
| denoiser φ₁ (τ-view, tube input) | 0.905 | **0.67** | ✓ |
| ZEROIN φ₁ | 0.649 | 0.66 | ≈ (+0.001) |
| SCRAMBLE φ₁ | 0.905 | **0.68** | ✓ |
| MSE φ₀ (t=0 view) | 0.654 | 0.56 | ✓ (weak, −0.011) |
| **MIP φ₀ (t=0 view)** | 0.697 | **0.74** | ≈ (0.0) |
| MIP φ₁ (τ-view) | 0.852 | **0.93** | ✗ (+0.021 marginal) |

(MSE/MIP rows use independently trained seed-1 checkpoints; full-model reference operators:
G_MIP cos 0.73, G_MSE indefinite. Tube-input rows are the chunk-aligned rerun; with an
intercept the on-support readout train R² is 1.000 — the penultimate features linearly
contain the executable chunk exactly, as they must since the network's own head is a
1×1 conv.)

Findings:
1. **A frozen-feature linear readout reproduces the full model's corrective operator**
   (0.70 vs 0.70 for the denoiser; 0.74 vs 0.73 for MIP). The servo law is encoded in the
   feature geometry φ(s), not in the nonlinear head — the "reshaped kernel" claim holds.
2. **Weight sharing transports it into the main slice**: MIP's t=0 features decode the law at
   cos 0.74 (≈ its φ₁ and ≈ the deployed step-1 operator), while MSE's t=0 features only
   reach 0.56 with visibly weaker gains. Auxiliary-view gradients measurably reshaped the
   main view's representation. This is the mechanism by which **step-1-only inference
   (byte-identical to MSE inference) inherits the corrective extension**.
3. Coordinate-independence at the feature level: SCRAMBLE φ₁ decodes at 0.75 — the feature
   reshaping does not rely on the action slot's physical coordinates (consistent with 4.8.1).
4. Two views are BOTH needed for SR: deploying the denoise slice alone closed-loop
   (f(τ, 0, s) of ZEROIN / f(τ, ε, s) of RANDIN) gives 75/78 — MSE-level, far below MIP's
   94–99. The aux view supplies the corrective extension, the main view supplies on-support
   precision; the deployed policy needs the shared-feature combination of both.
5. **HEAD-SURGERY closed-loop test (the features-vs-head verdict)**: replace the trained
   final head with a clean-data ridge readout on the frozen penultimate features (same
   linear function class as the original 1×1-conv head, fit on 200 clean demos, on-support
   train R² = 1.000 for both models) and deploy closed-loop, held-out seeds 21000–21100:
   - MSE features + ridge head: **74/100** (own head: 71) — no rescue. Although a ridge
     readout of MSE's features is weakly corrective and negative-definite (cos 0.56, gains
     −0.011 ≈ 10× smaller than G_GT's −0.08/−0.17/−0.05), that gain is far too small to pull
     the loop back. **MSE's escape is a FEATURE deficiency, not a head pathology.**
   - MIP features + ridge head: **95/100** (own head: 99, step1 94–99) — the entire MIP
     advantage survives replacing the head with a linear probe fit by ridge on clean data.
   Deployed function class and fitting data are now IDENTICAL between the two rows; the only
   difference is the frozen features. **The features are a sufficient statistic for the SR
   gap** — the strongest single confirmation of the feature-geometry (two-view kernel)
   account.
   - Combined with 4.8: correct *direction* with ~zero gain (MSE-ridge, 74) fails to rescue,
     and high gain with wrong *sign structure* (ZEROIN-FLAT, EQW) fails too; closed-loop
     stability needs GT-scale gain AND negative-definiteness — which is what the τ-weighted
     denoising view imprints into φ(s).

5b. **Layerwise probe (where does the reshaping live?)**: same ridge-probe per layer
   (enc = obs-encoder output → cond → down0-2 → mid → up0-1 → pen), cos vs G_GT:
   MSE-t0: 0.65/0.65/0.45/0.49/0.50/0.49/0.51/0.60/0.59;
   MIP-t0: 0.72/0.69/0.53/0.41/**0.78/0.77/0.78**/0.72/0.73 (MIP-τ identical per layer);
   standalone denoiser-τ: **0.78**/0.77/0.67/0.56/0.55/0.51/0.54/0.58/0.69.
   Readings: (i) divergence starts at the OBS ENCODER itself (0.65 vs 0.72/0.78) — MIP
   reshapes the state encoder, not only late action features; (ii) the largest gap opens in
   the deep trunk (down2/mid/up0: 0.50 vs 0.78) — MSE's trunk *degrades* the corrective
   content its own encoder carries (0.65 → 0.50), MIP's trunk enhances it; (iii) MIP's φ₀
   and φ₁ are identical at every layer — the weight-shared transport of the aux-view
   reshaping into the main slice is complete, layer by layer.
6. **Ridge-head λ sweep (is MSE's ceiling a head-hyperparameter artifact? — NO)**: sweeping
   the readout ridge λ over 10⁻⁸…10² for both feature sets (all with intercept; pairs
   protocol; bias = mean |pred − GT recovery action|, position+rot 6-d):
   - MSE features, λ ≤ 10⁻²: cos plateaus at 0.59–0.61, rot-block top-sv stays 0.13–0.14
     (the rotation amplifier is IN the features), bias[2,3) ≈ 0.36–0.37;
   - MIP features, λ ≤ 10⁻²: cos 0.73–0.74, rot-sv 0.086–0.089, bias[2,3) ≈ 0.24;
   - increasing λ to 1 suppresses MSE's rot-sv to 0.084 but costs on-support accuracy
     (bias[1,2) 0.09 → 0.15) and cos stays 0.60; λ = 10² destroys the readout for both
     (trainR² 0.65, bias 0.58).
   **No point on MSE's λ frontier reaches MIP's operator quality — the gap is not a
   head-regularization hyperparameter.** Closed-loop SR across the full sweep (held-out
   seeds, with support-distance tracking): MSE 53/75/**77**/46/1/0 vs MIP
   83/88/**94**/67/6/0 at λ = 1e-8/1e-6/1e-4/1e-2/1/1e2 — MIP dominates at every λ; MSE's
   best (77) « MIP's best (94). And in EVERY run with any non-crossing episodes,
   SR|never-crossed-PNR4 = 100%: all failures are support-escape failures; the SR gap
   reduces entirely to the crossing-rate gap set by the features' extrapolation operator.

7. **Bad-mode projection (sharper than cos)**: define MSE's two pathological modes from its
   fitted operator — the positive-feedback pose eigendirection (bad1) and the rank-1
   rotation-amplifier input direction. Signed gain along bad1 / rot response along the
   amplifier direction (one consistent de-z-scored convention):
   | model | gain@bad1 | rot@MSE-amp | own rot top-sv |
   |---|---|---|---|
   | G_GT | −0.010 | 0.013 | 0.063 |
   | MSE | **+0.003 (amplifying)** | **0.152 (= its own top mode)** | 0.152 |
   | MIP | **−0.020 (damps it, harder than GT)** | 0.036 | 0.075 |
   | standalone denoiser | −0.010 | **0.001** | 0.012 |
   | SCRAMBLE | −0.012 | 0.001 | 0.012 |
   | ZEROIN | −0.008 | 0.021 | 0.053 |
   | ZEROFLAT | **+0.014 (worse than MSE)** | 0.115 | 0.134 |
   | EQW | +0.008 | 0.099 | 0.122 |
   | DUP100 | +0.000 (flat) | 0.027 | 0.077 |
   | MSE-ridge λ=1e-4…1e-2 | −0.006…−0.008 | 0.050 | **0.140** |
   | MIP-ridge λ=1e-4…1e-2 | −0.003…−0.005 | 0.035 | 0.089 |
   MIP does not merely avoid MSE's bad modes — it actively contracts along MSE's
   positive-feedback direction and keeps its rotation spectrum at GT scale. The corrective
   family (denoiser/SCRAMBLE/ZEROIN/MIP) uniformly suppresses both bad modes; the failure
   family (ZEROFLAT/EQW) *recreates* them. **Head-vs-feature attribution of MSE's two bad
   modes**: the ridge readout of MSE's features KILLS the pose positive-feedback (bad1
   +0.003 → −0.006: head-attributable) but RETAINS the rotation-amplifier magnitude
   (rot top-sv 0.140 ≈ full model's 0.152: feature-locked, its input direction merely
   rotates) alongside the 10×-too-small servo gains. Closed loop, fixing the
   head-attributable part alone moves SR only 71 → 74–77 — the feature-locked deficits
   (weak gain + rotation amplifier) dominate the failure. (Caveat: abs-representation
   models sit outside this geometry — ABSMSE/ABSMIP both show all-positive projections in
   this delta-native convention; their comparison lives in §absrep, not here.)

## 5. Interpretation (current best account)

- **Data-side inductive bias, named**: the GT extension of the clean data into the band is,
  to R²≈0.6, a **rank-3 linear negative-feedback (proportional servo) law**
  δa\* ≈ −K δs with K's pose block negative-definite and modest gain. This is not an
  assumption — it is what the scripted corrective controller *is*, measured.
- **MIP learned approximately that operator** (same rank, same sign structure, per-entry
  similar gains, pred-cos 0.73), from clean data alone.
- **MSE learned a different operator**: rank-2, higher gain, an indefinite pose block (one
  positive-feedback direction) and a 5× rank-1 rotation amplifier. Closed loop, iterating an
  operator with a positive-feedback eigendirection amplifies deviation → escape → PNR. The
  escape phenomenology is a linear-instability statement.
- **Training-side origin — now RESOLVED (the decisive experiments)**: MIP's denoising head
  is trained as (a\* + ε, s) ↦ a\*, i.e. output = input − perturbation: an explicit
  **negative-feedback corrector in the action slot**, conditional on the observation.
  We trained this head as a **separate network** (same architecture, denoising loss only,
  clean-2k, no regression head, no weight sharing) and measured:
  1. **Its obs-slot response IS the corrective law**: response to state deviations (with the
     action input held at the anchor's clean chunk) is extremely lawful (R² = 0.93),
     directionally corrective (pred-cos vs G_GT = 0.64), pos-block negative-definite.
  2. **Static composition**: frozen MSE proposes, standalone denoiser refines:
     off-support |bias| 0.495 → **0.141** (shared-weight MIP-step1: 0.158). Matches sharing.
  3. **Closed-loop composition**: the frozen two-network pipeline achieves
     **98/100 held-out SR** (MSE alone 69; shared MIP-step1 94; shared MIP-full 96).
  **Weight sharing / co-training is NOT necessary.** The corrective law is created entirely
  by the conditional action-denoising objective and can be applied as a modular, bolt-on
  refiner at inference. (Earlier failures of bolt-on priors — inference projection onto an
  *unconditional* 20k-action manifold destroyed all policies; frozen unconditional manifold
  in-the-loss got 49% — are explained by *conditionality*: the corrective target must be
  a\*(s), not "any valid action".) The denoiser's action-slot behavior is full-removal with
  an error floor (residual/perturbation 0.74/0.40/0.19 at σ = 0.1/0.2/0.5).
  Dose–response checks now DONE — see §4.8 (τ sweep: weight-monotone; jacreg/obs-noise: do
  not reproduce; eqw: collapses toward MSE; flow: SR 93) and §4.9 (the operator is linearly
  decodable from frozen features, and weight sharing transports it into the t=0 slice —
  the two-view kernel account).
- Also relevant: with 3.5×/10× parameters, MSE's held-out SR *drops* 69→49→46 (capacity
  amplifies the unconstrained extrapolation); MIP-step1 = 94% at 20M.

## 6. Known caveats

1. Everything in normalized units; the pos-block "gain" mixes z-scored obs input and
   normalized action output (de-z-scored on input only). A fully physical-unit version
   (m → m) is a straightforward rescale by the action scale (0.05 m).
2. Rotation treated via raw quaternion deviations (4-dim) → linear block; axis–angle
   geometry not respected (small-angle regime makes this approximately linear, but the
   rot-block numbers should be read as magnitudes, not exact gains).
3. δz mixes eef and object dims; during carry the grasped frame moves with the eef, so
   deviations co-occur across blocks (an eef-vs-object attribution split is degenerate on
   this data: nearly all deviation energy sits in object dims that mirror the eef).
4. GT corrective actions come from ONE scripted recovery controller; K is that controller's
   gain. A different expert would define a different (but presumably still negative-feedback)
   law.
5. The 1-NN neighbor defines the "anchor" for δ quantities; results could depend on the
   anchor choice (k-NN averaging not tested).
6. Fitted on states visited by *perturbed scripted* rollouts, which may differ from states
   visited by the *policies'* own drift (though the band overlaps by construction of the
   distance).

## 7. Questions we want analyzed (for the theory write-up)

1. Correct formalization: given data on a manifold M = {(s, a\*(s))} and a GT extension that
   is locally a\*(s₀ + δ) = a\*(s₀) − Kδ (proportional servo), under what conditions does
   training f(t, a, s) with the two-head MIP objective yield ∂f/∂s|_{(0,0,s)} ≈ −K off the
   manifold? What is the role of (i) weight sharing, (ii) the action–state coupling induced
   by delta-action control (executing δa moves the pose by ≈ δa, so the corrupted-action
   distribution mimics the perturbed-state distribution one step later)?
2. Closed-loop stability: with s_{t+1} ≈ s_t + B·â(s_t) (delta control), derive the
   stability condition on the learned extrapolation operator G (pose block) and check that
   GT/MIP satisfy it and MSE's indefinite block violates it; connect the unstable
   eigendirection to the observed escape statistics quantitatively.
3. Is the negative-feedback structure of the *denoising field itself* (∂f/∂a ≈ contraction
   toward a\*(s)) sufficient, when composed with the dynamics, to imply corrective behavior
   — i.e., is inference-time composition (separately trained denoiser ∘ MSE) enough, or is
   co-training necessary? (Empirical answer pending; theory welcome.)
4. What implicit bias of the architecture/optimizer picks MSE's particular wrong operator
   (rank-2, indefinite, 5× rotation gain)? It is NOT the tangent/script-advance rule
   (measured ≈ orthogonal).

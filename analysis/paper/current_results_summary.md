# Consolidated results — scripted mechanism, chunking, multimodality, cross-domain (2026-07-17)

Status tags: [SOLID] = seed/eval-replicated or intervention-grade; [DIR] = directional, single-seed;
[OPEN] = live question; [RETRACTED] = withdrawn with cause. Full provenance: positive_validation_report.md PARTs CCXXXIII–CCLXI.

## 1. Scripted (f2i) — the mechanism

- [SOLID] **Execution dose**: same L2 ckpt, AS 8/4/2/1 = 75/83/94/99 (s1000; eval-replicated 98.0 pooled).
  Direction replicates on 3/3 training seeds (66→75, 70→92); magnitude is seed-dependent.
  All five losses converge at AS=1: L2 99 / HT 100 / HG 100 / MIP 98 / prog 100.
  l2cd (0/100) NOT rescued by AS=1 (4/100, SR|stay 60) — AS=1 is a diagnostic: separates
  execution-tolerance failures from fit-level failures.
- [SOLID] **Selector / variance collapse** (the headline): 3-seed families —
  L2 AS=8 {75,66,70} mean 70.3 | AS=1 {98,75,92} spread 23.
  HG AS=8 {96,97,88} mean 93.7 | AS=1 {100,95,95} spread 5. Families DISJOINT at AS=8.
  Same signature as the human-data minimum-selection account → one cross-domain mechanism.
- [SOLID] **Per-cycle mechanics**: in-tube creep equal (+.08); annulus [2,4): L2 corrective-head/
  expansive-tail (−.17/−.17/+.08) vs HG monotone contraction (−.28/−.35/−.42); past d≈4 both
  explode (L2 5× more often, 80× harder; readouts = saturated post/stroke).
  Asymptote framing: L2 tail +0.54(20k)→+0.08(300k) monotone, plateaus; HG at −0.44 by 60k.
  No crossover, no erosion — objectives converge to different fixed points.
- [SOLID] **Onset/amplification geography**: onset in the APPROACH (88% of L2 failed episodes;
  shared with HG), amplification at stroke/insert; settle/shoulder ~never the onset.
- [SOLID] **Fold universality**: off-tube settle purity 0–7% for ALL arms (incl. MIP 0%, prog 2%);
  nobody repairs the far extension; SR is monotone in escape rate (cross4 .29/.19/.11/.07/.06 →
  75/86/94/95/96). Fold = failure site, not cause.
- [SOLID] **σ mechanism = formation-window repricing**: @20k σ spread 30×, ρ(σ,|r|)=.93;
  near-degenerate decile amplified 14× relative to L2 (w 2.73 vs 0.19); outlier pocket suppressed
  30× (0.11 vs 3.57); σ collapses to floor by 300k. Human = top of the same 1/σ² lever
  (suppress noisy tail), scripted = bottom (amplify small distinctions).
- [SOLID] **Collision regime (H≤2)**: data dose-response 15.9/6.7/0.04/0% (H=1/2/4/8);
  H=2 on-support fold ON the predicted colliding pairs (SHO→PST 89%, SET→APP 79%; healthy H=10
  control) — label-funded-metric law by intervention; loss ladder at H=2/AS=1:
  L2 71.3 / MIP 81.0 / HG 90.7 (pooled 300 eps) — ordered by repricing strength.
- [DIR] **Horizon × execution matrix** (chiunet): AS=1: H4 79 / H8 {70.7, 95 — seed crossover} /
  H10 {98,75,92} / H16 96 / H32 26. AS=8: H8 54 / H10 70–75 / H16 75 / H32 60.
  Plateau [H10,H16]; below-plateau BEHAVIORAL gap RETRACTED (lottery); H32: execution-dose
  REVERSES (crutch regime); H32 on-support per-step fit at floor → its head failure is
  closed-loop only.
- [DIR] **headonly crucis**: {62, 77} vs full-supervision {75,92,98}; SR overlaps, but containment
  DISJOINT across seeds (cross4 {.42,.22} vs {.03,.14,.08}) — supervising never-executed tail
  steps improves the head's annulus containment (anti-execution-only-account evidence).
- [SOLID-NULL] **Junction geometry does NOT track the head lottery**: h8-s5 (95-scoring seed)
  shows the SAME shoulder→stroke absorption as the 70.7 seed (SHO→MID 78% vs 79%) and an even
  earlier stroke basin (0.46 already at α=0.3) — below-plateau geometry is horizon-typical, not
  quality-tracking. The lottery's representation-level carrier remains UNIDENTIFIED (consistent
  with the arc's repeated lesson: encoder-geometry statistics do not predict SR; head quality is
  a function/readout property).
- [OPEN] **sigaux** (σ-head aux-shaping vs transient-weighting path) — training, 3rd attempt.
- Nulls/refuted on scripted: within-chunk crowding (matched-quiet ratio 0.98); random-direction
  tail-robustness weak (1.3×); σ-landscape at convergence; late-training erosion.
  One clean pointwise fact: L2 on-support floor 2× higher (0.0016 vs 0.0008).

## 2. Chunking (both domains) — why we need it, decomposed

- [SOLID] **Targets = learning-side disambiguation** (duality with obs history):
  scripted collisions →0 by H=4; human conditional-mean collisions 17%→0.7% only by H=32
  (pacing stretches the dose-response). Raw human labels never collide (tremor separates).
- [SOLID] **Execution = deployment-side knob with head-quality-dependent sign**: cost in-plateau
  for conv heads (75→99 by removing it), crutch for MLP (71 vs 67) and long-H (60 vs 26).
  Simchowitz (2503.09722) adjudication: compounding is real but conditional on non-contractive
  heads; the deciding variable is random under L2, controlled under denoising.
- [SOLID] **Human H=2 cell**: 32 vs 81 (official); hetero-t at H=2 recovers 70 → the −49 collapse
  is ~78% gradient-noise channel (chunk targets average tremor in the gradient) + ~11 conditioning
  residue. Chunking is a partial LOSS-SUBSTITUTE on noisy data; chunk-specific irreducible residue
  ~10 pts on both domains.

## 3. Multimodality — level-attribution (one instrument, three datasets)

- [SOLID] Three phenomena, three cures: TREMOR (single-step noise; matched likelihood),
  POCKET BRANCH (action-level, TH-specific, 34–43% branchy states; EMERGES AT CHUNK SCALE:
  0.42 vs 0.08 tremor floor at H=8; orientation-keyed ⇒ feature resolution; history-immune —
  motion signal at the decision hover is at tremor SNR≈1), PACING (chunk-level, CV .2–.4,
  H- and history-invariant everywhere; quantile-estimator territory; fitted-speed shrinkage
  0.66–0.72 immune to all conditioning = density shrinkage).
- [SOLID] TH tail = flip-TIMING bimodality (grip 85%, 6.1× flip-adjacent, kurt 23.5) ⇒
  functionally noise ⇒ suppression (+30) > feature-supply (+13) > history (−15).
- [SOLID] Transport tail = grip-value disagreement (69–76%, both arms, kurt 4–5), NOT
  flip-adjacent, NOT motion-resolvable [verified ph] — trait-like, nature [OPEN].
- [SOLID] Transport direction aliasing = motion-state aliasing, velocity-resolvable (−35–50%);
  history's factorial main effect +9–17 (trmha 2/11/29/46, ~additive with ht).
  σ-mediation of the ht×history combination REFUTED (spearman(σ, alias) .02–.05 at all stages).
- History law: pays iff ambiguity is motion-keyed AND motion estimate beats the noise floor at the
  decisive states AND deployment stays on-distribution (closed-loop validity — [OPEN], inferred).
  Window length = velocity-estimate denoising (mh > ph > scripted).

## 4. Vision campaign (running)

- TH-image: ht peak 69 (s1000b @159k) vs MIP 56 — parity-to-better, in-train harness, single seeds.
- SQ-mh-image: ht 77 @199k (rising) vs MIP 83 @159k → 58 (late rollback).
- Tuning arms (cd/mu/wd/os4) at 70–92k; TP-ph image pair at ~14k (regenerated 4-cam data);
  TP-mh regeneration in progress. Final claims need pooled/3-seed protocol.

## 5. Retractions this arc (all with replacement accounts)

H8-vs-H10 AS=1 gap (seed crossover) · σ-landscape shrinkage · late-training erosion ·
within-chunk crowding · loss-independent collision damage · settle-corridor onset ·
σ-mediated ht×history synergy · "transport tail = resolvable aliasing" ·
"TH chunk-level = only speed" (pocket branch is chunk-scale-real) ·
"gap entirely execution-borne" (softened to seed-dependent magnitude) ·
2× concatenation-metric artifacts (standardized frames hide motion; near-constant-dim amplification).

## 6. How to understand more — open holes ranked by leverage

1. **Why does transient repricing set a permanent asymptote?** (the deepest hole)
   sigaux splits aux-shaping vs weighting-path; if weighting-path wins, the 0–20k window needs
   a dense snapshot study (σ discriminates from step ~0).
2. **What IS the carrier of the head lottery?** Junction geometry just refuted as the predictor
   (h8-s5 has bad-seed geometry with good-seed behavior). Candidates: closed-loop-only properties
   (per-cycle maps on cheap short rollouts as the predictor?), decoder/readout weights, or nothing
   state-static — a no-rollout quality predictor would still be the most practically valuable
   deliverable if any exists; the cheap-rollout version (50-episode cycle map) is the fallback.
3. **Selector statistics**: more seeds on HG/L2 to turn "variance collapse" into a distributional
   claim; + does progaux/HT also collapse variance or only shift mean?
4. **Transport gripper-tail nature**: trait (operator style) vs unobserved content — demo-level
   variance decomposition would settle it in one run.
5. **Closed-loop history validity**: on-policy history-feature OOD stats (the one inferred link
   in the history law).
6. **Scope**: vision completion (does the selector/AS structure transfer to image obs?);
   human AS-sweep never run (deployment-conditionality of the human gap unmeasured).
7. **Hardening**: training seeds for H4/H32 cells and the H=2 loss ladder.

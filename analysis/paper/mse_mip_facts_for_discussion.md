# MSE vs MIP: Design and Experimental Phenomena (purely descriptive, no analysis)

## 1. Task and data
- Task: the init→insertion slice of Robomimic/robosuite ToolHang (grasp tool → lift → align → insert), OSC delta control, 7-dimensional raw action per step (pos3 + axis-angle3 + grip1), converted to 10 dimensions for training (pos3 + rot6d + grip1).
- Data: 2000 successful demos collected by a scripted policy (waypoint + PD, no injected noise), each with T≈549. The expert demonstration tube is extremely narrow: the lateral distance of eef trajectories to the 300-demo mean tube is σ(p)≈8.9mm.
- Observations are 53-dimensional: object 44 (absolute pos3+quat4 for each of base/frame/tool, plus each one's pos3+quat4 relative to the eef, plus 2 trailing dims) + eef_pos 3 + eef_quat 4 + gripper_qpos 2. Observation window of 2 frames. Both obs and actions are normalized by dataset statistics.
- Label properties (closure-aligned statistics, 80 demos): aligning on the gripper-command −→+ flip step c1, at τ=0..10 all demos have a_z≈−0.01 (hold), and at τ=12 all demos jump synchronously to a_z=+1.0 (fully saturated lift); the sliced data contains no re-grasp branches (retry rate 0%); labels are unimodal at every alignment offset.

## 2. Network
- The same architecture is used for both objectives: chiunet (1D UNet, ~19.7M parameters), input (t, x, emb); x is a 16×10 action-chunk slot; t is a scalar time condition; emb comes from an MLP encoder (~225k parameters) encoding 2 frames of observation. At execution time, steps 2..9 of the chunk (8 steps) are executed open-loop, then the policy re-observes.
- Evaluation uses EMA weights.

## 3. Training objectives (as implemented in the actual code)
- MSE (regression):
  pred = f(t=0, x=0, emb(s)); loss = 100 · ‖pred − a‖²
- MIP:
  y0 = f(t=0, x=0, emb(s))
  y1 = f(t=0.9, a + 0.1·ε, emb(s)), ε ~ N(0, I) element-wise (noise magnitude = 1 − t_two_step = 0.1, applied in the normalized action space)
  loss = 100 · ( ‖(y0 − a)/0.9‖² + ‖(y1 − a)/0.1‖² )
  The internal weight ratio of the two terms is (0.9/0.1)² = 81; the same overall loss_scale=100 as MSE.
- Inference:
  MSE: a single forward pass.
  MIP (2-step): y0 = f(0, 0, emb); output = f(0.9, y0, emb).
  MIP-step1: use y0 only.
- Training data, network, optimizer, step count (300k), and normalizer are identical for the two; the only difference is the loss.

## 4. Success rate (SR)
Official train-harness protocol (100 seeds, 21000+):
| data | MSE | MIP | MIP-step1 |
|---|---|---|---|
| original (2000 clean scripted demos) | 71 | 95 | 89 |
| wpmatch (modified scripted data containing structured "toxic" labels) | 36 | 88 | 87 |
Other-objective controls on the same data (wpmatch): Cauchy loss 27; noise-free aux variant (y0=a) 79; aux target with the radial component removed 0; aux target with the radial component flipped 0; aux gradient blocked from the encoder (stop-grad) 64. On original, Cauchy 57.
Policy-splicing harness (48 seeds, 700-step cap, phase localized by a 240-knot demonstration-tube tracker; numbers carry a small systematic offset relative to the official protocol):
| configuration | SR |
|---|---|
| pure MSE | 67 |
| pure MIP | 96 |
| MSE@[0,70) + MIP@[70,∞) (switch before gripper closure) | 94-96 |
| MIP@[0,140) + MSE@[140,∞) (switch after lift completes) | 88 |
| MSE@[0,70) + MIP@[70,140) + MSE@[140,∞) (sandwich) | 87.5 |
| MSE@[0,140) + MIP@[140,∞) | 79 |
| MIP-step1-only (same harness) | 47/48 |
Knot reference: 70 ≈ just before the press-down toward the grasp column, 79 ≈ grasp bottom, 137 ≈ lift top, beyond 140 the align/insert phase begins.

## 5. Deployment command-stream statistics (full trajectories)
- MSE: high-frequency energy hf=0.146 (2.4× the data-label level); adjacent-displacement direction correlation flip=−0.168 (46% of adjacent steps reverse direction); 81 oscillation-onset events, 46% decay spontaneously, 26% grow.
- MIP: hf=0.057 (≈ data level); flip=+0.936; 41 onset events, 73% decay, 10% grow.

## 6. Phenomena at the closure→lift transition (all entries are measurements; all same-state comparisons are "dual queries": both policies are queried at the same deployed state, and only the on-duty one is executed)
1. Teacher-forced (feeding ground-truth data states): at every closure-aligned offset both models' outputs match the labels exactly (τ=10: −0.008/−0.009; τ=12: +1.000/+1.000).
2. Out-of-data interpolation (linearly interpolating hold-side and lift-side observations to synthesize intermediate states): both models' a_z transition width is 0.08 (the jump completes by α=0.1); unchanged when ±7mm lateral offsets are superimposed.
3. Closure-aligned deployment profiles (16 eps each): the lateral tube distance distribution at the closure instant is identical for the two (p50 13.9 vs 13.8mm, p90 20.5 vs 19.6); after closure the p90 diverges: +2 chunks 33.0 vs 27.3, +6 chunks 49.8 vs 14.3mm; p50 stays at the noise floor throughout for both.
4. Clean-state same-state comparison (MIP driving, MSE observing): paired chunk hf ratio p50=1.00, 92% within 2×; first-action difference at the switch handoff p50=4.2e-3.
5. MSE driving, MIP observing, first-action difference binned by lateral tube distance: at d<20mm |Δa₁|≈0.005-0.007 (noise floor); d 20-50mm: 0.031; d>50mm: 0.120. Radial component: at d 20-50mm a_MSE·r̂=+0.040 vs a_MIP·r̂=+0.021; d>50mm: +0.105 vs +0.003.
6. On the same batch of deep-region states (d>20mm, n=115), regressing a_r/|a| = k·(v_r/|v|) + b (v = arrival velocity): k_MSE=0.41, k_MIP=0.46 (≈ equal); b_MSE=+0.103 (outward), b_MIP=−0.101 (inward); the |a| magnitudes are the same for both (0.33/0.36).
7. Failure cases (16 eps pure MSE, 4 failures): every failed trajectory contains, before escape, a chunk where the eef is only 3-15mm from the tube yet the two policies' first-action difference reaches 0.3-0.9 (40-130× the median noise floor). Of the 4 failures, 1 is in the lift phase (21008), 1 in the align phase, 2 at the end of insertion.
8. Case 21008 (lift-phase failure) details:
   - The 20 nearest training neighbors of that state (1-2 chunks after closure; 106-dim z-scored) belong 100% to the pre-closure phase, with neighbor labels g=−1 (gripper reopening) and a_z=−0.35 (press down); neighbor-label dispersion is lower than controls (no bimodality).
   - MSE output: [micro-motion, g=−0.9]; next chunk [a_z=+0.57, g=+1]. MIP output: [≈0, g=+1] (for both chunks). MIP's output variance across 4 sampling seeds is 0.
   - That state's full-state NN distance is 0.165-0.232; the data's own NN baseline is 0.069 (p50) / 0.127 (p90); control chunks from deployed successful trajectories 0.093 (p50); the NNz profile of successful trajectories in the closure window p90=0.295-0.352. Of the 106 dims, 0 exceed the data min-max range; the largest single-dim deviation is <1σ.
   - Deviation by block: 80-95% of the z deviation lies in the object dims; absolute placements (base/frame/tool absolute poses) NN=0.081 (normal); relative geometry (each object relative to the eef) NN=0.104 (2× the baseline p90); episode initial placement NN=0.272 ≈ the inter-training-init distance 0.229.
   - Interpolation path (that state → nearest post-phase neighbor): MSE's g flips from −0.86 to +0.81 within α∈[0,0.12]; MIP stays at +0.98~1.00 throughout. That state's NN to the pre-phase slice is 0.165, to the post-phase slice 0.424 (the nearest post-side neighbor is already a lift state).
   - Dimension-controlled paths (moving toward the pre neighbor): moving only the 44 object dims, MIP flips to the pre-side output at α=0.25 (≈0.39 z-units); moving only the 2 gripper_qpos dims, MIP flips at α=0.5 (≈0.17 z-units). MSE already outputs pre-side values at the original state.
   - Encoder embedding space (each model's own): that state's pre/post slice distance ratio, MSE 0.90 / MIP 0.97 (chunk 8); 1.23 / 1.58 (chunk 9).
9. Phase-aliasing exposure rate (criterion: NNz>0.15 and all 10-NN are pre-phase): MIP driving 5/56 chunk-states, MSE driving 8/56. Of the 3 seeds where MIP was exposed, 2 succeeded; of the 4 where MSE was exposed, 1 succeeded; seed 21008 fails under both policies.
10. Functional form of the second step (post-convergence, MIP):
    - Scaling the input along the y0 direction with c∈[0,3]: the output component along that direction stays ≈‖y0‖ (full magnitude is recovered even when fed 0), gap states behave the same as control states, and output norms are of the same order (7.7-7.9 vs 7.4-10.3).
    - Random-direction perturbation gain ‖Δout‖/‖Δin‖ (26 states × 4 directions × magnitudes {0.5,2,8}): p50=0.00 (all magnitudes), p90≤0.08, max decays with magnitude (0.86→0.27→0.24).
    - Swapping the entire input chunk across states (‖Δin‖≈8.7): output change p50=0.14 (gain 1.6%), p90 gain 0.658.
    - Adversarially injecting a pre-phase action into a gap state (press down + gripper reopening): the position channels' output is unchanged (hold); the gripper channel +1.0 → +0.34.
11. Lateral channel (closure window +0..+6 chunks, same state): |a_xy| p50 0.080 (MSE) vs 0.075 (MIP); xy direction flip p50 +0.53 vs +0.69, worst decile −0.49 vs −0.34.
12. Off-support field comparison (probing after DART kicks the state past the boundary): MIP step1 and 2-step response fields are identical (edgejac); in the near region (2-4cm) MIP's differential response leads MSE by 1.4-2.6×.

## 7. Appendix: related ablation facts (wpmatch testbed, each changing exactly one thing)
- Noise-free aux view (the y1 input uses a itself): SR 79 (MIP 88, MSE 36); its deployed maximum lateral tube distance p90 7748mm (MIP 54mm).
- Aux target with the radial component removed (only tangential correction allowed): SR 0; after deployment all rollouts leave the tube within 14 steps.
- Aux target with the radial component flipped: SR 0, out of the tube within 8 steps.
- Aux gradient blocked from the encoder (stop-grad, only the action head receives it): SR 64, deep-region recovery rate drops, tails unbounded.
- On the training side, all three arms' losses converge normally (2.8e-4 / 6.8e-4 / 9.0e-5).

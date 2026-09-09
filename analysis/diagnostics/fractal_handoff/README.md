# Fractal close-drawer: paired continuation diagnostic

Status: running in pod `yuchen-fractal-handoff-0908-r3`. First scene (1234)
passed prefix replay validation: zero physical/controller difference, exact
image and gripper latch. Continuation results pending. This is a diagnostic,
not a new benchmark score or evidence about training-support membership.

First failed scene: seed 1235, bottom drawer, final qpos 0.200555 m.
Selected branch steps 49 and 69; replay at step 69 passed all exactness
checks (physical/controller max difference 0). See `first_failure.json`.

## Predeclared protocol

- Checkpoints: `ft_fr_nu224/checkpoint-20000` and
  `ft_fr_flow/checkpoint-20000` under `/mnt/pfs/yuchen/groot`.
- SIMPLER Google Robot close-drawer; original 256 x 320 input images,
  original preprocessing and sticky-gripper wrapper; execute one action from
  each predicted chunk; total horizon 300 environment steps.
- Ten scene seeds 1234 through 1243, fixed before evaluation. This is not
  the previous benchmark's five-environment chained-seed cohort.
- Collect full HT trajectories. For failures, find the first step t in
  [60, 220] whose preceding 20 steps have mean EEF displacement below
  1 mm/step and absolute drawer displacement below 1 mm. Branch at t and
  t-20. If no stall meets this definition, use steps 80 and 100 and explicitly
  label the case as a fixed-time fallback.
- From each selected state: HT control, stochastic Flow, zero-initialized
  deterministic Flow. Same remaining budget (300 minus branch time).
  One stochastic Flow draw per branch; do not interpret it as a recovery
  probability estimate. Early/late branches from one scene are not independent.
- Restore by resetting the same scene seed and replaying the exact recorded
  action prefix. Validate all native physical-state values and exposed
  controller state to max absolute error <1e-6, exact image hashes and exact
  sticky-gripper state. Abort on mismatch. No approximate handoffs accepted.
- HT-to-HT checks whether original failure and actions reproduce. Zero Flow
  is a zero-noise initialization, **not** the conditional mean.
- Simulator success is the native drawer-joint criterion, qpos <= 0.05 m;
  count any successful step, consistent with the original rollout harness.

## Artifacts

Cluster root: `/mnt/pfs/yuchen/fractal_handoff_20260908/`.

- `code/`: exact probe and server scripts.
- `collect/<seed>.pkl`: raw action prefixes, physical/controller snapshots,
  image hashes, EEF positions, drawer qpos, success flags, selected branches.
- `{ht,flow,flow_zero}/<seed>_<branch>.pkl`: continuation trajectories.
- Matching `.json`: outcomes and restoration errors.
- Matching `.mp4`: observation-camera videos (every third step).
- `server_<method>.log`: model-server logs.

Scripts: `scripts/probe_fractal_handoff.py`,
`scripts/serve_fractal_handoff.py`, `scripts/cluster/fractal_handoff.yaml`.
No training jobs or checkpoint contents modified.

Startup note: the first launch failed before simulation because this
Transformers tokenizer made a model-metadata request in offline mode. The
server now resolves the same cached processor to its local snapshot path.
The second launch lacked graphics-driver capabilities; the current pod uses
`NVIDIA_DRIVER_CAPABILITIES=all`, matching the existing simulation pods.

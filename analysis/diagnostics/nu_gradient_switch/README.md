# Fractal nu-switch gradient audit

This is a no-update diagnostic of the existing 7k-hold nu staircase. It does
not launch or change policy training. The checkpoint source is
`/mnt/pfs/yuchen/groot/ft_fr_nu1024_hold7k_to14_20260908`.

## Measurement

- At each 7k–12k checkpoint, compare the checkpoint's current nu against a
  10% reduction and its actual next staircase value.
- Same processed input tensors, same train-mode dropout RNG, same weights.
  Paired predictions and sigma are asserted bit-identical.
- Differentiate the joint Student-t training loss over the exact valid action
  mask (56 entries, including the gripper). Reconstructed loss must match the
  repository's actual forward loss within 2e-5.
- Include every trainable parameter: action decoder, sigma decoder, and the
  shared action-head network. The vision/language backbone is frozen, as in
  training. These are parameter gradients, not output-gradient proxies.
- Cosine, angle, and norm changes concern raw, accumulated loss gradients,
  before clipping or Adam. They are not measured optimizer-step directions.
- A same-nu, different-dropout comparison on the same input batch provides a
  stochasticity reference; it does not prove a smaller systematic perturbation
  is harmless.

## Sampling and scope

`small_batch/` contains six checkpoints × four 16-state batches: two randomly
selected mixed-dataset batches, one instruction-filtered move-near batch,
and one close-drawer batch. Each batch selects distinct episodes and one
valid chunk start per episode. These are training-demonstration diagnostics,
not held-out policy evaluations. Two mixed batches and one batch per targeted
family do not establish population-level uncertainty.

`global1024/` is the follow-up at 10k and 12k: a mixed-data batch with the
training global sample count (1024), accumulated in microbatches of eight.
It is not a replay of the original distributed training batch or RNG stream.

Full parameter vectors are used during measurement but not retained on disk.
The saved JSON contains groupwise gradient comparisons; NPZ files contain
per-sample squared error, sigma, standardized squared error, dimension, gate,
prediction, and episode/step identifiers. Protocol JSON and parameter-name
manifests make the test reproducible from the checkpoints.

## Locations

- Cluster: `/mnt/pfs/yuchen/fractal_nu_gradient_20260908/`
- Probe: `scripts/probe_nu_gradient_switch.py`
- Summarizer: `analysis/diagnostics/summarize_nu_gradient_switch.py`
- Cluster pods: `yuchen-fr-nu-gradient-0908`,
  `yuchen-fr-nu-gradient-global-0908`.

No conclusion about task success causality follows from gradient angles alone.

## Completion / local copies

Both measurement pods completed successfully. See `RESULTS.md` for the
findings. All 91 small-batch artifacts and all 13 global-batch artifacts
(five JSON manifests/results and eight NPZ files) have been downloaded and
SHA256 verified against the cluster originals. The complete raw per-sample
arrays are now available locally as well as on the cluster.

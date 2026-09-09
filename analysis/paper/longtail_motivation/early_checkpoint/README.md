# WidowX early-checkpoint tail diagnostic

Checkpoint: `/mnt/pfs/yuchen/groot/ft_wxht10k/checkpoint-2000`.
Objective: legacy HT, **not MSE**. This run starts from
`/mnt/pfs/yuchen/groot/model` (verified in its `experiment_cfg/config.yaml`).
The original MSE run retains only checkpoints 16000–20000; this diagnostic
must not be described as an early-to-late trajectory of that MSE run.

## Measurement

- Same 144 episodes / 1728 inputs as the previous WidowX zero-model and
  MSE-20000 probes: three most frequent instructions, 48 episodes each,
  12 full-chunk start positions per episode; seed 20260906.
- Frozen deterministic HT inference; one forward, no sampling noise.
- Residual = prediction minus training-normalized target, eight steps by
  six continuous channels. Exclude gripper, do not divide by predicted sigma.
- Episode/task/step IDs match the saved MSE probe exactly; targets match
  to absolute tolerance 1e-7.
- Reuse `probe_zero_model_tail.analyze`: five episode folds, per-coordinate
  RMS from the other folds, final pooled unit RMS, fixed zero location.
- Tail-fit and MLE use exactly the previous grids and objectives. These are
  **pooled scalar-coordinate fits**, not multivariate chunk fits and not
  estimates that directly prescribe the training loss's degrees of freedom.
- The demonstrations were available during policy training. Episode holdout
  here is for scale/density fitting, not policy generalization evaluation.

## Result

| Statistic | Zero model | HT 2k | MSE 20k |
|---|---:|---:|---:|
| Tail-fit nu | 300 (upper search bound) | 6.71255 | 5.71688 |
| MLE nu (original coarse grid) | 3.16228 | 2.51189 | 3.16228 |
| P(abs(z)>3) | 1.37563% | 1.80724% | 1.64448% |
| Maximum abs(z) | 4.22190 | 7.96302 | 8.60987 |

The early trained model no longer exhibits the zero-model probe's short
observed tail. This does not isolate the effect of training duration versus
training objective, or establish an optimal training nu.

## Reproduction and provenance

- Extraction: `scripts/probe_widowx_early_tail.py`, importing the original
  `probe_widowx_general_scale.py` input selection and forward pass.
- Cluster job: `scripts/cluster/widowx_early_tail_20260907.yaml`.
- Cluster outputs: `/mnt/pfs/yuchen/groot/early_tail_20260907/results/`.
- Local measured arrays: `early_ht_rank0.npz` in this directory.
- Local statistics: `early_checkpoint_tail_summary.json` in this directory.
- Figure: run `scripts/plot_zero_model_tail.py --early-dir
  analysis/paper/longtail_motivation/early_checkpoint`.
- Zero-model outputs are retained; the new figure is saved separately as
  `../fig_early_checkpoint_tail.{png,pdf}`.

Review: the figure labels identify both objective and step (clarity); all
numbers derive from saved predictions (evidence); identical targets and
folds are checked (protocol); lack of early MSE is disclosed (coverage);
scalar fits are not equated with joint-loss nu (method soundness).

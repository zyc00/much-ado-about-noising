# WidowX MSE 500-step diagnostic

Requested comparison: replace the zero-output baseline with an actual early
MSE policy, using the same observations and coordinate-tail statistics.

## Training

- Cluster run: `/mnt/pfs/yuchen/groot/ft_wxmse500_20260907`.
- Initialization: `/mnt/pfs/yuchen/groot/model`, the same pretrained base as
  the reference MSE run. This is a fresh fine-tune, not a resume of MSE 20k
  or HT 2k, and not an entirely randomly initialized robot model.
- Dataset: all 53,192 episodes in `bridge_orig_lerobot`, staged in the pod's
  private `/dev/shm/wxmse500`.
- Embodiment: `SIMPLER_ENV_WIDOWX`.
- Objective: original MSE (including the gripper during training).
- Eight GPUs, global batch 1024, AdamW, peak LR 1e-4, weight decay 1e-5,
  state dropout 0.8, eight data-loader workers per rank; other settings from
  the original finetuning launcher.
- Preserve the original 20,000-step cosine schedule with 5% warmup (1,000
  updates). A process-local callback saves and stops at update 500. Do not
  mistake the schedule's `max_steps=20000` for the actual run length.
- This makes the LR at update 500 approximately 5e-5; using a compressed
  500-step schedule would change the experiment.
- Original reference config and current model/trainer source are copied to
  the run's `audit/` directory. No shared model/trainer code is edited.

## Probe

After checkpoint 500 is saved, the job automatically evaluates the same
144 episodes / 1,728 inputs as the prior zero-model and MSE-20k diagnostics.
It checks episode/task/step IDs and normalized targets against the saved
MSE-20k arrays before fitting.

Only the six continuous channels enter the diagnostic. Residual = prediction
minus normalized label; eight-step chunks. No predicted sigma is used.
The same five episode folds, other-fold coordinate RMS, final pooled unit
RMS, zero-location tail fit, and original MLE grid are reused.

The data were available during policy training; the folds hold out episodes
for statistical scale/density fitting, not for policy training. All nu fits
are pooled scalar-coordinate fits, not direct estimates of a joint loss nu.

## Files

- Launcher: `scripts/cluster/widowx_mse500_launch.py`.
- Training and automatic probe: `scripts/cluster/widowx_mse500_run.sh`.
- Cluster pod: `scripts/cluster/widowx_mse500_20260907.yaml`.
- Probe: `scripts/probe_widowx_early_tail.py --objective mse --training-steps 500`.
- Cluster logs/results: `/mnt/pfs/yuchen/groot/mse500_tail_20260907/`.

Results must be read from the saved summary after the probe succeeds; none
are assumed from the earlier HT-2k diagnostic.

## Completed measurement

Training stopped at exactly 500 updates; checkpoint objective and global
step checks passed. Runtime reported by the trainer: 1006.62 seconds,
including initial cache filling. The original schedule's theoretical
throughput fields are not meaningful for an early-stopped run.

The full model/training/data config comparison with the original run found
only the output path and staging path changed. The first three logged
losses were 0.2484, 0.2444, 0.2355 in both runs, with identical learning
rates (gradient norms had small numerical differences).

The probe matched all 1728 episode/task/step IDs and normalized targets.

| Statistic | Zero model | MSE 500 | MSE 20k |
|---|---:|---:|---:|
| Tail-fit nu | 300 (search boundary) | 16.6730 | 5.71688 |
| MLE nu (original coarse grid) | 3.16228 | 2.51189 | 3.16228 |
| P(abs(z)>3) | 1.37563% | 1.90852% | 1.64448% |
| Maximum abs(z) | 4.22190 | 6.03462 | 8.60987 |

MSE 500 has 7.069 times the standard-normal probability beyond 3. Its
held-out density-fit NLL gain for Student-t is 0.08263 nat/coordinate.
This is not a held-out robot-success evaluation, nor evidence that every
coordinate or the joint action vector follows an exact Student-t law.

Local arrays and statistics: `mse500/early_mse_rank0.npz` and
`mse500/early_checkpoint_tail_summary.json`. Plot with
`scripts/plot_zero_model_tail.py --early-dir analysis/paper/longtail_motivation/mse500`.
The plot explicitly labels MSE 500 and MSE 20k; older zero/HT plots remain
unchanged.

Review: contribution is limited to the requested diagnostic; plot labels
identify the actual checkpoints; inputs/fitting protocol are matched;
robot success is not measured; pooled scalar-fit nu is not identified with
the multivariate training-loss nu.

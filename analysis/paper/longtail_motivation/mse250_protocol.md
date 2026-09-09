# WidowX MSE early-checkpoint replay

User requested checkpoint 250, additionally saving 50, 100, and 200.
The run stops at exactly 250 optimizer updates and retains all four saves.

- Run: `/mnt/pfs/yuchen/groot/ft_wxmse250_20260907`.
- Logs and probe outputs: `/mnt/pfs/yuchen/groot/mse250_tail_20260907/`.
- Initialization: same pretrained `/mnt/pfs/yuchen/groot/model`, fresh run.
- Recipe: unchanged original MSE configuration, 8 GPUs, batch 1024,
  seed 42, full 53192-episode BridgeData V2 dataset, state dropout 0.8.
- Preserve the original 20k cosine LR schedule and 1000-step warmup.
- `MSE_PROBE_SAVE_STEPS=50,100,200,250`; `MSE_PROBE_STOP_STEP=250`.
- Process-local trainer callback; no edits to shared model/trainer code.
- Four frozen probes run after training, one per GPU. Each uses the same
  144 episodes / 1728 inputs and fixed-coordinate RMS protocol as MSE 500
  and MSE 20k, excluding gripper from statistics but not from MSE training.
- Probe files are in `results/{50,100,200,250}/` under the log directory.
- Checkpoint global steps/objectives and probe input/target identity are
  asserted before reporting results.

The earlier MSE 500 checkpoint and its measured data are retained. This
replay is required because that run did not save earlier intermediate weights.

Launch files: `scripts/cluster/widowx_mse250_run.sh`,
`scripts/cluster/widowx_mse250_20260907.yaml`, and the configurable
`scripts/cluster/widowx_mse500_launch.py`.

## Completed results

All four checkpoint global-step and MSE-objective checks passed. Training
stopped at 250; reported runtime was 613.67 seconds including initial cache
filling and intermediate saves. Model/training/data configs match the
original run except output and local staging paths.

All four probes passed exact episode/task/step matching and normalized
target matching (absolute tolerance 1e-7) against the MSE-20k reference.

| Checkpoint | Tail-fit nu | MLE nu (original grid) | P(abs(z)>3) |
|---|---:|---:|---:|
| 50 | 300 (upper search boundary) | 1.99526 | 1.81327% |
| 100 | 300 (upper search boundary) | 3.16228 | 1.75781% |
| 200 | 41.41335 | 2.51189 | 1.94107% |
| 250 | 284.36656 | 2.51189 | 1.88320% |

These early tail fits are not monotonic in training step and are distinct
from full-distribution MLE. Boundary fits should not be treated as measured
degrees of freedom, and none of these scalar fits alone prescribes a joint
HT-loss hyperparameter.

Downloaded arrays and per-checkpoint summaries live under
`mse_early_checkpoints/{50,100,200,250}/`. Transfers verify SHA256 against
the cluster originals. `scripts/summarize_widowx_early_mse_tail.py` checks
the downloaded arrays against the reference again and aggregates these
results with the existing zero, MSE-500, and MSE-20k diagnostics.

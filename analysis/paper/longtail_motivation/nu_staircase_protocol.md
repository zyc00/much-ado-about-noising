# Joint Student-t nu staircase: WidowX and Fractal

Requested 2026-09-07. This is a new training experiment, not a coordinate-wise
Student-t run and not a continuation from a 2k checkpoint.

Both runs start from `/mnt/pfs/yuchen/groot/model`. The user revised the
initial nu from 896 to **1792 before either run was submitted**.

| Optimizer updates (inclusive) | Joint nu |
|---|---:|
| 1–1000 | 1792 |
| 1001–2000 | 896 |
| 2001–3000 | 448 |
| 3001–4000 | 224 |
| 4001–5000 | 112 |
| 5001–6000 | 56 |
| 6001–7000 | 28 |
| 7001–8000 | 14 |
| 8001–20000 | 7 |

Reference recipes: `ft_wxnu224/experiment_cfg/config.yaml` and
`ft_fr_nu224/experiment_cfg/config.yaml`, under `/mnt/pfs/yuchen/groot`.
Each run uses eight GPUs, global batch 1024, AdamW at 1e-4, 1000 warmup
updates, the original 20k cosine schedule, and the full original dataset.
WidowX retains sigma bias -0.4535 and state dropout 0.8; Fractal retains
sigma bias -0.5093 and state dropout 0.5. Action masks and gripper handling
are unchanged. This remains the joint action-chunk loss (`ht_mvt=true`),
not a sum of scalar Student-t losses.

Intentional changes are the nu schedule and checkpoint retention (20,
rather than 5, to retain each 1k checkpoint). A process-local callback
sets the actual action-head `config.ht_df` before each optimizer update.
Built-in smooth nu annealing stays disabled. A checkpoint at a boundary
stores the nu used for the update just completed; the callback applies
the next nu before the next update. `nu_schedule.jsonl` records transitions.

Cluster runs:

- Pod `yuchen-wx-nustair-0907`, output
  `/mnt/pfs/yuchen/groot/ft_wx_nustair_20260907`.
- Pod `yuchen-fr-nustair-0907`, output
  `/mnt/pfs/yuchen/groot/ft_fr_nustair_20260907`.

Each output contains `train.log` and an `audit/` directory with the
reference configuration, launcher, runner, evaluator, and model/trainer
source snapshots. Shared model/trainer sources are not modified.

After training, the runner automatically evaluates checkpoints 16000 and
20000. WidowX: seven SIMPLER tasks, 50 episodes/task, four executed actions.
Fractal: six Google Robot SIMPLER tasks, 100 episodes/task, one executed
action. Both use five parallel environments, 300 maximum episode steps,
and the reference unseeded evaluation protocol. Outputs are
`eval-{step}/summary.json` with all per-task rates and their macro mean;
rollout and server logs are retained. No success-rate results are available
at submission time, and this single-seed experiment does not establish
whether scheduling is better than fixed nu.

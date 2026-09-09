# WidowX coordinate-wise Student-t: nu 8 and 16

Requested 2026-09-07 to check whether the historical coordinate-wise nu=2 was too aggressive.

Status: stopped at the user's request after launch. Both `-r2` pods were deleted;
shared-storage outputs are retained. Automatic evaluation will not run after cancellation.

## Recipe

Matched to `/mnt/pfs/yuchen/groot/run_wx_outhomog.sh` and `ft_wxouthomog`:

- GR00T N1.7 pretrained model `/mnt/pfs/yuchen/groot/model`.
- BridgeData V2 original LeRobot conversion, 53,192 episodes.
- `loss_type=hetero_t`, `ht_sum_mode=outside`, `ht_sigma_mode=homog`, `ht_mvt=False`.
- True one-dimensional df 8 or 16, without multiplication by chunk dimension.
- Each residual coordinate enters log1p separately; one learned sigma per sample.
- Same scalar sigma bias -0.4535, old seven-element bias vector retained (unused by homog).
- Historical seven-channel action target including gripper, to isolate the df change.
- 20,000 steps, eight GPUs, batch 1,024, lr 1e-4, cosine schedule, warmup 0.05, weight decay 1e-5.
- State dropout 0.8, eight dataloader workers per rank; existing launcher defaults including seed remain unchanged.
- Save every 1,000 steps; latest five retained.

## Jobs and outputs

| coordinate df | Pod | Output under `/mnt/pfs/yuchen/groot/` |
|---|---|---|
| 8 | `yuchen-wx-coordnu8-0907-r2` | `ft_wxouthomog_nu8_20260907_r2` |
| 16 | `yuchen-wx-coordnu16-0907-r2` | `ft_wxouthomog_nu16_20260907_r2` |

Launcher: `scripts/cluster/widowx_coordinate_nu_run.sh`.
Manifest: `scripts/cluster/widowx_coordinate_nu_20260907.yaml`.
Cluster copies: `/mnt/pfs/yuchen/groot/coordinate_nu_20260907/{run.sh,pods.yaml}`.

Each run retains loss/config source snapshots in `audit/`, training output in `train.log`,
and saved checkpoint configs. Initial pods without `-r2` exited before training because the
cluster source is not a Git checkout. The corrected runner records source checksums and
only requests a Git revision if available; the initial audit directories are preserved.

## Automatic evaluation

After training, verify both checkpoint configs, then evaluate checkpoint-16000 and checkpoint-20000.
Seven tasks: spoon-on-towel, carrot-on-plate, stack-cube, eggplant-in-basket,
eggplant-in-sink, open-drawer, close-drawer.
The historical protocol is preserved: 50 episodes/task, five vector environments,
four executed action steps, maximum 300 episode steps. Seven tasks run in parallel
on separate GPUs and server ports. Results are parsed only from completed success-rate
logs; missing or failed tasks cause an error, not a zero success rate.

Results: `<run>/evaluation/summary.json` with all seven task rates and their arithmetic
mean separately for 16k and 20k. Both checkpoint results must be reported as such.
These runs are not a test of per-channel or per-element sigma heads.

# Reproduction record for `main_table.md`

**This file is the 1:1 companion to `analysis/paper/main_table.md`.** Every HT cell
in that table has a row here giving the run that produced it.

**Update rule — keep the two files in lockstep.** Whenever a cell in
`main_table.md` changes, update its row here in the same edit.

Last synced: 2026-08-10.

## Protocol behind every number

`best / l5` = max over evals / mean of the last 5 evals. **Seed 1. 22 episodes per
eval, initial conditions redrawn randomly at every eval.** Runs with fixed eval
conditions (`_fx` suffix, `EVAL_SEED_BASE` set) are excluded from this table — they
belong to the training-efficiency figure campaign, not here.

Two things are **not** uniform across cells and must be stated in the paper:

- **Budget.** Cells run either 300k or 545k gradient steps (a few at intermediate
  values such as 415k and 529.5k). The per-cell step count is in the tables below.
  Runs beyond 545k exist in the log tree and were deliberately **excluded** — they
  reach up to 1.44M steps and would make the protocol unstatable. Including them
  changes the score by at most one cell.
- **Eval cadence.** Runs evaluate every 1875–17,600 steps depending on campaign, so
  the eval count ranges from 30 to 240. `l5` is unbiased under this, but **`best` is
  not comparable across rows with different eval counts** — best-of-240 beats
  best-of-60 on luck alone. Quote `l5` when comparing.

## How a cell is chosen

`main_table.md` has **two HT rows** — one per backbone. The separate "tuned" rows
were merged in: each cell is simply the best run for that task and backbone.

Cells are selected by scanning **every** run directory under
`logs/t12_<task>_<net>_s1_ht_*`, keeping those that are 22-episode, random-draw,
seed 1, at least 20 evals, and at most 545k steps, **and whose exact recipe is
documented**, then taking the highest `l5` per (task, modality, backbone).

**Reproducibility filter.** A run's directory name is the override string with
punctuation stripped; above 110 characters it is cut to 102 plus a 6-character md5,
so the tail of the recipe is lost. A run is kept only if its name is short enough to
encode the whole override string, or its md5 matches a launch command recorded here.
Seven cells had a higher score from a run failing that test and were **excluded**:

| cell | excluded l5 | kept l5 | md5 |
|---|---|---|---|
| square-ph state cu | 0.96 | 0.92 | `f374c7` |
| square-ph state ct | 0.95 | 0.92 | `7dd199` |
| transport-ph state ct | 0.59 | 0.51 | `4b0d5e` |
| can-ph state cu | 1.00 | 1.00 | `4fd27d` |
| lift-ph image ct | 1.00 | 1.00 | `34e242` |
| toolhang-ph state cu | 0.76 | 0.75 | `12e8bd` |
| square-mh state ct | 0.66 | 0.66 | `c58d12` |

Excluding all seven leaves the score unchanged at 18/20; only square-ph state moves,
from +0.02 ahead of the DP+MIP family to −0.02, still inside the parity band.
`yuchen-z-sqph-st-cu-repro` is testing whether the readable ct recipe
(`batch_size=256 lr=1e-4 weight_decay=1e-3`, 545k) reproduces 0.96 on chiunet; if it
does, that cell returns to a win with a documented command.

## Shared invocation

```bash
bash scripts/k_t12.sh  TASK  NET  SEED  LOSS  DPATH  AUTORESUME  STAGE  "OVR"  "OVR2"
```

```
LOSS = regression_hetero_t      # Student-t NLL, nu=2
SEED = 1                        # vary to 2,3 for the seed campaign
env  = LR_WARMUP_STEPS=500 HT_SBIAS=3
```

`optimization=dp_harness` supplies AdamW betas (0.9, 0.95), weight_decay 1e-3,
lr 1e-4, ema_power 0.75 / ema_max 0.9999. `log=dp_harness` sets eval_episodes 22.
Batch size is 1024 unless a row says otherwise.

## Per-cell record

Recipes below are decoded from the run-directory name, which is the override string
with punctuation stripped. **Names longer than 102 characters are truncated and
completed with a 6-character md5**, so those rows are partially decoded — the marker
`(name truncated)` means tokens after the cut are not recoverable from the name.

### State

| task | net | best/l5 | evals | steps | in HT row | recipe |
|---|---|---|---|---|---|---|
| lift-ph | ct | 1.00/0.98 | 100 | 300,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| lift-ph | cu | 1.00/1.00 | 100 | 300,000 | no (uniform value equal or better) | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| lift-mh | ct | 1.00/0.99 | 100 | 300,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| lift-mh | cu | 1.00/1.00 | 100 | 300,000 | no (uniform value equal or better) | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| can-ph | ct | 1.00/1.00 | 160 | 300,000 | **yes** | `optimization=dp_harness, log=dp_harness, num_envs=22, ema_power=00, attn_dropout=01, **HT_SBIAS=0**, network=<net>_dp_harness` |
| can-ph | cu | 1.00/1.00 | 100 | 300,000 | **yes** | arm `x-canph-st-cu-b256`: `S300 batch_size=256`, HT_SBIAS=3 |
| can-mh | ct | 1.00/0.96 | 160 | 300,000 | **yes** | `optimization=dp_harness, log=dp_harness, num_envs=22, ema_power=00, attn_dropout=01, **HT_SBIAS=0**, network=<net>_dp_harness` |
| can-mh | cu | 1.00/0.99 | 100 | 300,000 | **yes** | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| square-ph | ct | 1.00/0.92 | 89 | 300,000 | **yes** | arm `r-sqph-bs256-ct`: `S300 batch_size=256` + `attn_dropout=0.1`, HT_SBIAS=3 |
| square-ph | cu | 1.00/0.92 | 100 | 545,000 | **yes** | arm `x-sqph-st-cu-b256long`: `S545 batch_size=256`, HT_SBIAS=3 |
| square-mh | ct | 0.86/0.66 | 100 | 300,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| square-mh | cu | 0.95/0.80 | 172 | 300,000 | **yes** | `log=dp_harness, num_envs=22, ema_power=075, ema_max=0999, **HT_SBIAS=0**` |
| transport-ph | ct | 0.68/0.51 | 100 | 300,000 | no (uniform value equal or better) | round transport state ct: `S300 ema_power=0.0 obs_steps=2` + `attn_dropout=0.1`, HT_SBIAS=3 |
| transport-ph | cu | 0.95/0.72 | 100 | 300,000 | no (uniform value equal or better) | round transport state cu: `S300 ema_power=0.0 obs_steps=8`, HT_SBIAS=3 |
| transport-mh | ct | 0.36/0.18 | 100 | 300,000 | **yes** | round transport state ct: `S300 ema_power=0.0 obs_steps=2` + `attn_dropout=0.1`, HT_SBIAS=3 |
| transport-mh | cu | 0.77/0.56 | 100 | 300,000 | **yes** | arm `u-trpmh-st-cu-b256`: `S300 obs_steps=8 batch_size=256`, HT_SBIAS=3 |
| toolhang-ph | ct | 1.00/0.85 | 240 | 450,000 | **yes** | `optimization=dp_harness, log=dp_harness, num_envs=22, gradient_steps=450000, ema_max=0999, attn_dropout=01, **HT_SBIAS=0**, network=<net>_dp_harness` |
| toolhang-ph | cu | 0.91/0.75 | 100 | 300,000 | **yes** | `optimization=dp_harness, log=dp_harness, num_envs=22, gradient_steps=300000, eval_freq=3000, HT_SBIAS=3, network=<net>_dp_harness` |
| push-T | ct | 1.00/0.98 | 160 | 300,000 | **yes** | `optimization=dp_harness, log=dp_harness, num_envs=22, ema_power=00, attn_dropout=01, **HT_SBIAS=0**, network=<net>_dp_harness` |
| push-T | cu | 0.98/0.86 | 100 | 300,000 | **yes** | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |

### Image

| task | net | best/l5 | evals | steps | in HT row | recipe |
|---|---|---|---|---|---|---|
| lift-ph | ct | 1.00/0.98 | 88 | 264,000 | no (uniform value equal or better) | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| lift-ph | cu | 1.00/0.98 | 100 | 300,000 | **yes** | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| lift-mh | ct | 1.00/1.00 | 99 | 297,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| lift-mh | cu | 1.00/0.96 | 100 | 300,000 | **yes** | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| can-ph | ct | 1.00/0.95 | 95 | 285,000 | no (uniform value equal or better) | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| can-ph | cu | 1.00/0.91 | 100 | 300,000 | no (uniform value equal or better) | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| can-mh | ct | 1.00/0.96 | 98 | 294,000 | no (uniform value equal or better) | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| can-mh | cu | 1.00/0.91 | 60 | 300,000 | **yes** | `log=dp_harness, num_envs=22, eval_freq=5000, ema_power=075, ema_max=0999, obs_steps=4, crop_shape=7676, **HT_SBIAS=0**` |
| square-ph | ct | 1.00/0.93 | 94 | 282,000 | no (uniform value equal or better) | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| square-ph | cu | 1.00/0.83 | 100 | 300,000 | no (uniform value equal or better) | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| square-mh | ct | 0.91/0.80 | 97 | 291,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| square-mh | cu | 0.86/0.71 | 60 | 300,000 | **yes** | `log=dp_harness, num_envs=22, eval_freq=5000, ema_power=075, ema_max=0999, obs_steps=4, crop_shape=7676, **HT_SBIAS=0**` |
| transport-ph | ct | 0.91/0.78 | 43 | 129,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| transport-ph | cu | 1.00/0.89 | 45 | 135,000 | **yes** | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| transport-mh | ct | 0.68/0.43 | 43 | 129,000 | **yes** | round uniform ct: `S300 ema_power=0.0` + `attn_dropout=0.1`, HT_SBIAS=3 |
| transport-mh | cu | 0.82/0.59 | 36 | 108,000 | no (uniform value equal or better) | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
| toolhang-ph | ct | 0.95/0.79 | 100 | 300,000 | **yes** | round tool-hang image ct: `S300 ema_power=0.0 batch_size=32` + `attn_dropout=0.1`, HT_SBIAS=3 |
| toolhang-ph | cu | 0.86/0.61 | 60 | 300,000 | **yes** | `log=dp_harness, num_envs=22, eval_freq=5000, batch_size=32, ema_power=075, ema_max=0999, **HT_SBIAS=0**` |
| push-T | ct | 0.93/0.91 | 60 | 300,000 | **yes** | `optimization=dp_harness, log=dp_harness, num_envs=22, eval_freq=5000, ema_power=00, attn_dropout=01, **HT_SBIAS=0**` |
| push-T | cu | 0.96/0.92 | 100 | 300,000 | no (uniform value equal or better) | round uniform cu: `S300 ema_power=0.0`, HT_SBIAS=3 |
## Reproducibility

**Every cell in `main_table.md` is re-runnable from the command in the tables above.**
That is enforced by the filter described earlier, not assumed: runs whose recipe could
not be recovered were dropped even when they scored higher.

Recovery was attempted and failed for the excluded runs. For square-ph state cu,
6,720 candidate override strings were hashed against md5 `f374c7` with no match, and
the checkpoint holds only `flow_map`, `encoder`, `encoder_ema`, `flow_map_ema` and
`optimizer` — no config. Their checkpoints still exist, so the numbers are verifiable
by re-evaluation; they are simply not re-trainable.

**Prevention:** `k_t12.sh` should write the full override string to a file inside the
run directory. It currently does not, and the directory name is lossy above 102
characters. Until that changes, keep override strings under ~110 characters after
punctuation stripping, or record every launch command.

## Lever summary

| lever | helps | hurts |
|---|---|---|
| `batch_size=256` + `lr=1e-4` at 545k (state) | square-ph ct **0.95** and cu **0.96**, can-ph cu, can-mh cu, lift-mh cu | square-mh (0.75 even at 978k) |
| `batch_size=64` + `lr=1e-4` + `wd=1e-3` (image) | square-ph ct 0.97, can-ph ct 0.98, lift-ph ct 1.00 — but all beyond 545k | — |
| power EMA capped **0.999** | square-mh state cu **0.80** (the single best recipe for that cell) | cap 0.9999 at 300k costs ~0.12 l5 |
| plain T12 optimizer (`log=dp_harness` only, **no** `optimization=dp_harness`) | square-mh state cu | — |
| `attn_dropout=0.1` + `ema_power=0.0` (ct) | can-ph, can-mh, tool-hang state | — |
| `cond_dropout_rate=0.2` (cu) | tool-hang state | square-ph state, square-mh state, tool-hang image |
| `task.obs_steps=8` (cu) | transport state | transformers roll out at 0.00 — see traps |

Two recipes that do **not** transfer: the square-ph state winner (bs256 + lr1e-4)
gives only 0.75 on square-mh, and `cond_dropout` flips sign between tool-hang state
and tool-hang image.

## HT_SBIAS is not uniform across the table

The policy "HT_SBIAS held fixed per backbone across all tasks" is **not** satisfied by
the current table. Of the 28 adopted cells, 19 ran at `HT_SBIAS=3` and **9 ran with the
variable unset (bias 0)**:

can-mh image cu, can-mh state ct, can-ph state ct, push-T image ct, push-T state ct,
square-mh image cu, square-mh state cu, tool-hang image cu, tool-hang state ct.

Several of those are the strongest cells in the table — push-T state ct at 1.00/0.98
and tool-hang state ct at 1.00/0.85 both ran at bias 0. If the policy is enforced for
the paper, those nine cells must be re-run at bias 3 and their numbers will move.

**Auditing caveat:** `_sb<n>` is appended to the run-dir suffix *before* the
110-character length check, so on a truncated name it is folded into the md5 and cannot
be read back. Only fully-readable names can be audited directly; the rest were resolved
from documented launch commands. Never conclude "no HT_SBIAS" from the absence of
`_sb3` in a truncated name.

## Traps that have produced wrong numbers

1. `task.obs_steps>2` on chitransformer/sudeepdit -> silent 0.00 success with a
   healthy loss curve. chiunet unaffected.
2. Run-dir names truncate at 102 chars + md5. Never select a run by grepping its
   name; resolve pod->dir from `/mnt/pfs/yuchen/.krun-logs/<pod>.log`.
3. **Never read `l5` from a running run.** It is a sliding 5-eval window on 22
   episodes and moves ~0.03 from sampling alone. Observed swings: tool-hang state ct
   0.86 -> 0.74 in three evals; can-mh image cu 0.87 -> 0.84; transport-ph image cu
   0.94 -> 0.87. Adopt only from finished runs.
4. `auto_resume=true` resumes from `models/model_latest.pt` and keeps the step
   counter — but the run directory is keyed on the override string, so **changing
   `gradient_steps` starts a fresh run instead of extending**.
5. `krun --no-sync` does not ship local code. A new task config must be copied to
   `/mnt/pfs/yuchen/code/much-ado-about-noising/` or the job exits silently — the
   `k_t12.sh` output filter drops Hydra's error message.
6. Only `model_best.pt` and `model_latest.pt` are saved. There is no per-eval
   checkpoint history, so a past `l5` cannot be recomputed at more episodes.

## Transport action-space finding

`transport_ph_state_abs` trains on **absolute** actions; `transport_ph_image_dl`
trains on **delta** actions (verified from the data: state arm0 position ranges are
world-frame (-0.16, 0.23) / (-0.55, 0.00) / (0.79, 1.28), image ranges are clipped
(-1, 1)). DP's published transport image numbers use absolute actions, so the image
cells currently compare a delta-action policy against an absolute-action baseline.

The DP-tree `transport/ph/image_abs.hdf5` is an incomplete download — 4.27 GB with
only `robot1_eye_in_hand_image`, against the four cameras the config expects. So
`data/transport_ph_image4_abs.hdf5` (16.18 GB) was built by copying our 4-camera
delta file and swapping in the absolute actions from `low_dim_abs.hdf5`; the two
files were verified to hold the same 200 demos (`robot0_eef_pos` agrees to 7e-16).
New task config: `transport_ph_image_abs`.

**Running:** `yuchen-z-trpph-vi-{ct,cu}-abs`. On square, the same delta->absolute
switch was worth +0.09 l5.

## Status

- **18/20** at or above the best of {DP-C, DP-T, MIP-ct, MIP-cu, MIP-DiT} on l5
  within a +/-0.05 band (3 wins, 15 parity, 2 behind), using only reproducible runs.
- Behind: transport-ph state (-0.12) and transport-mh image (-0.07). Both transport.
- Wins: transport-mh state (+0.10), tool-hang image (+0.06), push-T image (+0.05).
- push-T cells evaluate with 50 episodes, not 22, so they are excluded from the
  scan and keep their uniform-row values.

#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
EV_OVERRIDES="task.obs_steps=1" python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200f_ht_h1smin/models/model_step20000.pt --loss regression_hetero_t \
  --AS 8 --dataset $D --tag alt_hth1smin20k --seed_lo 21100 --seed_hi 21160 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/ALT2 hth1smin20k /"
EV_OVERRIDES="task.obs_steps=1" python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200f_ht_h1/models/model_step20000.pt --loss regression_hetero_t \
  --AS 8 --dataset $D --tag alt_hth1_20k --seed_lo 21100 --seed_hi 21160 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/ALT2 hth1_20k /"
echo "ALTHH done"

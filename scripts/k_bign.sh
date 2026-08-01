#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
EV_OVERRIDES="task.obs_steps=1" python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200f_ht_h1/models/model_step20000.pt --loss regression_hetero_t \
  --AS 8 --dataset $D --tag big_hth1 --seed_lo 21000 --seed_hi 21120 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/BIG ht_h1_20k_N120 /"
python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200f_ht_lr05/models/model_step60000.pt --loss regression_hetero_t \
  --AS 8 --dataset $D --tag big_htlr --seed_lo 21000 --seed_hi 21120 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/BIG ht_lr05_60k_N120 /"
echo "BIGN done"

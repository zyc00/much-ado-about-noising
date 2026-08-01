#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
for RANGE in "21000 21120 a" "21160 21280 b"; do
  set -- $RANGE
  python -u scripts/eval_twofactor.py --ckpt logs/mp200f_ht_lr05ema/models/model_step60000.pt \
    --loss regression_hetero_t --AS 8 --dataset $D --tag v_lr05ema_$3 \
    --seed_lo $1 --seed_hi $2 2>&1 | grep -a "TWOFACTOR.*SR=" | sed "s/^/VHT lr05ema_$3 /"
done
echo "VALHT done"

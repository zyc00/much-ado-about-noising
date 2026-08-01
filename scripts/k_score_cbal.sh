#!/bin/bash
# Score the cbal (channel-leverage-balance) arms with CHAN_SCALE exported at
# eval (pre-registered), plus the selfnorm confirmation. Prints "CB ... " lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5

echo "CB start cbal50"
CHAN_SCALE=0.5,2.0,2.0,2.0 python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200_cbal50_s1000/models/snap_300000.pt --loss regression --AS 8 \
  --dataset $D --tag cb_cbal50 --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/CB /"
echo "CB start cbal70"
CHAN_SCALE=0.7,1.5,1.5,1.5 python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200_cbal70_s1000/models/snap_300000.pt --loss regression --AS 8 \
  --dataset $D --tag cb_cbal70 --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/CB /"
echo "CB start selfnorm"
python -u scripts/eval_twofactor.py \
  --ckpt logs/mp200_selfnorm_s1000/models/snap_300000.pt --loss regression_selfnorm \
  --AS 8 --dataset $D --tag cb_selfnorm --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/CB /"
echo "CB done"

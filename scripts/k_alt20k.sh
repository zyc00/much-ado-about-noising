#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
python -u scripts/eval_twofactor.py --ckpt logs/mp200_hg_s1000/models/snap_20000.pt \
  --loss regression_hetero_gauss --AS 8 --dataset $D --tag alt_hg20k \
  --seed_lo 21100 --seed_hi 21160 2>&1 | grep -a "TWOFACTOR.*SR=" | sed "s/^/ALT hg20k /"
python -u scripts/eval_twofactor.py --ckpt logs/mp200_mip_s1000/models/snap_60000.pt \
  --loss mip --AS 8 --dataset $D --tag alt_mip60k \
  --seed_lo 21100 --seed_hi 21160 2>&1 | grep -a "TWOFACTOR.*SR=" | sed "s/^/ALT mip60k /"
echo "ALT done"

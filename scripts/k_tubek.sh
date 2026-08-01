#!/bin/bash
# Eval-time servo graft on frozen L2-200: ACT_TUBEK dose ladder.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
for K in 0.05 0.2; do
  ACT_TUBEK=$K python -u scripts/eval_twofactor.py \
    --ckpt logs/mp200_l2_s1000/models/snap_300000.pt --loss regression --AS 8 \
    --dataset $D --tag tubek_${K} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/TK k=$K /"
done
echo "TUBEK done"

#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
L2=logs/mp200_l2_s1000/models/snap_300000.pt
MIP=logs/mp200_mip_s1000/models/snap_300000.pt
for W in 0.3 0.5 0.7; do
  echo "BL start w=$W"
  BLEND2=$MIP:mip:$W python -u scripts/eval_twofactor.py --ckpt $L2 \
    --loss regression --AS 8 --dataset $D --tag bl_$W \
    --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/BL /"
done
echo "BL done"

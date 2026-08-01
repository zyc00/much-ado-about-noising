#!/bin/bash
# Causal test of the neighbor-averaging mediator at execution. "KX" lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
run() { # tag ckpt loss eta
  echo "KX start $1 eta=$4"
  KNN_EXTRAP=$4 python -u scripts/eval_twofactor.py --ckpt $2 --loss $3 --AS 8 \
    --dataset $D --tag kx_$1 --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/KX /"
}
L2=logs/mp200_l2_s1000/models/snap_300000.pt
MIP=logs/mp200_mip_s1000/models/snap_300000.pt
run l2_p15  $L2  regression  0.15
run l2_p30  $L2  regression  0.30
run l2_m15  $L2  regression  -0.15
run mip_m30 $MIP mip         -0.30
run mip_p30 $MIP mip         0.30
echo "KX done"

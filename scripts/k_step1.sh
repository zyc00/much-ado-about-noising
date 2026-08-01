#!/bin/bash
# Score STEP-1-ONLY inference of the MIP and ATK checkpoints (single-step
# regression deployment of two-view-trained nets).
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
for SPEC in "mip_s1:logs/mp200_mip_s1000/models/snap_300000.pt" "atk_s1:logs/mp200_nonoise_atk/models/snap_300000.pt" "nonoise_s1:logs/mp200_nonoise_s1000/models/snap_300000.pt"; do
  TAG="${SPEC%%:*}"; CK="${SPEC#*:}"
  python -u scripts/eval_twofactor.py --ckpt $CK --loss mip_step1 --AS 8 \
    --dataset $D --tag s1_${TAG} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/S1 $TAG /"
done
echo "STEP1 done"

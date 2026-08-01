#!/bin/bash
# Fresh-block verification: seeds 21160-21280 (120 new episodes, no overlap
# with 21000-21120 or 21100-21160). Usage: k_n240.sh <spec> ...
# spec = tag:ckpt:loss[:ovr]
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
for SPEC in "$@"; do
  IFS=: read -r TAG CK LOSS OVR <<< "$SPEC"
  EV_OVERRIDES="${OVR:-}" python -u scripts/eval_twofactor.py --ckpt $CK \
    --loss $LOSS --AS 8 --dataset $D --tag n240_${TAG} \
    --seed_lo 21160 --seed_hi 21280 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/N240 $TAG /"
done
echo "N240 done"

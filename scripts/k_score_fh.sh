#!/bin/bash
# Score flow + headmse causal arms, canonical twofactor.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
run() {
  local name=$1 loss=$2
  python -u scripts/eval_twofactor.py \
    --ckpt logs/${name}/models/snap_300000.pt --loss $loss --AS 8 \
    --dataset $D --tag fh_${name} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/FH /"
}
run mp200_flow_s1000 straight_flow
run mp200_headmse_s1000 regression_frozentrunk
echo "FH done"

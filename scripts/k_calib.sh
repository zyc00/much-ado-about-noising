#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
run() { local tag=$1 ck=$2 loss=$3 lo=$4 hi=$5 ovr=${6:-}
  EV_OVERRIDES="$ovr" python -u scripts/eval_twofactor.py --ckpt $ck --loss $loss --AS 8 \
    --dataset $D --tag cal_${tag} --seed_lo $lo --seed_hi $hi 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/CAL $tag /"; }
run ht300k_alt logs/mp200_ht_s1000/models/snap_300000.pt regression_hetero_t 21100 21160
run hg300k_alt logs/mp200_hg_s1000/models/snap_300000.pt regression_hetero_gauss 21100 21160
run mip300k_alt logs/mp200_mip_s1000/models/snap_300000.pt mip 21100 21160
echo "CALIB done"

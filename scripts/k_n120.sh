#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
run() { local tag=$1 ck=$2 loss=$3 ovr=${4:-}
  EV_OVERRIDES="$ovr" python -u scripts/eval_twofactor.py --ckpt $ck --loss $loss --AS 8 \
    --dataset $D --tag n120_${tag} --seed_lo 21000 --seed_hi 21120 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/N120 $tag /"; }
run mip300k logs/mp200_mip_s1000/models/snap_300000.pt mip
run ht300k logs/mp200_ht_s1000/models/snap_300000.pt regression_hetero_t
run hg300k logs/mp200_hg_s1000/models/snap_300000.pt regression_hetero_gauss
run hg20k logs/mp200_hg_s1000/models/snap_20000.pt regression_hetero_gauss
echo "N120 done"

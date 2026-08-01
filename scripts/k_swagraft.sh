#!/bin/bash
# Scoring-only arms: SWA averages of existing snapshots + eval-time graft
# on hg@20k. Prints SG lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
M=logs/mp200_hg_s1000/models
SWA_IN=$M/snap_20000.pt,$M/snap_40000.pt,$M/snap_60000.pt SWA_OUT=$M/swa_204060.pt python -u scripts/make_swa.py
SWA_IN=$M/snap_20000.pt,$M/snap_40000.pt SWA_OUT=$M/swa_2040.pt python -u scripts/make_swa.py
H=logs/mp200_ht_s1000/models
SWA_IN=$H/snap_60000.pt,$H/snap_100000.pt,$H/snap_140000.pt SWA_OUT=$H/swa_60140.pt python -u scripts/make_swa.py
run() { local tag=$1 ck=$2 loss=$3; shift 3
  env "$@" python -u scripts/eval_twofactor.py --ckpt $ck --loss $loss --AS 8 \
    --dataset $D --tag sg_${tag} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/SG $tag /"; }
run hg_swa204060 $M/swa_204060.pt regression_hetero_gauss
run hg_swa2040 $M/swa_2040.pt regression_hetero_gauss
run ht_swa60140 $H/swa_60140.pt regression_hetero_t
run hg20k_tk05 $M/snap_20000.pt regression_hetero_gauss ACT_TUBEK=0.05
run hg20k_tk10 $M/snap_20000.pt regression_hetero_gauss ACT_TUBEK=0.1
echo "SWAGRAFT done"

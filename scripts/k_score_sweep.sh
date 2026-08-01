#!/bin/bash
# Score the 16-arm hetero+condreg tuning sweep. "SWP" lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
score() { # tag loss
  [ -f logs/mp200_$1_s1000/models/snap_300000.pt ] || { echo "SWP $1 MISSING"; return; }
  python -u scripts/eval_twofactor.py --ckpt logs/mp200_$1_s1000/models/snap_300000.pt \
    --loss $2 --AS 8 --dataset $D --tag sw_$1 --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/SWP $1 /"
}
HG=regression_hetero_gauss_cnd
HT=regression_hetero_t_cnd
for T in hgcnd4 hgcnd5 hgc_l1e3 hgc_l1e5 hgc_l3e4t05 hgc_l1e4t05 hgc_l3e5t05 \
         hgc_l3e4t15 hgc_l1e4t15 hgc_l1e4e10 hgc_l1e4e20 hgc_l3e4e10 \
         hgc_l1e4k12 hgc_best; do
  score $T $HG
done
score htc_l1e4 $HT
score htc_l3e5 $HT
echo "SWP done"

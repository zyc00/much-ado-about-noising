#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
score() { # tag loss
  [ -f logs/mp200_$1_s1000/models/snap_300000.pt ] || { echo "ANI $1 MISSING"; return; }
  python -u scripts/eval_twofactor.py --ckpt logs/mp200_$1_s1000/models/snap_300000.pt \
    --loss $2 --AS 8 --dataset $D --tag an_$1 --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/ANI $1 /"
}
HGD=regression_hetero_diag
HTD=regression_hetero_t_diag
for T in hgdiag hgde999 hgdsd hgdsd999; do score $T $HGD; done
for T in htd htde999 htdsd htddf5 htddf10; do score $T $HTD; done
score flow straight_flow
echo "ANI done"

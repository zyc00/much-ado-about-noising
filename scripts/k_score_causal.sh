#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
score() { # tag loss
  [ -f logs/mp200_$1_s1000/models/snap_300000.pt ] || { echo "CSL $1 MISSING"; return; }
  python -u scripts/eval_twofactor.py --ckpt logs/mp200_$1_s1000/models/snap_300000.pt \
    --loss $2 --AS 8 --dataset $D --tag cs_$1 --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/CSL $1 /"
}
score cnde3 regression_condreg
score cnde6 regression_condreg
score dcr3 regression_dcr
score dcr4 regression_dcr
score headmse regression_frozentrunk
echo "CSL done"

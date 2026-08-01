#!/bin/bash
# Score EARLY snapshots of one arm with the canonical twofactor protocol.
# Usage: k_early.sh <run_dir_name> <loss>
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
NAME=$1; LOSS=$2
echo "EARLY snaps for $NAME:"; ls logs/${NAME}/models/ | grep snap | sort -t_ -k2 -n | tr '\n' ' '; echo
for ST in 20000 40000 60000 100000 140000 200000; do
  CK=logs/${NAME}/models/snap_${ST}.pt
  [ -f "$CK" ] || continue
  python -u scripts/eval_twofactor.py --ckpt $CK --loss $LOSS --AS 8 \
    --dataset $D --tag e_${NAME}_${ST} --seed_lo 21000 --seed_hi 21060 2>&1 \
    | grep -a "TWOFACTOR.*SR=" | sed "s/^/EARLY $NAME $ST /"
done
echo "EARLY done $NAME"

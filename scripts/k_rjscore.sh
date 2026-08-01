#!/bin/bash
# Score a finished RJ/AUG arm: model_latest == step-300000 model (save_freq
# 20k, 300k steps). Copies to snap_300000 for ledger consistency, then
# recovery-gain probe + canonical twofactor. Usage: k_rjscore.sh <name> <loss>
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
NAME=$1; LOSS=$2
cp -n logs/${NAME}/models/model_latest.pt logs/${NAME}/models/snap_300000.pt
RV_ARMS="${NAME}:${LOSS}:logs/${NAME}/models/snap_300000.pt" \
python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD" 
python -u scripts/eval_twofactor.py \
  --ckpt logs/${NAME}/models/snap_300000.pt --loss $LOSS --AS 8 \
  --dataset $D --tag rj_${NAME} --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/RJ /"
echo "RJSCORE done $NAME"

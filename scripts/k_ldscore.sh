#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
NAME=$1; DS=$2
cp -n logs/${NAME}/models/model_latest.pt logs/${NAME}/models/snap_300000.pt
RV_DATA=$DS RV_ARMS="${NAME}:regression:logs/${NAME}/models/snap_300000.pt" \
python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD"
python -u scripts/eval_twofactor.py \
  --ckpt logs/${NAME}/models/snap_300000.pt --loss regression --AS 8 \
  --dataset $DS --tag ld_${NAME} --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/LD /"
echo "LDSCORE done $NAME"

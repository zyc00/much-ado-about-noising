#!/bin/bash
# Data-ladder L2 arm: train on subset dataset, then recovery probe + twofactor
# (normalizers = training dataset). Usage: k_ladder.sh <name> <dataset>
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
NAME=$1; DS=$2
echo "LD train start $NAME $DS"
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$DS network=chiunet \
  optimization.loss_type=regression optimization.seed=1000 optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "Error|Traceback" -A6 | head -20
echo "LD train done $NAME"
cp -n logs/${NAME}/models/model_latest.pt logs/${NAME}/models/snap_300000.pt
RV_DATA=$DS RV_ARMS="${NAME}:regression:logs/${NAME}/models/snap_300000.pt" \
python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD|Traceback"
python -u scripts/eval_twofactor.py \
  --ckpt logs/${NAME}/models/snap_300000.pt --loss regression --AS 8 \
  --dataset $DS --tag ld_${NAME} --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/LD /"
echo "LD all done $NAME"

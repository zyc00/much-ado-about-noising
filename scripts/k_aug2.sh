#!/bin/bash
# Train one augmented/penalized arm then score (twofactor + recovery gain).
# Usage: k_aug2.sh <name> <loss> [ENV1=V1] [ENV2=V2] ...
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
NAME=$1; LOSS=$2; shift 2
for kv in "$@"; do export "$kv"; done
echo "AG train start $NAME $LOSS extra: $*"
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$D network=chiunet \
  optimization.loss_type=$LOSS optimization.seed=1000 optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "Error|Traceback" -A6 | head -20
echo "AG train done $NAME"
unset RECOVJIT RJ_NOCORR 2>/dev/null || true
cp -n logs/${NAME}/models/model_latest.pt logs/${NAME}/models/snap_300000.pt
RV_ARMS="${NAME}:${LOSS}:logs/${NAME}/models/snap_300000.pt" \
python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD|Traceback"
python -u scripts/eval_twofactor.py \
  --ckpt logs/${NAME}/models/snap_300000.pt --loss $LOSS --AS 8 \
  --dataset $D --tag ag_${NAME} --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/AG /"
echo "AG all done $NAME"

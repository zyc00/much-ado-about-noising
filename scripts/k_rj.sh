#!/bin/bash
# Train one RECOVJIT arm (recovery-jitter augmentation) then score it:
# canonical twofactor (AS=8, seeds 21000-21060, snap_300000) + recovery-gain
# property probe. Usage: k_rj.sh <name> <loss>
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=data/tool_hang_full2ins_mp_200.hdf5
NAME=$1; LOSS=$2
echo "RJ train start $NAME $LOSS"
RECOVJIT=1 RJ_P=0.5 RJ_LO=0.005 RJ_HI=0.05 RJ_K=4 \
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$D network=chiunet \
  optimization.loss_type=$LOSS optimization.seed=1000 optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "step.*30000|Error|Traceback" | tail -3
echo "RJ train done $NAME"
unset RECOVJIT
RV_ARMS="${NAME}:${LOSS}:logs/${NAME}/models/snap_300000.pt" \
python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD|Traceback"
python -u scripts/eval_twofactor.py \
  --ckpt logs/${NAME}/models/snap_300000.pt --loss $LOSS --AS 8 \
  --dataset $D --tag rj_${NAME} --seed_lo 21000 --seed_hi 21060 2>&1 \
  | grep -a "TWOFACTOR.*SR=" | sed "s/^/RJ /"
echo "RJ all done $NAME"

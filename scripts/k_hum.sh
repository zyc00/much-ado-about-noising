#!/bin/bash
# Human-data chain: train (full schedule, stop 60k) then OFFICIAL harness
# eval (mode=eval, 100 episodes). Usage: k_hum.sh <name> <loss> [ENV=V ...]
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
HD=data/tool_hang_human_lowdim_up.hdf5
NAME=$1; LOSS=$2; shift 2
for kv in "$@"; do export "$kv"; done
export SNAP_AT=20000,60000 STOP_AT=60000
echo "HUM train start $NAME $LOSS extra: $*"
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$HD network=chiunet \
  optimization.loss_type=$LOSS optimization.seed=1000 optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "Error|Traceback" -A6 | head -20
echo "HUM train done $NAME"
for ST in 20000 60000; do
  CK=logs/${NAME}/models/model_step${ST}.pt
  [ -f "$CK" ] || continue
  python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/$HD network=chiunet \
    optimization.loss_type=$LOSS optimization.auto_resume=false \
    optimization.model_path=$CK log.eval_episodes=100 \
    log.log_dir=logs/${NAME}_ev${ST} log.wandb_mode=disabled 2>&1 \
    | grep -aE "mean_success" | sed "s/^/HUM $NAME $ST /" | head -4
done
echo "HUM all done $NAME"

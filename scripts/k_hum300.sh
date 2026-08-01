#!/bin/bash
# Standard human protocol: FULL 300k training, periodic official in-run
# evals (best/last5 from metrics.jsonl), snapshots for post-hoc eval.
# Usage: k_hum300.sh <name> <loss> [ENV=V ...]
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
HD=data/tool_hang_human_lowdim_up.hdf5
NAME=$1; LOSS=$2; shift 2
for kv in "$@"; do export "$kv"; done
export SNAP_AT=60000,120000,180000,240000,300000
echo "HUM300 start $NAME $LOSS extra: $*"
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$HD network=chiunet \
  optimization.loss_type=$LOSS optimization.seed=1000 optimization.auto_resume=false \
  log.log_dir=logs/$NAME 2>&1 | grep -aE "mean_success|Error|Traceback" | tail -40
echo "HUM300 done $NAME"

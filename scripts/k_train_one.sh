#!/bin/bash
# Runs INSIDE krun GPU pod. Generic single-model trainer.
# args: $1=dataset_basename (in data/)  $2=loss(mip|regression)  $3=exp_name
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DS="$1"; LOSS="$2"; EXP="$3"
test -f "data/$DS" || { echo "MISSING data/$DS on PVC"; exit 2; }
rm -rf "logs/$EXP"
MUJOCO_GL=egl python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/$DS task.num_envs=1 network=chiunet \
  optimization.loss_type=$LOSS optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=$EXP log.log_dir=logs/$EXP
echo "=== $EXP TRAINED ==="

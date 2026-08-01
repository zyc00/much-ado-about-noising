#!/bin/bash
# args: <tag> -- human-demo (tool_hang PH) MSE training + official mode=eval; OBS_MASK from env
set -eu
export MUJOCO_GL=egl
TAG=$1
SEED=${SEED:-0}
LOSS=${LOSS:-regression}
EXTRA=${EXTRA:-}
NET=${NET:-chiunet}
GSTEPS=${GSTEPS:-300001}
DATA=${DATA:-tool_hang_human_lowdim_up.hdf5}
rm -rf logs/$TAG
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/data/$DATA network=$NET \
  task.num_envs=1 optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=$GSTEPS log.save_freq=5000 \
  optimization.seed=$SEED $EXTRA \
  optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG
python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/data/$DATA network=$NET \
  optimization.loss_type=$LOSS $EXTRA optimization.model_path=$(pwd)/logs/$TAG/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -aE "mean_assembled|mean_success" | sed "s/^/HUMAN_EVAL $TAG | /"

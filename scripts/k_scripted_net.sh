#!/bin/bash
# scripted full2ins_2000 trainer with network choice + official mode=eval
# args: <tag>; env: LOSS, NET, SEED
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; LOSS=${LOSS:-regression}; NET=${NET:-chiunet}; SEED=${SEED:-5}
rm -rf logs/$TAG
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_full2ins_2000.hdf5 network=$NET \
  task.num_envs=1 optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=300001 log.save_freq=100000 \
  optimization.seed=$SEED optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG
python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_full2ins_2000.hdf5 network=$NET \
  optimization.loss_type=$LOSS optimization.model_path=$(pwd)/logs/$TAG/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -aE "mean_assembled|mean_success" | sed "s/^/SCRIPTED_EVAL $TAG | /"

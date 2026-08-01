#!/bin/bash
# args: <tag>; OBS_MASK from env. Local human-demo MSE training + official mode=eval.
set -eu
export MUJOCO_GL=egl
TAG=$1
rm -rf logs/$TAG
.venv/bin/python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4 network=chiunet \
  task.num_envs=1 optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=regression log.exp_name=$TAG log.log_dir=logs/$TAG
.venv/bin/python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4 network=chiunet \
  optimization.loss_type=regression optimization.model_path=$(pwd)/logs/$TAG/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -aE "mean_assembled|mean_success" | sed "s/^/HUMAN_EVAL $TAG | /"

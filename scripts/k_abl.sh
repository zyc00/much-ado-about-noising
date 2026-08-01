#!/bin/bash
# args: <dataset_basename> <tag> — train MSE on ablation slice, then grasp census rollouts
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/$1
rm -rf logs/$2
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet \
  optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 \
  optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=regression log.exp_name=$2 log.log_dir=logs/$2
CKPT=logs/$2/models/model_latest.pt DS=$DSP TAG=$2 MUJOCO_GL=egl python -u scripts/grasp_census_generic.py

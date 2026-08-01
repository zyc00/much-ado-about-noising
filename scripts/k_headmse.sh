#!/bin/bash
# frozen MIP backbone (encoder + UNet trunk) + fresh final_conv head trained with MSE regression
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export INIT_CKPT=/mnt/pfs/yuchen/code/much-ado-about-noising/logs/mip_traj_2k/models/model_latest.pt
export REINIT_HEAD=1 TRUNK_LR_MULT=0
DS=data/tool_hang_full2ins_2000.hdf5
rm -rf logs/$1
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/$DS \
  task.num_envs=1 network=chiunet optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=150001 log.save_freq=5000 \
  optimization.loss_type=regression log.exp_name=$1 log.log_dir=logs/$1

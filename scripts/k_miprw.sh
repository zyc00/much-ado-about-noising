#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
[ -f logs/rw_assets.npz ] || python -u scripts/build_rw_assets.py
export RW_ASSETS=logs/rw_assets.npz
export RW_PROFILE="0.973,1.028,1.005,0.892,0.866"
rm -rf logs/mip_rw_2k
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_full2ins_2000.hdf5 \
  task.num_envs=1 network=chiunet optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=mip_rw log.exp_name=mip_rw_2k log.log_dir=logs/mip_rw_2k

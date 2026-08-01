#!/bin/bash
# args: <loss_type> <exp_name>
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
DS=data/tool_hang_full2ins_wpmatch_2000.hdf5
[ -f logs/rw_assets_wpmatch.npz ] || DS=$DS RW_OUT=logs/rw_assets_wpmatch.npz python -u scripts/build_rw_assets.py
export RW_ASSETS=logs/rw_assets_wpmatch.npz
rm -rf logs/$2
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/$DS \
  task.num_envs=1 network=chiunet optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=$1 log.exp_name=$2 log.log_dir=logs/$2

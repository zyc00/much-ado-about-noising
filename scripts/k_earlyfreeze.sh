#!/bin/bash
# args: <snap_path> <tag> -- frozen EARLY-MSE backbone + fresh head trained with MSE, then twofactor
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export INIT_CKPT=/mnt/pfs/yuchen/code/much-ado-about-noising/$1
export REINIT_HEAD=1 TRUNK_LR_MULT=0
DS=data/tool_hang_full2ins_2000.hdf5
rm -rf logs/$2
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/$DS \
  task.num_envs=1 network=chiunet optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=150001 log.save_freq=5000 \
  optimization.loss_type=regression log.exp_name=$2 log.log_dir=logs/$2
python scripts/eval_twofactor.py --ckpt logs/$2/models/model_latest.pt --loss regression --tag $2-final --dataset $DS
SNAP_GLOB="logs/$2/models/model_latest.pt" LOSS=regression TAG=$2 QDIR1=/mnt/pfs/yuchen/embq/msebulk QDIR2=/mnt/pfs/yuchen/embq/cfdump_mse2 python -u scripts/probe_foldsweep.py

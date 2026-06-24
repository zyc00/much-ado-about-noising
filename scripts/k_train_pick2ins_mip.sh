#!/bin/bash
# Runs INSIDE krun GPU pod. Train MIP insertion specialist (no-dagger), parallel
# to the dagger-mip pod's grasp_mip. Together = MIP two-stage trained in parallel.
set -eu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python
test -f data/tool_hang_pick2ins_2000.hdf5 || { echo "pick2ins data missing on PVC"; exit 2; }
rm -rf logs/pick2ins_mip_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_pick2ins_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=pick2ins_mip_2000 \
  log.log_dir=logs/pick2ins_mip_2000
echo "=== pick2ins_mip TRAINED ==="

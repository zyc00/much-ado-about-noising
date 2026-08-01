#!/bin/bash
# two-stage explicit coarse/fine boosting: stage1 coarse (20k early stop), stage2 residual student
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
DS=data/tool_hang_full2ins_2000.hdf5
DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/$DS
if [ ! -f logs/orig_coarse20k/models/model_latest.pt ]; then
  rm -rf logs/orig_coarse20k
  python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$DSP task.num_envs=1 network=chiunet \
    optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 \
    optimization.gradient_steps=20001 log.save_freq=5000 \
    optimization.loss_type=regression log.exp_name=orig_coarse20k log.log_dir=logs/orig_coarse20k
fi
export TEACHER_CKPT=/mnt/pfs/yuchen/code/much-ado-about-noising/logs/orig_coarse20k/models/model_latest.pt
rm -rf logs/orig_boost
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet \
  optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 \
  optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=regression_residual log.exp_name=orig_boost log.log_dir=logs/orig_boost
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt logs/orig_boost/models/model_latest.pt \
  --loss regression_residual --tag orig_boost-final --dataset $DS

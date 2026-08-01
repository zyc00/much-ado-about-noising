#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export SIG_MIN=0.003 SIG_MAX=3.0 SIG_STEPS=300000
DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_full2ins_2000.hdf5
rm -rf logs/orig_fadehint
( while true; do sleep 1800
    [ -f logs/orig_fadehint/models/model_latest.pt ] && \
      cp logs/orig_fadehint/models/model_latest.pt logs/orig_fadehint/models/snap_$(date +%H%M).pt
  done ) &
SNAPPID=$!
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet \
  optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 \
  optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=regression_fadehint log.exp_name=orig_fadehint log.log_dir=logs/orig_fadehint
kill $SNAPPID || true
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt logs/orig_fadehint/models/model_latest.pt \
  --loss regression_fadehint --tag orig_fadehint-final --dataset data/tool_hang_full2ins_2000.hdf5

#!/bin/bash
# sigma-annealed MIP with periodic checkpoint archiving for time-alignment analysis
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
export SIG_MIN=0.003 SIG_MAX=0.3 SIG_STEPS=300000
DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_full2ins_2000.hdf5
rm -rf logs/orig_siganneal
( while true; do sleep 1800
    [ -f logs/orig_siganneal/models/model_latest.pt ] && \
      cp logs/orig_siganneal/models/model_latest.pt logs/orig_siganneal/models/snap_$(date +%H%M).pt
  done ) &
SNAPPID=$!
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet \
  optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 \
  optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=mip_siganneal log.exp_name=orig_siganneal log.log_dir=logs/orig_siganneal
kill $SNAPPID || true
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt logs/orig_siganneal/models/model_latest.pt \
  --loss mip_siganneal --tag orig_siganneal-final --dataset data/tool_hang_full2ins_2000.hdf5

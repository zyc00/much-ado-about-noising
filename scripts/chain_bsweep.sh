#!/bin/bash
# Start-point sweep: train MSE at fixed backoff B (start = align_done - B), eval
# from matching held-out start, to locate where SR jumps 0% (pick) -> high.
# Then resume full_mip 200/2000/20000.
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python

for B in 50 60 70; do
  rm -rf logs/b${B}_regression_2000
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_b${B}_2000.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=b${B}_regression_2000 \
    log.log_dir=logs/b${B}_regression_2000 > /tmp/b${B}_regression_2000.log 2>&1
  echo "=== b${B}_regression_2000 TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt logs/b${B}_regression_2000/models/model_latest.pt \
    --dataset data/tool_hang_b${B}_2000.hdf5 --eval_states data/b${B}_eval_states.hdf5 \
    --loss regression --n 100 --max_chunks 30 2>&1 | grep -aE "INSERTION_SR" | sed "s/^/Bsweep B=${B} N=2000 /"
done
echo "ALL BSWEEP DONE"

for N in 200 2000 20000; do
  rm -rf logs/full_mip_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=full_mip_${N} \
    log.log_dir=logs/full_mip_${N} > /tmp/full_mip_${N}.log 2>&1
  echo "=== full_mip_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt logs/full_mip_${N}/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_${N}.hdf5 --loss mip --n 100 2>&1 \
    | grep -aE "FULL_SR" | sed "s/^/full mip N=${N} /"
done
echo "ALL FULL_MIP DONE"

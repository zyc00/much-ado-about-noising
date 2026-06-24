#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
# wide-range (backoff 0-80) training
rm -rf logs/wide_regression_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_wide_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=wide_regression_2000 \
  log.log_dir=logs/wide_regression_2000 > /tmp/wide_regression_2000.log 2>&1
echo "=== wide_regression_2000 TRAINED ==="
CK=logs/wide_regression_2000/models/model_latest.pt
for tag in insertion b40 b50 b60 b70 pick; do
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt $CK --dataset data/tool_hang_wide_2000.hdf5 \
    --eval_states data/${tag}_eval_states.hdf5 --loss regression --n 100 --max_chunks 35 2>&1 \
    | grep -aE "INSERTION_SR" | sed "s/^/WIDE on ${tag}: /"
done
echo "ALL WIDE DONE"
# full_mip
for N in 200 2000 20000; do
  rm -rf logs/full_mip_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=full_mip_${N} \
    log.log_dir=logs/full_mip_${N} > /tmp/full_mip_${N}.log 2>&1
  echo "=== full_mip_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt logs/full_mip_${N}/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_${N}.hdf5 --loss mip --n 100 2>&1 | grep -aE "FULL_SR" | sed "s/^/full mip N=${N} /"
done
echo "ALL FULL_MIP DONE"

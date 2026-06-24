#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
rm -rf logs/rr60_regression_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_rr60_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=rr60_regression_2000 \
  log.log_dir=logs/rr60_regression_2000 > /tmp/rr60_regression_2000.log 2>&1
echo "=== rr60_regression_2000 TRAINED ==="
# eval on range-0-60 states (matched) AND fixed B points to see the curve
for tag in range60 b40 b50 b60 insertion; do
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt logs/rr60_regression_2000/models/model_latest.pt \
    --dataset data/tool_hang_rr60_2000.hdf5 --eval_states data/${tag}_eval_states.hdf5 \
    --loss regression --n 100 --max_chunks 35 2>&1 | grep -aE "INSERTION_SR" | sed "s/^/RR60 on ${tag}: /"
done
echo "ALL RR60 DONE"

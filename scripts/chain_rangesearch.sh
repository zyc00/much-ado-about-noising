#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python

run_range () {  # $1=width(40|60) $2=eval_states_file
  local W=$1 ES=$2
  rm -rf logs/range${W}_regression_2000
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_range${W}_2000.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=range${W}_regression_2000 \
    log.log_dir=logs/range${W}_regression_2000 > /tmp/range${W}_regression_2000.log 2>&1
  echo "=== range${W}_regression_2000 TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt logs/range${W}_regression_2000/models/model_latest.pt \
    --dataset data/tool_hang_range${W}_2000.hdf5 --eval_states ${ES} \
    --loss regression --n 100 --max_chunks 35 2>&1 | grep -aE "INSERTION_SR" | sed "s/^/RANGE 0-${W}: /"
}

run_range 40 data/aligninsert_eval_states.hdf5
run_range 60 data/range60_eval_states.hdf5
echo "ALL RANGE SEARCH DONE"

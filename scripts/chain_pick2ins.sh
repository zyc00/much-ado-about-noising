#!/bin/bash
# pick2ins (pick->insertion, single policy) MSE 200/2000/20000, then resume the
# full_mip cells that were preempted. eval: reset to held-out pick state, run
# policy, success = frame assembled.
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python

# ---- pick2ins MSE ----
for N in 200 2000 20000; do
  rm -rf logs/pick2ins_regression_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_pick2ins_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=pick2ins_regression_${N} \
    log.log_dir=logs/pick2ins_regression_${N} > /tmp/pick2ins_regression_${N}.log 2>&1
  echo "=== pick2ins_regression_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt logs/pick2ins_regression_${N}/models/model_latest.pt \
    --dataset data/tool_hang_pick2ins_${N}.hdf5 --eval_states data/pick_eval_states.hdf5 \
    --loss regression --n 100 --max_chunks 30 2>&1 | grep -aE "INSERTION_SR" | sed "s/^/pick2ins regression N=${N} /"
done
echo "ALL PICK2INS DONE"

# ---- resume full_mip (preempted) ----
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

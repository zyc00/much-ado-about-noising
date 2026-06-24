#!/bin/bash
# Wait for b60 to finish in the running sweep, then preempt and run b40 (control
# for the 86%->3% cliff) first, then b70, then full_mip.
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
SWEEP=/tmp/claude-1001/-home-jigu-projects-much-ado-about-noising/469964b2-9364-47f5-8a44-7dda4fc11f85/tasks/bu4w2sh5q.output

echo "WAIT: for b60 eval to finish..."
while ! grep -qa "Bsweep B=60" "$SWEEP" 2>/dev/null; do sleep 30; done
echo "WAIT: b60 done -> preempting old sweep chain"
pkill -f chain_bsweep.sh 2>/dev/null
pkill -f "tool_hang_b70_2000" 2>/dev/null
pkill -f "tool_hang_full2ins" 2>/dev/null
sleep 5

run_mse_cell () {  # $1=tag(b40|b70) $2=B
  local TAG=$1 B=$2
  rm -rf logs/${TAG}_regression_2000
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_${TAG}_2000.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=${TAG}_regression_2000 \
    log.log_dir=logs/${TAG}_regression_2000 > /tmp/${TAG}_regression_2000.log 2>&1
  echo "=== ${TAG}_regression_2000 TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt logs/${TAG}_regression_2000/models/model_latest.pt \
    --dataset data/tool_hang_${TAG}_2000.hdf5 --eval_states data/${TAG}_eval_states.hdf5 \
    --loss regression --n 100 --max_chunks 30 2>&1 | grep -aE "INSERTION_SR" | sed "s/^/Bsweep B=${B} N=2000 /"
}

run_mse_cell b40 40
run_mse_cell b70 70
echo "ALL BSWEEP (with b40) DONE"

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

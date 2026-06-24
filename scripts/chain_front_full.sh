#!/bin/bash
# Wait for the running align+insertion mip_20000 to finish, then train+eval
# front (pick->handoff) then full (initial->insertion), MSE+MIP x 200/2000/20000.
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
AICHAIN=/tmp/claude-1001/-home-jigu-projects-much-ado-about-noising/469964b2-9364-47f5-8a44-7dda4fc11f85/tasks/b1cyizywm.output

echo "WAIT: for align+insert mip_20000 to finish..."
while ! grep -qa "mip N=20000 INSERTION_SR" "$AICHAIN" 2>/dev/null; do
  sleep 60
done
echo "WAIT: align+insert done, starting front/full chain"

run_cell () {  # $1=segment(front|full) $2=loss $3=N $4=evalscript
  local SEG=$1 LOSS=$2 N=$3 EV=$4
  rm -rf logs/${SEG}_${LOSS}_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_${SEG}_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=${LOSS} optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=${SEG}_${LOSS}_${N} \
    log.log_dir=logs/${SEG}_${LOSS}_${N} > /tmp/${SEG}_${LOSS}_${N}.log 2>&1
  echo "=== ${SEG}_${LOSS}_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/${EV} --ckpt logs/${SEG}_${LOSS}_${N}/models/model_latest.pt \
    --dataset data/tool_hang_${SEG}_${N}.hdf5 --loss ${LOSS} --n 100 2>&1 \
    | grep -aE "FRONT_SR|FULL_SR" | sed "s/^/${SEG} ${LOSS} N=${N} /"
}

# dataset basename uses full2ins for full segment
for LOSS in regression mip; do for N in 200 2000 20000; do
  run_cell front ${LOSS} ${N} eval_front.py
done; done
echo "ALL FRONT DONE"

for LOSS in regression mip; do for N in 200 2000 20000; do
  rm -rf logs/full_${LOSS}_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=${LOSS} optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=full_${LOSS}_${N} \
    log.log_dir=logs/full_${LOSS}_${N} > /tmp/full_${LOSS}_${N}.log 2>&1
  echo "=== full_${LOSS}_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt logs/full_${LOSS}_${N}/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_${N}.hdf5 --loss ${LOSS} --n 100 2>&1 \
    | grep -aE "FULL_SR" | sed "s/^/full ${LOSS} N=${N} /"
done; done
echo "ALL FULL DONE"

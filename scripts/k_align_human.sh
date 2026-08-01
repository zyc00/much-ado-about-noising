#!/bin/bash
# MIP-ALIGNED human-demo training: identical recipe to the official MIP runs
# (300k steps, default optimizer/EMA/etc.) except the loss/reg intervention.
# Checkpoints on the exact 20k grid (save_freq=20000 + fast sidecar), so the
# post-hoc ladder (k_ladder50.sh) reproduces the official in-train eval protocol
# (eval_freq=20000, eval_episodes=50, num_envs=1): best = max over grid,
# last-5 = mean(220k..300k). args: <tag>; env: LOSS, NET, SEED, EXTRA, GSTEPS
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; LOSS=${LOSS:-regression_hetero_t}; NET=${NET:-chiunet}; SEED=${SEED:-1000}
EXTRA=${EXTRA:-}; GSTEPS=${GSTEPS:-300001}
DS=${DATA:-/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5}
rm -rf logs/$TAG
( LAST=-1
  while true; do
    sleep 60
    step=$(python -c "import json;print(json.loads(open('logs/$TAG/metrics.jsonl').readlines()[-1])['step'])" 2>/dev/null) || continue
    [ -z "$step" ] && continue
    grid=$(( step / 20000 * 20000 ))
    [ "$grid" -lt 20000 ] && continue
    [ "$grid" = "$LAST" ] && continue
    cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_${grid}.pt 2>/dev/null || continue
    echo "SNAP saved snap_${grid}.pt"
    LAST=$grid
    [ "$grid" -ge $((GSTEPS-20001)) ] && break
  done ) &
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=$NET \
  task.num_envs=1 optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=$GSTEPS log.save_freq=20000 \
  optimization.seed=$SEED $EXTRA optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG
cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_300000.pt
echo "TRAIN-DONE $TAG"

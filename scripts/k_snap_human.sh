#!/bin/bash
# Human-demo training with step-named snapshots (sidecar copy every ~10 min) + official eval.
# args: <tag>; env: LOSS, NET, SEED
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; LOSS=${LOSS:-regression}; NET=${NET:-chiunet}; SEED=${SEED:-5}; EXTRA=${EXTRA:-}; GSTEPS=${GSTEPS:-300001}
DS=${DATA:-/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5}
rm -rf logs/$TAG
( LAST=-1
  while true; do
    sleep 600
    step=$(ls -t logs/$TAG/models/model_latest.pt 2>/dev/null >/dev/null && python -c "import json;print(json.loads(open('logs/$TAG/metrics.jsonl').readlines()[-1])['step'])" 2>/dev/null) || continue
    [ -z "$step" ] && continue
    [ "$step" = "$LAST" ] && continue
    cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_${step}.pt 2>/dev/null || continue
    echo "SNAP saved snap_${step}.pt"
    LAST=$step
    [ "$step" -ge $((GSTEPS-10001)) ] && break
  done ) &
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=$NET \
  task.num_envs=1 optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 optimization.gradient_steps=$GSTEPS log.save_freq=5000 \
  optimization.seed=$SEED $EXTRA optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG
python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=$NET \
  optimization.loss_type=$LOSS $EXTRA optimization.model_path=$(pwd)/logs/$TAG/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -aE "mean_assembled|mean_success" | sed "s/^/HUMAN_EVAL $TAG | /"

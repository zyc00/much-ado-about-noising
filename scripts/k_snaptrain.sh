#!/bin/bash
# args: <loss_type> <tag> -- fullMSE-style retrain with step-named snapshots + foldsweep
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
LOSS=$1; TAG=$2
DS=data/tool_hang_full2ins_2000.hdf5
rm -rf logs/$TAG
( LAST=-1
  while true; do
    sleep 300
    step=$(python -c "import json;print(json.loads(open('logs/$TAG/metrics.jsonl').readlines()[-1])['step'])" 2>/dev/null) || continue
    [ "$step" = "$LAST" ] && continue
    cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_${step}.pt 2>/dev/null || continue
    LAST=$step
    [ "$step" -ge 295000 ] && break
  done ) &
SNAPPID=$!
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/$DS \
  task.num_envs=1 network=chiunet optimization.auto_resume=false \
  log.wandb_mode=disabled log.eval_freq=100000000 optimization.gradient_steps=300001 \
  log.save_freq=5000 log.log_freq=1000 optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG
kill $SNAPPID 2>/dev/null || true
cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_300000.pt
SNAP_GLOB="logs/$TAG/models/snap_*.pt" LOSS=$LOSS TAG=$TAG QDIR1=/mnt/pfs/yuchen/embq/msebulk QDIR2=/mnt/pfs/yuchen/embq/cfdump_mse2 python -u scripts/probe_foldsweep.py

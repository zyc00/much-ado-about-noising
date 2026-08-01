#!/bin/bash
# args: <loss> <tag> — train with periodic snapshots, then amplitude-bin probe sweep
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
LOSS=$1; TAG=$2
DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_full2ins_2000.hdf5
rm -rf logs/$TAG
( while true; do sleep 1200
    if [ -f logs/$TAG/metrics.jsonl ]; then
      STEP=$(python -c "import json;print([json.loads(l) for l in open('logs/$TAG/metrics.jsonl')][-1]['step'])" 2>/dev/null || echo 0)
      [ -f logs/$TAG/models/model_latest.pt ] && cp logs/$TAG/models/model_latest.pt logs/$TAG/models/snap_$(printf %07d $STEP).pt
    fi
  done ) &
SNAPPID=$!
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet \
  optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 \
  optimization.gradient_steps=300001 log.save_freq=5000 \
  optimization.loss_type=$LOSS log.exp_name=$TAG log.log_dir=logs/$TAG
kill $SNAPPID || true
for f in logs/$TAG/models/snap_*.pt logs/$TAG/models/model_latest.pt; do
  CKPT=$f LOSS=$LOSS TAG=$TAG@$(basename $f .pt) MUJOCO_GL=egl python -u scripts/probe_ampbins.py || true
  if [ "$LOSS" = "mip" ]; then
    CKPT=$f LOSS=mip_step1 TAG=$TAG-step1@$(basename $f .pt) MUJOCO_GL=egl python -u scripts/probe_ampbins.py || true
  fi
done
MUJOCO_GL=egl python scripts/eval_twofactor.py --ckpt logs/$TAG/models/model_latest.pt --loss $LOSS --tag $TAG-final --dataset data/tool_hang_full2ins_2000.hdf5 || true

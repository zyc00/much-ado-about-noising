#!/bin/bash
# Official-protocol ladder: evaluate every 20k-grid snapshot exactly like the
# in-train eval of the original MIP runs (eval_episodes=50, num_envs=1).
# Prints ALIGN_SR <tag> <step> | ... ; then best and last-5 mean.
# args: <tag>; env: LOSS, NET
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; LOSS=${LOSS:-regression_hetero_t}; NET=${NET:-chiunet}; FROM=${FROM:-0}; TO=${TO:-999999}; EXTRA=${EXTRA:-}
DS=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5
for CK in $(ls logs/$TAG/models/snap_*.pt | sort -t_ -k2 -n); do
  STEP=$(basename $CK .pt | cut -d_ -f2)
  [ "$STEP" -lt "$FROM" ] && continue
  [ "$STEP" -gt "$TO" ] && continue
  python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=$NET \
    optimization.loss_type=$LOSS $EXTRA optimization.model_path=$(pwd)/$CK \
    task.num_envs=1 log.eval_episodes=50 log.wandb_mode=disabled log.save_video=false \
    optimization.auto_resume=false 2>&1 | grep -a "mean_success_1 - " | sed "s/^/ALIGN_SR $TAG $STEP | /"
done
echo "ALIGN-LADDER-DONE $TAG"

#!/bin/bash
# Evaluate every snapshot of a run under the official harness; print SR per step.
# args: <tag>; env: LOSS, NET
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; LOSS=${LOSS:-regression}; NET=${NET:-chiunet}
DS=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5
for CK in $(ls logs/$TAG/models/snap_*.pt | sort -t_ -k2 -n); do
  STEP=$(basename $CK .pt | cut -d_ -f2)
  python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=$NET \
    optimization.loss_type=$LOSS optimization.model_path=$(pwd)/$CK \
    task.num_envs=5 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
    optimization.auto_resume=false 2>&1 | grep -a "mean_success_1 - " | sed "s/^/PHASE_SR $TAG $STEP | /"
done
echo "SNAP-EVAL-DONE $TAG"

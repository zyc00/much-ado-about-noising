#!/bin/bash
# Official 100-ep eval of one snapshot. args: <tag> <step>; env: LOSS, NET
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; STEP=$2; LOSS=${LOSS:-regression_hetero_t}; NET=${NET:-chiunet}
DS=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5
python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=$NET \
  optimization.loss_type=$LOSS optimization.model_path=$(pwd)/logs/$TAG/models/$( [ "$STEP" = "best" ] && echo model_best.pt || { [ "$STEP" = "latest" ] && echo model_latest.pt || echo snap_${STEP}.pt; } ) \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -a "mean_success_1 - " | sed "s/^/EVAL100 $TAG $STEP | /"
echo "EVAL100-DONE $TAG $STEP"

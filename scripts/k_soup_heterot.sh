#!/bin/bash
# SWA soup of a run's snapshots (>=MINSTEP) + official mode=eval on the souped model.
# args: <tag> ; env: LOSS (for eval sampler), MINSTEP
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
TAG=$1; LOSS=${LOSS:-regression_hetero_t}; MINSTEP=${MINSTEP:-100000}
DS=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5
ls logs/$TAG/models/snap_*.pt | head -3
SNAPDIR=logs/$TAG/models MINSTEP=$MINSTEP python scripts/soup_eval.py
python -u examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy +task.dataset_path=$DS network=chiunet \
  optimization.loss_type=$LOSS optimization.model_path=$(pwd)/logs/$TAG/models/model_soup.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -aE "mean_assembled|mean_success" | sed "s/^/SOUP_EVAL $TAG | /"

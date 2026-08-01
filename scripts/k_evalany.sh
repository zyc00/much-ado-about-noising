#!/bin/bash
# Cross-task eval of one checkpoint. args: <tag> <ckpt-name>; env: TASK, LOSS, NET, EXTRA, EPS
set -eu
export MUJOCO_GL=egl
TAG=$1; CK=$2; TASK=${TASK:?}; LOSS=${LOSS:-regression_hetero_t}; NET=${NET:-chiunet}
EXTRA=${EXTRA:-}; EPS=${EPS:-50}
python -u examples/train_robomimic.py mode=eval task=$TASK network=$NET \
  optimization.loss_type=$LOSS optimization.model_path=$(pwd)/logs/$TAG/models/$CK $EXTRA \
  log.eval_episodes=$EPS log.wandb_mode=disabled log.save_video=false \
  optimization.auto_resume=false 2>&1 | grep -a "mean_success_1 - " | sed "s/^/EVALANY $TAG $CK | /"
echo "EVALANY-DONE $TAG $CK"

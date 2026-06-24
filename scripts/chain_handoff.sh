#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
# train insertion specialist on handoff (grasp-specialist-output) distribution
rm -rf logs/handoffins_regression_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_handoffins_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=handoffins_regression_2000 \
  log.log_dir=logs/handoffins_regression_2000 > /tmp/handoffins_regression_2000.log 2>&1
echo "=== handoffins_regression_2000 TRAINED ==="
# A) chain: grasp specialist -> handoff-matched insertion specialist
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_regression_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/handoffins_regression_2000/models/model_latest.pt --back_ds data/tool_hang_handoffins_2000.hdf5 \
  --loss regression --demos data/warmstart_demos.hdf5 --n 100 2>&1 | grep -aE "TWOSTAGE_FAITHFUL" | sed 's/^/[handoff-matched] /'
echo "HANDOFF CHAIN DONE"

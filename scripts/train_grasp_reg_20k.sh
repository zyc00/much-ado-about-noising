#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
rm -rf logs/grasp_regression_20000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_init2grasp_20000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=grasp_regression_20000 \
  log.log_dir=logs/grasp_regression_20000 > /tmp/grasp_reg_20k.log 2>&1
echo "GRASP_REG_20K TRAINED"

#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
# phase-indicator model (MSE + phase aux), clean_2000, settle eval
EVAL_SETTLE_STEPS=10 MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_clean_2000.hdf5 network=chiunet \
  +task.phase_indicator=true task.act_dim=13 optimization.loss_type=regression \
  optimization.model_path=$(pwd)/logs/phase_clean_2000/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
  | grep -aE "mean_assembled_1" | grep -a "logger:log" | sed 's/^/PHASE_MSE(clean,settle) | /'
# plain MSE baseline, clean_2000, settle eval
EVAL_SETTLE_STEPS=10 MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_clean_2000.hdf5 network=chiunet \
  optimization.loss_type=regression \
  optimization.model_path=$(pwd)/logs/chi_reg_clean2000/models/model_best.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
  | grep -aE "mean_assembled_1" | grep -a "logger:log" | sed 's/^/PLAIN_MSE(clean,settle) | /'
echo "PHASE EVAL DONE"

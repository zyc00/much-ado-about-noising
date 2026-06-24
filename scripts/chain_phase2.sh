#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
rm -rf logs/phase_clean_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_clean_2000.hdf5 task.num_envs=1 network=chiunet \
  +task.phase_indicator=true task.act_dim=13 \
  optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=phase_clean_2000 \
  log.log_dir=logs/phase_clean_2000 > /tmp/phase_clean_2000.log 2>&1
echo "=== phase_clean_2000 TRAINED ==="
MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_clean_2000.hdf5 network=chiunet \
  +task.phase_indicator=true task.act_dim=13 optimization.loss_type=regression \
  optimization.model_path=$(pwd)/logs/phase_clean_2000/models/model_best.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
  | grep -aE "mean_assembled_1|mean_success_1" | grep -a "logger:log" | sed 's/^/PHASE_clean | /'
echo "PHASE2 DONE"

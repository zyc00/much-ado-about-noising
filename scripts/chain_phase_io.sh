#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
# wait for OUTPUT (phaseout) training to finish
while [ "$(tail -1 logs/phaseout_full2ins_2000/metrics.jsonl 2>/dev/null | python3 -c 'import json,sys;print(json.loads(sys.stdin.read()).get("step",0))' 2>/dev/null)" != "299999" ]; do sleep 60; done
sleep 20
# eval OUTPUT (phase-as-output), settle
EVAL_SETTLE_STEPS=10 MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_full2ins_2000.hdf5 network=chiunet \
  +task.phase_indicator=true task.act_dim=13 optimization.loss_type=regression \
  optimization.model_path=$(pwd)/logs/phaseout_full2ins_2000/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
  | grep -aE "mean_assembled_1" | grep -a "logger:log" | sed 's/^/PHASE_OUTPUT(full2ins) | /'
# train INPUT (phase-as-input), obs_dim 56
rm -rf logs/phasein_full2ins_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_full2ins_2000.hdf5 task.num_envs=1 network=chiunet \
  +task.phase_input=true task.obs_dim=56 \
  optimization.loss_type=regression optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=phasein_full2ins_2000 \
  log.log_dir=logs/phasein_full2ins_2000 > /tmp/phasein_full2ins_2000.log 2>&1
echo "=== phasein TRAINED ==="
# eval INPUT (oracle phase + settle)
EVAL_SETTLE_STEPS=10 EVAL_PHASE_INPUT=1 MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_full2ins_2000.hdf5 network=chiunet \
  +task.phase_input=true task.obs_dim=56 optimization.loss_type=regression \
  optimization.model_path=$(pwd)/logs/phasein_full2ins_2000/models/model_latest.pt \
  task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
  | grep -aE "mean_assembled_1" | grep -a "logger:log" | sed 's/^/PHASE_INPUT(full2ins,oracle) | /'
echo "PHASE_IO DONE"

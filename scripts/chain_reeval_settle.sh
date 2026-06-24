#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
ev () {  # $1=logdir $2=loss $3=N $4=tag
  EVAL_SETTLE_STEPS=10 MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_${3}.hdf5 network=chiunet optimization.loss_type=$2 \
    optimization.model_path=$(pwd)/logs/$1/models/model_latest.pt \
    task.num_envs=1 log.eval_episodes=100 log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
    | grep -aE "mean_assembled_1" | grep -a "logger:log" | sed "s/^/$4 /"
}
ev full_regression_200   regression 200   "MSE_200"
ev full_regression_2000  regression 2000  "MSE_2000"
ev full_regression_20000 regression 20000 "MSE_20000"
ev full_mip_200          mip 200   "MIP_200"
ev full_mip_2000         mip 2000  "MIP_2000"
ev full_mip_20000        mip 20000 "MIP_20000(undertrained-133k)"
echo "REEVAL DONE"

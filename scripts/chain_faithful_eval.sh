#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
run () {  # $1=ckpt $2=dataset $3=loss $4=label
  echo ">>>> $4"
  MUJOCO_GL=egl $PY examples/train_robomimic.py mode=eval task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/$2 network=chiunet optimization.loss_type=$3 \
    optimization.model_path=$(pwd)/$1 task.num_envs=1 log.eval_episodes=100 \
    log.wandb_mode=disabled log.save_video=false optimization.auto_resume=false 2>&1 \
    | grep -aE "mean_assembled_|mean_success_" | grep -aE "logger:log" | sed "s/^/$4 | /"
}
run logs/chi_mip_clean/models/model_best.pt       data/tool_hang_markovian_200.hdf5 mip        "MIP_200"
run logs/chi_reg_clean/models/model_best.pt       data/tool_hang_markovian_200.hdf5 regression "MSE_200"
run logs/chi_mip_clean2000/models/model_best.pt   data/tool_hang_clean_2000.hdf5    mip        "MIP_2000"
run logs/chi_reg_clean2000/models/model_best.pt   data/tool_hang_clean_2000.hdf5    regression "MSE_2000"
run logs/chi_mip_clean20000/models/model_best.pt  data/tool_hang_clean_20000.hdf5   mip        "MIP_20000"
run logs/chi_reg_clean20000/models/model_best.pt  data/tool_hang_clean_20000.hdf5   regression "MSE_20000"
echo "ALL FAITHFUL EVAL DONE"

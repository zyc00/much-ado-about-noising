#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python; D=data/warmstart_demos.hdf5
train_eval () {  # $1=loss $2=tag
  rm -rf logs/grasp_${1}_2000
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_init2grasp_2000.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=$1 optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=grasp_${1}_2000 \
    log.log_dir=logs/grasp_${1}_2000 > /tmp/grasp_${1}_2000.log 2>&1
  echo "=== grasp_${1}_2000 TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/grasp_${1}_2000/models/model_latest.pt \
    --dataset data/tool_hang_init2grasp_2000.hdf5 --loss $1 --demos $D --warm_to 0 --success grasp --n 100 2>&1 \
    | grep -aE "WARMSTART" | sed "s/^/SPECIALIST ${2} grasp /"
}
train_eval regression MSE
train_eval mip MIP
echo "GRASP SPECIALIST DONE"

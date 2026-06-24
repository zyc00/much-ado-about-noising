#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
train(){  # $1 dataset_basename  $2 exp
  rm -rf logs/$2
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/$1 task.num_envs=1 network=chiunet optimization.loss_type=regression \
    optimization.auto_resume=false log.wandb_mode=disabled log.eval_freq=100000000 log.save_freq=100000 \
    log.exp_name=$2 log.log_dir=logs/$2 > /tmp/$2.log 2>&1
  echo "TRAINED $2"
}
bd(){  # $1 exp  $2 dataset  $3 tag
  MUJOCO_GL=egl $PY scripts/eval_grasp_breakdown.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss regression --tag "$3" --n 50 2>&1 | grep -aE "GRASPBD"
}
sr(){  # $1 exp  $2 dataset  $3 tag
  MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss regression --demos data/warmstart_demos.hdf5 --warm_to 0 --success grasp --init_mode reset_settle --settle 10 --n 100 2>&1 | grep -aE "WARMSTART" | sed "s/^/GRASPSR $3 /"
}
train tool_hang_init2graspP40_2000.hdf5 grasp_reg_ext40_2000
train tool_hang_init2align_2000.hdf5    grasp_reg_align_2000
echo "=== EXTENSION ABLATION EVAL ==="
bd grasp_regression_2000   tool_hang_init2grasp_2000.hdf5    "ext20(c1+20,baseline)"
bd grasp_reg_ext40_2000    tool_hang_init2graspP40_2000.hdf5 "ext40(c1+40)"
bd grasp_reg_align_2000    tool_hang_init2align_2000.hdf5    "extAlign(af0)"
sr grasp_regression_2000   tool_hang_init2grasp_2000.hdf5    "ext20"
sr grasp_reg_ext40_2000    tool_hang_init2graspP40_2000.hdf5 "ext40"
sr grasp_reg_align_2000    tool_hang_init2align_2000.hdf5    "extAlign"
echo "EXT ABLATION DONE"

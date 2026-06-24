#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
D=data/tool_hang_full2ins_2000.hdf5   # same full trajectories (train seeds) for all
run () { MUJOCO_GL=egl $PY scripts/eval_train_loss.py --ckpt "$1" --dataset "$2" --loss "$3" --demos $D --tag "$4" --n 50 --stride 2 2>&1 | grep -aE "TRAINLOSS"; }
run logs/full_regression_2000/models/model_latest.pt     data/tool_hang_full2ins_2000.hdf5    regression "[MSE_generalist]"
run logs/full_mip_2000/models/model_latest.pt            data/tool_hang_full2ins_2000.hdf5    mip        "[MIP_generalist]"
run logs/grasp_regression_2000/models/model_latest.pt    data/tool_hang_init2grasp_2000.hdf5  regression "[MSE_grasp_spec]"
run logs/grasp_mip_2000/models/model_latest.pt           data/tool_hang_init2grasp_2000.hdf5  mip        "[MIP_grasp_spec]"
run logs/pick2ins_regression_2000/models/model_latest.pt data/tool_hang_pick2ins_2000.hdf5    regression "[MSE_insert_spec]"
run logs/pick2ins_mip_2000/models/model_latest.pt        data/tool_hang_pick2ins_2000.hdf5    mip        "[MIP_insert_spec]"
echo "TRAINLOSS ALL DONE"

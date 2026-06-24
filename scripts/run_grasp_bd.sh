#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
run(){ MUJOCO_GL=egl $PY scripts/eval_grasp_breakdown.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss $3 --tag "$4" --n 50 2>&1 | grep -aE "GRASPBD"; }
run full_regression_2000   tool_hang_full2ins_2000.hdf5   regression "MSE_gen"
run grasp_regression_2000  tool_hang_init2grasp_2000.hdf5 regression "MSE_graspSpec"
run full_mip_2000          tool_hang_full2ins_2000.hdf5   mip        "MIP_gen"
run grasp_mip_2000         tool_hang_init2grasp_2000.hdf5 mip        "MIP_graspSpec"
echo "GRASPBD ALL DONE"

#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
run () { MUJOCO_GL=egl $PY scripts/eval_val_loss.py --ckpt "$1" --dataset "$2" --loss "$3" --tag "$4" --n 50 --stride 2 2>&1 | grep -aE "VALLOSS"; }
run logs/full_regression_20000/models/model_latest.pt     data/tool_hang_full2ins_20000.hdf5    regression "[MSE_gen_20k]"
run logs/full_mip_20000/models/model_latest.pt            data/tool_hang_full2ins_20000.hdf5    mip        "[MIP_gen_20k]"
run logs/pick2ins_regression_20000/models/model_latest.pt data/tool_hang_pick2ins_20000.hdf5   regression "[MSE_insert_spec_20k]"
echo "VAL20K EXISTING DONE"

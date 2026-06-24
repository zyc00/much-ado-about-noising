#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
run(){ MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss $3 --demos data/warmstart_demos.hdf5 --warm_to 0 --init_mode reset_settle --settle 10 --success assembled --n 100 2>&1 | grep -aE "WARMSTART" | sed "s/^/BASE2k $4 /"; }
run full_mip_2000        tool_hang_full2ins_2000.hdf5 mip        MIP_gen_2k_warm0
run full_regression_2000 tool_hang_full2ins_2000.hdf5 regression MSE_gen_2k_warm0
echo "BASE2k DONE"

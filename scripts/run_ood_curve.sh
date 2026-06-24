#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
run(){ MUJOCO_GL=egl $PY scripts/eval_ood_perturb.py --ckpt logs/$1/models/model_latest.pt --dataset data/$2 --loss $3 --mag $4 --psteps 5 --n 40 --tag "$5" 2>&1 | grep -aE "OODPERTURB"; }
for MAG in 0.0 0.15 0.3 0.5; do
  run pick2ins_regression_2000 tool_hang_pick2ins_2000.hdf5 regression $MAG "MSEins"
  run pick2ins_mip_2000        tool_hang_pick2ins_2000.hdf5 mip        $MAG "MIPins"
done
echo "OOD CURVE DONE"

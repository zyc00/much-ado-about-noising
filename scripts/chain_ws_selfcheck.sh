#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
D=data/warmstart_demos.hdf5
# (a) mechanism: replay all, no policy -> should ~100%
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/chi_mip_clean2000/models/model_best.pt --dataset data/tool_hang_clean_2000.hdf5 --loss mip --demos $D --warm_to all --n 100 2>&1 | grep -aE "WARMSTART"
# (b) cross-check standalone==official: chi_mip full, warm_to=0 -> expect ~41%
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/chi_mip_clean2000/models/model_best.pt --dataset data/tool_hang_clean_2000.hdf5 --loss mip --demos $D --warm_to 0 --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/$/  (expect ~41 = official init->ins)/'
# (c) chi_mip full, warm_to=align_done -> expect high (insertion from align easy)
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/chi_mip_clean2000/models/model_best.pt --dataset data/tool_hang_clean_2000.hdf5 --loss mip --demos $D --warm_to align_done --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/$/  (expect ~99 insertion-only)/'
# (d) KEY: existing pick2ins SLICED model, warm_to=c1 -> expect >>0 (set_state gave 0)
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/pick2ins_regression_2000/models/model_latest.pt --dataset data/tool_hang_pick2ins_2000.hdf5 --loss regression --demos $D --warm_to c1 --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/$/  (pick2ins sliced; set_state gave 0)/'
echo "SELFCHECK DONE"

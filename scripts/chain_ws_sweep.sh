#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python; D=data/warmstart_demos.hdf5
sweep () {  # $1=ckpt $2=dataset $3=loss $4=tag
  for W in 0 mid_reach c1 mid_align align_done; do
    MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt $1 --dataset $2 --loss $3 --demos $D --warm_to $W --n 100 2>&1 \
      | grep -aE "WARMSTART" | sed "s/^/$4 /"
  done
}
sweep logs/chi_mip_clean2000/models/model_best.pt data/tool_hang_clean_2000.hdf5 mip        "MIP"
sweep logs/chi_reg_clean2000/models/model_best.pt data/tool_hang_clean_2000.hdf5 regression "MSE"
echo "WS SWEEP DONE"

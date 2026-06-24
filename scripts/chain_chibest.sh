#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
PY=./.venv/bin/python
SEEDS=data/full_eval_seeds.npy
run () { MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt $1 --dataset $2 --loss $3 --seeds_file $SEEDS --n 200 --max_chunks 40 2>&1 | grep -aE "FULL_SR" | sed "s/^/$4: /"; }
run logs/chi_mip_clean/models/model_best.pt       data/tool_hang_markovian_200.hdf5 mip        "BEST chi_MIP N=200"
run logs/chi_reg_clean/models/model_best.pt       data/tool_hang_markovian_200.hdf5 regression "BEST chi_MSE N=200"
run logs/chi_mip_clean2000/models/model_best.pt   data/tool_hang_clean_2000.hdf5    mip        "BEST chi_MIP N=2000"
run logs/chi_reg_clean2000/models/model_best.pt   data/tool_hang_clean_2000.hdf5    regression "BEST chi_MSE N=2000"
run logs/chi_mip_clean20000/models/model_best.pt  data/tool_hang_clean_20000.hdf5   mip        "BEST chi_MIP N=20000"
run logs/chi_reg_clean20000/models/model_best.pt  data/tool_hang_clean_20000.hdf5   regression "BEST chi_MSE N=20000"
echo "ALL CHIBEST DONE"

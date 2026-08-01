#!/bin/bash
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
RV_DATA=data/tool_hang_full2ins_2000.hdf5 \
RV_ARMS="L2-2K:regression:logs/full_regression_2000/models/model_latest.pt,MIP-2K:mip:logs/full_mip_2000_s1/models/model_latest.pt" \
  python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD|Traceback" -A4
echo "MED2K done"

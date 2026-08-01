#!/bin/bash
# Avoidance/robustness probe with cluster paths baked in. args: names...
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
export DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5
export NAMES="$*"
python -u scripts/probe_offsup_compare.py
echo "OFFSUP-DONE"

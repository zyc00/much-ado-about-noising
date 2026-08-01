#!/bin/bash
# Recovery-gain mediator sweep: existing intervention arms (200-normalizers)
# + MP-2k arms (2k-normalizers). Prints RD lines.
set -u
cd /mnt/pfs/yuchen/code/much-ado-about-noising
source .venv/bin/activate
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
RV_ARMS="CND2A:regression_condreg:logs/mp200_cnd2a_s1000/models/snap_300000.pt,CND2B:regression_condreg:logs/mp200_cnd2b_s1000/models/snap_300000.pt,NONOISE:mip_nonoise:logs/mp200_nonoise_s1000/models/snap_300000.pt" \
  python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD|Traceback"
RV_DATA=data/tool_hang_full2ins_2000.hdf5 \
RV_ARMS="L2-2K:regression:logs/full_regression_2000/models/snap_300000.pt,MIP-2K:mip:logs/full_mip_2000/models/snap_300000.pt" \
  python -u scripts/probe_recovdir.py 2>&1 | grep -aE "^RD|Traceback"
echo "MEDSWEEP done"

#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
cd /mnt/pfs/yuchen/code/much-ado-about-noising
i=0
for f in $(ls -tr logs/orig_fadehint/models/snap_*.pt); do
  i=$((i+1))
  CKPT=$f LOSS=regression_fadehint TAG=fadehint-snap$i-$(basename $f .pt) python -u scripts/probe_ampbins.py || true
done

#!/bin/bash
# Gradient-stream anatomy across training phases: run probe_grad_stream at the
# given snapshots of one run, all four loss views at each point.
# args: <tag> <step1> [step2 ...]; env: GLOSSES (default all four)
set -eu
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
export DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_human_lowdim_up.hdf5
TAG=$1; shift
export GLOSSES="${GLOSSES:-L2 Cauchy HT MIPv2}"
for STEP in "$@"; do
  CKPT=logs/$TAG/models/snap_${STEP}.pt TAG=${TAG}@${STEP} python -u scripts/probe_grad_stream.py
done
echo "GRADPHASE-DONE $TAG"

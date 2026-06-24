#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
# MSE stitching (expect some FAILs) on first 6 held-out seeds
MUJOCO_GL=egl $PY scripts/render_stitch_faithful.py \
  --grasp_ckpt logs/grasp_regression_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/pick2ins_regression_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss regression --demos data/warmstart_demos.hdf5 --n 6 --out analysis/videos/MSE_stitch 2>&1 | grep -aE "RENDERED"
# MIP stitching (expect OK) on the SAME 6 seeds
MUJOCO_GL=egl $PY scripts/render_stitch_faithful.py \
  --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/pick2ins_mip_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss mip --demos data/warmstart_demos.hdf5 --n 6 --out analysis/videos/MIP_stitch 2>&1 | grep -aE "RENDERED"
echo "RENDER DONE"

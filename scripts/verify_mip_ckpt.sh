#!/bin/bash
set -u; cd /home/jigu/projects/much-ado-about-noising; PY=./.venv/bin/python
# same pick2ins_mip_2000 ckpt, sampled two ways
for L in mip regression; do
  MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/pick2ins_mip_2000/models/model_latest.pt \
    --dataset data/tool_hang_pick2ins_2000.hdf5 --loss $L --demos data/warmstart_demos.hdf5 \
    --warm_to c1 --init_mode state0 --success assembled --n 50 2>&1 | grep -aE "WARMSTART" | sed "s/^/CKPTCHECK pick2ins_mip_ckpt+sampler=$L /"
done
echo "CKPTCHECK DONE"

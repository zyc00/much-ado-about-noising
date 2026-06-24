#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
rm -rf logs/rr80_mip_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_rr80_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=rr80_mip_2000 \
  log.log_dir=logs/rr80_mip_2000 > /tmp/rr80_mip_2000.log 2>&1
echo "=== rr80_mip_2000 TRAINED ==="
for tag in insertion b40 b50 b60 b70 pick range80; do
  MUJOCO_GL=egl $PY scripts/eval_insertion.py --ckpt logs/rr80_mip_2000/models/model_latest.pt \
    --dataset data/tool_hang_rr80_2000.hdf5 --eval_states data/${tag}_eval_states.hdf5 \
    --loss mip --n 100 --max_chunks 40 2>&1 | grep -aE "INSERTION_SR" | sed "s/^/RR80MIP on ${tag}: /"
done
echo "ALL RR80MIP DONE"

#!/bin/bash
# After b40 eval finishes, skip the redundant b70 redo (already have b70=0%) and
# go straight to full_mip 200/2000/20000.
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
W=/tmp/claude-1001/-home-jigu-projects-much-ado-about-noising/469964b2-9364-47f5-8a44-7dda4fc11f85/tasks/b40farnf6.output

echo "WAIT: for b40 eval..."
while ! grep -qa "Bsweep B=40" "$W" 2>/dev/null; do sleep 30; done
echo "WAIT: b40 done -> killing b40insert chain (skip redundant b70), start full_mip"
pkill -f chain_b40insert.sh 2>/dev/null
pkill -f "tool_hang_b70_2000" 2>/dev/null
sleep 5

for N in 200 2000 20000; do
  rm -rf logs/full_mip_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=full_mip_${N} \
    log.log_dir=logs/full_mip_${N} > /tmp/full_mip_${N}.log 2>&1
  echo "=== full_mip_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt logs/full_mip_${N}/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_${N}.hdf5 --loss mip --n 100 2>&1 \
    | grep -aE "FULL_SR" | sed "s/^/full mip N=${N} /"
done
echo "ALL FULL_MIP DONE"

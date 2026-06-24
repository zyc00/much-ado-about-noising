#!/bin/bash
set -u
cd /home/jigu/projects/much-ado-about-noising
unset WAYPOINT WP_SPACING DART_DIST DART_POS DART_ACT IMPULSE DART_GAIN DART_P SKEW_TILT_DEG MISALIGN MISALIGN_PROB INSERT_RETRIES GRASP_RETRIES
PY=./.venv/bin/python
SEEDS=data/full_eval_seeds.npy
# wait for held-out eval seeds
while [ ! -f "$SEEDS" ]; do sleep 20; done

# re-eval existing MSE full2ins models on the 200 held-out seeds (natural reset)
for N in 200 2000 20000; do
  MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt logs/full_regression_${N}/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_${N}.hdf5 --loss regression --seeds_file $SEEDS --n 200 --max_chunks 35 2>&1 \
    | grep -aE "FULL_SR" | sed "s/^/INIT2INS MSE N=${N}: /"
done

# train MIP on full2ins and eval on the same 200 seeds
for N in 200 2000 20000; do
  rm -rf logs/full_mip_${N}
  MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_${N}.hdf5 task.num_envs=1 network=chiunet \
    optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
    log.eval_freq=100000000 log.save_freq=100000 log.exp_name=full_mip_${N} \
    log.log_dir=logs/full_mip_${N} > /tmp/full_mip_${N}.log 2>&1
  echo "=== full_mip_${N} TRAINED ==="
  MUJOCO_GL=egl $PY scripts/eval_full.py --ckpt logs/full_mip_${N}/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_${N}.hdf5 --loss mip --seeds_file $SEEDS --n 200 --max_chunks 35 2>&1 \
    | grep -aE "FULL_SR" | sed "s/^/INIT2INS MIP N=${N}: /"
done
echo "ALL INIT2INS MSE+MIP DONE"

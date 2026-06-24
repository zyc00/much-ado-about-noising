#!/bin/bash
# Runs INSIDE the krun GPU pod (cwd=repo, venv active). Train MIP insertion
# specialist, then eval MIP no-dagger stitching (grasp_mip + pick2ins_mip),
# to compare against the MSE no-dagger stitching = 42%.
set -u
PY=python
# 1. train MIP insertion specialist (pick2ins, loss=mip)
rm -rf logs/pick2ins_mip_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_pick2ins_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=pick2ins_mip_2000 \
  log.log_dir=logs/pick2ins_mip_2000
echo "=== pick2ins_mip_2000 TRAINED ==="
# 2. sanity: MIP insertion specialist from clean c1 (expect ~100, == MSE 100)
MUJOCO_GL=egl $PY scripts/eval_warmstart.py --ckpt logs/pick2ins_mip_2000/models/model_latest.pt \
  --dataset data/tool_hang_pick2ins_2000.hdf5 --loss mip --demos data/warmstart_demos.hdf5 \
  --warm_to c1 --init_mode state0 --success assembled --n 100 2>&1 | grep -aE "WARMSTART" | sed 's/^/MIP_INS_from_c1 /'
# 3. MIP no-dagger stitching: grasp_mip + pick2ins_mip (compare vs MSE 42%)
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
  --back_ckpt logs/pick2ins_mip_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
  --loss mip --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed 's/^/MIP_STITCH_nodagger /'
echo "MIP STITCH DONE"

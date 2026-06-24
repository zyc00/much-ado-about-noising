#!/bin/bash
# Cluster-native DAgger-MIP stitching pipeline (runs INSIDE krun GPU pod).
# Zero large transfer: only init2grasp_2000 + warmstart_demos needed on PVC.
#   1) train grasp_mip from init2grasp (model not on PVC -> retrain fresh)
#   2) collect handoffins_mip from grasp_mip's grasp distribution
#   3) train handoffins_mip (MIP insertion on matched-handoff data)
#   4) eval DAgger-MIP matched-handoff stitching (vs MIP no-dagger 91, MSE dagger 97)
set -eu
# Fully local pipeline: never touch HuggingFace (cluster proxy can't HTTPS to HF).
# Fail fast instead of hanging on the proxy if any stray HF call slips through.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python
ID=data/tool_hang_init2grasp_2000.hdf5
# 1) grasp_mip (always retrain fresh; avoid partial-file traps)
rm -rf logs/grasp_mip_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/$ID task.num_envs=1 network=chiunet \
  optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=grasp_mip_2000 \
  log.log_dir=logs/grasp_mip_2000
test -f logs/grasp_mip_2000/models/model_latest.pt || { echo "grasp_mip train FAILED"; exit 1; }
echo "=== grasp_mip READY ==="
# 2) collect handoffins_mip: grasp_mip drives grasp from settled init, scripted insertion recorded
MUJOCO_GL=egl $PY scripts/collect_handoff_demos.py \
  --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt \
  --grasp_ds $ID --grasp_loss mip --src $ID --n_demos 2000 \
  --output data/tool_hang_handoffins_mip_2000.hdf5
echo "=== handoffins_mip COLLECTED ==="
# 3) train MIP insertion specialist on the matched-handoff data
rm -rf logs/handoffins_mip_2000
MUJOCO_GL=egl $PY -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$(pwd)/data/tool_hang_handoffins_mip_2000.hdf5 task.num_envs=1 network=chiunet \
  optimization.loss_type=mip optimization.auto_resume=false log.wandb_mode=disabled \
  log.eval_freq=100000000 log.save_freq=100000 log.exp_name=handoffins_mip_2000 \
  log.log_dir=logs/handoffins_mip_2000
echo "=== handoffins_mip TRAINED ==="
# 4) eval DAgger-MIP matched-handoff stitching
MUJOCO_GL=egl $PY scripts/eval_twostage_faithful.py \
  --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt --grasp_ds $ID \
  --back_ckpt logs/handoffins_mip_2000/models/model_latest.pt --back_ds data/tool_hang_handoffins_mip_2000.hdf5 \
  --loss mip --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100 2>&1 \
  | grep -aE "TWOSTAGE_FAITHFUL" | sed 's/^/MIP_STITCH_dagger /'
echo "DAGGER MIP DONE"

#!/bin/bash
# Oracle-phase MSE arm: build phase dataset if missing, train 300k, census + Jacobian probe.
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
DSP=data/tool_hang_full2ins_2000_phase.hdf5
[ -f $DSP ] || python -u scripts/make_phase_dataset.py
TAG=orig_phasemse
rm -rf logs/$TAG
OBS_DIM_OVERRIDE=56 python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=/mnt/pfs/yuchen/code/much-ado-about-noising/$DSP \
  ++task.obs_dim=56 task.num_envs=1 network=chiunet optimization.auto_resume=false \
  log.wandb_mode=disabled log.eval_freq=100000000 optimization.gradient_steps=300001 \
  log.save_freq=5000 optimization.loss_type=regression log.exp_name=$TAG log.log_dir=logs/$TAG
CKPT=logs/$TAG/models/model_latest.pt DS=$DSP TAG=phasemse python -u scripts/grasp_census_phase.py
DS=$DSP CKPT=logs/$TAG/models/model_latest.pt LOSS=regression TAG=phasemse PHASE_PAD=1,0,0 python -u scripts/probe_jac2.py

#!/bin/bash
set -eu
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
DSP=/mnt/pfs/yuchen/code/much-ado-about-noising/data/tool_hang_full2ins_wpmatch_2000.hdf5
# stage 1: denoiser only, 150k
rm -rf logs/wpm_dseq
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet optimization.auto_resume=false \
  log.wandb_mode=disabled log.eval_freq=100000000 optimization.gradient_steps=150001 \
  log.save_freq=5000 optimization.loss_type=denoise_only log.exp_name=wpm_dseq log.log_dir=logs/wpm_dseq
# stage 2: distill step1 from the denoiser, warm-start, 150k more
python -u examples/train_robomimic.py task=tool_hang_ph_state_delta_legacy \
  +task.dataset_path=$DSP task.num_envs=1 network=chiunet optimization.auto_resume=false \
  optimization.model_path=logs/wpm_dseq/models/model_latest.pt \
  log.wandb_mode=disabled log.eval_freq=100000000 optimization.gradient_steps=300001 \
  log.save_freq=5000 optimization.loss_type=mip_distill log.exp_name=wpm_dseq log.log_dir=logs/wpm_dseq
echo DSEQ-DONE

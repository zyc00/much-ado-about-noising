#!/usr/bin/env bash
# Dataset name is the sole positional argument; Slurm resources are submitted externally.
set -euo pipefail
dataset="${1:?expected widowx, fractal, or gr1}"
case "$dataset" in widowx|fractal|gr1) ;; *) exit 2 ;; esac
root=/vast/projects/jiayuanm/mao-lab/yuchen
module load anaconda3/2023.09-0
export PYTHONPATH="$root/code/hf-packages"
export HF_TOKEN_PATH=/vast/home/z/zyc0187/.cache/huggingface/token
export HF_HOME="$root/code/hf-cache/$dataset"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$root/code/logs" "$root/data/_hf_download_20260908"
python "$root/code/parcc_hf_download.py" "$dataset"

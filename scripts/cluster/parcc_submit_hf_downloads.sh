#!/usr/bin/env bash
set -euo pipefail
root=/vast/projects/jiayuanm/mao-lab/yuchen
module load anaconda3/2023.09-0
export PYTHONPATH="$root/code/hf-packages"
export HF_TOKEN_PATH=/vast/home/z/zyc0187/.cache/huggingface/token
python -c "from huggingface_hub import whoami; x=whoami(); assert x.get('name') == 'yuchen0187' and x.get('isPro') is True; print('HF Pro authentication verified')"
chmod 750 "$root/code/parcc_hf_download.slurm"
chmod 640 "$root/code/parcc_hf_download.py"
mkdir -p "$root/code/logs" "$root/data/_hf_download_20260908"
for dataset in widowx fractal gr1; do
    job=$(sbatch --parsable \
        --account=jiayuanm-mao-lab \
        --partition=genoa-std-mem \
        --cpus-per-task=8 \
        --time=2-00:00:00 \
        --job-name="hf-$dataset" \
        --output="$root/code/logs/hf-$dataset-%j.out" \
        --chdir="$root/code" \
        "$root/code/parcc_hf_download.slurm" "$dataset")
    printf '%s %s\n' "$dataset" "$job"
done

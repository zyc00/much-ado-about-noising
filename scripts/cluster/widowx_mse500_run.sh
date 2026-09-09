#!/usr/bin/env bash
set -euo pipefail
root=/mnt/pfs/yuchen/groot
probe_dir="$root/mse500_tail_20260907"
run_dir="$root/ft_wxmse500_20260907"
cd "$root/Isaac-GR00T"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD:$probe_dir"
export HF_HOME=/mnt/pfs/yuchen/hf_home WANDB_MODE=disabled
export GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
mkdir "$run_dir"
mkdir "$run_dir/audit"
cp "$root/ft_wxmse/experiment_cfg/config.yaml" "$run_dir/audit/reference_config.yaml"
cp "$probe_dir/widowx_mse500_launch.py" "$run_dir/audit/launcher.py"
cp gr00t/model/gr00t_n1d7/gr00t_n1d7.py "$run_dir/audit/model_source.py"
cp gr00t/experiment/trainer.py "$run_dir/audit/trainer_source.py"
mkdir /dev/shm/wxmse500
cp -r "$root/bridge_orig_lerobot/." /dev/shm/wxmse500/
episode_count=$(wc -l < /dev/shm/wxmse500/meta/episodes.jsonl)
test "$episode_count" -eq 53192
echo "STAGED $episode_count episodes"
torchrun --nproc_per_node=8 --master_port=29500 "$probe_dir/widowx_mse500_launch.py" \
  --base_model_path "$root/model" --dataset_path /dev/shm/wxmse500 \
  --embodiment_tag SIMPLER_ENV_WIDOWX --num_gpus 8 --output_dir "$run_dir" \
  --save_steps 1000 --save_total_limit 5 --max_steps 20000 \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size 1024 --dataloader_num_workers 8 \
  --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --no-use-wandb --loss-type=mse --state-dropout-prob 0.8 \
  2>&1 | tee "$probe_dir/train.log"
python - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'checkpoint-500'
assert json.loads((p/'trainer_state.json').read_text())['global_step'] == 500
assert json.loads((p/'config.json').read_text())['loss_type'] == 'mse'
print('MSE500_CHECKPOINT_VERIFIED', flush=True)
PY
CUDA_VISIBLE_DEVICES=0 python "$probe_dir/probe_widowx_early_tail.py" \
  --checkpoint "$run_dir/checkpoint-500" --objective mse --training-steps 500 \
  --reference /mnt/pfs/yuchen/widowx_general_scale_20260906/raw \
  --output "$probe_dir/results" 2>&1 | tee "$probe_dir/probe.log"

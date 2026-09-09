#!/usr/bin/env bash
# Fresh replay of the original MSE recipe, saving the requested early snapshots.
set -euo pipefail
root=/mnt/pfs/yuchen/groot
probe_dir="$root/mse250_tail_20260907"
run_dir="$root/ft_wxmse250_20260907"
cd "$root/Isaac-GR00T"
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD:$probe_dir"
export HF_HOME=/mnt/pfs/yuchen/hf_home WANDB_MODE=disabled
export GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export MSE_PROBE_STOP_STEP=250 MSE_PROBE_SAVE_STEPS=50,100,200,250
mkdir "$run_dir"
mkdir "$run_dir/audit"
cp "$root/ft_wxmse/experiment_cfg/config.yaml" "$run_dir/audit/reference_config.yaml"
cp "$probe_dir/widowx_mse500_launch.py" "$run_dir/audit/launcher.py"
cp "$probe_dir/widowx_mse250_run.sh" "$run_dir/audit/run.sh"
cp gr00t/model/gr00t_n1d7/gr00t_n1d7.py "$run_dir/audit/model_source.py"
cp gr00t/experiment/trainer.py "$run_dir/audit/trainer_source.py"
mkdir /dev/shm/wxmse250
cp -r "$root/bridge_orig_lerobot/." /dev/shm/wxmse250/
episode_count=$(wc -l < /dev/shm/wxmse250/meta/episodes.jsonl)
test "$episode_count" -eq 53192
echo "STAGED $episode_count episodes"
torchrun --nproc_per_node=8 --master_port=29500 "$probe_dir/widowx_mse500_launch.py" \
  --base_model_path "$root/model" --dataset_path /dev/shm/wxmse250 \
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
for step in (50, 100, 200, 250):
    p = Path(sys.argv[1]) / f'checkpoint-{step}'
    assert json.loads((p/'trainer_state.json').read_text())['global_step'] == step
    assert json.loads((p/'config.json').read_text())['loss_type'] == 'mse'
    print(f'MSE_CHECKPOINT_VERIFIED {step}', flush=True)
PY
pids=()
steps=(50 100 200 250)
for gpu in "${!steps[@]}"; do
  step=${steps[$gpu]}
  CUDA_VISIBLE_DEVICES="$gpu" python "$probe_dir/probe_widowx_early_tail.py" \
    --checkpoint "$run_dir/checkpoint-$step" --objective mse --training-steps "$step" \
    --reference /mnt/pfs/yuchen/widowx_general_scale_20260906/raw \
    --output "$probe_dir/results/$step" > "$probe_dir/probe_$step.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
test "$failed" -eq 0
echo ALL_FOUR_PROBES_COMPLETE

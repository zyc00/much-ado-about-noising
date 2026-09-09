#!/usr/bin/env bash
set -euo pipefail
dataset=$1
root=/mnt/pfs/yuchen/groot
scripts="$root/nu_hold7k_20260908"
case "$dataset" in
  widowx)
    short=wx; source_data=bridge_orig_lerobot; expected=53192
    tag=SIMPLER_ENV_WIDOWX; sbias=-0.4535; dropout=0.8
    reference=ft_wxnu224 ;;
  fractal)
    short=fr; source_data=fractal_lerobot; expected=87212
    tag=SIMPLER_ENV_GOOGLE; sbias=-0.5093; dropout=0.5
    reference=ft_fr_nu224 ;;
  *) exit 2 ;;
esac
run_dir="$root/ft_${short}_nu1024_hold7k_to14_20260908"
staged="/dev/shm/${short}_nustair"
cd "$root/Isaac-GR00T"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD:$scripts"
export HF_HOME=/mnt/pfs/yuchen/hf_home WANDB_MODE=disabled
export GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
mkdir "$run_dir"
mkdir "$run_dir/audit"
cp "$root/$reference/experiment_cfg/config.yaml" "$run_dir/audit/reference_config.yaml"
cp "$scripts/nu_hold7k_launch.py" "$scripts/nu_hold7k_train.sh" "$scripts/nu_hold7k_eval.py" "$run_dir/audit/"
cp gr00t/model/gr00t_n1d7/gr00t_n1d7.py "$run_dir/audit/model_source.py"
cp gr00t/experiment/trainer.py "$run_dir/audit/trainer_source.py"
if [[ ! -f "$staged/meta/episodes.jsonl" ]]; then
  mkdir -p "$staged"
  cp -rL "$root/$source_data/." "$staged/"
fi
episode_count=$(wc -l < "$staged/meta/episodes.jsonl")
test "$episode_count" -eq "$expected"
if [[ "$dataset" == fractal ]]; then
  video_count=$(find "$staged/videos" -name '*.mp4' | wc -l)
  test "$video_count" -eq 87212
fi
echo "STAGED $dataset $episode_count episodes"
torchrun --nproc_per_node=8 --master_port=29500 "$scripts/nu_hold7k_launch.py" \
  --base_model_path "$root/model" --dataset_path "$staged" \
  --embodiment_tag "$tag" --num_gpus 8 --output_dir "$run_dir" \
  --save_steps 1000 --save_total_limit 20 --max_steps 18000 \
  --warmup_ratio 0.05555555555555555 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size 1024 --dataloader_num_workers 8 \
  --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --no-use-wandb --loss-type=hetero_t --ht-sbias="$sbias" --ht-df=1024.0 \
  --ht-mvt --state-dropout-prob "$dropout" \
  2>&1 | tee "$run_dir/train.log"
python "$scripts/nu_hold7k_eval.py" --dataset "$dataset" --run-dir "$run_dir" --step 18000 \
  2>&1 | tee "$run_dir/eval.log"

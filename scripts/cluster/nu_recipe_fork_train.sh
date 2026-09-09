#!/usr/bin/env bash
set -euo pipefail
arm=$1
case "$arm" in stair14|smooth14|smooth7) ;; *) exit 2 ;; esac
root=/mnt/pfs/yuchen/groot
code=/mnt/pfs/yuchen/nu_recipe_debug_20260908/code
source_ckpt="$root/ft_fr_nu1024_hold7k_to14_20260908/checkpoint-7000"
attempt_suffix=${NU_DEBUG_ATTEMPT_SUFFIX:-}
run_dir="$root/ft_fr_nudebug_${arm}_20260908${attempt_suffix}"
staged=/dev/shm/fr_nu_debug
cd "$root/Isaac-GR00T"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD:$code"
export HF_HOME=/mnt/pfs/yuchen/hf_home WANDB_MODE=disabled
export GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export NU_DEBUG_ARM="$arm"
mkdir "$run_dir"
mkdir "$run_dir/audit"
cp "$code/nu_recipe_fork_launch.py" "$code/nu_recipe_fork_train.sh" \
   "$code/nu_recipe_fork_eval.py" "$code/nu_fixed_scenes_rollout.py" \
   "$code/nu_hold7k_server.py" "$run_dir/audit/"
cp gr00t/model/gr00t_n1d7/gr00t_n1d7.py "$run_dir/audit/model_source.py"
cp gr00t/experiment/trainer.py "$run_dir/audit/trainer_source.py"
cmp "$run_dir/audit/model_source.py" "$root/ft_fr_nu1024_hold7k_to14_20260908/audit/model_source.py"
cmp "$run_dir/audit/trainer_source.py" "$root/ft_fr_nu1024_hold7k_to14_20260908/audit/trainer_source.py"
test -f "$source_ckpt/global_step7000/bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt"
mkdir -p "$staged"
cp -rL "$root/fractal_lerobot/." "$staged/"
test "$(wc -l < "$staged/meta/episodes.jsonl")" -eq 87212
test "$(find "$staged/videos" -name '*.mp4' | wc -l)" -eq 87212
torchrun --nproc_per_node=8 --master_port=29500 "$run_dir/audit/nu_recipe_fork_launch.py" \
  --base_model_path "$source_ckpt" --dataset_path "$staged" \
  --embodiment_tag SIMPLER_ENV_GOOGLE --num_gpus 8 --output_dir "$run_dir" \
  --save_steps 1000 --save_total_limit 8 --max_steps 18000 --resume-from-checkpoint \
  --warmup_ratio 0.05555555555555555 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size 1024 --dataloader_num_workers 8 \
  --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --no-use-wandb --loss-type=hetero_t --ht-sbias=-0.5093 --ht-df=1024.0 \
  --ht-mvt --state-dropout-prob 0.5 \
  2>&1 | tee "$run_dir/train.log"
test -f "$run_dir/resume_verified.json"
python "$run_dir/audit/nu_recipe_fork_eval.py" --run-dir "$run_dir" --step 13000 \
  2>&1 | tee "$run_dir/eval.log"

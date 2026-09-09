#!/usr/bin/env bash
set -euo pipefail
root=/mnt/pfs/yuchen/groot
code=/mnt/pfs/yuchen/nu224_to64_hgcomp_20260909/code
run_dir="$root/ft_fr_nu224to64_hgcomp_20260909"
staged=/dev/shm/fr_nu224to64_hgcomp
cd "$root/Isaac-GR00T"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD:$code"
export HF_HOME=/mnt/pfs/yuchen/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled GROOT_ROPE_CACHE=1 PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1 GROOT_LEARN_NU=0
mkdir "$run_dir"
mkdir "$run_dir/audit"
cp "$code/nu224_to64_hgcomp_launch.py" "$code/nu224_to64_hgcomp_train.sh" \
   "$code/test_nu224_to64_hgcomp.py" "$code/nu_recipe_fork_eval.py" \
   "$code/nu_fixed_scenes_rollout.py" "$code/nu_hold7k_server.py" \
   "$run_dir/audit/"
cp gr00t/model/gr00t_n1d7/gr00t_n1d7.py "$run_dir/audit/model_source.py"
cp gr00t/experiment/trainer.py "$run_dir/audit/trainer_source.py"
cp "$root/ft_fr_nu224/experiment_cfg/config.yaml" "$run_dir/audit/reference_config.yaml"
python "$run_dir/audit/test_nu224_to64_hgcomp.py" 2>&1 | tee "$run_dir/audit/unit_tests.log"
mkdir "$staged"
cp -rL "$root/fractal_lerobot/." "$staged/"
test "$(wc -l < "$staged/meta/episodes.jsonl")" -eq 87212
test "$(find "$staged/videos" -name '*.mp4' | wc -l)" -eq 87212
torchrun --nproc_per_node=8 --master_port=29500 \
  "$run_dir/audit/nu224_to64_hgcomp_launch.py" \
  --base_model_path "$root/model" --dataset_path "$staged" \
  --embodiment_tag SIMPLER_ENV_GOOGLE --num_gpus 8 --output_dir "$run_dir" \
  --save_steps 1000 --save_total_limit 20 --max_steps 20000 \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size 1024 --dataloader_num_workers 8 \
  --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --no-use-wandb --loss-type=hetero_t --ht-sbias=-0.5093 --ht-df=224.0 \
  --ht-mvt --state-dropout-prob 0.5 \
  2>&1 | tee "$run_dir/train.log"
test -f "$run_dir/recipe.json"
python "$run_dir/audit/nu_recipe_fork_eval.py" \
  --run-dir "$run_dir" --step 20000 --protocol paired --gpus 0,1,2,3,4,5 \
  2>&1 | tee "$run_dir/eval-seed1234.log"

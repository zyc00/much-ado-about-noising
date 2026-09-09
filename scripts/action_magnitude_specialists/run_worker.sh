#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pfs/yuchen/action_mag_specialists_20260905
GROOT=/mnt/pfs/yuchen/groot
PIROOT=/mnt/pfs/yuchen/pi05
STEPS=5000

export HF_HOME=/mnt/pfs/yuchen/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACTION_MAG_SPLIT=train

stage_gr1() {
    if [[ ! -f /dev/shm/.action_mag_gr1_ready ]]; then
        mkdir -p /dev/shm/gr1
        cp -r "$GROOT/lerobot/LeRobot/." /dev/shm/gr1/
        local count
        count=$(find /dev/shm/gr1 -name '*.parquet' | wc -l)
        [[ "$count" -gt 0 ]]
        touch /dev/shm/.action_mag_gr1_ready
        echo "[stage] GR1 parquet=$count"
    fi
}

stage_widowx() {
    if [[ ! -f /dev/shm/.action_mag_widowx_ready ]]; then
        mkdir -p /dev/shm/wx
        cp -r "$GROOT/bridge_orig_lerobot/." /dev/shm/wx/
        local count
        count=$(wc -l < /dev/shm/wx/meta/episodes.jsonl)
        [[ "$count" -eq 53192 ]]
        touch /dev/shm/.action_mag_widowx_ready
        echo "[stage] WidowX episodes=$count"
    fi
}

stage_pi05() {
    if [[ ! -f /dev/shm/.action_mag_pi05_ready ]]; then
        mkdir -p /dev/shm/lr_libero
        cp -r "$PIROOT/libero_lerobot/." /dev/shm/lr_libero/
        local count
        count=$(find /dev/shm/lr_libero -name '*.parquet' | wc -l)
        [[ "$count" -gt 0 ]]
        touch /dev/shm/.action_mag_pi05_ready
        echo "[stage] LIBERO parquet=$count"
    fi
}

run_groot() {
    local stack=$1
    local bin=$2
    local data_path embodiment base batch state_args manifest_dir output
    if [[ "$stack" == gr1 ]]; then
        stage_gr1
        data_path=$(find /dev/shm/gr1 -mindepth 1 -maxdepth 1 -type d -name 'gr1_unified.*' -print | sort | paste -sd:)
        embodiment=ROBOCASA_GR1_TABLETOP
        base="$GROOT/ft_mse/checkpoint-60000"
        batch=512
        state_args=()
        manifest_dir="$ROOT/manifests/gr1"
    else
        stage_widowx
        data_path=/dev/shm/wx
        embodiment=SIMPLER_ENV_WIDOWX
        # Earliest retained checkpoint: 10k MSE prefit followed by 4k HT.
        # Every specialist below is then optimized solely with MSE for 5k steps.
        base="$GROOT/ft_wxmsepre/checkpoint-14000"
        batch=1024
        state_args=(--state-dropout-prob 0.8)
        manifest_dir="$ROOT/manifests/widowx"
    fi
    output="$ROOT/runs/${stack}_q${bin}"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output"; return 22; }
    export ACTION_MAG_BIN=$bin
    export ACTION_MAG_MANIFEST_DIR=$manifest_dir
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    cd "$GROOT/Isaac-GR00T"
    export PATH="$PWD/.venv/bin:$PATH"
    echo "[run] stack=$stack bin=$bin base=$base output=$output"
    torchrun --standalone --nproc_per_node=8 "$ROOT/scripts/launch_groot_bin.py" \
        --base-model-path "$base" \
        --dataset-path "$data_path" \
        --embodiment-tag "$embodiment" \
        --num-gpus 8 \
        --output-dir "$output" \
        --save-steps "$STEPS" \
        --save-total-limit 1 \
        --save-only-model \
        --max-steps "$STEPS" \
        --warmup-ratio 0.05 \
        --weight-decay 1e-5 \
        --learning-rate 1e-4 \
        --global-batch-size "$batch" \
        --dataloader-num-workers 8 \
        --shard-size 1024 \
        --num-shards-per-epoch 100000 \
        --episode-sampling-rate 0.1 \
        --no-use-wandb \
        --loss-type mse \
        "${state_args[@]}" \
        2>&1 | tee "$ROOT/logs/${stack}_q${bin}.log"
}

run_pi05() {
    local bin=$1
    local output="$ROOT/runs/pi05_q${bin}"
    stage_pi05
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output"; return 22; }
    export ACTION_MAG_BIN=$bin
    export ACTION_MAG_MANIFEST="$ROOT/manifests/pi05/pi05.npz"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export LIBERO_CONFIG_PATH=/mnt/pfs/yuchen/oft/libero_cfg
    export PATH="$PIROOT/venv/bin:$PATH"
    export PYTHONPATH="$PIROOT/lerobot_src/src:${PYTHONPATH:-}"
    cd "$PIROOT"
    echo "[run] stack=pi05 bin=$bin output=$output"
    "$PIROOT/venv/bin/torchrun" --standalone --nproc-per-node=8 \
        "$ROOT/scripts/launch_pi05_bin.py" \
        --policy.path="$PIROOT/run_mse/checkpoints/030000/pretrained_model" \
        --policy.loss_type=mse \
        --policy.push_to_hub=false \
        --policy.dtype=bfloat16 \
        --policy.gradient_checkpointing=true \
        --policy.scheduler_warmup_steps=167 \
        --policy.scheduler_decay_steps="$STEPS" \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --dataset.root=/dev/shm/lr_libero \
        --dataset.eval_split=0.0 \
        --batch_size=4 \
        --steps="$STEPS" \
        --save_freq="$STEPS" \
        --num_workers=8 \
        --output_dir="$output" \
        --wandb.enable=false \
        2>&1 | tee "$ROOT/logs/pi05_q${bin}.log"
}

echo "[worker] $(date --iso-8601=seconds) host=$(hostname) schedule=$*"
for item in "$@"; do
    stack=${item%%:*}
    bin=${item##*:}
    case "$stack" in
        gr1|widowx) run_groot "$stack" "$bin" ;;
        pi05) run_pi05 "$bin" ;;
        *) echo "Unknown schedule item: $item"; exit 2 ;;
    esac
done
echo "[worker] complete $(date --iso-8601=seconds)"

# ToolHang Composition Study — MSE vs MIP

Code to reproduce the ToolHang `init → insertion` experiments: behavior cloning
with **MSE (regression)** vs **MIP (two-step flow map)**, comparing an
**end-to-end generalist** against **two-stage grasp+insert specialists**
(with/without DAgger handoff matching).

## Install

```bash
uv sync
# headless / cluster:
export MUJOCO_GL=egl
# if EGL/GL is missing on Ubuntu:
sudo apt-get install -y libglew-dev libosmesa6-dev patchelf
```

All commands below assume `export MUJOCO_GL=egl`.

---

## Get the data (download from HuggingFace)

Easiest path — one command (no scripted collection needed). The HF repo holds the
clean source demos + eval set; segments are regenerated locally on download.

```bash
python scripts/download_data.py --repo yuchen0187/toolhang-mip-data            # 2k + 20k
python scripts/download_data.py --repo yuchen0187/toolhang-mip-data --scale 2k # 2k only
# or set the default once:  export HF_DATA_REPO=yuchen0187/toolhang-mip-data
```

This downloads `tool_hang_clean_{2000,20000}.hdf5` + `warmstart_demos.hdf5` +
`full_eval_seeds.npy` into `data/`, then slices `full2ins` / `init2grasp` /
`pick2ins` for each scale. Run `uv sync` first so versions match (reproducible
slicing). To skip re-slicing and pull the segment files directly: `--segments download`.

**Publishing/updating the data** (owner only, needs `huggingface-cli login`):
```bash
python scripts/upload_data.py --repo yuchen0187/toolhang-mip-data              # clean + eval (~16 GB)
python scripts/upload_data.py --repo yuchen0187/toolhang-mip-data --with-segments  # also segments (~30 GB)
```

If you instead want to regenerate everything from the scripted policy, use Section 1.

---

## 1. Collect data (regenerate from scratch — optional)

> ⚠️ **Use `collect_scriptB_full.py` (built on `scripted_tool_hang_v2.py`).**
> Do **not** use `collect_tool_hang_demos.py` — it is a *different* scripted
> policy whose trajectory distribution does **not** match the eval reference,
> which silently tanks success rate.

```bash
# (a) Collect clean expert demos (scripted policy). 2000 demos:
python scripts/collect_scriptB_full.py \
    --n_demos 2000 --start_seed 0 --output data/tool_hang_clean_2000.hdf5
#   For 20k: --n_demos 20000  (slow single-process; shard by --start_seed and
#   merge with scripts/merge_hdf5.py)

# (b) Slice the three task segments from the clean demos:
python scripts/slice_segments.py --segment full \
    --src data/tool_hang_clean_2000.hdf5 --out data/tool_hang_full2ins_2000.hdf5   --n 2000  # generalist: init->insertion
python scripts/slice_segments.py --segment init2grasp \
    --src data/tool_hang_clean_2000.hdf5 --out data/tool_hang_init2grasp_2000.hdf5 --n 2000  # grasp specialist
python scripts/slice_segments.py --segment pick2ins \
    --src data/tool_hang_clean_2000.hdf5 --out data/tool_hang_pick2ins_2000.hdf5   --n 2000  # insert specialist

# (c) Held-out eval demos (seeds 21000+, disjoint from training seeds 0-19999):
python scripts/collect_warmstart_demos.py \
    --seeds_file data/full_eval_seeds.npy --output data/warmstart_demos.hdf5
```

DAgger handoff-matched insertion data (insert specialist trained on the grasp
specialist's own output): `scripts/collect_handoff_demos.py`.

---

## 2. Train (2k / 20k, MSE / MIP)

Main entry point: `examples/train_robomimic.py` (Hydra).
**2k vs 20k = which dataset file you point at; MSE vs MIP = `optimization.loss_type`.**

```bash
python examples/train_robomimic.py \
    task=tool_hang_ph_state_delta_legacy \
    +task.dataset_path=$(pwd)/data/tool_hang_full2ins_2000.hdf5 \
    network=chiunet task.num_envs=1 \
    optimization.loss_type=mip \
    optimization.auto_resume=false log.wandb_mode=disabled \
    log.exp_name=full_mip_2000 log.log_dir=logs/full_mip_2000
```

- **MSE**: `optimization.loss_type=regression`
- **20k**: point `+task.dataset_path` at the `..._20000.hdf5` file
- **Specialists**: point `dataset_path` at `init2grasp_*` (grasp) or `pick2ins_*` (insert)
- Checkpoint is written to `logs/<exp_name>/models/model_latest.pt`

---

## 3. Eval

### Generalist full task / sub-segments — `scripts/eval_warmstart.py`

```bash
# Full init->insertion success rate
python scripts/eval_warmstart.py \
    --ckpt logs/full_mip_2000/models/model_latest.pt \
    --dataset data/tool_hang_full2ins_2000.hdf5 --loss mip \
    --demos data/warmstart_demos.hdf5 \
    --warm_to 0 --init_mode reset_settle --settle 10 --success assembled --n 100
```

- **Grasp segment** (from init): `--warm_to 0  --success grasp`
- **Insertion segment** (from expert c1): `--warm_to c1 --success assembled --init_mode state0`

### Two-stage specialist stitching — `scripts/eval_twostage_faithful.py`

```bash
python scripts/eval_twostage_faithful.py \
    --grasp_ckpt logs/grasp_mip_2000/models/model_latest.pt --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
    --back_ckpt  logs/pick2ins_mip_2000/models/model_latest.pt --back_ds data/tool_hang_pick2ins_2000.hdf5 \
    --loss mip --demos data/warmstart_demos.hdf5 --stage1 policy --init_mode reset_settle --n 100
```
Cross-method combos: use per-stage `--grasp_loss` / `--back_loss`.

### Three rules (or the numbers will be wrong)

1. **Always `--init_mode reset_settle --settle 10`** — reproduces the scripted
   demos' 10-step leading settle; without it, brittle models score ~0.
2. **`--loss` must match training** (a MIP checkpoint needs `--loss mip`).
3. **Eval on `warmstart_demos.hdf5`** (held-out seeds 21000+, disjoint from
   training seeds 0-19999).

---

## Other useful scripts

| Script | Purpose |
|---|---|
| `scripts/eval_val_loss.py` / `eval_train_loss.py` | Per-phase open-loop action-prediction loss (val / train) |
| `scripts/eval_grasp_breakdown.py` | Grasp-phase loss split early/mid/late (handoff divergence) |
| `scripts/eval_ood_perturb.py` | OOD-robustness curve under handoff perturbation |
| `scripts/render_stitch_faithful.py` | Render two-stage stitching rollouts to mp4 |
| `scripts/merge_hdf5.py` | Merge collection shards |
| `analysis/plot_*.py` | Figures (composition, scaling, segment ablations) |

## Reference results (2000 demos, ToolHang, settle-fixed eval, n=100)

| Setting | MSE | MIP |
|---|---|---|
| Generalist (end-to-end) | 80 | 95 |
| Two specialists, no dagger | 42 | 91 |
| Two specialists, + DAgger handoff | 97 | 98 |

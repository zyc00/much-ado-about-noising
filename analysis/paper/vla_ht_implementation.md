# HT on GR00T N1.7: implementation summary

Goal (paper §6.6): show single-step heteroscedastic Student-t regression (HT)
matches the flow-matching action head on a 3B VLA at 1 network evaluation
instead of K=4, using NVIDIA's own finetuning recipe and published benchmark
(RoboCasa GR1 Tabletop, 24 tasks × 20 trials, released average 44.5%).

## 1. Model changes (4 files, flow path byte-identical)

All changes gated by a new `loss_type` config field (`"flow"` default,
`"hetero_t"` ours) in `gr00t/configs/model/gr00t_n1d7.py`, plumbed through
`finetune_config.py` and `launch_finetune.py` exactly like their
`tune_llm`-style flags.

In `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`:

- **Training forward.** Their flow branch samples t ~ Beta-derived schedule,
  builds `x_t = (1−t)·ε + t·a` and supervises the decoder output as velocity
  `a − ε` with masked MSE. The hetero_t branch instead feeds a **zero action
  trajectory** at **fixed t=0** (bucket 0) — no sampled noise anywhere — and
  supervises the decoder output **directly as the action chunk**.
- **σ head.** A parallel `sigma_decoder` (`CategorySpecificMLP`, same shape as
  their action decoder) on the shared DiT output, freshly initialized,
  instantiated only for hetero_t (no dead params in flow mode, no DDP
  unused-parameter issues, base checkpoint loads with one "newly initialized"
  warning). Wired into their `tune_projector` freeze/eval groups.
- **Inference.** `get_action` short-circuits the K-step Euler loop: one DiT
  forward at (zeros, t=0), decoder output returned as the action.
  1 network evaluation per chunk, deterministic.

Tests: `tests/gr00t/model/test_action_head_hetero_t.py` (12 tests) + their
original suites, 34/34 green. Hook-counted: hetero_t runs exactly 1 DiT
forward at inference, flow runs exactly K.

## 2. The loss (harness-exact)

Matches `regression_hetero_t` in `mip/losses.py` exactly — a **per-sample
scalar σ** and a **multivariate Student-t NLL** on the whole-chunk residual
(not per-dim independent t; per-dim was our first draft and is a different
estimator than the one validated on robomimic):

```
σ_i  = mean_masked(softplus(s_raw_i + sbias)) + 1e-3          # scalar per sample
NLL_i = 0.5·(ν+1)·log1p(‖r_i‖²_masked / (ν·σ_i²·D_i))·D_i + D_i·log σ_i
loss  = Σ_i NLL_i / Σ masked dims,   ν = 2
```

Computed in fp32 under bf16 autocast (the log/divide chain is less
bf16-tolerant than plain MSE). Harness experimental knobs (HT_MIX etc.)
are not ported.

## 3. Why the pretrained flow head transfers (and the proof)

Their convention puts pure noise at t=0 and velocity target `a − ε`; the
converged head therefore computes `v̂(ε, 0, obs) = E[a|obs] − ε`, so at our
operating point `v̂(0, 0, obs) = E[a|obs]` — the pretrained decoder already
outputs the posterior-mean action at (zeros, t=0), in the same output space
and scale. Bucket 0 is also the *best-trained* timestep (their time-sampling
density peaks at t=0, and every inference trajectory starts there).

**Zero-shot probe (measured, base checkpoint, real GR1 data, n=240):**
residual RMS median **0.29** (q10–q90: 0.26–0.33) on ±1-normalized actions —
a competent regressor before any finetuning. The probe also calibrates σ
initialization: `ht_sbias = softplus⁻¹(0.29) = −1.088`. The naive default
(+3) would have initialized σ ≈ 3 — 10× too large, suppressing early
μ-gradients by ~two orders of magnitude.

## 4. Training setup

NVIDIA's recipe, one flag added:

```
NUM_GPUS=8 MAX_STEPS=60000 GLOBAL_BATCH_SIZE=512 SAVE_STEPS=2000 \
bash examples/finetune.sh --base-model-path <N1.7-3B> \
  --dataset-path <24 task dirs, colon-joined> \
  --embodiment-tag ROBOCASA_GR1_TABLETOP --output-dir ... \
  -- --no-use-wandb --loss-type=hetero_t --ht-sbias=-1.088 \
     --dataloader-num-workers 8
```

Launch fixes catalogued (each would have killed the 8-GPU run):
dataset paths must be individual LeRobot dirs (no parent expansion);
`generate_stats` upstream bug deletes valid stats for dtype-`object`
features (patched: drop only features absent from info.json);
pre-2026 `initial_actions.npz` rejected by their loader (renamed aside;
absence is a graceful no-op); wandb on by default (disabled); extra flags
must follow the `--` separator and use tyro kebab-case.

Health at launch: first loss 1.05 vs probe-predicted 1.09; grad norm ~0.2;
1.2 it/s (≈14 h for 60k steps).

## 5. Training efficiency (measured, then engineered)

Measured on the live run: ~50% GPU duty cycle, ~8 of 128 host cores busy.
The pipeline decodes MP4 frames per sample on CPU with effectively one
active decode worker per rank (2 vs 8 workers: identical throughput;
batch 512→1024: throughput halves). More GPUs (scales per node) and more
CPUs (existing ones idle) cannot fix it.

**Fix: frozen-backbone feature cache.** The default finetune freezes the
entire 2B VLM; everything trainable sits after `backbone_features`, so
backbone outputs are training-constant. Determinism verified empirically
(identical collate outputs; no train-time augmentation in the recipe).
Stage A precomputes features once (seq 79→pad 96 × 2048, fp16, ~310 GB,
one decode-epoch fanned over 8 GPUs); Stage B trains the action head only
on cached shards (same optimizer/schedule/loss; DDP; no video decode, no
VL processor, no backbone forward). Training was paying the decode pipeline
~44× (60k×512 samples / 700k unique); the cache pays it once.

## 6. Evaluation

Their protocol exactly: RoboCasa sim (installed & verified),
24 tasks × 20 trials, `max_episode_steps 720`, `n_action_steps 8`,
vs the **published** 44.5% (their recipe, their eval, most favorable to
them). Decision rule: HT ≥ ~44.5% → done (harness friction only hurts us);
HT ≪ 40% → investigate; HT 40–44% → ambiguous (eval SE ≈ ±2.3 pts at 480
episodes) → run the flow control under identical conditions to attribute
the gap.

Latency (measured, A800, eager): action head 80 → 22 ms (K=4 → 1, 3.6×);
model forward 177 → 109 ms; E2E 297 → 275 ms (preprocessing-bound in eager;
head share dominates in their TRT deployment path).

## 7. Artifacts

- Patched repo: desktop `/home/jigu/data/groot/Isaac-GR00T`, cluster
  `/mnt/pfs/yuchen/groot/Isaac-GR00T` (editable venv, portable interpreter)
- Probe: `probe_vla.py`, result `probe_result.json`
- Cache: `precache_vla.py` (Stage A), `cached_train_vla.py` (Stage B),
  features at `/mnt/pfs/yuchen/groot/feat_cache/`
- Headline run: `/mnt/pfs/yuchen/groot/ft_ht` (checkpoints every 2k steps)
- Latency logs: `bench_k4.log` / `bench_k1.log`

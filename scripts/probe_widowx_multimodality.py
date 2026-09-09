#!/usr/bin/env python3
"""Held-out same-observation multimodality probe for GR00T WidowX Flow.

This script supports two complementary state sources:

* ``SOURCE=train``: random Bridge training states through the finetuning pipeline.
* ``SOURCE=rollout``: observations logged during actual Flow-policy SimplerEnv
  rollouts (close drawer and put eggplant in basket).

For every state, K=8 stochastic action chunks are screened.  Candidate states
are selected independently for full-chunk, arm-only, and gripper-only structure.
Fresh samples are then split into discovery and confirmation sets: the PCA axis
is estimated only on discovery samples, while a one-vs-two Gaussian BIC test is
performed only on confirmation projections.  This avoids reusing the samples
that selected a state or an axis.  Benjamini-Hochberg correction is applied per
action group.  A confirmed mode additionally requires delta-BIC >= 10, at least
20% mass in each component, and component separation >= 2 pooled standard
deviations.

Outputs are a compressed NPZ with raw confirmation chunks and a JSON summary.
All action-space analysis is in the checkpoint's normalized coordinates.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


SOURCE = os.environ.get("SOURCE", "train")
N_SCREEN = int(os.environ.get("N_SCREEN", "3000" if SOURCE == "train" else "1000"))
K_SCREEN = int(os.environ.get("K_SCREEN", "8"))
K_DISCOVERY = int(os.environ.get("K_DISCOVERY", "32"))
K_CONFIRM = int(os.environ.get("K_CONFIRM", "64"))
DRAW_BATCH = int(os.environ.get("DRAW_BATCH", "4"))
N_NULL = int(os.environ.get("N_NULL", "10000"))
SEED = int(os.environ.get("SEED", "20260903"))
N_RANDOM_DEEP = int(os.environ.get("N_RANDOM_DEEP", "0"))
RANDOM_ONLY = os.environ.get("RANDOM_ONLY", "0") == "1"
CHECKPOINT = os.environ.get(
    "CKPT", "/mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000"
)
ROOT = Path("/mnt/pfs/yuchen/groot")
OUT_STEM = Path(os.environ.get("OUT", str(ROOT / f"widowx_mm_{SOURCE}")))

GROUP_COLUMNS = {
    "full": tuple(range(7)),
    "arm": tuple(range(6)),
    "translation": tuple(range(3)),
    "rotation": tuple(range(3, 6)),
    "gripper": (6,),
}


def group_view(x: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    """Select action columns from [..., T, A] and flatten T and A."""
    return x[..., list(columns)].reshape(x.shape[0], -1)


def kmeans_1d(z: np.ndarray) -> np.ndarray:
    c0, c1 = float(np.min(z)), float(np.max(z))
    labels = np.zeros(len(z), dtype=bool)
    for _ in range(100):
        new_labels = np.abs(z - c1) < np.abs(z - c0)
        if new_labels.all() or (~new_labels).all():
            return np.zeros(len(z), dtype=bool)
        n0, n1 = float(z[~new_labels].mean()), float(z[new_labels].mean())
        if np.array_equal(labels, new_labels) or abs(n0 - c0) + abs(n1 - c1) < 1e-12:
            labels = new_labels
            break
        labels, c0, c1 = new_labels, n0, n1
    return labels


def pooled_separation(z: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    n0, n1 = int((~labels).sum()), int(labels.sum())
    if min(n0, n1) == 0:
        return 0.0, 0.0
    weight = min(n0, n1) / len(z)
    if weight < 0.2:
        return 0.0, weight
    v0 = float(z[~labels].var(ddof=1)) if n0 > 1 else 0.0
    v1 = float(z[labels].var(ddof=1)) if n1 > 1 else 0.0
    pooled = math.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(len(z) - 2, 1))
    sep = abs(float(z[labels].mean() - z[~labels].mean())) / max(pooled, 1e-12)
    return sep, weight


def pca_screen_stat(x: np.ndarray) -> dict[str, float]:
    c = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    if not len(s) or s[0] < 1e-12:
        return {"sep": 0.0, "spread": 0.0, "eff_rank": 0.0}
    z = c @ vt[0]
    sep, _ = pooled_separation(z, kmeans_1d(z))
    lam = s * s
    eff = float(lam.sum() ** 2 / max(float((lam * lam).sum()), 1e-18))
    return {
        "sep": float(sep),
        "spread": float(np.sqrt(np.mean(np.sum(c * c, axis=1)))),
        "eff_rank": eff,
    }


def normal_logpdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mean) ** 2 / var)


def fit_gmm2(z: np.ndarray) -> dict[str, Any]:
    """Deterministic 1-D two-Gaussian EM with several quantile starts."""
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    total_var = max(float(z.var()), 1e-10)
    floor = max(total_var * 1e-3, 1e-10)
    best = None
    for q0, q1 in ((0.2, 0.8), (0.3, 0.7), (0.4, 0.6)):
        means = np.quantile(z, [q0, q1]).astype(np.float64)
        variances = np.array([total_var * 0.5, total_var * 0.5])
        weights = np.array([0.5, 0.5])
        old_ll = -np.inf
        for _ in range(200):
            logp = np.column_stack(
                [
                    math.log(max(weights[j], 1e-12))
                    + normal_logpdf(z, float(means[j]), float(variances[j]))
                    for j in range(2)
                ]
            )
            row_max = logp.max(axis=1, keepdims=True)
            log_norm = row_max + np.log(np.exp(logp - row_max).sum(axis=1, keepdims=True))
            resp = np.exp(logp - log_norm)
            ll = float(log_norm.sum())
            nk = resp.sum(axis=0).clip(min=1e-8)
            weights = nk / n
            means = (resp * z[:, None]).sum(axis=0) / nk
            variances = (resp * (z[:, None] - means) ** 2).sum(axis=0) / nk
            variances = np.maximum(variances, floor)
            if abs(ll - old_ll) < 1e-8:
                break
            old_ll = ll
        if best is None or ll > best[0]:
            best = (ll, weights.copy(), means.copy(), variances.copy(), resp.copy())
    assert best is not None
    ll2, weights, means, variances, resp = best
    order = np.argsort(means)
    weights, means, variances, resp = (
        weights[order],
        means[order],
        variances[order],
        resp[:, order],
    )
    mean1, var1 = float(z.mean()), max(float(z.var()), 1e-10)
    ll1 = float(normal_logpdf(z, mean1, var1).sum())
    # One Gaussian has 2 parameters; two univariate Gaussians have 5.
    delta_bic = (2 * math.log(n) - 2 * ll1) - (5 * math.log(n) - 2 * ll2)
    pooled = math.sqrt(float(np.sum(weights * variances)))
    sep = abs(float(means[1] - means[0])) / max(pooled, 1e-12)
    labels = np.argmax(resp, axis=1).astype(np.int8)
    return {
        "delta_bic": float(delta_bic),
        "sep": float(sep),
        "min_weight": float(weights.min()),
        "weights": weights,
        "means": means,
        "variances": variances,
        "labels": labels,
    }


def null_delta_bic(n: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        vals[i] = fit_gmm2(rng.standard_normal(n))["delta_bic"]
    return np.sort(vals)


def empirical_p(value: float, sorted_null: np.ndarray) -> float:
    at_least = len(sorted_null) - int(np.searchsorted(sorted_null, value, side="left"))
    return float((at_least + 1) / (len(sorted_null) + 1))


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1].clip(max=1.0)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q


def mc_screen_null(k: int, d: int, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for i in range(reps):
        vals[i] = pca_screen_stat(rng.standard_normal((k, d)))["sep"]
    return {
        "median": float(np.median(vals)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
    }


def power_check(sorted_null: np.ndarray, n: int, reps: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 99)
    out = {}
    for weight in (0.5, 0.2):
        for sep in (2.0, 3.0, 4.0):
            detected = 0
            for _ in range(reps):
                labels = rng.random(n) < weight
                z = rng.standard_normal(n) + np.where(labels, sep / 2, -sep / 2)
                fit = fit_gmm2(z)
                p = empirical_p(fit["delta_bic"], sorted_null)
                detected += int(
                    p <= 0.05
                    and fit["delta_bic"] >= 10.0
                    and fit["sep"] >= 2.0
                    and fit["min_weight"] >= 0.2
                )
            out[f"weight={weight:.1f},sep={sep:.1f}"] = detected / reps
    return out


def to_cuda(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, Mapping):
        return type(x)({k: to_cuda(v) for k, v in x.items()})
    return x


def to_cpu(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, Mapping):
        return type(x)({k: to_cpu(v) for k, v in x.items()})
    return x


def repeat_batch(x: Any, n: int) -> Any:
    if torch.is_tensor(x):
        if x.ndim > 0 and x.shape[0] == 1:
            return x.repeat((n,) + (1,) * (x.ndim - 1))
        return x
    if isinstance(x, Mapping):
        return type(x)({k: repeat_batch(v, n) for k, v in x.items()})
    if isinstance(x, list) and len(x) == 1:
        return x * n
    if isinstance(x, tuple) and len(x) == 1:
        return x * n
    return x


def extract_prediction(output: Any) -> torch.Tensor:
    action = (
        output["action_pred"]
        if isinstance(output, Mapping) and "action_pred" in output
        else output[0]
        if isinstance(output, tuple)
        else output
    )
    if isinstance(action, Mapping):
        action = next(iter(action.values()))
    if not torch.is_tensor(action):
        raise TypeError(f"Cannot extract action tensor from {type(output)}")
    return action


def encode_condition(model: Any, inputs: Any) -> dict[str, Any]:
    """Encode vision/language/state once; stochasticity enters only in the action head."""
    inner = inputs["inputs"] if "inputs" in inputs else inputs
    backbone_inputs, action_input = model.prepare_input(inner)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        backbone_output = model.backbone(backbone_inputs)
        features = model.action_head._encode_features(backbone_output, action_input)
    return {
        "backbone_features": features.backbone_features,
        "state_features": features.state_features,
        "embodiment_id": action_input.embodiment_id,
        "backbone_output": backbone_output,
        "action_input": action_input,
    }


def draw_chunks(model: Any, condition: dict[str, Any], k: int, select: Any) -> np.ndarray:
    chunks = []
    for start in range(0, k, DRAW_BATCH):
        b = min(DRAW_BATCH, k - start)
        repeated = repeat_batch(condition, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.action_head.get_action_with_features(**repeated)
        action = extract_prediction(output).float().cpu().numpy()
        chunks.append(select(action))
    return np.concatenate(chunks, axis=0).astype(np.float64)


def score_chunks(chunks: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: pca_screen_stat(group_view(chunks, columns))
        for name, columns in GROUP_COLUMNS.items()
    }


def add_to_buckets(
    buckets: dict[str, list[dict[str, Any]]], entry: dict[str, Any], prefix: str
) -> None:
    specs = {
        "full": (20 if prefix == "train" else 10, entry["scores"]["full"]["sep"]),
        "arm": (20 if prefix == "train" else 10, entry["scores"]["arm"]["sep"]),
        "gripper": (10 if prefix == "train" else 5, entry["scores"]["gripper"]["sep"]),
        "spread": (10 if prefix == "train" else 5, entry["scores"]["full"]["spread"]),
    }
    for kind, (limit, value) in specs.items():
        key = f"{prefix}:{kind}"
        bucket = buckets.setdefault(key, [])
        bucket.append(entry)
        bucket.sort(
            key=lambda e: e["scores"]["full"]["spread"]
            if kind == "spread"
            else e["scores"][kind]["sep"],
            reverse=True,
        )
        del bucket[limit:]


def unique_candidates(buckets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates = {}
    for bucket in buckets.values():
        for entry in bucket:
            candidates[entry["uid"]] = entry
    return sorted(candidates.values(), key=lambda e: e["uid"])


def setup_training_source():
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model import MODEL_REGISTRY

    tag = EmbodimentTag.resolve("SIMPLER_ENV_WIDOWX").value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(ROOT / "bridge_orig_lerobot")],
                        "mix_ratio": 1.0,
                        "embodiment_tag": tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model.loss_type = "flow"
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.state_dropout_prob = 0.8
    config.training.start_from_checkpoint = CHECKPOINT
    config.training.num_gpus = 1
    config.training.global_batch_size = 1
    config.training.output_dir = str(ROOT / "widowx_mm_train_setup")
    config.training.use_wandb = False
    config.validate()
    cfg_dir = Path(config.training.output_dir) / "experiment_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, cfg_dir)
    pipeline.setup()
    model = pipeline.return_model().cuda().eval()
    dataset, _ = pipeline.return_dataset()
    collator = pipeline.return_collator()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator, num_workers=2)
    iterator = iter(loader)

    def get_state(i: int) -> dict[str, Any]:
        batch = to_cuda(next(iterator))
        inner = batch["inputs"] if "inputs" in batch else batch
        mask = inner["action_mask"].float()
        mask_flat = mask.reshape(-1).cpu().numpy() > 0
        inputs = {k: v for k, v in inner.items() if k != "action"}
        true_action = inner["action"].float().reshape(-1).cpu().numpy()

        def select(action: np.ndarray) -> np.ndarray:
            flat = action.reshape(action.shape[0], -1)[:, : mask_flat.size]
            active = flat[:, mask_flat]
            if active.shape[1] % 7:
                raise ValueError(f"Expected WidowX active action dimension divisible by 7; got {active.shape}")
            return active.reshape(action.shape[0], -1, 7)

        return {
            "uid": f"train:{i}",
            "source_group": "train",
            "source_index": i,
            "inputs": to_cpu(inputs),
            "select": select,
            "true_action": true_action[: mask_flat.size][mask_flat].reshape(-1, 7),
        }

    return model, get_state


def setup_rollout_source():
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve("SIMPLER_ENV_WIDOWX"),
        model_path=CHECKPOINT,
        device="cuda",
        strict=True,
    )
    records = [
        (
            "close_drawer",
            ROOT / "rollout_dumps/flow_closedr.npz",
            "close the drawer",
        ),
        (
            "eggplant_basket",
            ROOT / "rollout_dumps/flow_eggbask.npz",
            "put the eggplant in the yellow basket",
        ),
    ]
    # NPZ members are decompressed on every ``z[key]`` access.  Materialize the
    # observation arrays once; otherwise the ~400 MB video member is inflated
    # again for every screened state.
    loaded = []
    state_keys = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
    for name, path, instruction in records:
        with np.load(path) as archive:
            arrays = {"obs.video.image_0": archive["obs.video.image_0"]}
            arrays.update(
                {f"obs.state.{key}": archive[f"obs.state.{key}"] for key in state_keys}
            )
        loaded.append((name, arrays, instruction))
    rng = np.random.default_rng(SEED)
    per_task = N_SCREEN // len(loaded)
    refs = []
    for task_no, (name, z, instruction) in enumerate(loaded):
        nt, ne = z["obs.video.image_0"].shape[:2]
        count = min(per_task + (task_no < N_SCREEN % len(loaded)), nt * ne)
        chosen = rng.choice(nt * ne, size=count, replace=False)
        refs.extend((name, z, instruction, int(q // ne), int(q % ne)) for q in chosen)
    rng.shuffle(refs)

    def get_state(i: int) -> dict[str, Any]:
        name, z, instruction, t, e = refs[i]
        states = {
            key: z[f"obs.state.{key}"][t, e][None].astype(np.float32)
            for key in state_keys
        }
        video = z["obs.video.image_0"][t, e][None]
        step = VLAStepData(
            images={"image_0": video[0]},
            states={key: value[0] for key, value in states.items()},
            actions={},
            text=instruction,
            embodiment=policy.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        processed = policy.processor(messages)
        collated = policy.collate_fn([processed])
        collated = to_cuda(_rec_to_dtype(collated, dtype=torch.bfloat16))

        def select(action: np.ndarray) -> np.ndarray:
            return action[:, :8, :7]

        return {
            "uid": f"{name}:{t}:{e}",
            "source_group": name,
            "source_index": (t, e),
            "inputs": collated,
            "select": select,
            "rgb": video[0, 0],
        }

    return policy.model, get_state


def summarize_deep(
    candidates: list[dict[str, Any]], deep: list[dict[str, Any]], null_bic: np.ndarray
) -> dict[str, Any]:
    # Apply BH correction separately for each scientifically distinct group.
    for group in GROUP_COLUMNS:
        qvals = bh_qvalues(np.array([row["tests"][group]["p"] for row in deep]))
        for row, q in zip(deep, qvals):
            test = row["tests"][group]
            test["q"] = float(q)
            test["detected"] = bool(
                q <= 0.05
                and test["delta_bic"] >= 10.0
                and test["sep"] >= 2.0
                and test["min_weight"] >= 0.2
            )

    group_summary = {}
    for group in GROUP_COLUMNS:
        tests = [row["tests"][group] for row in deep]
        detected = [t for t in tests if t["detected"]]
        group_summary[group] = {
            "n_candidates": len(tests),
            "n_detected": len(detected),
            "max_delta_bic": float(max(t["delta_bic"] for t in tests)),
            "min_q": float(min(t["q"] for t in tests)),
            "detected_uids": [row["uid"] for row in deep if row["tests"][group]["detected"]],
        }

    return {
        "source": SOURCE,
        "checkpoint": CHECKPOINT,
        "n_screen": N_SCREEN,
        "k_screen": K_SCREEN,
        "n_deep_candidates": len(candidates),
        "k_discovery": K_DISCOVERY,
        "k_confirmation": K_CONFIRM,
        "draw_batch": DRAW_BATCH,
        "confirmation_rule": {
            "axis": "PC1 fit on fresh discovery samples only",
            "test": "1-Gaussian versus 2-Gaussian delta BIC on independent confirmation projections",
            "multiple_testing": "Benjamini-Hochberg q<=0.05 separately within each action group",
            "delta_bic_min": 10.0,
            "pooled_sd_separation_min": 2.0,
            "component_weight_min": 0.2,
        },
        "null_delta_bic": {
            "repetitions": len(null_bic),
            "median": float(np.median(null_bic)),
            "q95": float(np.quantile(null_bic, 0.95)),
            "q99": float(np.quantile(null_bic, 0.99)),
        },
        "single_candidate_power": power_check(null_bic, K_CONFIRM),
        "group_summary": group_summary,
    }


def main() -> None:
    if SOURCE not in {"train", "rollout"}:
        raise ValueError("SOURCE must be 'train' or 'rollout'")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, get_state = setup_training_source() if SOURCE == "train" else setup_rollout_source()
    nfe = getattr(getattr(model, "action_head", None), "num_inference_timesteps", None)
    if nfe != 4:
        raise RuntimeError(f"Expected exact eval NFE=4, got {nfe}")
    print(
        f"SOURCE={SOURCE} N_SCREEN={N_SCREEN} K_SCREEN={K_SCREEN} "
        f"K_DISCOVERY={K_DISCOVERY} K_CONFIRM={K_CONFIRM} DRAW_BATCH={DRAW_BATCH} NFE={nfe}",
        flush=True,
    )

    if N_RANDOM_DEEP > N_SCREEN:
        raise ValueError("N_RANDOM_DEEP cannot exceed N_SCREEN")
    random_deep_indices = set(
        np.random.default_rng(SEED + 501).choice(
            N_SCREEN, size=N_RANDOM_DEEP, replace=False
        ).tolist()
    )
    random_candidates = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    screen_rows = []
    with torch.inference_mode():
        for i in range(N_SCREEN):
            state = get_state(i)
            inputs = to_cuda(state.pop("inputs"))
            select = state.pop("select")
            condition = encode_condition(model, inputs)
            chunks = draw_chunks(model, condition, K_SCREEN, select)
            scores = score_chunks(chunks)
            entry = {
                **state,
                "condition": to_cpu(condition),
                "select": select,
                "scores": scores,
            }
            add_to_buckets(buckets, entry, entry["source_group"])
            if i in random_deep_indices:
                random_candidates.append(entry)
            screen_rows.append(
                {
                    "uid": entry["uid"],
                    "source_group": entry["source_group"],
                    **{
                        f"{group}_{metric}": value
                        for group, stats in scores.items()
                        for metric, value in stats.items()
                    },
                }
            )
            if i == 0 or (i + 1) % 50 == 0:
                print(
                    f"screen {i + 1}/{N_SCREEN} uid={entry['uid']} "
                    f"sep(full/arm/grip)={scores['full']['sep']:.2f}/"
                    f"{scores['arm']['sep']:.2f}/{scores['gripper']['sep']:.2f}",
                    flush=True,
                )

    if RANDOM_ONLY:
        if not random_candidates:
            raise ValueError("RANDOM_ONLY=1 requires N_RANDOM_DEEP > 0")
        candidates = sorted(random_candidates, key=lambda entry: entry["uid"])
        selection = "uniform random screened states"
    else:
        candidates = unique_candidates(buckets)
        # Add uniform random candidates when requested, de-duplicating any that
        # were also selected by the extreme-score buckets.
        candidates = sorted(
            {entry["uid"]: entry for entry in candidates + random_candidates}.values(),
            key=lambda entry: entry["uid"],
        )
        selection = "extreme-score buckets plus optional uniform random states"
    print(f"deep candidates: {len(candidates)} ({selection})", flush=True)
    null_bic = null_delta_bic(K_CONFIRM, N_NULL, SEED + 7)
    print(
        "confirmation null delta-BIC: "
        f"median={np.median(null_bic):.2f} q95={np.quantile(null_bic, .95):.2f} "
        f"q99={np.quantile(null_bic, .99):.2f}",
        flush=True,
    )

    deep = []
    raw_discovery, raw_confirmation, rgbs = [], [], []
    with torch.inference_mode():
        for j, entry in enumerate(candidates):
            condition = to_cuda(entry["condition"])
            xd = draw_chunks(model, condition, K_DISCOVERY, entry["select"])
            xc = draw_chunks(model, condition, K_CONFIRM, entry["select"])
            tests = {}
            for group, columns in GROUP_COLUMNS.items():
                gd = group_view(xd, columns)
                gc = group_view(xc, columns)
                center = gd.mean(axis=0, keepdims=True)
                _, s, vt = np.linalg.svd(gd - center, full_matrices=False)
                axis = vt[0] if len(s) and s[0] >= 1e-12 else np.zeros(gd.shape[1])
                zc = (gc - center) @ axis
                fit = fit_gmm2(zc)
                labels = fit.pop("labels")
                delta = xc[labels == 1].mean(axis=0) - xc[labels == 0].mean(axis=0)
                energy = np.sum(delta * delta, axis=0)
                total_energy = max(float(energy.sum()), 1e-18)
                tests[group] = {
                    "delta_bic": fit["delta_bic"],
                    "sep": fit["sep"],
                    "min_weight": fit["min_weight"],
                    "p": empirical_p(fit["delta_bic"], null_bic),
                    "energy_translation": float(energy[:3].sum() / total_energy),
                    "energy_rotation": float(energy[3:6].sum() / total_energy),
                    "energy_gripper": float(energy[6] / total_energy),
                    "centroid_delta": delta.tolist(),
                }
            deep.append(
                {
                    "uid": entry["uid"],
                    "source_group": entry["source_group"],
                    "source_index": entry["source_index"],
                    "screen_scores": entry["scores"],
                    "tests": tests,
                }
            )
            raw_discovery.append(xd.astype(np.float32))
            raw_confirmation.append(xc.astype(np.float32))
            if SOURCE == "rollout":
                rgbs.append(entry["rgb"])
            print(
                f"deep {j + 1}/{len(candidates)} {entry['uid']} "
                f"deltaBIC(full/arm/grip)={tests['full']['delta_bic']:.1f}/"
                f"{tests['arm']['delta_bic']:.1f}/{tests['gripper']['delta_bic']:.1f}",
                flush=True,
            )

    summary = summarize_deep(candidates, deep, null_bic)
    summary["deep_selection"] = selection
    summary["n_random_deep_requested"] = N_RANDOM_DEEP
    screen_arrays = {
        key: np.array([row[key] for row in screen_rows], dtype=np.float64)
        for key in screen_rows[0]
        if key not in {"uid", "source_group"}
    }
    screen_null = {}
    for gi, (group, columns) in enumerate(GROUP_COLUMNS.items()):
        d = 8 * len(columns)
        null = mc_screen_null(K_SCREEN, d, min(N_NULL, 10000), SEED + 100 + gi)
        values = screen_arrays[f"{group}_sep"]
        screen_null[group] = {
            **null,
            "observed_median": float(np.median(values)),
            "observed_frac_above_null_q95": float(np.mean(values > null["q95"])),
        }
    summary["screen_null"] = screen_null
    summary["deep"] = deep

    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{OUT_STEM}.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        f"{OUT_STEM}.npz",
        **screen_arrays,
        screen_uid=np.array([row["uid"] for row in screen_rows]),
        screen_group=np.array([row["source_group"] for row in screen_rows]),
        deep_uid=np.array([row["uid"] for row in deep]),
        discovery=np.stack(raw_discovery),
        confirmation=np.stack(raw_confirmation),
        rgb=np.stack(rgbs) if rgbs else np.empty((0, 0, 0, 3), dtype=np.uint8),
        null_delta_bic=null_bic,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "deep"}, indent=2), flush=True)
    print(f"WIDOWX_MM_{SOURCE.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()

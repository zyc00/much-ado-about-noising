#!/usr/bin/env python3
"""Independent adjacent-state audit for modes found by the WidowX screen.

For each pre-registered candidate, a discovery batch at the center state fixes
one action-space PCA direction. Fresh samples at nearby rollout states are then
projected onto that same direction. This tests whether a candidate is a stable
two-action choice or a temporal boundary whose mixture weight moves with state.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks

from probe_widowx_multimodality import (
    GROUP_COLUMNS,
    bh_qvalues,
    draw_chunks,
    empirical_p,
    encode_condition,
    fit_gmm2,
    group_view,
    null_delta_bic,
    to_cuda,
)


ROOT = Path("/mnt/pfs/yuchen/groot")
CHECKPOINT = os.environ.get("CKPT", str(ROOT / "ft_wxflow/checkpoint-20000"))
OUT_STEM = Path(os.environ.get("OUT", str(ROOT / "widowx_mm_offsets_v1")))
K_DISCOVERY = int(os.environ.get("K_DISCOVERY", "32"))
K_CONFIRM = int(os.environ.get("K_CONFIRM", "64"))
N_NULL = int(os.environ.get("N_NULL", "5000"))
SEED = int(os.environ.get("SEED", "20260904"))
OFFSETS = tuple(int(value) for value in os.environ.get("OFFSETS", "-4,-2,-1,0,1,2,4").split(","))

CASES = (
    ("close_drawer:11:2", "rotation"),
    ("close_drawer:267:1", "rotation"),
    ("close_drawer:385:0", "arm"),
    ("close_drawer:440:4", "rotation"),
    ("eggplant_basket:73:3", "arm"),
    ("eggplant_basket:148:1", "gripper"),
)
RECORDS = {
    "close_drawer": (ROOT / "rollout_dumps/flow_closedr.npz", "close the drawer"),
    "eggplant_basket": (
        ROOT / "rollout_dumps/flow_eggbask.npz",
        "put the eggplant in the yellow basket",
    ),
}
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")


def density_geometry(fit: dict[str, Any], z: np.ndarray) -> tuple[int, float]:
    pad = max(float(z.std()), 1e-3)
    grid = np.linspace(float(z.min() - pad), float(z.max() + pad), 5000)
    density = np.zeros_like(grid)
    for weight, mean, variance in zip(fit["weights"], fit["means"], fit["variances"]):
        density += (
            weight
            / math.sqrt(2 * math.pi * variance)
            * np.exp(-0.5 * (grid - mean) ** 2 / variance)
        )
    peaks, _ = find_peaks(density, prominence=density.max() * 1e-5)
    if len(peaks) < 2:
        return len(peaks), 0.0
    chosen = sorted(sorted(peaks, key=lambda i: density[i], reverse=True)[:2])
    valley = density[chosen[0] : chosen[1] + 1].min()
    depth = 1.0 - valley / min(density[chosen[0]], density[chosen[1]])
    return len(peaks), float(depth)


def build_input(policy: Any, arrays: dict[str, np.ndarray], instruction: str, t: int, e: int):
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    states = {
        key: arrays[f"obs.state.{key}"][t, e][None].astype(np.float32)
        for key in STATE_KEYS
    }
    image_sequence = arrays["obs.video.image_0"][t, e]
    step = VLAStepData(
        images={"image_0": image_sequence},
        states={key: value[0] for key, value in states.items()},
        actions={},
        text=instruction,
        embodiment=policy.embodiment_tag,
    )
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    return to_cuda(_rec_to_dtype(collated, dtype=torch.bfloat16)), image_sequence[0]


def main() -> None:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve("SIMPLER_ENV_WIDOWX"),
        model_path=CHECKPOINT,
        device="cuda",
        strict=True,
    )
    model = policy.model
    if model.action_head.num_inference_timesteps != 4:
        raise RuntimeError("Expected the evaluation setting NFE=4")

    needed_tasks = sorted({uid.split(":", 1)[0] for uid, _ in CASES})
    loaded = {}
    for task in needed_tasks:
        path, instruction = RECORDS[task]
        with np.load(path) as archive:
            arrays = {"obs.video.image_0": archive["obs.video.image_0"]}
            arrays.update(
                {f"obs.state.{key}": archive[f"obs.state.{key}"] for key in STATE_KEYS}
            )
        loaded[task] = (arrays, instruction)

    null = null_delta_bic(K_CONFIRM, N_NULL, SEED + 7)
    rows, raw, images = [], [], []
    with torch.inference_mode():
        for case_no, (uid, group) in enumerate(CASES):
            task, t_text, e_text = uid.split(":")
            center_t, ensemble = int(t_text), int(e_text)
            arrays, instruction = loaded[task]
            center_input, _ = build_input(policy, arrays, instruction, center_t, ensemble)
            center_condition = encode_condition(model, center_input)
            select = lambda action: action[:, :8, :7]
            discovery = draw_chunks(model, center_condition, K_DISCOVERY, select)
            gd = group_view(discovery, GROUP_COLUMNS[group])
            center = gd.mean(axis=0, keepdims=True)
            _, singular_values, vh = np.linalg.svd(gd - center, full_matrices=False)
            if not len(singular_values) or singular_values[0] < 1e-12:
                raise ValueError(f"Degenerate discovery batch for {uid}")
            axis = vh[0]

            case_raw, case_images = [], []
            for offset in OFFSETS:
                t = center_t + offset
                inputs, image = build_input(policy, arrays, instruction, t, ensemble)
                condition = encode_condition(model, inputs)
                confirmation = draw_chunks(model, condition, K_CONFIRM, select)
                projected = (group_view(confirmation, GROUP_COLUMNS[group]) - center) @ axis
                fit = fit_gmm2(projected)
                labels = fit.pop("labels")
                n_modes, valley_depth = density_geometry(fit, projected)
                delta = confirmation[labels == 1].mean(axis=0) - confirmation[labels == 0].mean(axis=0)
                energy = np.sum(delta * delta, axis=0)
                total = max(float(energy.sum()), 1e-18)
                rows.append(
                    {
                        "uid": uid,
                        "group": group,
                        "offset": offset,
                        "t": t,
                        "delta_bic": fit["delta_bic"],
                        "sep": fit["sep"],
                        "min_weight": fit["min_weight"],
                        "component_weights": fit["weights"].tolist(),
                        "component_means_on_center_axis": fit["means"].tolist(),
                        "projection_mean": float(projected.mean()),
                        "projection_sd": float(projected.std()),
                        "p": empirical_p(fit["delta_bic"], null),
                        "fitted_density_modes": n_modes,
                        "valley_depth": valley_depth,
                        "energy_translation": float(energy[:3].sum() / total),
                        "energy_rotation": float(energy[3:6].sum() / total),
                        "energy_gripper": float(energy[6] / total),
                    }
                )
                case_raw.append(confirmation.astype(np.float32))
                case_images.append(image)
                print(
                    f"{uid} offset={offset:+d} deltaBIC={fit['delta_bic']:.1f} "
                    f"sep={fit['sep']:.2f} w={fit['min_weight']:.2f} valley={valley_depth:.2f}",
                    flush=True,
                )
            raw.append(np.stack(case_raw))
            images.append(np.stack(case_images))
            print(f"case {case_no + 1}/{len(CASES)} done", flush=True)

    qvalues = bh_qvalues(np.array([row["p"] for row in rows]))
    for row, qvalue in zip(rows, qvalues):
        row["q"] = float(qvalue)
        row["detected"] = bool(
            qvalue <= 0.05
            and row["delta_bic"] >= 10.0
            and row["sep"] >= 2.0
            and row["min_weight"] >= 0.2
        )
    result = {
        "checkpoint": CHECKPOINT,
        "nfe": 4,
        "k_discovery": K_DISCOVERY,
        "k_confirmation": K_CONFIRM,
        "offsets": OFFSETS,
        "cases": CASES,
        "axis": "center-state PC1 from a fresh discovery batch; fixed across offsets",
        "multiple_testing": "BH across all case-offset confirmations",
        "rows": rows,
    }
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{OUT_STEM}.json", "w") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        f"{OUT_STEM}.npz",
        chunks=np.stack(raw),
        rgb=np.stack(images),
        uid=np.array([uid for uid, _ in CASES]),
        group=np.array([group for _, group in CASES]),
        offsets=np.array(OFFSETS),
        null_delta_bic=null,
    )
    print("WIDOWX_MM_OFFSETS_DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matched residual-bin gradient probe for RoboMimic Flow checkpoints.

The protocol mirrors the VLA E8 probe: obtain action residuals from one
converged Flow policy, bin samples by residual RMS, and evaluate the relative
output-gradient mass of MSE, Flow, and fixed-scale Student-t on those exact
samples. Binary gripper coordinates are excluded.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset


ROOT = Path(__file__).resolve().parents[1]
HF_SNAPSHOT = Path(
    "/home/jigu/.cache/huggingface/hub/models--ChaoyiPan--mip-checkpoints/"
    "snapshots/126fc8515dfb2da9b60c596778969bf764111d98"
)
OLD_CHECKPOINTS = Path("/home/jigu/projects/much-ado-about-noising-old/checkpoints")
OUT = ROOT / "analysis/paper/gradient_distribution/robomimic_gradient_probes.json"

TASKS = {
    "Tool-Hang": {
        "task": "tool_hang_ph_state_delta_legacy",
        "obs_dim": 53,
        "checkpoint": HF_SNAPSHOT
        / "robomimic/tool_hang_ph_state/delta_legacy/flow_chiunet_256_seed1_success92.pt",
    },
    "Transport-MH": {
        "task": "transport_mh_state_delta_legacy",
        "obs_dim": 59,
        "checkpoint": OLD_CHECKPOINTS
        / "transport_mh_state_flow_chiunet_256_seed2_success55.pt",
    },
    "Transport-PH": {
        "task": "transport_ph_state_delta_legacy",
        "obs_dim": 59,
        "checkpoint": OLD_CHECKPOINTS
        / "transport_ph_state_flow_chiunet_256_seed0_success75.pt",
    },
    "Square-PH": {
        "task": "square_ph_state_delta_legacy",
        "obs_dim": 23,
        "checkpoint": HF_SNAPSHOT
        / "robomimic/square_ph_state/delta_legacy/flow_chiunet_256_seed3_success97.pt",
    },
}

BIN_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0, np.inf])
BIN_LABELS = ["<0.5x", "0.5-1x", "1-2x", "2-4x", ">4x"]


def continuous_mask(action_dim: int, device: torch.device) -> torch.Tensor:
    """Rot6D action blocks have nine continuous coordinates and one gripper."""
    if action_dim % 10 != 0:
        raise ValueError(f"Expected 10-D action blocks, got action_dim={action_dim}")
    keep = torch.ones(action_dim, dtype=torch.bool, device=device)
    keep[9::10] = False
    return keep


def summarize(ratio: np.ndarray, masses: dict[str, np.ndarray]) -> dict:
    ids = np.digitize(ratio, BIN_EDGES[1:-1], right=False)
    result = {"bins": BIN_LABELS, "sample_percent": []}
    for method in masses:
        result[f"{method.lower()}_gradient_percent"] = []
    for bin_id in range(len(BIN_LABELS)):
        selected = ids == bin_id
        result["sample_percent"].append(float(100.0 * selected.mean()))
        for method, mass in masses.items():
            share = 100.0 * mass[selected].sum() / mass.sum() if selected.any() else 0.0
            result[f"{method.lower()}_gradient_percent"].append(float(share))
    result["top1_residual_gradient_percent"] = {}
    cutoff = np.quantile(ratio, 0.99)
    tail = ratio >= cutoff
    for method, mass in masses.items():
        result["top1_residual_gradient_percent"][method] = float(
            100.0 * mass[tail].sum() / mass.sum()
        )
    return result


def run_task(name: str, spec: dict, n_samples: int, draws: int) -> dict:
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "examples/configs")
    ):
        cfg = compose(
            config_name="main",
            overrides=[
                f"task={spec['task']}",
                "network=chiunet",
                "optimization.loss_type=flow",
                "optimization.auto_resume=false",
                "log.wandb_mode=disabled",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.task.horizon = 16
    cfg.task.obs_dim = spec["obs_dim"]
    cfg.optimization.sample_mode = "zero"

    dataset = make_dataset(cfg.task)
    indices = np.linspace(0, len(dataset) - 1, n_samples).astype(int)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=64, shuffle=False)

    agent = TrainingAgent(cfg)
    agent.load(str(spec["checkpoint"]), load_optimizer=False)
    agent.eval()
    device = torch.device(cfg.optimization.device)

    residual_sums = []
    flow_masses = []
    for batch_idx, batch in enumerate(loader):
        action = batch["action"][:, : cfg.task.horizon].to(device)
        obs = batch["obs"]["state"][:, : cfg.task.obs_steps].to(device)
        keep = continuous_mask(action.shape[-1], device)

        with torch.no_grad():
            prediction = agent.sample(
                act_0=torch.zeros_like(action), obs={"state": obs}, num_steps=9, use_ema=True
            )
            residual = prediction[..., keep] - action[..., keep]
            residual_sums.append(residual.square().sum(dim=(1, 2)).cpu().numpy())

            embedding = agent.encoder_ema(obs, None)
            draw_mass = []
            for draw_idx in range(draws):
                torch.manual_seed(100_000 * batch_idx + draw_idx)
                t = torch.rand(action.shape[0], device=device)
                action_0 = torch.randn_like(action)
                action_t = agent.interpolant.calc_It(t, action_0, action)
                target_velocity = agent.interpolant.calc_It_dot(t, action_0, action)
                predicted_velocity = agent.flow_map_ema.get_velocity(t, action_t, embedding)
                velocity_error = predicted_velocity[..., keep] - target_velocity[..., keep]
                draw_mass.append(velocity_error.square().sum(dim=(1, 2)))
            flow_masses.append(torch.stack(draw_mass).mean(dim=0).cpu().numpy())

    squared_residual = np.concatenate(residual_sums)
    flow_mass = np.concatenate(flow_masses)
    n_continuous = cfg.task.horizon * int(continuous_mask(cfg.task.act_dim, device).sum())
    sigma2 = squared_residual.mean() / n_continuous
    residual_ratio = np.sqrt(squared_residual / n_continuous) / np.sqrt(sigma2)

    nu = 2.0 * n_continuous
    weight = (nu + n_continuous) / (nu + squared_residual / sigma2)
    masses = {
        "MSE": squared_residual,
        "Flow": flow_mass,
        "HT": np.square(weight) * squared_residual,
    }
    result = summarize(residual_ratio, masses)
    result.update(
        {
            "task_config": spec["task"],
            "checkpoint": str(spec["checkpoint"]),
            "n_samples": int(len(squared_residual)),
            "flow_draws": draws,
            "flow_inference_steps": 9,
            "continuous_dimensions": n_continuous,
            "gripper_excluded": True,
            "pooled_rms": float(np.sqrt(sigma2)),
            "nu": float(nu),
        }
    )
    print(name, json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    torch.set_float32_matmul_precision("high")
    n_samples = int(os.environ.get("RM_GRAD_N", "1024"))
    draws = int(os.environ.get("RM_GRAD_DRAWS", "8"))
    output = {
        "protocol": "matched Flow-residual bins; continuous action channels only",
        "tasks": {
            name: run_task(name, spec, n_samples, draws) for name, spec in TASKS.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

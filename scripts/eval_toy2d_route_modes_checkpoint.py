"""Reload a saved route-mode checkpoint and independently rerun evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toy2d_mip_nfl import ViewNet
from toy2d_route_modes_benchmark import branch_metrics, rollout_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--n-eval", type=int, default=401)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if bundle.get("format") != "toy2d-route-modes-v1":
        raise ValueError(f"unsupported checkpoint format: {bundle.get('format')}")
    action_dim = int(bundle["action_dim"])
    width = int(bundle["width"])
    device = torch.device("cpu")
    evaluated = {}
    specs = (
        ("regression", "regression", "regression", False),
        ("mip_step1", "mip", "mip_step1", False),
        ("mip_full", "mip", "mip_full", False),
        ("hg", "hg", "hg", True),
        ("ht", "ht", "ht", True),
    )
    for result_name, weight_name, sampler, scalar_head in specs:
        net = ViewNet(action_dim, width, scalar_head=scalar_head)
        net.load_state_dict(bundle["models"][weight_name])
        net.eval()
        metrics = {
            **branch_metrics(net, sampler, device),
            **rollout_metrics(net, sampler, device, args.n_eval),
        }
        recorded = bundle["metrics"][result_name]
        evaluated[result_name] = metrics
        print(
            f"{result_name:11s} "
            f"loaded SR={metrics['success_rate']:.6f} "
            f"recorded SR={recorded['success_rate']:.6f} "
            f"loaded first={metrics['first_lateral_action_mean']:+.6f} "
            f"recorded first={recorded['first_lateral_action_mean']:+.6f}",
            flush=True,
        )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": int(bundle["seed"]),
        "evaluation": evaluated,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

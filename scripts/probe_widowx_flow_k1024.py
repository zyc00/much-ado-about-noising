#!/usr/bin/env python3
"""Flow-only rerun of probe_widowx_general_scale.py with K = 1024 samples per state on the SAME 1,728 states
(same episode selection, same 12 chunk starts), so the per-state conditional scale of the Flow policy can be
estimated from 1024 draws instead of 16. Output: <output>/flow_rank{RANK}.npz (samples [n, 1024, 8, 6])."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_widowx_general_scale import ROOT, select_episodes, load_model, probe  # noqa: E402

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "bridge_orig_lerobot"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=48); p.add_argument("--steps", type=int, default=12)
    p.add_argument("--k", type=int, default=1024); p.add_argument("--draw-batch", type=int, default=64)
    args = p.parse_args()
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0"))); args.output.mkdir(parents=True, exist_ok=True)
    episodes, tasks = select_episodes(args.data, args.episodes); episodes = episodes[rank::world]
    checkpoint = ROOT / "ft_wxflow" / "checkpoint-20000"; path = args.output / f"flow_rank{rank}.npz"
    if path.exists(): raise FileExistsError(path)
    model, processor = load_model(checkpoint, "flow")
    result = probe(model, processor, episodes, args, "flow")
    meta = dict(checkpoint=str(checkpoint), objective="flow", task_names=tasks, k=args.k,
                task_selection="three most frequent nonempty exact instructions among episodes >=24 frames",
                sample_selection="48 random episodes/task, 12 uniformly spaced full-chunk start positions (identical to the k=16 probe)",
                continuous_channels=list(range(6)), gripper_excluded=True)
    np.savez_compressed(path, **result, metadata=json.dumps(meta)); print("SAVED", path, flush=True)

if __name__ == "__main__":
    main()

"""Dataset-wide general-MSE audit with instruction-stratified episode sampling.

All instruction groups enter the sampling frame, including an unlabeled group.
Draw up to 12 episodes per group, then one full-chunk start per progress decile.
Store inverse episode-selection weights for dataset-wide summaries; only groups
with >=12 selected episodes qualify for the multi-demonstration task display.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

SEED = 20260907
SETTINGS = {
    "bridge": ("bridge_orig_lerobot", "ft_wxmse", "SIMPLER_ENV_WIDOWX"),
    "fractal": ("fractal_lerobot", "ft_fr_mse", "SIMPLER_ENV_GOOGLE"),
}

def make_manifest(root, dataset, per_task):
    data_dir, checkpoint_dir, tag = SETTINGS[dataset]
    path = root / data_dir / "meta/episodes.jsonl"
    groups = defaultdict(list)
    total = short = 0
    for ordinal, line in enumerate(path.open()):
        row = json.loads(line)
        total += 1
        if row["length"] < 8:
            short += 1
            continue
        instruction = (row.get("tasks") or [""])[0].strip().lower()
        groups[instruction].append(dict(ordinal=ordinal, episode=int(row["episode_index"]), length=int(row["length"])))
    rng = np.random.default_rng(SEED)
    tasks, episodes = [], []
    for tid, name in enumerate(sorted(groups)):
        pool = groups[name]
        selected = np.sort(rng.choice(len(pool), min(per_task, len(pool)), replace=False))
        tasks.append(dict(task_id=tid, instruction=name, total_episodes=len(pool), sampled_episodes=len(selected),
                          display_eligible=bool(name) and len(selected) >= 12))
        for j in selected:
            ep = dict(pool[int(j)], task_id=tid, episode_weight=len(pool)/len(selected))
            # No padded labels. Some short episodes cannot contribute to late bins.
            starts = np.arange(ep["length"] - 7)
            stages = np.minimum(9, (starts / max(ep["length"] - 1, 1) * 10).astype(int))
            ep["steps"] = [int(rng.choice(starts[stages == b])) for b in range(10) if np.any(stages == b)]
            episodes.append(ep)
    return dict(dataset=dataset, root=str(root/data_dir), checkpoint=str(root/checkpoint_dir/"checkpoint-20000"),
                embodiment=tag, seed=SEED, sample_cap_per_instruction=per_task,
                metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                total_episodes=total, excluded_short_episodes=short,
                eligible_episodes=total-short, instruction_groups=len(tasks),
                display_task_count=sum(t["display_eligible"] for t in tasks),
                sampled_episodes=len(episodes), sampled_states=sum(len(e["steps"]) for e in episodes),
                scope="Training-demonstration diagnostic, not policy-held-out. Instruction groups use first instruction, stripped/lowercased, no semantic merging.",
                tasks=tasks, episodes=episodes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SETTINGS, required=True)
    ap.add_argument("--root", type=Path, default=Path("/mnt/pfs/yuchen/groot"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=12)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args.root, args.dataset, args.per_task)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    path = args.output / "sampling_manifest.json"
    if rank == 0:
        if path.exists():
            assert json.loads(path.read_text()) == manifest, "Do not replace a different sampling manifest"
        else:
            path.write_text(json.dumps(manifest) + "\n")
        print(json.dumps({k:v for k,v in manifest.items() if k not in ["tasks", "episodes"]}), flush=True)
    if args.manifest_only:
        return
    import torch
    from probe_widowx_general_scale import load_model
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    model, processor = load_model(Path(manifest["checkpoint"]), "mse")
    tag = EmbodimentTag.resolve(manifest["embodiment"])
    loader = LeRobotEpisodeLoader(Path(manifest["root"]), processor.modality_configs[tag.value])
    selected = manifest["episodes"][rank::world]
    if args.max_episodes is not None:
        selected = selected[:args.max_episodes]
    rows, shard_no, start_time = [], 0, time.monotonic()
    def flush():
        nonlocal rows, shard_no
        if not rows:
            return
        target_path = args.output / f"rank{rank:02d}_part{shard_no:04d}.npz"
        if target_path.exists():
            raise FileExistsError(target_path)
        meta = dict(dataset=args.dataset, checkpoint=manifest["checkpoint"], rank=rank, world=world,
                    objective="mse", action_channels=list(range(6)), horizon=8,
                    manifest=str(path), smoke_only=args.max_episodes is not None)
        np.savez_compressed(target_path, **{k:np.asarray([r[k] for r in rows]) for k in rows[0]}, metadata=json.dumps(meta))
        rows, shard_no = [], shard_no+1
    for ei, ep in enumerate(selected):
        episode = loader[ep["ordinal"]]
        steps = ep["steps"]
        for offset in range(0, len(steps), args.batch_size):
            active = steps[offset:offset+args.batch_size]
            features = []
            for step in active:
                item = extract_step_data(episode, step, processor.modality_configs[tag.value], tag, allow_padding=False)
                features.append(processor([{"type": MessageType.EPISODE_STEP.value, "content": item}]))
            batch = processor.collator(features)
            inner = batch["inputs"] if "inputs" in batch else batch
            mask = inner["action_mask"].bool()
            assert mask[:, :8, :6].all() and not mask[:, 8:].any(), mask.shape
            target = inner["action"][:, :8, :6].float().cpu().numpy()
            inputs = {k:v for k,v in inner.items() if k != "action"}
            torch.manual_seed(SEED+ep["episode"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.get_action(inputs)
                pred = out["action_pred"][:, :8, :6].float().cpu().numpy()
            assert pred.shape == target.shape and np.isfinite(pred).all() and np.isfinite(target).all()
            for k, step in enumerate(active):
                rows.append(dict(episode=ep["episode"], task_id=ep["task_id"], step=step, length=ep["length"],
                                 progress=step/(ep["length"]-1), episode_weight=ep["episode_weight"],
                                 target=target[k], prediction=pred[k]))
        if (ei+1)%100 == 0:
            flush()
            print(f"{args.dataset} rank={rank} episodes={ei+1}/{len(selected)} elapsed={time.monotonic()-start_time:.1f}s", flush=True)
    flush()
    (args.output / f"rank{rank:02d}_done.json").write_text(json.dumps(dict(episodes=len(selected), shards=shard_no, elapsed=time.monotonic()-start_time)))
    print("PROBE_DONE", args.dataset, rank, flush=True)

if __name__ == "__main__":
    main()

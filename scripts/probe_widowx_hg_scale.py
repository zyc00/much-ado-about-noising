#!/usr/bin/env python3
"""Per-state residual and per-sample sigma of a heteroscedastic-Gaussian (or HT) GR00T checkpoint on the SAME
1,728 WidowX states used by probe_widowx_general_scale.py (same episodes, same chunk starts, same target selection).
Prediction = the head's single-pass action (t = 0, zero trajectory) captured from action_decoder; sigma = the loss's
per-sample scalar: masked chunk-mean softplus(s_raw + ht_sbias) + 1e-3 over ALL valid action dims (incl. gripper),
exactly as in the training loss. Output: <output>/hg_rank0.npz with target/prediction/residual (8 x 6 continuous
dims), sigma, m (mean squared residual over all valid dims), and episode/task/step/progress ids."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_widowx_general_scale import ROOT, SEED, select_episodes, load_model  # noqa: E402

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "bridge_orig_lerobot"); p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--steps", type=int, default=12); p.add_argument("--tag", default="hg")
    args = p.parse_args()
    episodes, tasks = select_episodes(args.data, args.episodes)
    model, processor = load_model(args.checkpoint, "hetero_t")
    sbias = float(model.config.ht_sbias); print("ht_sbias", sbias, "hg_steps", getattr(model.config, "ht_hg_steps", None), flush=True)
    captured = {}
    model.action_head.action_decoder.register_forward_hook(lambda m, i, o: captured.__setitem__("pred", o.detach()))
    model.action_head.sigma_decoder.register_forward_hook(lambda m, i, o: captured.__setitem__("s_raw", o.detach()))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    tag = EmbodimentTag.resolve("SIMPLER_ENV_WIDOWX"); modalities = processor.modality_configs[tag.value]
    loader = LeRobotEpisodeLoader(args.data, modalities); rows = []; start = time.monotonic()
    for ep_idx, ep in enumerate(episodes):
        episode = loader[ep["ordinal"]]
        steps = np.unique(np.rint(np.linspace(0, ep["length"] - 8, args.steps)).astype(int))
        for step in steps:
            torch.manual_seed(SEED + ep["episode_index"] * 1000 + int(step))
            data = extract_step_data(episode, int(step), modalities, tag, allow_padding=False)
            feature = processor([{"type": MessageType.EPISODE_STEP.value, "content": data}])
            batch = processor.collator([feature]); inner = batch["inputs"] if "inputs" in batch else batch
            mask = inner["action_mask"].bool()
            tv = torch.nonzero(mask[0].any(dim=1)).flatten(); av_all = torch.nonzero(mask[0].any(dim=0)).flatten(); av = av_all[:6]
            assert len(tv) == 8 and len(av) == 6, (tv, av_all)
            act = inner["action"].float()
            batch_cuda = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()} if "inputs" not in batch else \
                {"inputs": {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch["inputs"].items()}}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**batch_cuda)
            pred = captured["pred"][0].float().cpu(); s_raw = captured["s_raw"][0].float().cpu()
            T = act.shape[1]; pred = pred[-T:]; s_raw = s_raw[-T:]
            m_all = mask[0].float()
            sp = torch.nn.functional.softplus(s_raw + sbias) * m_all
            sigma = float(sp.sum() / m_all.sum() + 1e-3)
            r_all = (act[0] - pred) * m_all
            m = float((r_all ** 2).sum() / m_all.sum())
            target = act[0][tv][:, av].numpy(); prediction = pred[tv][:, av].numpy()
            rows.append(dict(episode=ep["episode_index"], task_id=ep["task_id"], step=int(step), length=ep["length"],
                             progress=step / (ep["length"] - 1), target=target, prediction=prediction, residual=prediction - target,
                             sigma=sigma, m=m, d=float(m_all.sum())))
        print(f"{args.tag} episodes={ep_idx+1}/{len(episodes)} states={len(rows)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    result = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
    meta = dict(checkpoint=str(args.checkpoint), objective=args.tag, task_names=tasks, ht_sbias=sbias,
                sigma="masked chunk-mean softplus(s_raw + ht_sbias) + 1e-3 over all valid dims (loss definition)",
                continuous_channels=[0, 1, 2, 3, 4, 5], gripper_excluded_in_residual=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.tag}_rank0.npz", **result, metadata=json.dumps(meta))
    print("saved", args.output / f"{args.tag}_rank0.npz", "sigma median", float(np.median(result["sigma"])), "rms residual", float(np.sqrt(np.mean(result["residual"] ** 2))))

if __name__ == "__main__":
    main()

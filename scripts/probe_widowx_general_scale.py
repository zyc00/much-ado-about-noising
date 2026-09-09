#!/usr/bin/env python3
"""Probe frozen general MSE and Flow on identical BridgeData observations.

No action-magnitude selection and no specialist checkpoints. Task selection is
by instruction frequency only. Original policies saw these demonstrations;
episode holdout is used later only for the variance-fit diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

SEED = 20260906
ROOT = Path('/mnt/pfs/yuchen/groot')


def map_tensors(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, Mapping):
        return type(value)({k: map_tensors(v, fn) for k, v in value.items()})
    return value


def select_episodes(data_root, count):
    meta = [json.loads(s) for s in (data_root / 'meta/episodes.jsonl').open()]
    eligible = [dict(row, ordinal=i, name=row['tasks'][0].strip().lower())
                for i, row in enumerate(meta) if row['length'] >= 24 and row['tasks'][0].strip()]
    tasks = [name for name, _ in Counter(row['name'] for row in eligible).most_common(3)]
    rng = np.random.default_rng(SEED)
    selected = []
    for tid, task in enumerate(tasks):
        candidates = [r for r in eligible if r['name'] == task]
        for pos in rng.choice(len(candidates), min(count, len(candidates)), replace=False):
            selected.append(dict(candidates[pos], task_id=tid))
    return sorted(selected, key=lambda r: (r['task_id'], r['episode_index'])), tasks


def load_model(checkpoint, expected):
    from transformers import AutoModel, AutoProcessor
    import gr00t.model  # noqa: F401
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    backbone = '/mnt/pfs/yuchen/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
    if not hasattr(processing, '_scale_original_builder'):
        processing._scale_original_builder = processing.build_processor
    processing.build_processor = lambda name, kwargs: processing._scale_original_builder(backbone, kwargs)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True,
        model_name=backbone, transformers_loading_kwargs={'local_files_only': True})
    processor.eval()
    model, info = AutoModel.from_pretrained(str(checkpoint), local_files_only=True,
        output_loading_info=True, transformers_loading_kwargs={'local_files_only': True})
    assert not info.get('missing_keys') and not info.get('unexpected_keys'), info
    assert model.config.loss_type == expected, model.config.loss_type
    return model.cuda().eval(), processor


def probe(model, processor, episodes, args, mode):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    tag = EmbodimentTag.resolve('SIMPLER_ENV_WIDOWX')
    modalities = processor.modality_configs[tag.value]
    loader = LeRobotEpisodeLoader(args.data, modalities)
    rows = []
    start = time.monotonic()
    for ep_idx, ep in enumerate(episodes):
        episode = loader[ep['ordinal']]
        steps = np.unique(np.rint(np.linspace(0, ep['length'] - 8, args.steps)).astype(int))
        for step in steps:
            torch.manual_seed(SEED + ep['episode_index'] * 1000 + int(step))
            data = extract_step_data(episode, int(step), modalities, tag, allow_padding=False)
            feature = processor([{'type': MessageType.EPISODE_STEP.value, 'content': data}])
            batch = processor.collator([feature])
            inner = batch['inputs'] if 'inputs' in batch else batch
            mask = inner['action_mask'].bool()
            tv = torch.nonzero(mask[0].any(dim=1)).flatten()
            av = torch.nonzero(mask[0].any(dim=0)).flatten()[:6]
            assert len(tv) == 8 and len(av) == 6, (tv, av)
            target = inner['action'][:, tv][:, :, av].float().cpu().numpy()[0]
            inputs = {k: v for k, v in inner.items() if k != 'action'}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                backbone_inputs, action_input = model.prepare_input(inputs)
                backbone_output = model.backbone(backbone_inputs)
                features = model.action_head._encode_features(backbone_output, action_input)
                condition = dict(backbone_features=features.backbone_features,
                    state_features=features.state_features, embodiment_id=action_input.embodiment_id,
                    backbone_output=backbone_output, action_input=action_input)
                predictions = []
                k = 1 if mode == 'mse' else args.k
                for offset in range(0, k, args.draw_batch):
                    draws = min(args.draw_batch, k-offset)
                    expanded = map_tensors(condition, lambda x: x.repeat((draws,) + (1,)*(x.ndim-1))
                        if x.ndim and x.shape[0] == 1 else x)
                    out = model.action_head.get_action_with_features(**expanded)['action_pred']
                    predictions.append(out[:, tv.to(out.device)][:, :, av.to(out.device)].float().cpu().numpy())
                predictions = np.concatenate(predictions)
                state = action_input.state.float().cpu().numpy().reshape(-1)
                embedding = features.backbone_features.float().mean(dim=1).cpu().numpy()[0]
            row = dict(episode=ep['episode_index'], task_id=ep['task_id'], step=int(step),
                length=ep['length'], progress=step / (ep['length']-1), target=target,
                prediction=predictions.mean(axis=0), residual=predictions.mean(axis=0)-target)
            if mode == 'mse':
                row.update(state=state, embedding=embedding)
            else:
                row.update(samples=predictions, spread=np.sqrt(np.mean(
                    (predictions-predictions.mean(axis=0, keepdims=True))**2, dtype=np.float64)))
            rows.append(row)
        print(f'{mode} rank={os.environ.get("RANK",0)} episodes={ep_idx+1}/{len(episodes)} states={len(rows)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'bridge_orig_lerobot')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=48)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--draw-batch', type=int, default=8)
    args = p.parse_args()
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, tasks = select_episodes(args.data, args.episodes)
    episodes = episodes[rank::world]
    for mode, directory in [('mse', 'ft_wxmse'), ('flow', 'ft_wxflow')]:
        checkpoint = ROOT/directory/'checkpoint-20000'
        path = args.output/f'{mode}_rank{rank}.npz'
        if path.exists():
            raise FileExistsError(path)
        model, processor = load_model(checkpoint, mode)
        result = probe(model, processor, episodes, args, mode)
        meta = dict(checkpoint=str(checkpoint), objective=mode, task_names=tasks,
            task_selection='three most frequent nonempty exact instructions among episodes >=24 frames',
            sample_selection='48 random episodes/task, 12 uniformly spaced full-chunk start positions',
            policy_split='diagnostic on training demonstrations; not policy-held-out',
            continuous_channels=list(range(6)), gripper_excluded=True, k=(1 if mode=='mse' else args.k))
        np.savez_compressed(path, **result, metadata=json.dumps(meta))
        print('SAVED', path, flush=True)
        del model, processor, result
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

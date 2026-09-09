"""Frozen HT diagnostic: exact shared sigma, channel errors, temporal change, images.

No fitting or checkpoint/config changes. Frequency-selected tasks, random episodes,
uniform valid chunk starts, and original checkpoint normalization throughout.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path('/mnt/pfs/yuchen/groot')
SETTINGS = {
 'bridge': ('bridge_orig_lerobot', 'ft_wxnu224/checkpoint-20000', 'SIMPLER_ENV_WIDOWX'),
 'fractal': ('fractal_lerobot', 'ft_fr_nu224/checkpoint-20000', 'SIMPLER_ENV_GOOGLE'),
 'gr1': ('lerobot/LeRobot', 'ft_gr1c2/checkpoint-60000', 'ROBOCASA_GR1_TABLETOP'),
}
SEED = 20260908

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=SETTINGS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=int, default=24)
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--steps', type=int, default=20)
    args = p.parse_args()
    import torch
    from PIL import Image
    from probe_widowx_general_scale import load_model, map_tensors
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    data_dir, ckpt, embodiment = SETTINGS[args.dataset]
    paths = sorted((ROOT/data_dir).glob('gr1_unified.*')) if args.dataset == 'gr1' else [ROOT/data_dir]
    groups = defaultdict(list)
    for path in paths:
        for ordinal, line in enumerate((path/'meta/episodes.jsonl').open()):
            row = json.loads(line)
            name = path.name if args.dataset == 'gr1' else (row.get('tasks') or [''])[0].strip().lower()
            if row['length'] >= 24 and name:
                groups[name].append(dict(path=str(path), ordinal=ordinal, episode=row['episode_index'], length=row['length']))
    names = sorted(groups, key=lambda n: (-len(groups[n]), n))[:args.tasks]
    rng = np.random.default_rng(SEED)
    selected = []
    for tid, name in enumerate(names):
        pool = groups[name]
        for j in np.sort(rng.choice(len(pool), min(args.episodes, len(pool)), replace=False)):
            selected.append(dict(pool[int(j)], task_id=tid))
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'probe.npz').exists()
    model, proc = load_model(ROOT/ckpt, 'hetero_t')
    cfg = model.action_head.config
    assert cfg.ht_mvt and not getattr(cfg, 'ht_gripper_bce', False)
    tag = EmbodimentTag.resolve(embodiment)
    modalities = proc.modality_configs[tag.value]
    meta = dict(dataset=args.dataset, checkpoint=str(ROOT/ckpt), seed=SEED, tasks=names,
        task_selection='24 most frequent nonempty instructions; GR1 all 24 task datasets',
        episodes_per_task=args.episodes, steps_per_episode=args.steps,
        scope='Training-demonstration descriptive diagnostic, not held-out or contact ground truth.',
        ht_df=cfg.ht_df, ht_sbias=cfg.ht_sbias, action_keys=modalities['action'].modality_keys,
        sigma='masked mean softplus(s_raw + checkpoint ht_sbias) + 1e-3; all valid channels',
        episodes=selected)
    (args.output/'protocol.json').write_text(json.dumps(meta, indent=2))
    cap = {}
    model.action_head.action_decoder.register_forward_hook(lambda m,i,o: cap.__setitem__('pred', o.detach()))
    model.action_head.sigma_decoder.register_forward_hook(lambda m,i,o: cap.__setitem__('raw', o.detach()))
    loader = None; last_path = None; rows = []; thumbnails = []; start = time.monotonic()
    for ei, ep in enumerate(selected):
        if ep['path'] != last_path:
            loader = LeRobotEpisodeLoader(Path(ep['path']), modalities); last_path = ep['path']
        episode = loader[ep['ordinal']]
        steps = np.unique(np.rint(np.linspace(0, ep['length']-8, args.steps)).astype(int))
        for offset in range(0, len(steps), 8):
            active = steps[offset:offset+8]
            feats = []; thumbs = []
            for step in active:
                item = extract_step_data(episode, int(step), modalities, tag, allow_padding=False)
                frame = np.asarray(next(iter(item.images.values()))[-1])
                thumbs.append(np.asarray(Image.fromarray(frame).convert('RGB').resize((192,128))))
                feats.append(proc([{'type': MessageType.EPISODE_STEP.value, 'content': item}]))
            batch = proc.collator(feats); inner = batch.get('inputs', batch)
            target = inner['action'].float().cpu()
            mask = inner['action_mask'].bool().cpu()
            valid_t = torch.nonzero(mask[0].any(1)).flatten()
            valid_a = torch.nonzero(mask[0].any(0)).flatten()
            assert len(valid_t) == 8 and len(valid_a) == (29 if args.dataset == 'gr1' else 7)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(map_tensors(inner, lambda t: t.cuda()))
            pred = cap['pred'][:, -target.shape[1]:].float().cpu()
            sp = torch.nn.functional.softplus(cap['raw'][:, -target.shape[1]:].float().cpu() + cfg.ht_sbias)
            sigma = (sp * mask).sum((1,2))/mask.sum((1,2)) + .001
            assert torch.isfinite(sigma).all()
            if ei == 0 and offset == 0:
                # Check zero-input training path agrees with deployment prediction.
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    inf = model.get_action({k:v for k,v in inner.items() if k != 'action'})['action_pred'].float().cpu()
                err = float((inf[:,valid_t][:,:,valid_a]-pred[:,valid_t][:,:,valid_a]).abs().max())
                print('INFERENCE_PATH_MAX_DIFF', err, flush=True)
                assert err < .03, err
                meta['inference_forward_max_difference'] = err
                (args.output/'protocol.json').write_text(json.dumps(meta, indent=2))
            for k, step in enumerate(active):
                sub = lambda v: v[k,valid_t][:,valid_a].numpy()
                rows.append(dict(task_id=ep['task_id'], episode=ep['episode'], episode_uid=ei,
                    step=int(step), length=ep['length'], progress=step/(ep['length']-1),
                    sigma=float(sigma[k]), target=sub(target), prediction=sub(pred), sigma_components=sub(sp)))
                thumbnails.append(thumbs[k])
        if (ei+1)%12 == 0:
            print(f"{args.dataset} {ei+1}/{len(selected)} episodes {len(rows)} states {time.monotonic()-start:.0f}s", flush=True)
    np.savez_compressed(args.output/'probe.npz', **{k:np.asarray([r[k] for r in rows]) for k in rows[0]}, metadata=json.dumps(meta))
    np.savez_compressed(args.output/'thumbnails.npz', images=np.asarray(thumbnails))
    (args.output/'done.json').write_text(json.dumps(dict(states=len(rows), episodes=len(selected), elapsed=time.monotonic()-start)))
    print('SIGMA_CONTEXT_DONE', args.dataset, flush=True)

if __name__ == '__main__':
    main()

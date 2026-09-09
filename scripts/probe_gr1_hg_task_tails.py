"""Frozen pure-HG GR1 probe with task/episode identities, no training changes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path('/mnt/pfs/yuchen/groot')
SEED = 20260908


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=24)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--memory-fraction', type=float, default=.28)
    p.add_argument('--start-task', type=int, default=0)
    p.add_argument('--stop-task', type=int, default=24)
    args = p.parse_args()
    import torch
    from probe_widowx_general_scale import load_model, map_tensors
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    checkpoint = ROOT/'ft_gr1purehg/checkpoint-22000'
    config = json.loads((checkpoint/'config.json').read_text())
    assert config['ht_hg_steps'] > 22000 and config['hg_mode_now']
    paths = sorted((ROOT/'lerobot/LeRobot').glob('gr1_unified.*'))
    assert len(paths) == 24
    tasks = [path.name for path in paths]
    selected = []
    for tid, path in enumerate(paths):
        candidates = []
        for ordinal, line in enumerate((path/'meta/episodes.jsonl').open()):
            row = json.loads(line)
            if row['length'] >= 24:
                candidates.append(dict(ordinal=ordinal, episode=row['episode_index'],
                                       length=row['length'], task_id=tid))
        rng = np.random.default_rng(SEED+tid)
        selected.append([candidates[int(i)] for i in sorted(rng.choice(
            len(candidates), min(args.episodes, len(candidates)), replace=False))])
    args.output.mkdir(parents=True, exist_ok=True)
    meta = dict(checkpoint=str(checkpoint), objective='hg', dataset='RoboCasa-GR1',
                task_names=tasks, episodes=selected, seed=SEED,
                ht_sbias=config['ht_sbias'], hg_steps=config['ht_hg_steps'],
                sample_selection='All 24 tasks; random episodes per task; uniformly spaced valid full-chunk starts.',
                policy_split='Training demonstration diagnostic, not policy-held-out.',
                continuous_channels=list(range(29)), gripper_excluded_in_residual=False,
                hands='All continuous GR1 hand joints retained.',
                sigma='masked mean softplus(s_raw + checkpoint ht_sbias) + 1e-3 over all valid loss coordinates',
                steps_per_episode=args.steps, episodes_per_task=args.episodes)
    (args.output/'protocol.json').write_text(json.dumps(meta, indent=2))
    model, proc = load_model(checkpoint, 'hetero_t')
    assert model.action_head.config.ht_sbias == config['ht_sbias']
    tag = EmbodimentTag.resolve('ROBOCASA_GR1_TABLETOP')
    modalities = proc.modality_configs[tag.value]
    meta['action_keys'] = modalities['action'].modality_keys
    captured = {}
    model.action_head.action_decoder.register_forward_hook(
        lambda m,i,o: captured.__setitem__('pred', o.detach()))
    model.action_head.sigma_decoder.register_forward_hook(
        lambda m,i,o: captured.__setitem__('raw', o.detach()))
    start = time.monotonic()
    for tid, path in enumerate(paths):
        if not args.start_task <= tid < args.stop_task:
            continue
        dest = args.output/f'hg_task{tid:02d}.npz'
        if dest.exists():
            print('EXISTS', dest, flush=True)
            continue
        loader = LeRobotEpisodeLoader(path, modalities)
        rows = []
        for ei, ep in enumerate(selected[tid]):
            episode = loader[ep['ordinal']]
            steps = np.unique(np.rint(np.linspace(0, ep['length']-8, args.steps)).astype(int))
            for offset in range(0, len(steps), args.batch):
                active = steps[offset:offset+args.batch]
                feats = []
                for step in active:
                    item = extract_step_data(episode, int(step), modalities, tag, allow_padding=False)
                    feats.append(proc([{'type': MessageType.EPISODE_STEP.value, 'content': item}]))
                batch = proc.collator(feats)
                inner = batch.get('inputs', batch)
                target = inner['action'].float().cpu()
                mask = inner['action_mask'].bool().cpu()
                tv = torch.nonzero(mask[0].any(1)).flatten()
                av = torch.nonzero(mask[0].any(0)).flatten()
                assert len(tv) == 8 and len(av) == 29
                assert torch.equal(mask, mask[:1].expand_as(mask))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    model(map_tensors(inner, lambda t: t.cuda()))
                pred = captured['pred'][:, -target.shape[1]:].float().cpu()
                raw = captured['raw'][:, -target.shape[1]:].float().cpu()
                sp = torch.nn.functional.softplus(raw + config['ht_sbias'])
                sigma = (sp*mask).sum((1,2))/mask.sum((1,2)) + .001
                assert torch.isfinite(sigma).all() and torch.isfinite(pred).all()
                if tid == 0 and ei == 0 and offset == 0:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        inferred = model.get_action({k:v for k,v in inner.items() if k != 'action'})['action_pred'].float().cpu()
                    difference = float((inferred[:,tv][:,:,av]-pred[:,tv][:,:,av]).abs().max())
                    assert difference < .03, difference
                    meta['inference_forward_max_difference'] = difference
                    print('INFERENCE_PATH_MAX_DIFF', difference, flush=True)
                for k, step in enumerate(active):
                    sub = lambda v: v[k,tv][:,av].numpy()
                    a, mu = sub(target), sub(pred)
                    rows.append(dict(task_id=tid, episode=ep['episode'], step=int(step),
                                     length=ep['length'], progress=step/(ep['length']-1),
                                     sigma=float(sigma[k]), target=a, prediction=mu, residual=mu-a))
            if (ei+1)%6 == 0:
                print(f'task={tid+1}/24 ep={ei+1}/{len(selected[tid])} elapsed={time.monotonic()-start:.0f}s', flush=True)
        np.savez_compressed(dest, **{k:np.asarray([r[k] for r in rows]) for k in rows[0]},
                            metadata=json.dumps(meta))
        print('SAVED', dest, 'states', len(rows), 'peak_GB', torch.cuda.max_memory_allocated()/1e9, flush=True)
    if 'inference_forward_max_difference' in meta:
        (args.output/'protocol.json').write_text(json.dumps(meta, indent=2))
    if len(list(args.output.glob('hg_task*.npz'))) == len(paths):
        (args.output/'done.json').write_text(json.dumps(dict(tasks=len(paths), elapsed=time.monotonic()-start)))
    print('GR1_HG_TASK_PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()

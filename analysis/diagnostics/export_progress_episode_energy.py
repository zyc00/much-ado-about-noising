"""Lossless-to-float32 compact episode/stage energy export from existing MSE probes."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for dataset in ('bridge', 'fractal'):
    folder = args.root / dataset
    manifest = json.loads((folder/'sampling_manifest.json').read_text())
    tasks = [t for t in manifest['tasks'] if t['display_eligible']]
    selected = {int(t['task_id']): {} for t in tasks}
    for rank in range(2):
        done = json.loads((folder/f'rank{rank:02d}_done.json').read_text())
        for part in range(done['shards']):
            with np.load(folder/f'rank{rank:02d}_part{part:04d}.npz') as z:
                meta = json.loads(str(z['metadata']))
                assert meta['checkpoint'] == manifest['checkpoint'] and not meta['smoke_only']
                keep = np.isin(z['task_id'], list(selected))
                e = np.mean((z['prediction'][keep].astype(float)-z['target'][keep].astype(float))**2, axis=(1,2))
                for tid, ep, progress, energy in zip(z['task_id'][keep],z['episode'][keep],z['progress'][keep],e):
                    bins = selected[int(tid)].setdefault(int(ep), [[] for _ in range(10)])
                    bins[min(9,int(progress*10))].append(float(energy))
    matrices, episodes = [], []
    for task in tasks:
        per_ep = selected[int(task['task_id'])]
        assert len(per_ep) == task['sampled_episodes'] == 12
        ids = sorted(per_ep)
        matrices.append([[np.mean(b) if b else np.nan for b in per_ep[ep]] for ep in ids])
        episodes.append(ids)
    np.savez_compressed(args.output/f'{dataset}_episode_energy.npz',
                        energy=np.asarray(matrices,dtype=np.float32), episodes=episodes,
                        tasks=[t['instruction'] for t in tasks],
                        metadata=json.dumps(dict(source=str(folder), checkpoint=manifest['checkpoint'],
                                                 metric='MSE residual energy, continuous channels, original probe normalization')))
    print(dataset,len(tasks),flush=True)

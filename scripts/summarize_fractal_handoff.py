"""Summarize all paired continuations without treating branches as independent."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    originals = [json.loads(f.read_text()) for f in sorted((args.root/'collect').glob('*.json'))]
    rows = []
    for method in ('ht', 'flow', 'flow_zero'):
        for path in sorted((args.root/method).glob('*.pkl')):
            with path.open('rb') as f:
                data = pickle.load(f)
            row = dict(data['summary'])
            check = row.pop('replay_check')
            row.update(check)
            states, actions = data['states'], data['actions']
            xyz = np.stack([s['xyz'] for s in states])
            drawer = np.array([s['drawer'] for s in states])
            command = np.stack([np.concatenate([a['action.'+k] for k in ('x','y','z')]) for a in actions])
            row.update(first20_eef_mm_per_step=float(1000*np.linalg.norm(np.diff(xyz[:21],axis=0),axis=1).mean()),
                       first20_command_norm=float(np.linalg.norm(command[:20],axis=1).mean()),
                       first20_drawer_closure_mm=float(1000*(drawer[0]-drawer[min(20,len(drawer)-1)])),
                       initial_drawer=float(drawer[0]),
                       min_drawer=float(drawer.min()))
            rows.append(row)
    expected = [(r['seed'], b) for r in originals for b in r['branches']]
    per_method = {}
    for method in ('ht', 'flow', 'flow_zero'):
        selected = [r for r in rows if r['method'] == method]
        per_method[method] = dict(completed=len(selected), expected=len(expected),
                                  successful_branches=sum(r['success'] for r in selected),
                                  successful_scenes=len({r['seed'] for r in selected if r['success']}))
    summary = dict(collected=len(originals), ht_original_successes=sum(r['success'] for r in originals),
                   failure_scenes=sum(not r['success'] for r in originals),
                   all_replays_valid=all(r['valid'] for r in rows), methods=per_method,
                   rows=rows,
                   caveat='HT-failure-selected pilot; early and late branches are paired within scene, not independent trials.')
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2))
    if rows:
        with (args.root/'paired_results.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__ == '__main__':
    main()

"""Summarize paired parameter-gradient measurements without pooling task scopes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--write-manifest', action='store_true')
    args = p.parse_args()
    if args.write_manifest:
        files = sorted(x for x in args.directory.iterdir() if x.suffix in ('.json','.npz')
                       and x.name != 'transfer_manifest.json' and not x.name.startswith('partial_'))
        (args.directory/'transfer_manifest.json').write_text(json.dumps(dict(tasks=[
            dict(source=str(f), source_sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files
        ]), indent=2))
    rows = []
    for step in [7000, 8000, 9000, 10000, 11000, 12000]:
        path = args.directory/f'checkpoint_{step}.json'
        if not path.exists():
            path = args.directory/f'partial_{step}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for batch in data['batches']:
            for setting, comp in batch['comparisons'].items():
                groups = comp['groups']
                total = groups['all']
                rows.append(dict(step=step, before=data['nu_before'], after=comp['nu'],
                                 batch=batch['batch'], comparison=setting,
                                 cosine=total['cosine'], angle=total['angle_degrees'],
                                 norm_ratio=total['norm_ratio'],
                                 action_cosine=groups['action_decoder']['cosine'],
                                 sigma_cosine=groups['sigma_decoder']['cosine'],
                                 sigma_norm_ratio=groups['sigma_decoder'].get('norm_ratio'),
                                 shared_cosine=groups['shared_action_head']['cosine']))
    print('| Step | Batch | Comparison | Nu | Total cosine | Angle | Norm ratio | Action cosine | Sigma cosine | Sigma norm ratio | Shared cosine |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['step']} | {r['batch']} | {r['comparison']} | {r['before']:g} → {r['after']:g} | "
              f"{r['cosine']:.5f} | {r['angle']:.2f}° | {r['norm_ratio']:.3f} | "
              f"{r['action_cosine']:.5f} | {r['sigma_cosine']:.5f} | {r['sigma_norm_ratio']:.3f} | {r['shared_cosine']:.5f} |")


if __name__ == '__main__':
    main()

"""Compare only complete, seed- AND initial-observation-matched evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def compare(left,right):
    rows=[]
    for a in sorted(left.glob('*.scenes.json')):
        b=right/a.name
        if not b.exists():continue
        x=json.loads(a.read_text());y=json.loads(b.read_text())
        task=a.name.removesuffix('.scenes.json')
        row=dict(task=task,left_completed=x['completed'],right_completed=y['completed'])
        rows.append(row)
        if x['completed']!=x['requested'] or y['completed']!=y['requested']:
            row['status']='incomplete';continue
        assert x['protocol']==y['protocol']=='fixed_episode_seeds_v1'
        assert x['requested']==y['requested']==100
        xx=x['episodes'];yy=y['episodes']
        assert [r['seed'] for r in xx]==[r['seed'] for r in yy]
        matched=[u['initial_observation_sha256']==v['initial_observation_sha256'] for u,v in zip(xx,yy)]
        row['matched_initial_observations']=sum(matched)
        if not all(matched):
            row['status']='observation_mismatch_no_paired_inference'
            row['mismatch_seeds']=[r['seed'] for r,m in zip(xx,matched) if not m]
            continue
        p=np.array([r['success'] for r in xx],int);q=np.array([r['success'] for r in yy],int)
        gains=int(np.sum((p==1)&(q==0)));losses=int(np.sum((p==0)&(q==1)))
        rng=np.random.default_rng(20260908)
        delta=p-q
        boot=np.mean(delta[rng.integers(0,100,size=(10000,100))],axis=1)
        row.update(status='verified_scene_pairs',left_success=float(p.mean()),right_success=float(q.mean()),
                   difference=float(delta.mean()),paired_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                   left_only_success=gains,right_only_success=losses,
                   mcnemar_exact_p=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.)
    return dict(left=str(left),right=str(right),tasks=rows,
                scope='Diagnostic scene set; no adjustment for recipe/model selection. '
                      'Does not isolate different training histories or establish a multi-seed recipe ranking.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--left',type=Path,required=True)
    p.add_argument('--right',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=compare(args.left,args.right)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text)

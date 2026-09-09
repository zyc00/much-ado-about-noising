import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);a=json.loads(Path(sys.argv[2]).read_text());z=np.load(root/'probe.npz')
rows=[]
for key,(lo,hi) in a['windows'].items():
    uid=int(key);ix=z['episode_uid']==uid
    push=ix&(z['step']>=lo)&(z['step']<=hi);before=ix&(z['step']<lo)
    def s(m):
        return dict(n=int(m.sum()),sigma=float(np.median(z['sigma'][m])),sigma_max=float(z['sigma'][m].max()),
                    weight=float(np.median(z['weight'][m])),weight_min=float(z['weight'][m].min()),
                    gate=float(np.median(z['gate'][m])),gate_min=float(z['gate'][m].min()),
                    rms=float(np.median(np.sqrt(z['S'][m]/56))),
                    gradient_norm=float(np.median(z['weight'][m]*np.sqrt(z['S'][m])/56)))
    b=s(before);p=s(push)
    rows.append(dict(uid=uid,episode=int(z['episode'][ix][0]),interval=[lo,hi],before=b,push=p,
                     ratios={k:p[k]/b[k] for k in ['sigma','weight','gate','gradient_norm']}))
result=dict(selection=a['selection'],rows=rows,
            median_within_episode_ratios={k:float(np.median([r['ratios'][k] for r in rows])) for k in ['sigma','weight','gate','gradient_norm']},
            downweighted_episodes=sum(r['ratios']['weight']<1 for r in rows),episodes=len(rows))
(root/'visual_push_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

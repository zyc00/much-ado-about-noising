"""Aggregate HT, 16-sample Flow mean and zero-initialized Flow on identical inputs."""
from pathlib import Path
import sys,json
import numpy as np
root=Path(sys.argv[1]);z=np.load(root/'probe.npz');f=np.load(root/'matched_flow.npz');ix=f['index']
ann=json.loads((root/'visual_push_intervals.json').read_text())['windows']
ck=Path('/mnt/pfs/yuchen/groot/ft_fr_nu224/checkpoint-20000');st=json.loads((ck/'statistics.json').read_text())
tag=next(k for k in st if 'google' in k.lower());lo=st[tag]['action']['x']['q01'][0];hi=st[tag]['action']['x']['q99'][0]
physical=lambda a:(np.clip(a,-1,1)+1)/2*(hi-lo)+lo
methods={'ht':z['prediction'][ix], 'flow_mean16':f['samples'].mean(1),'flow_zero':f['zero']}
px={'ht':physical(methods['ht'][:,0,0]),'flow_mean16':physical(f['samples'][:,:,0,0]).mean(1),
    'flow_zero':physical(methods['flow_zero'][:,0,0])}
y=z['target'][ix];yy=physical(y[:,0,0]);rows=[]
for uid,(start,end) in ann.items():
    m=(z['episode_uid'][ix]==int(uid))&(z['step'][ix]>=start)&(z['step'][ix]<=end)
    def stat(name,p):
        r=p[m]-y[m];pos=yy[m]>0
        return dict(n=int(m.sum()),chunk_rms=float(np.sqrt(np.mean(r*r))),
                    first_x_underprediction_fraction=float(np.mean(r[:,0,0]<0)),
                    first_x_normalized_bias=float(r[:,0,0].mean()),
                    physical_first_x_mean=float(px[name][m].mean()),
                    positive_x_prediction_target_ratio=float(px[name][m][pos].sum()/max(yy[m][pos].sum(),1e-8)))
    rows.append(dict(episode=int(z['episode'][ix][m][0]),uid=int(uid),methods={n:stat(n,p) for n,p in methods.items()}))
out=dict(rows=rows,notes='Flow mean is 16 Monte Carlo samples, not exact mean. Physical forward values average decoded/clipped samples; normalized residuals use mean before clipping. Zero Flow is not mean Flow.')
(root/'matched_flow_analysis.json').write_text(json.dumps(out,indent=2))
print(json.dumps(next(r for r in rows if r['episode']==19034),indent=2))

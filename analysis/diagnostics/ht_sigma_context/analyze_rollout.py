"""GR1 rollout associations; commanded hand change is not a contact sensor."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
RAW=ROOT.parents[1]/'paper/sigma_videos/raw'
results=[]
for path in sorted(RAW.iterdir()):
    z=np.load(path/'traj_slim.npz')
    rows=[json.loads(line) for line in (path/'sigma.jsonl').open()]
    sigma=np.array([r['sigma'] for r in rows])
    assert len(sigma)==len(z['done'])
    action=np.concatenate([z['act.action.left_arm'][:,0],z['act.action.right_arm'][:,0]],-1)
    state=np.concatenate([z['obs.state.left_arm'][:,0],z['obs.state.right_arm'][:,0]],-1)
    hand=np.concatenate([z['act.action.left_hand'][:,0],z['act.action.right_hand'][:,0]],-1)
    amplitude=np.sqrt(((action-state)**2).mean((1,2)))
    hand_change=np.sqrt((np.ptp(hand,axis=1)**2).mean(1))
    keep=np.ones(len(sigma),bool);keep[0]=False
    keep[np.flatnonzero(z['done'][:-1,0])+1]=False
    high=sigma>=np.quantile(sigma[keep],.8)
    results.append(dict(task=path.name,n=int(keep.sum()),raw_calls=len(sigma),
        rho_sigma_arm_offset=float(spearmanr(sigma[keep],amplitude[keep]).statistic),
        rho_sigma_commanded_hand_range=float(spearmanr(sigma[keep],hand_change[keep]).statistic),
        amplitude_high_sigma_vs_rest=float(np.median(amplitude[keep&high])/np.median(amplitude[keep&~high]))))
(ROOT/'rollout_analysis.json').write_text(json.dumps(dict(
    scope='All saved episodes, successes and failures. First call of each episode excluded. No demonstration labels or verified contact states.',
    per_task=results),indent=2))
print(json.dumps(results,indent=2))

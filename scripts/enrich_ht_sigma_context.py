"""Physical-unit checks, hand transition proxy, residual attribution and matched pairs."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr

root=Path(sys.argv[1])
out={}
for dataset in ['gr1','bridge','fractal']:
    z=np.load(root/dataset/'probe.npz');meta=json.loads(str(z['metadata']))
    y=z['target'].astype(float);p=z['prediction'].astype(float);s=z['sigma'];tid=z['task_id']
    ck=Path(meta['checkpoint']); cfg=json.load(open(ck/'processor_config.json'))['processor_kwargs'];stats=json.load(open(ck/'statistics.json'))
    tag={'gr1':'robocasa_gr1_tabletop','bridge':'simpler_env_widowx','fractal':'simpler_env_google'}[dataset]
    # Resolve key via action-key identity if capitalization/tag spelling differs.
    if tag not in stats:
        tag=next(k for k in stats if ('widowx' if dataset=='bridge' else 'google') in k.lower())
    ac=cfg['modality_configs'][tag]['action'];offset=0;raw_y=[];raw_p=[]
    for j,key in enumerate(ac['modality_keys']):
        dim=len(stats[tag]['action'][key]['min']);rs=(ac.get('action_configs') or [None]*len(ac['modality_keys']))[j]
        relative=bool(rs and rs['rep']=='RELATIVE' and cfg['use_relative_action'])
        st=stats[tag]['relative_action' if relative else 'action'][key]
        lo=np.asarray(st['min' if relative or not cfg['use_percentiles'] else 'q01'])
        hi=np.asarray(st['max' if relative or not cfg['use_percentiles'] else 'q99'])
        if lo.ndim==2: lo=lo[:8];hi=hi[:8]
        raw_y.append((np.clip(y[:,:,offset:offset+dim],-1,1)+1)/2*(hi-lo)+lo)
        raw_p.append((np.clip(p[:,:,offset:offset+dim],-1,1)+1)/2*(hi-lo)+lo)
        offset+=dim
    yy=np.concatenate(raw_y,-1);pp=np.concatenate(raw_p,-1)
    arms=list(range(14 if dataset=='gr1' else 6));hands=list(range(14,26)) if dataset=='gr1' else [6]
    # GR1: radians of relative arm joints. Bridge/Fractal: translation meters
    # separately from rotation; do not mix different physical units.
    motion=list(range(14)) if dataset=='gr1' else [0,1,2]
    amp=np.sqrt((yy[:,:,motion]**2).mean((1,2)))
    hr=np.max(np.ptp(yy[:,:,hands],axis=1),axis=1)
    hand_excursion=hr
    event=hand_excursion>(.2 if dataset=='gr1' else .5)
    r=p-y;energy=(r*r).sum((1,2));he=(r[:,:,hands]**2).sum((1,2))
    high=np.zeros(len(s),bool);low=high.copy();matched=[]
    per_task=[]
    for t in np.unique(tid):
        ix=np.flatnonzero(tid==t);high[ix]=s[ix]>=np.quantile(s[ix],.8);low[ix]=s[ix]<=np.quantile(s[ix],.2)
        per_task.append(dict(task=meta['tasks'][int(t)],rho_sigma_physical_amplitude=float(spearmanr(s[ix],amp[ix]).statistic)))
        for ii in np.array_split(ix[np.argsort(amp[ix])],4):
            a=ii[event[ii]];b=ii[~event[ii]]
            if min(len(a),len(b))>=3:
                matched.append(dict(task=meta['tasks'][int(t)],n_event=len(a),n_no_event=len(b),
                    sigma_ratio=float(np.median(s[a])/np.median(s[b])),amplitude_ratio=float(np.median(amp[a])/max(np.median(amp[b]),1e-9))))
    def summary(mask):
        return dict(n=int(mask.sum()),sigma_median=float(np.median(s[mask])),
            physical_arm_amplitude_median=float(np.median(amp[mask])),
            hand_event_fraction=float(event[mask].mean()),
            hand_energy_share=float(he[mask].sum()/energy[mask].sum()),
            hand_error_constant_in_chunk_fraction=float((8*(r[mask][:,:,hands].mean(1)**2).sum())/he[mask].sum()))
    out[dataset]=dict(scope=meta['scope'],physical_amplitude_units='rad RMS of relative arm joints' if dataset=='gr1' else 'm RMS of translational action',
        hand_event_definition='max in-chunk hand joint command range > 0.2 rad' if dataset=='gr1' else 'in-chunk gripper command range > 0.5',
        physical_amplitude_correlation=float(spearmanr(s,amp).statistic),
        hand_excursion_correlation=float(spearmanr(s,hand_excursion).statistic),
        event=summary(event),no_event=summary(~event),high_sigma=summary(high),low_sigma=summary(low),
        matched=matched,matched_sigma_ratio_median=float(np.median([x['sigma_ratio'] for x in matched])) if matched else None,
        per_task=per_task)
    np.savez_compressed(root/dataset/'physical_features.npz',arm_amplitude=amp,hand_excursion=hand_excursion,event=event,
        target_physical=yy,prediction_physical=pp)
    print(dataset,json.dumps({k:v for k,v in out[dataset].items() if k not in ['matched','per_task']}),flush=True)
(root/'physical_analysis.json').write_text(json.dumps(out,indent=2))

"""Measured associations and paired visual audit of shared HT sigma.

Hand changes are motion proxies, not contact labels. No causal/aleatoric claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rms(a, axis=None):
    return np.sqrt(np.mean(np.square(a), axis=axis))

def corr(x, y):
    return float(spearmanr(x, y).statistic) if np.std(x)>0 and np.std(y)>0 else None

def analyze(root, dataset):
    folder = root/dataset
    z = np.load(folder/'probe.npz')
    meta = json.loads(str(z['metadata']))
    y, pred, s = z['target'].astype(float), z['prediction'].astype(float), z['sigma'].astype(float)
    r = pred-y
    arm = list(range(14 if dataset == 'gr1' else 6))
    hand = list(range(14,26)) if dataset == 'gr1' else [6]
    values = dict(sigma=s, arm_label_rms=rms(y[:,:,arm], (1,2)), arm_pred_rms=rms(pred[:,:,arm], (1,2)),
        arm_residual_rms=rms(r[:,:,arm], (1,2)), hand_residual_rms=rms(r[:,:,hand], (1,2)),
        total_residual_rms=rms(r, (1,2)), hand_label_range=rms(np.ptp(y[:,:,hand],axis=1),1),
        hand_pred_range=rms(np.ptp(pred[:,:,hand],axis=1),1),
        arm_label_range=rms(np.ptp(y[:,:,arm],axis=1),1), progress=z['progress'])
    energy=(r*r).sum((1,2)); he=(r[:,:,hand]**2).sum((1,2)); ae=(r[:,:,arm]**2).sum((1,2))
    tid=z['task_id']; uid=z['episode_uid']
    per_task=[]; hi=np.zeros(len(s),bool); low=hi.copy()
    ranks={k:np.zeros(len(s)) for k in values}
    matched=[]
    for t in np.unique(tid):
        sel=tid==t; ix=np.flatnonzero(sel)
        hi[sel]=s[sel]>=np.quantile(s[sel],.8); low[sel]=s[sel]<=np.quantile(s[sel],.2)
        for k,v in values.items(): ranks[k][sel]=rankdata(v[sel])/sel.sum()
        per_task.append(dict(task=meta['tasks'][int(t)], n=int(sel.sum()),
            correlation={k:corr(s[sel],v[sel]) for k,v in values.items() if k!='sigma'}))
        # Control action magnitude using within-task bins. Compare upper/lower
        # hand-change terciles inside each bin, excluding tied strata.
        for ii in np.array_split(ix[np.argsort(values['arm_label_rms'][ix])],4):
            hr=values['hand_label_range'][ii]; a,b=np.quantile(hr,[1/3,2/3])
            if b-a<.02: continue
            il,ih=ii[hr<=a],ii[hr>=b]
            if min(len(il),len(ih))<3: continue
            matched.append(dict(task_id=int(t), n_low=len(il), n_high=len(ih),
                sigma_ratio=float(np.median(s[ih])/np.median(s[il])),
                arm_amplitude_ratio=float(np.median(values['arm_label_rms'][ih])/max(np.median(values['arm_label_rms'][il]),1e-9)),
                arm_residual_ratio=float(rms(r[ih][:,:,arm])/rms(r[il][:,:,arm])),
                hand_residual_ratio=float(rms(r[ih][:,:,hand])/max(rms(r[il][:,:,hand]),1e-9))))
    def stats(sel):
        return dict(n=int(sel.sum()), sigma_median=float(np.median(s[sel])),
            arm_label_rms_median=float(np.median(values['arm_label_rms'][sel])),
            arm_residual_rms=float(rms(r[sel][:,:,arm])), hand_residual_rms=float(rms(r[sel][:,:,hand])),
            hand_error_energy_share=float(he[sel].sum()/energy[sel].sum()),
            arm_error_energy_share=float(ae[sel].sum()/energy[sel].sum()),
            hand_label_range_median=float(np.median(values['hand_label_range'][sel])),
            residual_rms_by_step=rms(r[sel],axis=(0,2)).tolist())
    # Partial rank association after removing task-specific rank action magnitude.
    X=np.column_stack([np.ones(len(s)),ranks['arm_label_rms']])
    partial=lambda v: v-X@np.linalg.lstsq(X,v,rcond=None)[0]
    summary=dict(dataset=dataset,n=len(s),episodes=len(np.unique(uid)),tasks=len(per_task),metadata=meta,
        pooled_correlations={k:corr(s,v) for k,v in values.items() if k!='sigma'},
        within_task_rank_correlations={k:corr(ranks['sigma'],v) for k,v in ranks.items() if k!='sigma'},
        hand_range_partial_rank_given_arm_magnitude=float(np.corrcoef(partial(ranks['sigma']),partial(ranks['hand_label_range']))[0,1]),
        per_task=per_task, top_sigma_quintile=stats(hi), bottom_sigma_quintile=stats(low), all=stats(np.ones(len(s),bool)),
        amplitude_matched_hand_change=matched,
        amplitude_matched_median_sigma_ratio=float(np.median([v['sigma_ratio'] for v in matched])) if matched else None)
    (folder/'analysis.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(folder/'derived.npz',**values,task_id=tid,episode_uid=uid,step=z['step'])
    print(dataset, json.dumps({k:v for k,v in summary.items() if k not in ['metadata','per_task','amplitude_matched_hand_change']}),flush=True)
    if not (folder/'thumbnails.npz').exists(): return
    ims=np.load(folder/'thumbnails.npz')['images']
    # First 8 tasks by predefined frequency order, select each task's maximum sigma
    # and closest-amplitude low-sigma control. Clearly selected examples, not prevalence.
    visual=[]
    fig,axs=plt.subplots(8,2,figsize=(8,16))
    for t in range(min(8,len(per_task))):
        ix=np.flatnonzero(tid==t)
        peak=ix[np.argmax(s[ix])]
        candidates=ix[s[ix]<=np.median(s[ix])]
        control=candidates[np.argmin(abs(np.log(values['arm_label_rms'][candidates]+1e-6)-np.log(values['arm_label_rms'][peak]+1e-6)))]
        for j,i in enumerate([peak,control]):
            axs[t,j].imshow(ims[i]); axs[t,j].axis('off')
            axs[t,j].set_title(f"{t}: {'high sigma' if j==0 else 'amplitude-matched control'} | ep{int(z['episode'][i])}, step{int(z['step'][i])}\n"
              f"sigma={s[i]:.3f}, arm |a|={values['arm_label_rms'][i]:.3f}, hand range={values['hand_label_range'][i]:.2f}\n"
              f"error arm={values['arm_residual_rms'][i]:.3f}, hand={values['hand_residual_rms'][i]:.3f}",fontsize=8)
            visual.append(dict(index=int(i),task=meta['tasks'][t],role='peak' if j==0 else 'control'))
    fig.tight_layout(); fig.savefig(folder/'visual_pairs.jpg',dpi=140); plt.close(fig)
    (folder/'visual_pairs.json').write_text(json.dumps(visual,indent=2))
    # Whole episode for peak-sigma episode in each of first four tasks: no inferred
    # contact shading; images themselves supply context for manual inspection.
    for t in range(min(4,len(per_task))):
        ix=np.flatnonzero(tid==t); peak=ix[np.argmax(s[ix])]; ep=uid[peak]; ii=np.flatnonzero(uid==ep)
        fig=plt.figure(figsize=(12,4.7)); gs=fig.add_gridspec(2,6,height_ratios=[1,1.4])
        chosen=ii[np.linspace(0,len(ii)-1,6).round().astype(int)]
        for j,i in enumerate(chosen):
            ax=fig.add_subplot(gs[0,j]); ax.imshow(ims[i]); ax.axis('off'); ax.set_title(f"step {int(z['step'][i])}",fontsize=9)
        ax=fig.add_subplot(gs[1,:]); ax.plot(z['step'][ii],s[ii],label='predicted sigma',color='#b43b30',lw=2)
        ax.plot(z['step'][ii],values['arm_residual_rms'][ii],label='arm residual RMS',color='#2878a9')
        ax.plot(z['step'][ii],values['hand_residual_rms'][ii],label='hand residual RMS',color='#159570')
        ax.plot(z['step'][ii],values['arm_label_rms'][ii],label='arm action RMS',color='.6',ls='--')
        ax.set_xlabel('Demonstration step'); ax.set_ylabel('Normalized action units'); ax.legend(fontsize=8,ncol=4)
        fig.suptitle(meta['tasks'][t],fontsize=9); fig.tight_layout(); fig.savefig(folder/f'episode_task{t}.jpg',dpi=130);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--dataset',required=True)
    a=p.parse_args();analyze(a.root,a.dataset)

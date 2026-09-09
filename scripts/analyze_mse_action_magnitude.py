#!/usr/bin/env python3
"""Residual scale versus predicted, physically centered movement magnitude.

Exploratory analysis requested after the progress-bin probe was launched.
Bin within task by predicted movement magnitude, never by target magnitude.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from analyze_mse_stage_scale import metric_arrays,NAMES,COLORS,load_npz


def run(z,affine):
    meta=json.loads(str(z['metadata']));start=meta['executed_start'];H=meta['executed_horizon']
    pred=z['pred_final'][:,start:start+H].astype(np.float64)
    gt=z['gt'][:,start:start+H].astype(np.float64)
    assert z['valid_time'][:,start:start+H].all()
    offset=np.array(affine['offset']);slope=np.array(affine['slope']);ch=affine['magnitude_channels']
    physical=pred*slope+offset
    amp=np.sqrt(np.mean(physical[:,:,ch]**2,axis=(1,2)))
    energy=np.mean(metric_arrays(z,'pred_final')**2,axis=1)
    # Matching physical channels as an independent representation check.
    motion_energy=np.mean(((pred-gt)*slope)[:,:,ch]**2,axis=(1,2))
    bins=np.full(len(amp),-1);normalized=np.full(len(amp),np.nan)
    normalized_motion=np.full(len(amp),np.nan)
    rows=[];by_task=[]
    for task in sorted(set(z['task'])):
        idx=np.flatnonzero(z['task']==task);order=idx[np.argsort(amp[idx])]
        for b,g in enumerate(np.array_split(order,5)):bins[g]=b
        baseline=np.mean(energy[idx]);base_motion=np.mean(motion_energy[idx])
        normalized[idx]=energy[idx]/baseline
        normalized_motion[idx]=motion_energy[idx]/base_motion
        values=[np.sqrt(np.mean(normalized[idx][bins[idx]==b])) for b in range(5)]
        physical_values=[np.sqrt(np.mean(normalized_motion[idx][bins[idx]==b])) for b in range(5)]
        rows.append(dict(task=task,rho=float(spearmanr(amp[idx],energy[idx]).statistic),
            relative_rms=list(map(float,values)),relative_motion_channel_rms=list(map(float,physical_values)),
            median_movement_magnitude=[float(np.median(amp[idx][bins[idx]==b])) for b in range(5)],
            high_low_ratio=float(values[-1]/values[0])))
        eps=sorted(set(z['episode'][idx]));sums=np.zeros((len(eps),5));counts=np.zeros_like(sums)
        for i,ep in enumerate(eps):
            for b in range(5):
                m=(z['task']==task)&(z['episode']==ep)&(bins==b)
                sums[i,b]=np.sum(normalized[m]);counts[i,b]=m.sum()
        by_task.append((sums,counts))
    curves=np.array([row['relative_rms'] for row in rows]);mean=curves.mean(0)
    rng=np.random.default_rng(20260904);boot=[]
    for _ in range(500):
        curves_b=[]
        for sums,counts in by_task:
            ii=rng.integers(len(sums),size=len(sums))
            curves_b.append(np.sqrt(sums[ii].sum(0)/np.maximum(counts[ii].sum(0),1)))
        boot.append(np.mean(curves_b,axis=0))
    boot=np.array(boot)
    return dict(n=len(amp),task_count=len(rows),per_task=rows,curve=mean.tolist(),
        ci=np.quantile(boot,[.025,.975],axis=0).tolist(),
        high_low_ratio=float(mean[-1]/mean[0]),
        high_low_ratio_ci=np.quantile(boot[:,-1]/boot[:,0],[.025,.975]).tolist(),
        median_rho=float(np.median([r['rho'] for r in rows])),
        magnitude_definition=affine['magnitude_description'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    affine={}
    for p in args.root.glob('action_affine_*.json'):affine.update(json.loads(p.read_text()))
    results={}
    for key in NAMES:
        path=args.root/(key+'.npz')
        if path.exists() and key in affine:
            results[key]=run(load_npz(path),affine[key])
            print(key,{k:results[key][k] for k in ('high_low_ratio','high_low_ratio_ci','median_rho')},flush=True)
    (args.root/'magnitude_summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    if not results:return
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
        'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig,axes=plt.subplots(1,len(results),figsize=(5.5,2.15),sharey=True,squeeze=False)
    for ax,(key,s) in zip(axes[0],results.items()):
        curve=np.array(s['curve']);ci=np.array(s['ci']);x=np.arange(5)
        ax.axhline(1,color='0.4',ls='--',lw=.7)
        ax.fill_between(x,ci[0],ci[1],color=COLORS[key],alpha=.18,lw=0)
        ax.plot(x,curve,'o-',color=COLORS[key],lw=1.5,ms=3)
        ax.set_xticks([0,2,4],['Small','Mid','Large']);ax.set_xlim(-.45,4.45)
        ax.set_xlabel('Predicted movement',labelpad=3);ax.set_title(NAMES[key],pad=5)
        ax.tick_params(length=2.5,pad=2)
        ax.text(.04,.96,f"{s['high_low_ratio']:.2f}×",transform=ax.transAxes,va='top',fontsize=8)
    top=max(np.max(s['ci'][1]) for s in results.values())
    bottom=min(np.min(s['ci'][0]) for s in results.values())
    axes[0,0].set_ylim(max(0,bottom-.15),top+.3)
    axes[0,0].set_ylabel('Residual RMS\n(relative)',labelpad=3)
    fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.77,wspace=.2)
    fig.savefig(args.root/'mse_action_magnitude.pdf')
    fig.savefig(args.root/'mse_action_magnitude.png',dpi=240)


if __name__=='__main__':main()

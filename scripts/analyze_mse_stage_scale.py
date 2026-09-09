#!/usr/bin/env python3
"""Episode-aware analysis and compact vector figures for frozen-MSE probes."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


NAMES = {'gr1': 'GR00T N1.7\nGR1 / RoboCasa', 'pi05': '$\\pi_{0.5}$\nLIBERO',
         'tool_hang': 'Chi-UNet\nTool-Hang', 'transport': 'Chi-UNet\nTransport'}
COLORS = {'gr1':'#0072B2','pi05':'#009E73','tool_hang':'#D55E00','transport':'#7C5C9B'}


def available_mean(values, axis=None):
    """Leave unsampled stages missing; never impute terminal padded actions."""
    values=np.asarray(values,dtype=float)
    valid=np.isfinite(values)
    count=valid.sum(axis=axis)
    total=np.where(valid,values,0).sum(axis=axis)
    return np.divide(total,count,out=np.full_like(total,np.nan,dtype=float),where=count>0)


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def load_npz(path):
    # Older running probes save between tasks. Materialize a complete snapshot
    # now rather than keep lazy references to an archive that might be replaced.
    for attempt in range(10):
        try:
            with np.load(io.BytesIO(Path(path).read_bytes()),allow_pickle=False) as archive:
                return {k:archive[k] for k in archive.files}
        except (zipfile.BadZipFile,EOFError,OSError):
            if attempt==9:raise
            time.sleep(.5)


def metric_arrays(z, key, single=False, channels=None):
    meta=json.loads(str(z['metadata']))
    start=meta['executed_start']
    stop=start+(1 if single else meta['executed_horizon'])
    channels=meta['continuous_channels'] if channels is None else channels
    assert z['valid_time'][:,start:stop].all(), 'Padded execution target entered primary metric'
    r=(z[key][:,start:stop,channels]-z['gt'][:,start:stop,channels]).astype(np.float64)
    assert np.isfinite(r).all()
    return r.reshape(len(r),-1)


def task_matrices(z, energy):
    mats=[]
    for task in sorted(set(z['task'])):
        eps=sorted(set(z['episode'][z['task']==task]))
        matrix=np.full((len(eps),10),np.nan)
        splits=[]
        for ei,ep in enumerate(eps):
            keep=(z['task']==task)&(z['episode']==ep)
            splits.append(int(z['split'][keep][0]))
            for b in range(10):
                v=energy[keep&(z['stage']==b)]
                if len(v):matrix[ei,b]=v.mean()
        mats.append((task,matrix,np.array(splits)))
    return mats


def stage_curve(mats):
    curves=[]
    for _,m,_ in mats:
        e=available_mean(m,axis=0)
        curves.append(np.sqrt(e/np.nanmean(e)))
    curves=np.array(curves)
    return curves,np.nanmedian(curves,axis=0)


def bootstrap_curve(mats,rng,n=400):
    values=[]
    for _ in range(n):
        sampled=[(t,m[rng.integers(len(m),size=len(m))],s) for t,m,s in mats]
        values.append(stage_curve(sampled)[1])
    return np.nanquantile(values,[.025,.975],axis=0)


def replication(mats):
    """Rank progress bins by scale on one episode half, score the other half.

    Both directions are evaluated; query residuals never define group membership.
    Each fold uses one pooled query-set denominator for all three groups, which
    cancels in the high/low ratio. Each task has its own normalization.
    """
    records=[]
    for task,m,split in mats:
        for train_split in (0,1):
            cal=available_mean(m[split==train_split],axis=0)
            test=available_mean(m[split!=train_split],axis=0)
            valid=np.isfinite(cal)&np.isfinite(test)&(cal>0)
            if valid.sum()<6:continue
            ids=np.flatnonzero(valid)
            ordered=ids[np.argsort(cal[ids])]
            groups=np.array_split(ordered,3)
            test_pool=np.mean(test[ids])
            v=np.array([np.sqrt(test[g].mean()/test_pool) for g in groups])
            global_var=np.mean(cal[ids])
            nll_global=.5*(np.log(global_var)+test[ids]/global_var)
            nll_stage=.5*(np.log(cal[ids])+test[ids]/cal[ids])
            records.append(dict(task=task,calibration_split=train_split,
                low_mid_high=v.tolist(),high_low_ratio=float(v[2]/v[0]),
                scale_rank_correlation=float(spearmanr(cal[ids],test[ids]).statistic),
                stage_nll_gain_per_coordinate=float(np.mean(nll_global-nll_stage))))
    return records


def summarize(z, key='pred_final',single=False,channels=None):
    r=metric_arrays(z,key,single,channels)
    energy=np.mean(r*r,axis=1)
    mats=task_matrices(z,energy)
    curves,median=stage_curve(mats)
    records=replication(mats)
    per_task=[]
    for task,m,_ in mats:
        scale=np.sqrt(available_mean(m,axis=0))
        per_task.append(dict(task=task,episodes=len(m),stage_rms=scale.tolist(),
            stage_episode_counts=np.isfinite(m).sum(0).tolist(),
            stage_max_min=float(np.nanmax(scale)/np.nanmin(scale))))
    # Coarse stage-centering is descriptive, not irreducible conditional variance.
    centered_ratios=[];bias_fracs=[]
    for task in sorted(set(z['task'])):
        sd=[]
        for b in range(10):
            vals=r[(z['task']==task)&(z['stage']==b)]
            if len(vals)>1:
                sd.append(np.sqrt(np.var(vals,axis=0,ddof=1).mean()))
                bias_fracs.append(float(np.mean(vals.mean(0)**2)/np.mean(vals**2)))
        if sd: centered_ratios.append(float(max(sd)/min(sd)))
    return dict(n=len(r),task_count=len(mats),episode_count=sum(len(m) for _,m,_ in mats),
        dimension=r.shape[1],rms=float(np.sqrt(energy.mean())),
        sample_rms_q10_q50_q90=np.quantile(np.sqrt(energy),[.1,.5,.9]).tolist(),
        median_task_stage_max_min=float(np.median([v['stage_max_min'] for v in per_task])),
        median_centered_stage_max_min=float(np.median(centered_ratios)),
        median_coarse_cell_mean_energy_fraction=float(np.median(bias_fracs)),
        normalized_median_curve=median.tolist(),per_task=per_task,replication=records,
        replicated_high_low_ratio_median=float(np.median([v['high_low_ratio'] for v in records])),
        replicated_stage_rank_correlation_median=float(np.nanmedian([v['scale_rank_correlation'] for v in records])),
        replicated_stage_nll_gain_mean=float(np.mean([v['stage_nll_gain_per_coordinate'] for v in records]))),mats


def draw(all_results,root):
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
        'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65})
    keys=list(all_results)
    width=5.5
    fig,axes=plt.subplots(1,len(keys),figsize=(width,2.05),sharey=True,squeeze=False)
    x=np.arange(10)*10+5
    rng=np.random.default_rng(20260904)
    for ax,key in zip(axes[0],keys):
        summary,mats=all_results[key]
        curves,median=stage_curve(mats)
        if len(curves)>1:
            for row in curves:ax.plot(x,row,color='0.73',lw=.45,alpha=.65,zorder=1)
        ci=bootstrap_curve(mats,rng)
        ax.fill_between(x,ci[0],ci[1],color=COLORS[key],alpha=.18,lw=0,zorder=2)
        ax.plot(x,median,'o-',color=COLORS[key],lw=1.55,ms=2.5,zorder=3)
        ax.axhline(1,color='0.25',lw=.65,ls=(0,(3,2)),zorder=0)
        ax.set_title(NAMES[key],pad=5)
        ax.set_xlim(-3,103);ax.set_xticks([0,50,100]);ax.set_xticklabels(['0','50','100'])
        ax.set_xlabel('Progress (%)',labelpad=2)
        ax.set_yscale('log',base=2)
        ax.set_yticks([.25,.5,1,2,4]);ax.set_yticklabels(['0.25','0.5','1','2','4'])
        ax.tick_params(length=2.5,pad=2)
        ax.text(.04,.96,f"{summary['median_task_stage_max_min']:.1f}× stage range",
                transform=ax.transAxes,va='top',fontsize=8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
    axes[0,0].set_ylabel('Relative residual RMS',labelpad=3)
    limits=np.concatenate([stage_curve(v[1])[0].reshape(-1) for v in all_results.values()])
    axes[0,0].set_ylim(max(.1,float(np.nanmin(limits))*.8),min(8,float(np.nanmax(limits))*1.35))
    fig.subplots_adjust(left=.09,right=.97,bottom=.23,top=.77,wspace=.20)
    fig.savefig(root/'mse_stage_scale.pdf')
    fig.savefig(root/'mse_stage_scale.png',dpi=240)
    plt.close(fig)

    # Independent-episode validation of stage-scale ranking; no residual-based
    # binning on the same observations being displayed.
    fig,axes=plt.subplots(1,len(keys),figsize=(width,1.95),sharey=True,squeeze=False)
    for ax,key in zip(axes[0],keys):
        summary,_=all_results[key]
        rows=np.array([r['low_mid_high'] for r in summary['replication']])
        # Average the two cross-fitting directions within each task first.
        tasks=sorted(set(r['task'] for r in summary['replication']))
        means=np.array([rows[[r['task']==t for r in summary['replication']]].mean(0) for t in tasks])
        for row in means:ax.plot([0,1,2],row,color='0.75',lw=.6,alpha=.7)
        ax.plot([0,1,2],np.median(means,axis=0),'o-',color=COLORS[key],lw=1.7,ms=4)
        ax.axhline(1,color='0.3',ls='--',lw=.7)
        ax.set_xticks([0,1,2],['Low','Mid','High']);ax.set_xlim(-.2,2.2)
        ax.set_title(NAMES[key],pad=5);ax.set_xlabel('Stage scale group',labelpad=2)
        ax.tick_params(length=2.5,pad=2)
    axes[0,0].set_ylabel('Separate-episode RMS\n(relative)',labelpad=3)
    fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.77,wspace=.17)
    fig.savefig(root/'mse_stage_scale_replication.pdf')
    fig.savefig(root/'mse_stage_scale_replication.png',dpi=240)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path)
    args=ap.parse_args();root=args.root
    results={};raw_summary={}
    for key in NAMES:
        path=root/(key+'.npz')
        if not path.exists():continue
        z=load_npz(path)
        final,mats=summarize(z)
        late,_=summarize(z,'pred_late')
        first,_=summarize(z,single=True)
        r=metric_arrays(z,'pred_final');rl=metric_arrays(z,'pred_late')
        late_check=dict(final_to_late_rms=final['rms']/late['rms'],
            per_sample_rms_rank_correlation=float(spearmanr(np.mean(r*r,1),np.mean(rl*rl,1)).statistic),
            stage_profile_rank_correlation=float(spearmanr(
                np.array([t['stage_rms'] for t in final['per_task']]).ravel(),
                np.array([t['stage_rms'] for t in late['per_task']]).ravel(),nan_policy='omit').statistic))
        channels=list(range(14)) if key=='gr1' else ([0,1,2,10,11,12] if key=='transport' else [0,1,2])
        restricted,_=summarize(z,channels=channels)
        raw_summary[key]=dict(metadata=json.loads(str(z['metadata'])),final=final,late=late,
                              single_action=first,restricted_channels=restricted,late_checkpoint_check=late_check)
        results[key]=(final,mats)
        print(key,json.dumps({k:final[k] for k in ['n','episode_count','task_count','rms',
            'median_task_stage_max_min','replicated_high_low_ratio_median',
            'replicated_stage_rank_correlation_median','replicated_stage_nll_gain_mean']}),
            'late',late_check,flush=True)
    (root/'summary.json').write_text(json.dumps(json_safe(raw_summary),indent=2,allow_nan=False))
    draw(results,root)


if __name__=='__main__':main()

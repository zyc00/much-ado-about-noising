"""Real Bridge HG per-task tails; no synthetic data or pooled-task scale mixing."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import norm, t as student_t

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from tail_stats_single import fit_student_t, t_logpdf

RAW=REPO/'analysis/paper/widowx_heterogeneous_scale/raw/hg_rank0.npz'
OUT=Path(__file__).resolve().parent/'crossdataset_hg_tails'
OUT.mkdir(exist_ok=True)
SEED=20260908
with np.load(RAW) as z:
    meta=json.loads(str(z['metadata']))
    r=z['residual'].astype(float)
    sigma=z['sigma'].astype(float)
    ids=z['task_id'].copy()
    episode=z['episode'].copy()
    assert np.allclose(r,z['prediction']-z['target'],atol=1e-6)
assert r.shape[1:]==(8,6) and np.all(sigma>0)
assert meta['objective']=='hg'
thresholds=np.linspace(.5,6,111)
g3=float(2*norm.sf(3))
records=[]
all_z=np.empty_like(r)
all_fold=np.empty(len(r),dtype=int)
for tid,task in enumerate(meta['task_names']):
    keep=ids==tid
    x=r[keep]/sigma[keep,None,None]
    eps=episode[keep]
    unique=np.unique(eps)
    order=np.random.default_rng(SEED+tid).permutation(unique)
    fold=np.asarray([{int(ep):i%5 for i,ep in enumerate(order)}[int(ep)] for ep in eps])
    standardized=np.empty_like(x)
    nll_g=nll_t=nll_l=0.
    nll_count=0
    t_curves=[]
    fold_fits=[]
    for f in range(5):
        train=x[fold!=f]
        center=train.mean(axis=0)
        scale=train.std(axis=0)
        assert np.all(scale>0)
        tr=((train-center)/scale).ravel()
        te=(x[fold==f]-center)/scale
        standardized[fold==f]=te
        nu,s=fit_student_t(tr)
        b=np.mean(np.abs(tr))
        nll_g+=float((-norm.logpdf(te)).sum())
        nll_t+=float((-t_logpdf(te,nu,s)).sum())
        nll_l+=float((np.log(2*b)+np.abs(te)/b).sum())
        nll_count+=te.size
        t_curves.append(2*student_t.sf(thresholds/s,nu))
        fold_fits.append(dict(fold=f,nu=float(nu),scale=float(s),test_coordinates=int(te.size)))
    all_z[keep]=standardized
    all_fold[keep]=fold
    absz=np.abs(standardized)
    probability=float((absz>3).mean())
    ep_hits=np.asarray([int((absz[eps==ep]>3).sum()) for ep in unique])
    ep_sizes=np.asarray([int(absz[eps==ep].size) for ep in unique])
    picks=np.random.default_rng(SEED+100+tid).integers(len(unique),size=(2000,len(unique)))
    boot=ep_hits[picks].sum(axis=1)/ep_sizes[picks].sum(axis=1)/g3
    ci=np.quantile(boot,[.025,.975])
    record=dict(task=task,episodes=len(unique),states=len(x),coordinates=absz.size,
                p_gt3=probability,gaussian_gt3=g3,tail_ratio=probability/g3,
                ratio_ci=ci.tolist(),empirical_tail=[float((absz>t).mean()) for t in thresholds],
                student_predictive_tail=np.average(t_curves,axis=0,
                    weights=[fit['test_coordinates'] for fit in fold_fits]).tolist(),
                gaussian_nll=nll_g/nll_count,student_nll=nll_t/nll_count,
                laplace_nll=nll_l/nll_count,student_nll_gain=(nll_g-nll_t)/nll_count,
                fits=fold_fits)
    records.append(record)
    print(task,'ratio',round(record['tail_ratio'],3),'CI',ci,'t gain',record['student_nll_gain'],flush=True)

summary=dict(dataset='Bridge / WidowX',scope='Only the three previously probed instruction groups; all three are shown.',
             source=str(RAW),source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),metadata=meta,
             normalization='Divide residual by HG own predicted sigma. Within each task, fit per-coordinate mean/std on four episode folds and apply to the fifth. No final test-set RMS rescaling.',
             difference_from_old_figure='Old plot calibrated coordinates across pooled tasks and finally rescaled all test values to unit RMS; this preview uses within-task centered cross-fitting only.',
             uncertainty='2000 whole-episode bootstrap resamples of cross-fitted exceedance outcomes, conditional on fitted calibration; calibration is not refitted in the bootstrap.',
             density_fit='Zero-location univariate Student-t and Laplace fit on calibration folds. Gaussian is N(0,1) after calibration. Held-out refers only to density/calibration, not policy training.',
             seed=SEED,thresholds=thresholds.tolist(),tasks=records)
(OUT/'bridge_hg_tail_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(OUT/'bridge_hg_crossfitted_z.npz',z=all_z,task_id=ids,episode=episode,fold=all_fold)

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'stix',
    'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1,1.5]})
fig.subplots_adjust(left=.17,right=.985,bottom=.25,top=.76,wspace=.32)
colors=['#0072B2','#D55E00','#009E73']
labels=['Sweep into pile','Open drawer','Close drawer']
for i,(rec,c,label) in enumerate(zip(records,colors,labels)):
    value=rec['tail_ratio']; lo,hi=rec['ratio_ci']
    ax.errorbar(value,2-i,xerr=[[value-lo],[hi-value]],fmt='o',color=c,
                ms=5.5,lw=1.7,capsize=3,zorder=3)
    ax.text(value,2-i+.21,f'{value:.1f}×',ha='center',va='bottom',fontsize=8,color=c)
    curve=np.asarray(rec['empirical_tail'])
    bx.plot(thresholds,np.where(curve>0,curve,np.nan),color=c,lw=1.5,label=label)
    bx.plot(thresholds,rec['student_predictive_tail'],color=c,lw=.85,ls=(0,(2,2)),alpha=.75)
ax.axvline(1,color='.3',lw=1,ls='--')
ax.set(yticks=[2,1,0],yticklabels=labels,ylim=(-.5,2.65),xlabel='Tail frequency / Gaussian',
       xlim=(0,max(rec['ratio_ci'][1] for rec in records)*1.18))
ax.grid(axis='x',color='.92',lw=.5)
ax.text(.98,.02,'Bars: episode-bootstrap 95% CI',transform=ax.transAxes,ha='right',fontsize=6.5,color='.4')
bx.plot(thresholds,2*norm.sf(thresholds),color='.15',lw=1.3,ls=(0,(5,3)))
bx.axvline(3,color='.7',lw=.7,ls=':')
bx.set(yscale='log',ylim=(1e-5,1),xlim=(.5,6),xticks=[1,2,3,4,5,6],
       xlabel=r'Standardized residual $|z|$',ylabel=r'Tail probability $P(|Z|>|z|)$')
bx.grid(axis='y',color='.92',lw=.5)
ax.set_title('a  Excess 3σ events',loc='left',fontsize=9,fontweight='bold',pad=8)
bx.set_title('b  Residual tails after HG scaling',loc='left',fontsize=9,fontweight='bold',pad=8)
fig.text(.17,.92,'Bridge / WidowX · 3 tasks · 144 demonstrations',fontsize=10,fontweight='bold')
fig.legend(handles=[Line2D([],[],color='.3',lw=1.5,label='Observed (task colors)'),
                    Line2D([],[],color='.3',lw=.9,ls=(0,(2,2)),label='Student-t fit'),
                    Line2D([],[],color='.15',lw=1.3,ls=(0,(5,3)),label='Gaussian')],
           loc='lower center',bbox_to_anchor=(.6,.025),ncol=3,frameon=False,fontsize=7)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'fig_bridge_hg_task_tails.{ext}',dpi=260)
plt.close(fig)

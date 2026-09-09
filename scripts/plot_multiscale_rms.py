#!/usr/bin/env python3
"""Compact ICLR residual-spread and learned-scale diagnostic; no policy inference.

Left: equally task-weighted empirical histograms of within-task-normalized MSE
chunk RMS. Right: GR1 HT sigma versus its own RMS on a separate training sample.
The two probes are not a paired MSE/HT comparison or a demonstration-noise test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.stats import spearmanr

from analyze_mse_stage_scale import load_npz, metric_arrays


SETTINGS = {
    "gr1": ("GR00T N1.7\nGR1", "#0072B2"),
    "pi05": (r"$\pi_{0.5}$" + "\nLIBERO", "#009E73"),
    "tool_hang": ("Chi-UNet\nTool-Hang", "#D55E00"),
    "transport": ("Chi-UNet\nTransport", "#7C5C9B"),
}


def task_normalized_rms(rms, tasks):
    """Remove task-wide scale differences without removing within-task spread."""
    rms=np.asarray(rms,dtype=float)
    tasks=np.asarray(tasks)
    assert np.isfinite(rms).all() and (rms>0).all()
    relative=np.empty_like(rms)
    weight=np.empty_like(rms)
    names=np.unique(tasks)
    for task in names:
        keep=tasks==task
        relative[keep]=rms[keep]/np.median(rms[keep])
        weight[keep]=1/(len(names)*keep.sum())
    assert np.isclose(weight.sum(),1)
    return relative,weight


def weighted_quantiles(values, weights, q):
    # Empirical inverse-CDF definition; no Gaussian/KDE/mixture fit.
    return np.quantile(values,q,weights=weights,method="inverted_cdf")


def load_mse(path):
    z=load_npz(path)
    r=metric_arrays(z,"pred_final")
    rms=np.sqrt(np.mean(r*r,axis=1))
    relative,weight=task_normalized_rms(rms,z["task"])
    q=weighted_quantiles(relative,weight,[.01,.1,.5,.9,.99])
    rawq=np.quantile(rms,[.1,.5,.9])
    return dict(relative=relative,weight=weight,summary=dict(
        source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n=len(rms),tasks=len(np.unique(z["task"])),
        executed_dimension=r.shape[1],
        relative_q01_q10_q50_q90_q99=q.tolist(),
        relative_q90_q10=float(q[3]/q[1]),
        raw_rms_q10_q50_q90=rawq.tolist(),raw_rms_q90_q10=float(rawq[2]/rawq[0]),
        normalization="Divide sample RMS by median RMS within its task; equal task weights",
        policy_split="training demonstrations"))


def load_ht(path):
    z=load_npz(path)
    sigma=z["sigma"].astype(float)
    rms=np.sqrt(z["m"].astype(float))
    assert sigma.shape==rms.shape and len(sigma)==2400
    assert np.isfinite(sigma).all() and (sigma>0).all()
    assert np.isfinite(rms).all() and (rms>0).all()
    assert z["r"].shape==(len(sigma),8,29)
    np.testing.assert_allclose(np.mean(z["r"].astype(float)**2,axis=(1,2)),z["m"],rtol=2e-6)
    np.testing.assert_allclose(z["sp"].astype(float).mean(axis=(1,2))+.001,sigma,rtol=2e-6)
    groups=np.array_split(np.argsort(sigma,kind="stable"),5)
    binned=[dict(n=len(g),sigma_rms=float(np.sqrt(np.mean(sigma[g]**2))),
                 sigma_median=float(np.median(sigma[g])),
                 residual_rms=float(np.sqrt(np.mean(rms[g]**2)))) for g in groups]
    # Student-t scale is not exactly marginal standard deviation. The reference
    # uses the actual incumbent nu=2*d=464, giving std=sigma*sqrt(nu/(nu-2)).
    nu=464.
    qsig=np.quantile(sigma,[.1,.5,.9]); qr=np.quantile(rms,[.1,.5,.9])
    return dict(sigma=sigma,rms=rms,binned=binned,nu=nu,summary=dict(
        source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cluster_source="/mnt/pfs/yuchen/groot/resid_dump3_gr1_ht60k_all.npz",
        checkpoint=str(z["ckpt"]),n=len(sigma),nu=nu,chunk_shape=[8,29],
        sigma_q10_q50_q90=qsig.tolist(),sigma_q90_q10=float(qsig[2]/qsig[0]),
        rms_q10_q50_q90=qr.tolist(),rms_q90_q10=float(qr[2]/qr[0]),
        sigma_rms_spearman=float(spearmanr(sigma,rms).statistic),
        rms_over_sigma_q10_q50_q90=np.quantile(rms/sigma,[.1,.5,.9]).tolist(),
        quintiles=binned,policy_split="training demonstrations",
        grouping="Five equal-count bins by predicted sigma only; no residual-based grouping",
        sampling="Existing 2400-input probe; all 24 datasets supplied for correct normalization; no per-input episode/task IDs saved",
        uncertainty="No confidence intervals: episode IDs unavailable; no independent-sample bootstrap"))


def style():
    plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral"],
        "mathtext.fontset":"stix","font.size":9,"axes.titlesize":9,
        "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":.65,
        "axes.spines.top":False,"axes.spines.right":False})


def draw(mse,ht,out):
    style()
    fig=plt.figure(figsize=(5.5,2.55))
    ax=fig.add_axes([.15,.23,.34,.60])
    right=fig.add_axes([.665,.23,.32,.60])
    all_rel=np.concatenate([d["relative"] for d in mse.values()])
    lo=np.floor(np.log10(all_rel.min())*12)/12
    hi=np.ceil(np.log10(all_rel.max())*12)/12
    edges=np.logspace(lo,hi,int(round((hi-lo)*12))+1)
    for row,(key,d) in enumerate(mse.items()):
        base=3-row
        label,color=SETTINGS[key]
        hist,_=np.histogram(d["relative"],bins=edges,weights=d["weight"])
        assert np.isclose(hist.sum(),1),"Histogram must include all measured samples"
        heights=.62*hist/hist.max()  # Equal peak height per strip; shape, not counts.
        ax.stairs(base+heights,edges,baseline=base,fill=True,color=color,alpha=.22,lw=0)
        ax.stairs(base+heights,edges,baseline=base,color=color,lw=.85)
        q=d["summary"]["relative_q01_q10_q50_q90_q99"]
        ax.plot([q[1],q[3]],[base-.10]*2,color=color,lw=2.5,solid_capstyle="butt")
        ax.plot(q[2],base-.10,"|",color="0.1",ms=6,mew=.9)
        ax.text(.98,(base+.48)/4.15,f"{q[3]/q[1]:.1f}×",transform=ax.transAxes,
                ha="right",va="center",fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(edges[0],edges[-1]);ax.set_ylim(-.30,3.85)
    ticks=[.25,.5,1,2,4,8,16,32]
    ticks=[t for t in ticks if edges[0]<=t<=edges[-1]]
    ax.set_xticks(ticks,[f"{t:g}" for t in ticks]);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([3.22,2.22,1.22,.22],[SETTINGS[k][0] for k in mse])
    ax.tick_params(axis="y",length=0,pad=5,labelsize=8.5)
    ax.tick_params(axis="x",length=3,pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Residual RMS / task median",labelpad=4)
    ax.axvline(1,color="0.35",ls=(0,(2,2)),lw=.6,zorder=0)
    ax.set_title("(a) MSE: residual-scale distributions",loc="left",pad=10)

    sigma,rms=ht["sigma"],ht["rms"]
    right.scatter(sigma,rms,s=3,color="#0072B2",alpha=.11,edgecolors="none",rasterized=False)
    low=min(sigma.min(),rms.min())*.8
    high=max(sigma.max(),rms.max())*1.2
    span=np.array([low,high])
    right.plot(span,span*np.sqrt(ht["nu"]/(ht["nu"]-2)),color="0.35",lw=.8,
               ls=(0,(3,2)),label="Model-implied RMS")
    xx=[v["sigma_rms"] for v in ht["binned"]]
    yy=[v["residual_rms"] for v in ht["binned"]]
    right.plot(xx,yy,"o-",color="#D55E00",lw=1.3,ms=3.7,mec="white",mew=.5,
               label="Measured RMS (5 bins)")
    right.set_xscale("log");right.set_yscale("log")
    right.set_xlim(low,high);right.set_ylim(low,high)
    ticks=[.01,.03,.1,.3]
    right.set_xticks(ticks,[f"{t:g}" for t in ticks]);right.set_yticks(ticks,[f"{t:g}" for t in ticks])
    right.xaxis.set_minor_locator(NullLocator());right.yaxis.set_minor_locator(NullLocator())
    right.tick_params(length=3,pad=2)
    right.set_xlabel(r"Predicted scale $\sigma(o)$",labelpad=4)
    right.set_ylabel("Measured residual RMS",labelpad=3)
    right.set_title("(b) HT: input-dependent scale",loc="left",pad=10)
    right.text(.04,.95,"GR00T / GR1\n"+rf"Spearman $\rho={ht['summary']['sigma_rms_spearman']:.2f}$",
               transform=right.transAxes,va="top",fontsize=8.5)
    right.legend(loc="lower right",bbox_to_anchor=(1.02,-.015),fontsize=7.2,
                 frameon=False,handlelength=1.5,labelspacing=.3,handletextpad=.4)
    fig.savefig(out/"multiscale_rms.pdf")
    fig.savefig(out/"multiscale_rms.png",dpi=300)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=Path("analysis/paper/mse_scale"))
    ap.add_argument("--out",type=Path,default=Path("analysis/paper/multiscale_rms"))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    mse={key:load_mse(args.data/(key+".npz")) for key in SETTINGS}
    ht=load_ht(args.data/"ht_gr1_60k_raw.npz")
    summary=dict(mse={k:d["summary"] for k,d in mse.items()},ht=ht["summary"],
        status="Training-fit diagnostic; no data Gaussianity, discrete scale clusters, or precision-phase mechanism established",
        histogram="Empirical histogram on common equal-log-width bins; equal peak heights; interval is weighted P10-P90, not a confidence interval")
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    draw(mse,ht,args.out)
    print(json.dumps({"mse_relative_p90_p10":{k:d["summary"]["relative_q90_q10"] for k,d in mse.items()},
                      "ht_rho":ht["summary"]["sigma_rms_spearman"]},indent=2))


if __name__=="__main__":main()

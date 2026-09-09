#!/usr/bin/env python3
"""Data-side phase-scale and tail probe on human Tool-Hang demonstrations.

No trained policy is used. For each query action chunk, estimate its local
conditional mean and scale from state-nearest chunks in disjoint demonstrations
of the same semantic phase. The resulting cross-demonstration residual is a
local action-dispersion proxy, not an identified aleatoric-noise sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import gammaln
from scipy.stats import kurtosis, norm, t


SEED=20260905
HORIZON=8
K_NEIGHBORS=32
QUERY_PER_DEMO_PHASE=12
PHASES=("Frame transit","Frame insertion","Tool transit","Tool hanging")
OBS_KEYS=("object","robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos")


def sustained(values,positive,k=5):
    active=values>=0 if positive else values<0
    for i in range(len(active)-k+1):
        if active[i:i+k].all():return i
    return None


def events(gripper):
    c1=sustained(gripper,True)
    o1=None if c1 is None else sustained(gripper[c1:],False)
    o1=None if o1 is None else c1+o1
    c2=None if o1 is None else sustained(gripper[o1:],True)
    c2=None if c2 is None else o1+c2
    o2=None if c2 is None else sustained(gripper[c2:],False)
    o2=None if o2 is None else c2+o2
    return c1,o1,c2,o2


def phase_ranges(gripper):
    """Non-overlapping, gripper-event-aligned carrying and precision windows."""
    e=events(gripper)
    if any(v is None for v in e):return None
    c1,o1,c2,o2=e
    # A gap separates the transit and final pre-release windows. The stop is
    # exclusive and query construction additionally requires a complete chunk.
    return ((c1+10,o1-45),(o1-40,o1-8),(c2+10,o2-55),(o2-50,o2-8))


def build_pool(path):
    X=[];Y=[];phase=[];demo=[];step=[]
    with h5py.File(path,"r") as h:
        names=sorted(h["data"],key=lambda k:int(k.split("_")[-1]))
        for di,name in enumerate(names):
            d=h["data"][name];a=np.clip(np.asarray(d["actions"],dtype=np.float32),-1,1)
            obs=np.concatenate([np.asarray(d["obs"][k],dtype=np.float32) for k in OBS_KEYS],axis=1)
            ranges=phase_ranges(a[:,6])
            if ranges is None:continue
            for p,(lo,hi) in enumerate(ranges):
                lo=max(1,lo);hi=min(len(a),hi)
                for tt in range(lo,hi-HORIZON+1,2):
                    # Two observation frames retain local velocity information.
                    X.append(np.stack([obs[tt-1],obs[tt]]).reshape(-1))
                    Y.append(a[tt:tt+HORIZON,:6].reshape(-1))
                    phase.append(p);demo.append(di);step.append(tt)
    return (np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(phase,np.int8),
            np.asarray(demo,np.int16),np.asarray(step,np.int16))


def standardize_states(X,cal):
    center=X[cal].mean(0,dtype=np.float64)
    scale=X[cal].std(0,dtype=np.float64)
    active=scale>max(1e-7,scale.max()*1e-5)
    scale=np.maximum(scale[active],np.median(scale[active])*.05)
    return ((X[:,active]-center[active])/scale).astype(np.float32),int(active.sum())


def fit_student_zero(x):
    """MLE of zero-centered Student-t df and scale; deterministic subsample."""
    x=np.asarray(x,dtype=np.float64)
    rng=np.random.default_rng(SEED)
    if len(x)>200_000:x=x[rng.choice(len(x),200_000,replace=False)]
    def nll(par):
        df=np.exp(par[0])+.25;scale=np.exp(par[1])
        z=x/scale
        ll=(gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*np.pi)
            -np.log(scale)-.5*(df+1)*np.log1p(z*z/df))
        return -ll.mean()
    out=minimize(nll,[np.log(3-.25),np.log(np.median(np.abs(x))/.6745)],method="Nelder-Mead",
                 options={"maxiter":800,"xatol":1e-7,"fatol":1e-8})
    assert out.success,out.message
    return float(np.exp(out.x[0])+.25),float(np.exp(out.x[1]))


def fit_heldout_models(z_matrix, query_fold, coordinate_rescale=False, valid=None):
    """Cross-fit zero-centered Gaussian and Student-t residual models.

    When ``coordinate_rescale`` is true, each of the 48 chunk coordinates is
    additionally divided by its training-fold RMS. This deliberately removes
    fixed position/rotation and horizon-coordinate scale differences before
    testing whether the remaining residual shape is heavy-tailed.
    """
    if valid is None:
        valid=np.ones_like(z_matrix,dtype=bool)
    assert valid.shape==z_matrix.shape
    fits=[]
    for train_fold in (0,1):
        tr=query_fold==train_fold;te=~tr
        coord_scale=np.ones(z_matrix.shape[1],dtype=np.float64)
        if coordinate_rescale:
            coord_scale=np.array([
                np.sqrt(np.mean(z_matrix[tr,j][valid[tr,j]]**2))
                for j in range(z_matrix.shape[1])],dtype=np.float64)
            assert np.all(coord_scale>0)
        z_train=(z_matrix[tr]/coord_scale)[valid[tr]]
        z_test=(z_matrix[te]/coord_scale)[valid[te]]
        gauss_scale=float(np.sqrt(np.mean(z_train**2)))
        df,t_scale=fit_student_zero(z_train)
        llg=float(np.mean(norm.logpdf(z_test,loc=0,scale=gauss_scale)))
        llt=float(np.mean(t.logpdf(z_test,df,loc=0,scale=t_scale)))
        fits.append(dict(train_fold=train_fold,n_train=int(len(z_train)),n_test=int(len(z_test)),
            coordinate_rescaled=coordinate_rescale,
            gaussian_scale=gauss_scale,student_df=df,student_scale=t_scale,
            heldout_nll_gaussian=-llg,heldout_nll_student=-llt,
            heldout_student_nll_gain=llt-llg,
            heldout_excess_kurtosis=float(kurtosis(z_test,fisher=True,bias=False)),
            retained_fraction_train=float(valid[tr].mean()),
            retained_fraction_test=float(valid[te].mean()),
            coordinate_scale=coord_scale.tolist()))
    return fits


def tail_curves(z_matrix, query_fold, fit, coordinate_rescale=False):
    """Held-out two-sided survival curves using parameters fit on fold zero."""
    coord_scale=np.asarray(fit["coordinate_scale"],dtype=np.float64)
    evalz=np.abs((z_matrix[query_fold==1]/coord_scale).reshape(-1))
    upper=min(8.0,float(np.quantile(evalz,.9999)))
    grid=np.geomspace(.5,upper,120)
    empirical=np.array([(evalz>=v).mean() for v in grid])
    gaussian=2*norm.sf(grid,scale=fit["gaussian_scale"])
    student=2*t.sf(grid,fit["student_df"],scale=fit["student_scale"])
    thresholds={}
    for threshold in (2.,3.,4.,5.):
        observed=float(np.mean(evalz>=threshold))
        expected=float(2*norm.sf(threshold,scale=fit["gaussian_scale"]))
        thresholds[str(int(threshold))]=dict(
            empirical=observed,gaussian=expected,
            empirical_over_gaussian=observed/expected)
    return dict(evaluation_fold=1,fit_fold=0,
        coordinate_rescaled=coordinate_rescale,grid=grid.tolist(),
        empirical=empirical.tolist(),gaussian=gaussian.tolist(),student=student.tolist(),
        threshold_survival=thresholds)


def run(path,out,k_neighbors=K_NEIGHBORS):
    X,Y,P,D,S=build_pool(path)
    assert len(np.unique(D))==200 and Y.shape[1]==HORIZON*6
    rng=np.random.default_rng(SEED)
    fold=np.array([hashlib.sha256(f"{SEED}:{int(d)}".encode()).digest()[0]%2 for d in D],dtype=np.int8)
    query_mask=np.zeros(len(X),bool)
    # Equal cap per demo/phase prevents long demonstrations from dominating.
    for d in np.unique(D):
        for p in range(len(PHASES)):
            ids=np.flatnonzero((D==d)&(P==p))
            if len(ids):query_mask[rng.choice(ids,min(QUERY_PER_DEMO_PHASE,len(ids)),replace=False)]=True
    residual=np.full_like(Y,np.nan,dtype=np.float32)
    local_mean=np.full_like(Y,np.nan,dtype=np.float32)
    local_scale=np.full(len(Y),np.nan,dtype=np.float32)
    neighbor_radius=np.full(len(Y),np.nan,dtype=np.float32)
    active_dims=[]
    for qfold in (0,1):
        cal=fold!=qfold
        Z,ad=standardize_states(X,cal);active_dims.append(ad)
        for p in range(len(PHASES)):
            ci=np.flatnonzero(cal&(P==p))
            qi=np.flatnonzero((fold==qfold)&query_mask&(P==p))
            assert len(ci)>k_neighbors and len(qi)>0
            tree=cKDTree(Z[ci])
            dist,nn=tree.query(Z[qi],k=k_neighbors,workers=-1)
            yn=Y[ci[nn]].astype(np.float64)
            mu=yn.mean(1)
            local_mean[qi]=mu.astype(np.float32)
            residual[qi]=(Y[qi]-mu).astype(np.float32)
            # Unbiased neighbor variance, pooled across the exact 48-D chunk.
            local_scale[qi]=np.sqrt(np.var(yn,axis=1,ddof=1).mean(1)).astype(np.float32)
            neighbor_radius[qi]=np.median(dist,axis=1).astype(np.float32)
    keep=query_mask
    assert np.isfinite(residual[keep]).all() and np.isfinite(local_scale[keep]).all()
    assert (local_scale[keep]>0).all()
    rq=residual[keep];pq=P[keep];dq=D[keep];fq=fold[keep]
    rms=np.sqrt(np.mean(rq.astype(np.float64)**2,axis=1))
    # Standardize each residual vector by a scale estimated without its action.
    z_matrix=(rq/local_scale[keep,None]).astype(np.float64)
    z=z_matrix.reshape(-1)
    element_phase=np.repeat(pq,Y.shape[1]);element_demo=np.repeat(dq,Y.shape[1]);element_fold=np.repeat(fq,Y.shape[1])
    # The primary tail control additionally removes fixed coordinate scale.
    # Raw locally standardized fits are retained as a diagnostic.
    fits=fit_heldout_models(z_matrix,fq,coordinate_rescale=False)
    coordinate_fits=fit_heldout_models(z_matrix,fq,coordinate_rescale=True)
    # Saturated controller targets are a plausible source of apparent tails.
    # Removing query elements at either action bound leaves the result intact.
    unclipped=(np.abs(Y[keep])<.999)&(np.abs(local_mean[keep])<.999)
    unclipped_coordinate_fits=fit_heldout_models(
        z_matrix,fq,coordinate_rescale=True,valid=unclipped)
    phase_mixture_fits=[]
    for train_fold in (0,1):
        tr=fq==train_fold;te=~tr
        coord_mean=np.mean(rq[tr],axis=0,dtype=np.float64)
        coord_std=np.std(rq[tr],axis=0,ddof=1,dtype=np.float64)
        standardized=(rq[te]-coord_mean)/coord_std
        transit=standardized[np.isin(pq[te],(0,2))].reshape(-1)
        precision=standardized[np.isin(pq[te],(1,3))].reshape(-1)
        transit_sigma=float(np.sqrt(np.mean(transit**2)))
        precision_sigma=float(np.sqrt(np.mean(precision**2)))
        phase_mixture_fits.append(dict(train_fold=train_fold,test_fold=1-train_fold,
            transit_n=int(len(transit)),precision_n=int(len(precision)),
            transit_mean=float(np.mean(transit)),precision_mean=float(np.mean(precision)),
            transit_sigma=transit_sigma,precision_sigma=precision_sigma,
            precision_over_transit_sigma=precision_sigma/transit_sigma,
            precision_percent_smaller=float(100*(1-precision_sigma/transit_sigma)),
            transit_weight=float(len(transit)/(len(transit)+len(precision))),
            coordinate_mean=coord_mean.tolist(),coordinate_std=coord_std.tolist()))
    local_scale_calibration=[]
    query_local_scale=local_scale[keep].astype(np.float64)
    for query_fold in (1,0):
        q=fq==query_fold
        predicted=query_local_scale[q]
        query_residual=rq[q].astype(np.float64)
        query_demo=dq[q]
        query_phase=pq[q]
        cuts=np.quantile(predicted,np.linspace(0,1,6))
        groups=np.searchsorted(cuts[1:-1],predicted,side="right")
        unique_demo=np.unique(query_demo)
        sumsq=np.zeros((len(unique_demo),5),dtype=np.float64)
        counts=np.zeros((len(unique_demo),5),dtype=np.int64)
        demo_row={int(d):i for i,d in enumerate(unique_demo)}
        group_summary=[]
        for group in range(5):
            gm=groups==group
            values=query_residual[gm]
            group_summary.append(dict(quintile=group+1,n_queries=int(gm.sum()),
                predicted_radius_median=float(np.median(predicted[gm])),
                heldout_residual_rms=float(np.sqrt(np.mean(values**2))),
                heldout_residual_mean=float(np.mean(values)),
                precision_phase_fraction=float(np.mean(np.isin(query_phase[gm],(1,3))))))
            for d in np.unique(query_demo[gm]):
                dm=gm&(query_demo==d);row=demo_row[int(d)]
                sumsq[row,group]=np.sum(query_residual[dm]**2)
                counts[row,group]=int(dm.sum()*query_residual.shape[1])
        boot_rms=np.empty((4000,5),dtype=np.float64)
        for b in range(len(boot_rms)):
            rows=rng.integers(len(unique_demo),size=len(unique_demo))
            boot_rms[b]=np.sqrt(sumsq[rows].sum(0)/counts[rows].sum(0))
        for group,row in enumerate(group_summary):
            row["episode_bootstrap_95_ci"]=np.quantile(
                boot_rms[:,group],[.025,.975]).tolist()
        ratio=boot_rms[:,-1]/boot_rms[:,0]
        sample_rms=np.sqrt(np.mean(query_residual**2,axis=1))
        observed_ratio=(group_summary[-1]["heldout_residual_rms"]
                        /group_summary[0]["heldout_residual_rms"])
        null_ratio=np.empty(4000,dtype=np.float64)
        for b in range(len(null_ratio)):
            shuffled=groups.copy()
            # Preserve phase composition while destroying state-local pairing.
            for phase_id in range(len(PHASES)):
                ids=np.flatnonzero(query_phase==phase_id)
                shuffled[ids]=rng.permutation(shuffled[ids])
            null_ratio[b]=(np.sqrt(np.mean(query_residual[shuffled==4]**2))
                           /np.sqrt(np.mean(query_residual[shuffled==0]**2)))
        permutation_p=float((1+np.sum(null_ratio>=observed_ratio))
                            /(1+len(null_ratio)))
        local_scale_calibration.append(dict(query_fold=query_fold,
            reference_fold=1-query_fold,n_queries=int(q.sum()),
            local_radius_quantile_edges=cuts.tolist(),groups=group_summary,
            high_over_low_rms=float(observed_ratio),
            high_over_low_bootstrap_95_ci=np.quantile(ratio,[.025,.975]).tolist(),
            within_phase_permutation_p=permutation_p,
            within_phase_null_ratio_q95_q99=np.quantile(
                null_ratio,[.95,.99]).tolist(),
            local_radius_q90_over_q10=float(np.quantile(predicted,.9)
                                             /np.quantile(predicted,.1)),
            log_radius_log_sample_rms_correlation=float(
                np.corrcoef(np.log(predicted),np.log(sample_rms))[0,1])))
    # Per-demo phase RMS: a unit for episode bootstrap and visible dots.
    demo_phase=[]
    for d in np.unique(dq):
        for p in range(len(PHASES)):
            m=(dq==d)&(pq==p)
            if m.any():demo_phase.append((int(d),p,int(m.sum()),float(np.sqrt(np.mean(rms[m]**2)))))
    phase_summary=[]
    boot=[]
    by_phase={p:np.array([(d,v) for d,pp,n,v in demo_phase if pp==p]) for p in range(len(PHASES))}
    for p in range(len(PHASES)):
        m=pq==p
        phase_summary.append(dict(phase=PHASES[p],n=int(m.sum()),demos=len(np.unique(dq[m])),
            residual_rms=float(np.sqrt(np.mean(rms[m]**2))),median_sample_rms=float(np.median(rms[m])),
            local_neighbor_scale_median=float(np.median(local_scale[keep][m])),
            neighbor_radius_median=float(np.median(neighbor_radius[keep][m])),
            standardized_abs_q90_q99=np.quantile(np.abs(rq[m]/local_scale[keep][m,None]),[.9,.99]).tolist()))
    for _ in range(1000):
        row=[]
        for p in range(len(PHASES)):
            vals=by_phase[p][:,1]
            row.append(float(np.median(vals[rng.integers(len(vals),size=len(vals))])))
        boot.append(row)
    boot=np.asarray(boot)
    pair_summary=[]
    for transit,precision,label in ((0,1,"Frame insertion / transit"),(2,3,"Tool hanging / transit")):
        transit_map={int(d):float(v) for d,v in by_phase[transit]}
        precision_map={int(d):float(v) for d,v in by_phase[precision]}
        common=np.array(sorted(set(transit_map)&set(precision_map)),dtype=int)
        ratios=np.array([precision_map[int(d)]/transit_map[int(d)] for d in common])
        boot_ratio=np.empty(4000)
        for b in range(len(boot_ratio)):
            ids=rng.integers(len(ratios),size=len(ratios))
            boot_ratio[b]=np.median(ratios[ids])
        pair_summary.append(dict(comparison=label,n_demos=int(len(common)),
            median_paired_ratio=float(np.median(ratios)),
            median_percent_lower=float(100*(1-np.median(ratios))),
            bootstrap_95_ci=np.quantile(boot_ratio,[.025,.975]).tolist(),
            fraction_precision_lower=float(np.mean(ratios<1))))
    raw_tail_plot=tail_curves(z_matrix,fq,fits[0],coordinate_rescale=False)
    controlled_tail_plot=tail_curves(
        z_matrix,fq,coordinate_fits[0],coordinate_rescale=True)
    summary=dict(dataset=str(path),dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seed=SEED,horizon=HORIZON,continuous_action_channels=list(range(6)),gripper_excluded=True,
        observation_keys=list(OBS_KEYS),nearest_neighbors=k_neighbors,
        query_per_demo_phase=QUERY_PER_DEMO_PHASE,pool_n=len(X),query_n=int(keep.sum()),
        demos=len(np.unique(D)),active_state_dimensions_by_fold=active_dims,
        phase_definitions={"Frame transit":"10 frames after first grasp to 45 frames before first release",
            "Frame insertion":"40 to 8 frames before first release",
            "Tool transit":"10 frames after second grasp to 55 frames before final release",
            "Tool hanging":"50 to 8 frames before final release"},
        metric=f"RMS of query 8-step x 6-D continuous action minus {k_neighbors}-neighbor mean; controller-normalized action units",
        phase_summary=phase_summary,paired_phase_summary=pair_summary,
        phase_gaussian_mixture_fit=phase_mixture_fits,
        local_scale_calibration=local_scale_calibration,
        student_fit=fits,coordinate_controlled_student_fit=coordinate_fits,
        unclipped_coordinate_controlled_student_fit=unclipped_coordinate_fits,
        raw_tail_plot=raw_tail_plot,tail_plot=controlled_tail_plot,
        scope="Cross-demonstration local action dispersion; includes local mean-estimation and state-matching error; not identified aleatoric noise")
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"probe.npz",phase=pq,demo=dq,fold=fq,residual=rq,
        target=Y[keep],local_mean=local_mean[keep],
        rms=rms,local_scale=local_scale[keep],neighbor_radius=neighbor_radius[keep],
        demo_phase=np.asarray(demo_phase,float),bootstrap_phase_median=boot,z=z,
        z_matrix=z_matrix,
        element_phase=element_phase,element_demo=element_demo,element_fold=element_fold)
    print(json.dumps({"pool_n":len(X),"query_n":int(keep.sum()),"phase_summary":phase_summary,
                      "paired_phase_summary":pair_summary,
                      "coordinate_controlled_student_fit":coordinate_fits},indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=Path("/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"))
    ap.add_argument("--out",type=Path,default=Path("analysis/paper/data_ht_motivation"))
    ap.add_argument("--neighbors",type=int,default=K_NEIGHBORS)
    args=ap.parse_args();run(args.data,args.out,args.neighbors)


if __name__=="__main__":main()

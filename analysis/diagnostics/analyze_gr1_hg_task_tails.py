"""Episode cross-fitting for identified GR1 or Bridge pure-HG residual shards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm, t as student_t

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
from tail_stats_single import fit_student_t, t_logpdf


def analyze(path):
    with np.load(path) as data:
        meta = json.loads(str(data['metadata']))
        r = data['residual'].astype(float)
        sigma = data['sigma'].astype(float)
        eps = data['episode']
        tid = int(data['task_id'][0])
        assert np.all(data['task_id'] == tid)
        np.testing.assert_allclose(r, data['prediction']-data['target'], atol=1e-6)
    assert r.shape[1:] == (8,len(meta['continuous_channels'])) and meta['objective'] == 'hg'
    assert r.shape[2] in (6,29)
    x = r/sigma[:,None,None]
    unique = np.unique(eps)
    order = np.random.default_rng(20260908+tid).permutation(unique)
    mapping = {int(ep):i%5 for i,ep in enumerate(order)}
    fold = np.array([mapping[int(ep)] for ep in eps])
    standardized = np.empty_like(x)
    thresholds = np.linspace(.5,6,111)
    fits = []; curves = []; total = 0
    gnll = tnll = lnll = 0.
    for f in range(5):
        train = x[fold!=f]
        mean, std = train.mean(0), train.std(0)
        assert np.all(std > 0), (path, std.min())
        tr = ((train-mean)/std).ravel()
        te = (x[fold==f]-mean)/std
        standardized[fold==f] = te
        nu, s = fit_student_t(tr)
        b = np.abs(tr).mean()
        gnll += float(-norm.logpdf(te).sum())
        tnll += float(-t_logpdf(te,nu,s).sum())
        lnll += float((np.log(2*b)+np.abs(te)/b).sum())
        total += te.size
        curves.append(2*student_t.sf(thresholds/s,nu))
        fits.append(dict(fold=f,nu=float(nu),scale=float(s),test_coordinates=te.size,
                         calibration_coordinate_scale_min=float(std.min())))
    absz = np.abs(standardized)
    g3 = float(2*norm.sf(3))
    hit = np.array([(absz[eps==ep]>3).sum() for ep in unique])
    size = np.array([absz[eps==ep].size for ep in unique])
    pick = np.random.default_rng(20260908+100+tid).integers(len(unique),size=(2000,len(unique)))
    boot = hit[pick].sum(1)/size[pick].sum(1)/g3
    record = dict(task=meta['task_names'][tid],task_id=tid,episodes=len(unique),states=len(x),
        coordinates=absz.size,p_gt3=float((absz>3).mean()),gaussian_gt3=g3,
        tail_ratio=float((absz>3).mean()/g3),ratio_ci=np.quantile(boot,[.025,.975]).tolist(),
        empirical_tail=[float((absz>v).mean()) for v in thresholds],
        student_predictive_tail=np.average(curves,axis=0,weights=[f['test_coordinates'] for f in fits]).tolist(),
        gaussian_nll=gnll/total,student_nll=tnll/total,laplace_nll=lnll/total,
        student_nll_gain=(gnll-tnll)/total,fits=fits,source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return record, standardized, fold, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--dataset',choices=['gr1','bridge'],default='gr1')
    p.add_argument('--expected-tasks',type=int,default=24)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    paths = sorted(args.raw_dir.glob('hg_task*.npz'))
    if not args.allow_partial:
        assert len(paths)==args.expected_tasks, len(paths)
    records=[]
    for path in paths:
        dest = args.output/f'{path.stem}_summary.json'
        if dest.exists():
            record = json.loads(dest.read_text())
            assert record['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            record,z,fold,meta=analyze(path)
            dest.write_text(json.dumps(record,indent=2)+'\n')
            np.savez_compressed(args.output/f'{path.stem}_crossfitted_z.npz',z=z,fold=fold)
        records.append(record)
        print(record['task'], 'ratio',round(record['tail_ratio'],3),'gain',round(record['student_nll_gain'],4),flush=True)
    summary=dict(dataset='RoboCasa-GR1' if args.dataset=='gr1' else 'Bridge / WidowX',task_count=len(records),tasks=records,
                 thresholds=np.linspace(.5,6,111).tolist(),
                 normalization='Same as Bridge preview: HG own sigma; within-task per-coordinate centered 5-fold episode cross-fitting; no test RMS rescaling.',
                 scope='Training-demonstration diagnostic; only calibration and density fits are episode-held-out.')
    (args.output/f'{args.dataset}_hg_tail_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()

"""Fractal zero-output probe, using the same coordinate protocol as WidowX.

Input gt is already training-normalized, extracted by fit_dump_fractal.py.
The Google embodiment orders each step as x,y,z,roll,pitch,yaw,gripper.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_zero_model_tail import analyze
from tail_stats_single import make_folds


def main():
    outdir = Path("analysis/paper/longtail_motivation")
    source = outdir / "fractal_raw/fit_dump_fractal.npz"
    mapping = json.loads((outdir / "fractal_raw/fractal_ep2inst.json").read_text())
    with np.load(source) as raw:
        target = raw["gt"].reshape(-1, 8, 7)[:, :, :6].astype(np.float64)
        episode, step = raw["ep"], raw["step"]
    assert np.isfinite(target).all() and (episode >= 0).all()
    task = np.array([mapping[str(e)] for e in episode])
    order = np.lexsort((step, episode, task))
    target, episode, task = target[order], episode[order], task[order]
    fold = make_folds(task, episode)
    assert len(np.unique(fold)) == 5
    for ep in np.unique(episode):
        assert len(np.unique(fold[episode == ep])) == 1
    result = {
        "definition": "f(o)=0 in training-normalized action space, residual=-target; no mean subtraction",
        "source": str(source),
        "cluster_source": "/mnt/pfs/yuchen/groot/fit_dump_fractal.npz",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sampling": "Existing first 2000 training-loader items, seed 0; not the WidowX task-balanced subset",
        "states": len(target), "episodes": len(np.unique(episode)),
        "instructions": len(np.unique(task)), "coordinates": target.size,
        "channels": "eight steps, six continuous channels; gripper excluded",
        "protocol": "Same v4 panel-b helper and zero-model analyzer as WidowX: task-stratified episode folds, other-fold coordinate RMS, final pooled unit RMS, fixed zero location",
        "fold_state_counts": np.bincount(fold, minlength=5).tolist(),
        "zero_model": analyze(-target, fold),
    }
    # Many Fractal instructions have only one or two episodes. The exact
    # WidowX task-stratified helper puts those disproportionately in fold 0.
    # Check a second split balanced by episode count, without task strata.
    balanced_fold = make_folds(np.zeros(len(episode), dtype=int), episode)
    result["balanced_episode_split_sensitivity"] = {
        "protocol": "Same statistics, five episode folds without task stratification",
        "fold_state_counts": np.bincount(balanced_fold, minlength=5).tolist(),
        "zero_model": analyze(-target, balanced_fold),
    }
    output = outdir / "fractal_zero_model_tail_summary.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

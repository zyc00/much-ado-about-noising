"""Reduce COMPLETE dataset probes to per-instruction and dataset-wide statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def summarize(root, dataset):
    folder = root / dataset
    manifest = json.loads((folder / "sampling_manifest.json").read_text())
    done = [json.loads((folder / f"rank{rank:02d}_done.json").read_text()) for rank in range(2)]
    assert sum(d["episodes"] for d in done) == manifest["sampled_episodes"]
    chunks = []
    seen = set()
    for rank, d in enumerate(done):
        for part in range(d["shards"]):
            p = folder / f"rank{rank:02d}_part{part:04d}.npz"
            with np.load(p) as z:
                meta = json.loads(str(z["metadata"]))
                assert not meta["smoke_only"] and meta["checkpoint"] == manifest["checkpoint"]
                a, pred = z["target"].astype(float), z["prediction"].astype(float)
                e = np.mean((pred-a)**2, axis=(1,2))
                label = np.sqrt(np.mean(a*a, axis=(1,2)))
                for ep,step in zip(z["episode"],z["step"]):
                    assert (int(ep),int(step)) not in seen
                    seen.add((int(ep),int(step)))
                chunks.append(dict(energy=e,label=label,**{k:z[k] for k in ["episode","task_id","progress","episode_weight"]}))
    arrays = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    assert len(seen) == manifest["sampled_states"]
    stage = np.minimum(9,(10*arrays["progress"]).astype(int))
    rows, eligible = [], []
    for task in manifest["tasks"]:
        keep = arrays["task_id"] == task["task_id"]
        energy, magnitude = arrays["energy"][keep], arrays["label"][keep]
        assert len(np.unique(arrays["episode"][keep])) == task["sampled_episodes"]
        curve = [float(np.sqrt(np.mean(energy[stage[keep]==b]))) if np.any(stage[keep]==b) else None for b in range(10)]
        rho = float(spearmanr(magnitude,np.sqrt(energy)).statistic) if len(energy)>2 and np.ptp(magnitude)>0 and np.ptp(energy)>0 else None
        row = dict(dataset=dataset, task=task["instruction"], n=len(energy),episodes=task["sampled_episodes"],
                   population_episodes=task["total_episodes"], label_residual_rho=rho, stage_rms=curve,
                   display_eligible=task["display_eligible"])
        rows.append(row)
        if task["display_eligible"]:
            assert rho is not None
            eligible.append(row)
    overall = []
    for b in range(10):
        keep = stage==b
        overall.append(float(np.sqrt(np.average(arrays["energy"][keep],weights=arrays["episode_weight"][keep]))) if keep.any() else None)
    return dict(tasks=len(eligible), states=len(seen), episodes=manifest["sampled_episodes"],
                total_episodes=manifest["total_episodes"], eligible_episodes=manifest["eligible_episodes"],
                instruction_groups=manifest["instruction_groups"], nonempty_instruction_groups=sum(bool(t["instruction"]) for t in manifest["tasks"]),
                task_display_rule="Nonempty instruction groups with 12 sampled episodes; all other groups retained in dataset-wide weighted statistics.",
                displayed_group_population_episodes=sum(r["population_episodes"] for r in eligible),
                dataset_wide_stage_rms=overall, checkpoint=manifest["checkpoint"]), eligible, rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    base=json.loads(args.base.read_text())
    result=dict(scope="Training-demonstration diagnostics. Broad instruction sampling, not policy-held-out.",
                base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
                datasets={k:base["datasets"][k] for k in ["gr1","pi05"]},
                per_task=[r for r in base["per_task"] if r["dataset"] in ["gr1","pi05"]])
    all_rows={}
    for s in ["bridge","fractal"]:
        stats, rows, all_rows[s]=summarize(args.root,s)
        result["datasets"][s]=stats
        result["per_task"].extend(rows)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output.parent/"all_instruction_groups.json").write_text(json.dumps(all_rows,allow_nan=False)+"\n")
    print(json.dumps(result["datasets"],indent=2))

if __name__=="__main__":
    main()

"""Slice task segments out of the clean source demos (real expert actions).

Segments (per demo; c1=grasp-close, af0=align-done, o1=first release):
  full  : [0 : o1+tail]                  initial -> insertion (+release tail)
  front : [c1 : af0 - backoff(0..40)]    pick -> handoff (== align+insert start)

backoff is deterministic per seed (RandomState(seed)) so front's end matches
the existing align+insertion (back-half) start exactly -> the two tile.

Usage:
  python scripts/slice_segments.py --segment full  --n 20000 \
     --src data/tool_hang_clean_20000.hdf5 --out data/tool_hang_full2ins_20000.hdf5
  python scripts/slice_segments.py --segment front --n 20000 \
     --src data/tool_hang_clean_20000.hdf5 --out data/tool_hang_front_20000.hdf5
"""
import argparse
import h5py
import numpy as np
from tqdm import tqdm

TAIL = 10  # frames after first release to include the settle of a done insertion


def frames(d):
    a = d["actions"][:]; ez = d["obs/robot0_eef_pos"][:, 2]; g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl:
        return None
    c1 = cl[0]
    # release must come AFTER grasp (guard against spurious early gripper-open)
    op = [t for t in range(1, len(g)) if g[t-1] >= 0 and g[t] < 0 and t > c1]
    if not op or op[0] <= c1:
        return None
    o1 = op[0]
    af0 = c1 + int(np.argmax(ez[c1:o1]))
    return c1, af0, o1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True, choices=["full", "front", "pick2ins", "fixbo", "rangebo", "init2grasp", "init2graspP", "init2align"])
    ap.add_argument("--backoff", type=int, default=0)  # for segment=fixbo: start = align_done - backoff
    ap.add_argument("--src", default="data/tool_hang_clean_20000.hdf5")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backoff_max", type=int, default=40)
    args = ap.parse_args()

    sf = h5py.File(args.src, "r")
    src_keys = sorted(sf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    out = h5py.File(args.out, "w")
    dg = out.create_group("data")
    for ak, av in sf["data"].attrs.items():
        dg.attrs[ak] = av

    n = 0; total = 0
    pbar = tqdm(total=args.n, desc=args.segment)
    for k in src_keys:
        if n >= args.n:
            break
        d = sf["data/" + k]
        fr = frames(d)
        if fr is None:
            continue
        c1, af0, o1 = fr
        sd = int(d.attrs.get("seed", 0))
        if args.segment == "full":
            s, e = 0, min(o1 + TAIL, d["actions"].shape[0])
        elif args.segment == "init2grasp":
            s, e = 0, min(c1 + 20, d["actions"].shape[0])  # init -> grasp + initial lift
        elif args.segment == "init2graspP":
            s, e = 0, min(c1 + args.backoff, d["actions"].shape[0])  # init -> grasp + backoff frames
        elif args.segment == "init2align":
            s, e = 0, min(af0 + 5, d["actions"].shape[0])  # init -> grasp -> lift/align peak
        elif args.segment == "pick2ins":
            s, e = c1, min(o1 + TAIL, d["actions"].shape[0])
        elif args.segment == "fixbo":
            s, e = max(c1 + 2, af0 - args.backoff), min(o1 + TAIL, d["actions"].shape[0])
        elif args.segment == "rangebo":
            bo = int(np.random.RandomState(sd).randint(0, args.backoff_max + 1))
            s, e = max(c1 + 2, af0 - bo), min(o1 + TAIL, d["actions"].shape[0])
        else:  # front
            bo = int(np.random.RandomState(sd).randint(0, args.backoff_max + 1))
            s, e = c1, max(c1 + 2, af0 - bo)
        if e - s < 2:
            continue
        grp = dg.create_group(f"demo_{n}")
        for ak, av in d.attrs.items():
            grp.attrs[ak] = av
        grp.attrs["num_samples"] = e - s
        for dk in ["actions", "states", "rewards", "dones"]:
            if dk in d:
                grp.create_dataset(dk, data=d[dk][s:e])
        og = grp.create_group("obs")
        for ok in d["obs"].keys():
            og.create_dataset(ok, data=d["obs/" + ok][s:e])
        n += 1; total += e - s
        pbar.update(1)
    pbar.close()
    dg.attrs["total"] = total
    out.close(); sf.close()
    print(f"SLICED {args.out}: {n} demos, {total} samples, avg {total/max(n,1):.0f}/demo")


if __name__ == "__main__":
    main()

"""Make a first-N-demos subset of a demo hdf5, preserving structure/attrs.

Usage:
  python scripts/make_subset.py --src data/tool_hang_aligninsert_20000.hdf5 \
     --n 200 --out data/tool_hang_aligninsert_200.hdf5
"""
import argparse
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sf = h5py.File(args.src, "r")
    keys = sorted(sf["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:args.n]
    with h5py.File(args.out, "w") as of:
        dg = of.create_group("data")
        for ak, av in sf["data"].attrs.items():
            dg.attrs[ak] = av
        total = 0
        for i, k in enumerate(keys):
            src = sf["data/" + k]
            grp = dg.create_group(f"demo_{i}")
            for ak, av in src.attrs.items():
                grp.attrs[ak] = av
            for dk in ["actions", "states", "rewards", "dones"]:
                if dk in src:
                    grp.create_dataset(dk, data=src[dk][:])
            og = grp.create_group("obs")
            for ok in src["obs"].keys():
                og.create_dataset(ok, data=src["obs/" + ok][:])
            total += src["actions"].shape[0]
        dg.attrs["total"] = total
    sf.close()
    print(f"SUBSET {args.out}: {len(keys)} demos, {total} samples")


if __name__ == "__main__":
    main()

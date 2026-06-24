"""Concatenate demos from multiple robomimic hdf5 files into one (renumbered).
Used to build DAgger-augmented training data = original demos + on-policy
resampled mid-state demos, for training a SINGLE policy.

Usage:
  python scripts/merge_hdf5.py --out data/tool_hang_dagger_2000.hdf5 \
     data/tool_hang_full2ins_2000.hdf5 data/tool_hang_handoffins_2000.hdf5
"""
import argparse
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("srcs", nargs="+")
    args = ap.parse_args()

    out = h5py.File(args.out, "w")
    dg = out.create_group("data")
    n = 0; total = 0; env_args = None
    for src in args.srcs:
        sf = h5py.File(src, "r")
        if env_args is None and "env_args" in sf["data"].attrs:
            env_args = sf["data"].attrs["env_args"]
        keys = sorted(sf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for k in keys:
            s = sf["data/" + k]
            grp = dg.create_group(f"demo_{n}")
            for ak, av in s.attrs.items():
                grp.attrs[ak] = av
            for dk in ["actions", "states", "rewards", "dones"]:
                if dk in s:
                    grp.create_dataset(dk, data=s[dk][:])
            og = grp.create_group("obs")
            for ok in s["obs"].keys():
                og.create_dataset(ok, data=s["obs/" + ok][:])
            total += s["actions"].shape[0]; n += 1
        sf.close()
    if env_args is not None:
        dg.attrs["env_args"] = env_args
    dg.attrs["total"] = total
    out.close()
    print(f"MERGED {args.out}: {n} demos, {total} samples from {len(args.srcs)} files")


if __name__ == "__main__":
    main()

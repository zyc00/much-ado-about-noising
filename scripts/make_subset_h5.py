"""Slice first N demos of the MP-2k hdf5 into a subset file. Env: N_DEMOS."""
import os

import h5py

N = int(os.environ["N_DEMOS"])
src = "data/tool_hang_full2ins_2000.hdf5"
dst = f"data/tool_hang_full2ins_sub{N}.hdf5"
with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
    fo.create_group("data")
    for k, v in fi["data"].attrs.items():
        fo["data"].attrs[k] = v
    names = sorted(fi["data"].keys(), key=lambda d: int(d.split("_")[1]))[:N]
    for dn in names:
        fi.copy(f"data/{dn}", fo["data"], name=dn)
    if "mask" in fi:
        fi.copy("mask", fo)
print("SUBSET done", dst, N, flush=True)

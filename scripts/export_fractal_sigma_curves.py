"""Keep full raw frames on PFS; export small, lossless numeric arrays locally."""
import argparse
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
args=p.parse_args()
for seed in [1235,1236,1240]:
    source=args.root/f'{seed}.npz'
    if source.exists():
        with np.load(source) as d:
            np.savez_compressed(args.root/f'{seed}_curves.npz',
                                **{k:d[k] for k in d.files if k!='static_frames'})

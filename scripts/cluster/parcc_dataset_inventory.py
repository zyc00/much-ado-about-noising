"""Run on source utility pod: inventory file paths/sizes, never edit datasets."""
import gzip
import json
import os
from pathlib import Path

ROOTS = {
    'widowx': '/mnt/pfs/yuchen/groot/bridge_orig_lerobot',
    'fractal': '/mnt/pfs/yuchen/groot/fractal_lerobot',
    'gr1': '/mnt/pfs/yuchen/groot/lerobot/LeRobot',
}
out = Path('/mnt/pfs/yuchen/parcc_setup_20260908')
out.mkdir(exist_ok=True)
for name, root in ROOTS.items():
    rows=[]
    for folder, dirs, files in os.walk(root,followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in ('.cache','._____temp','__pycache__'))
        for file in sorted(files):
            if file in ('.msc','.mv','.DS_Store'):
                continue
            path=Path(folder)/file
            if path.is_file():
                rows.append([str(path.relative_to(root)),path.stat().st_size])
    with gzip.open(out/f'{name}.inventory.json.gz','wt') as stream:
        json.dump(dict(source=root,files=rows),stream)
    print(name,len(rows),sum(x[1] for x in rows),flush=True)

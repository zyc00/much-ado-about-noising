"""Copy scientific probe data in small chunks to tolerate cluster stream truncation."""
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


PYTHON = "/mnt/pfs/yuchen/groot/Isaac-GR00T/.venv/bin/python"
SOURCE = "/mnt/pfs/yuchen/groot/fit_dump_fractal.npz"


def remote(code):
    for attempt in range(4):
        try:
            return subprocess.check_output(
                ["kubectl", "exec", "yuchen-vla-util", "--", PYTHON, "-c", code],
                timeout=45, stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source = args.source
    meta = json.loads(remote(
        f"from pathlib import Path; import hashlib,json; b=Path({source!r}).read_bytes(); "
        "print(json.dumps([len(b),hashlib.sha256(b).hexdigest()]))"
    ))
    def chunk(start):
        code = (f"from pathlib import Path; import base64; b=Path({source!r}).read_bytes(); "
                f"print(base64.b64encode(b[{start}:{start+24576}]).decode())")
        return base64.b64decode(remote(code))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        data = b"".join(pool.map(chunk, range(0, meta[0], 24576)))
    assert len(data) == meta[0] and hashlib.sha256(data).hexdigest() == meta[1]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        print(f"Verified {len(data)} bytes, SHA256 {meta[1]}")
        return
    out = Path("analysis/paper/longtail_motivation/fractal_raw")
    (out / "fit_dump_fractal.npz").write_bytes(data)
    mapping = remote(
        f"import numpy as np,json; x=np.load({SOURCE!r}); "
        "m=json.load(open('/mnt/pfs/yuchen/groot/fractal_ep2inst.json')); "
        "print(json.dumps({str(e):m[str(e)] for e in np.unique(x['ep'])}))"
    )
    subset = json.loads(mapping)
    (out / "fractal_ep2inst.json").write_text(json.dumps(subset, indent=2))
    print(f"Verified {len(data)} bytes, {len(subset)} episode instructions, SHA256 {meta[1]}")


if __name__ == "__main__":
    main()

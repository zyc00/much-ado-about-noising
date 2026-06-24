"""Download the ToolHang MIP dataset from HuggingFace into ./data/.

One command for collaborators. The HF repo only needs the CLEAN source demos +
eval set; the task segments (full2ins / init2grasp / pick2ins) are regenerated
locally by slicing (default), so the download stays small (~16 GB for 2k+20k).

Examples:
  python scripts/download_data.py --repo yuchen0187/toolhang-mip-data
  python scripts/download_data.py --repo yuchen0187/toolhang-mip-data --scale 2k
  python scripts/download_data.py --repo yuchen0187/toolhang-mip-data --segments download
  HF_DATA_REPO=yuchen0187/toolhang-mip-data python scripts/download_data.py
"""
import argparse
import os
import subprocess
import sys
from huggingface_hub import hf_hub_download

DEFAULT_REPO = os.environ.get("HF_DATA_REPO", "yuchen0187/toolhang-mip-data")

EVAL_FILES = ["warmstart_demos.hdf5", "full_eval_seeds.npy"]
CLEAN = {"2k": "tool_hang_clean_2000.hdf5", "20k": "tool_hang_clean_20000.hdf5"}
N = {"2k": 2000, "20k": 20000}
SEGMENTS = [  # (slice_segments --segment, output basename suffix)
    ("full", "full2ins"),
    ("init2grasp", "init2grasp"),
    ("pick2ins", "pick2ins"),
]


def pull(repo, fname, out_dir):
    print(f"  ↓ {fname}", flush=True)
    try:
        return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=out_dir)
    except Exception as e:
        print(f"    [skip] {fname}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id (or set HF_DATA_REPO)")
    ap.add_argument("--scale", choices=["2k", "20k", "both"], default="both")
    ap.add_argument("--segments", choices=["slice", "download", "skip"], default="slice",
                    help="slice: regenerate from clean (small download); download: pull from HF; skip: clean only")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    if "CHANGE_ME" in args.repo:
        sys.exit("Set --repo <user>/<dataset> (or HF_DATA_REPO env var).")
    os.makedirs(args.out, exist_ok=True)
    scales = ["2k", "20k"] if args.scale == "both" else [args.scale]
    print(f"Downloading from {args.repo} (scales={scales}, segments={args.segments}) -> {args.out}/")

    print("[eval set]")
    for f in EVAL_FILES:
        pull(args.repo, f, args.out)

    for s in scales:
        print(f"[{s} clean source]")
        clean_path = pull(args.repo, CLEAN[s], args.out)
        if args.segments == "download":
            print(f"[{s} segments: download]")
            for _, suf in SEGMENTS:
                pull(args.repo, f"tool_hang_{suf}_{N[s]}.hdf5", args.out)
        elif args.segments == "slice":
            if not clean_path or not os.path.exists(clean_path):
                print(f"    [skip slicing] clean {s} not available")
                continue
            print(f"[{s} segments: slicing from clean]")
            for seg, suf in SEGMENTS:
                out = os.path.join(args.out, f"tool_hang_{suf}_{N[s]}.hdf5")
                print(f"  ✂ {seg} -> {out}")
                subprocess.run([sys.executable, "scripts/slice_segments.py",
                                "--segment", seg, "--src", clean_path, "--out", out, "--n", str(N[s])],
                               check=True)

    print("\nDone. Files in", args.out + "/:")
    for f in sorted(os.listdir(args.out)):
        if f.endswith((".hdf5", ".npy")):
            print(f"  {os.path.getsize(os.path.join(args.out, f))/1e9:6.2f} GB  {f}")


if __name__ == "__main__":
    main()

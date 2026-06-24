"""Upload / update the ToolHang MIP dataset on HuggingFace. Run by the data owner.

Requires a write token once: `huggingface-cli login`.
By default uploads only the CLEAN source demos + eval set (~16 GB for 2k+20k);
collaborators regenerate the segments via download_data.py. Use --with-segments
to also publish the 6 sliced segment files (then ~30 GB, no slicing needed).

Examples:
  python scripts/upload_data.py --repo yuchen0187/toolhang-mip-data
  python scripts/upload_data.py --repo yuchen0187/toolhang-mip-data --scale 2k
  python scripts/upload_data.py --repo yuchen0187/toolhang-mip-data --with-segments --private
"""
import argparse
import os
import sys
from huggingface_hub import HfApi, create_repo

EVAL_FILES = ["warmstart_demos.hdf5", "full_eval_seeds.npy"]
CLEAN = {"2k": "tool_hang_clean_2000.hdf5", "20k": "tool_hang_clean_20000.hdf5"}
SEG = {"2k": ["tool_hang_full2ins_2000.hdf5", "tool_hang_init2grasp_2000.hdf5", "tool_hang_pick2ins_2000.hdf5"],
       "20k": ["tool_hang_full2ins_20000.hdf5", "tool_hang_init2grasp_20000.hdf5", "tool_hang_pick2ins_20000.hdf5"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. yuchen0187/toolhang-mip-data")
    ap.add_argument("--scale", choices=["2k", "20k", "both"], default="both")
    ap.add_argument("--with-segments", action="store_true", help="also upload the 6 sliced segment files")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    scales = ["2k", "20k"] if args.scale == "both" else [args.scale]
    files = list(EVAL_FILES)
    for s in scales:
        files.append(CLEAN[s])
        if args.with_segments:
            files += SEG[s]

    # only upload what exists locally
    present, missing = [], []
    for f in files:
        (present if os.path.exists(os.path.join(args.data_dir, f)) else missing).append(f)
    if missing:
        print("[warning] not found locally, skipping:", ", ".join(missing))
    if not present:
        sys.exit("Nothing to upload (no listed files in " + args.data_dir + "/).")

    total = sum(os.path.getsize(os.path.join(args.data_dir, f)) for f in present)
    print(f"Uploading {len(present)} files (~{total/1e9:.1f} GB) to {args.repo} "
          f"({'private' if args.private else 'public'}):")
    for f in present:
        print(f"  {os.path.getsize(os.path.join(args.data_dir, f))/1e9:6.2f} GB  {f}")

    create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
    api = HfApi()
    for f in present:
        print(f"↑ {f} ...", flush=True)
        api.upload_file(path_or_fileobj=os.path.join(args.data_dir, f),
                        path_in_repo=f, repo_id=args.repo, repo_type="dataset")
    print(f"\nDone. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

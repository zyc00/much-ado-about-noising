"""Upload trained checkpoints to a HuggingFace model repo. Owner only (needs
`huggingface-cli login`). Uploads logs/<exp>/models/model_latest.pt for the
canonical 2k + 20k models (generalist / grasp-spec / insert-spec × MSE / MIP),
laid out as <exp>/model_latest.pt so download_ckpt.py can restore them.

  python scripts/upload_ckpt.py --repo yuchen0187/toolhang-mip-checkpoints
"""
import argparse
import os
import sys
from huggingface_hub import HfApi, create_repo

EXPS_2K = ["full_regression_2000", "full_mip_2000", "grasp_regression_2000",
           "grasp_mip_2000", "pick2ins_regression_2000", "pick2ins_mip_2000"]
EXPS_20K = ["full_regression_20kB", "full_mip_20kB", "grasp_regression_20kB",
            "grasp_mip_20kB", "pick2ins_regression_20kB", "pick2ins_mip_20kB"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF model repo id, e.g. yuchen0187/toolhang-mip-checkpoints")
    ap.add_argument("--scale", choices=["2k", "20k", "both"], default="both")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    EXPS = (EXPS_2K if args.scale in ("2k", "both") else []) + (EXPS_20K if args.scale in ("20k", "both") else [])
    present = [(e, f"logs/{e}/models/model_latest.pt") for e in EXPS
               if os.path.exists(f"logs/{e}/models/model_latest.pt")]
    missing = [e for e in EXPS if not os.path.exists(f"logs/{e}/models/model_latest.pt")]
    if missing:
        print("[warning] missing locally, skipping:", ", ".join(missing))
    if not present:
        sys.exit("No checkpoints found under logs/<exp>/models/.")
    total = sum(os.path.getsize(p) for _, p in present)
    print(f"Uploading {len(present)} ckpts (~{total/1e9:.1f} GB) to {args.repo} "
          f"({'private' if args.private else 'public'}):")
    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api = HfApi()
    for e, p in present:
        print(f"↑ {e}/model_latest.pt", flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{e}/model_latest.pt",
                        repo_id=args.repo, repo_type="model")
    print(f"\nDone. https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

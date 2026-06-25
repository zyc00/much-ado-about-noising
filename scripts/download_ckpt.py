"""Download trained checkpoints from HuggingFace into logs/<exp>/models/, so the
eval scripts work out-of-the-box (no training needed). Public repo → no auth.

  python scripts/download_ckpt.py                       # all canonical 2k+20k
  python scripts/download_ckpt.py --scale 2k            # only 2k
  python scripts/download_ckpt.py --exps full_mip_2000 grasp_mip_2000
  HF_CKPT_REPO=<user>/repo python scripts/download_ckpt.py

After downloading, evaluate directly, e.g.:
  python scripts/eval_warmstart.py --ckpt logs/full_mip_2000/models/model_latest.pt \\
     --dataset data/tool_hang_full2ins_2000.hdf5 --loss mip \\
     --demos data/warmstart_demos.hdf5 --warm_to 0 --init_mode reset_settle --settle 10 \\
     --success assembled --n 100
"""
import argparse
import os
import sys
from huggingface_hub import hf_hub_download

DEFAULT_REPO = os.environ.get("HF_CKPT_REPO", "yuchen0187/toolhang-mip-checkpoints")
EXPS_2K = ["full_regression_2000", "full_mip_2000", "grasp_regression_2000",
           "grasp_mip_2000", "pick2ins_regression_2000", "pick2ins_mip_2000"]
EXPS_20K = ["full_regression_20kB", "full_mip_20kB", "grasp_regression_20kB",
            "grasp_mip_20kB", "pick2ins_regression_20kB", "pick2ins_mip_20kB"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--scale", choices=["2k", "20k", "both"], default="both")
    ap.add_argument("--exps", nargs="+", default=None, help="explicit exp names (overrides --scale)")
    args = ap.parse_args()
    if "CHANGE_ME" in args.repo:
        sys.exit("Set --repo <user>/<repo> (or HF_CKPT_REPO).")
    exps = args.exps or ((EXPS_2K if args.scale in ("2k", "both") else []) +
                         (EXPS_20K if args.scale in ("20k", "both") else []))
    print(f"Downloading {len(exps)} checkpoints from {args.repo} -> logs/<exp>/models/")
    for e in exps:
        dst = f"logs/{e}/models"
        os.makedirs(dst, exist_ok=True)
        try:
            hf_hub_download(repo_id=args.repo, filename=f"{e}/model_latest.pt",
                            repo_type="model", local_dir="logs/_hf_ckpt")
            src = f"logs/_hf_ckpt/{e}/model_latest.pt"
            os.replace(src, f"{dst}/model_latest.pt")
            print(f"  ✓ {e}/model_latest.pt")
        except Exception as ex:
            print(f"  [skip] {e}: {ex}")
    # cleanup staging
    import shutil
    shutil.rmtree("logs/_hf_ckpt", ignore_errors=True)
    print("Done. Checkpoints in logs/<exp>/models/model_latest.pt")


if __name__ == "__main__":
    main()

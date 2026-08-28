"""Entry point for HT-in-Diffusion-Policy runs.

Only reason this file exists: in the legacy DP environment (py3.9,
torch 1.12.1+cu116, mujoco-py 2.0.2.13) `import torch` segfaults during
pybind11/ONNX binding init unless mujoco-py has been loaded first.
Importing mujoco_py here fixes the load order, then DP's own train.py is
executed unmodified.
"""
import runpy
import sys

import mujoco_py  # noqa: F401  MUST precede torch

DP_ROOT = "/mnt/pfs/yuchen/dp"

if __name__ == "__main__":
    sys.argv[0] = DP_ROOT + "/train.py"
    runpy.run_path(DP_ROOT + "/train.py", run_name="__main__")

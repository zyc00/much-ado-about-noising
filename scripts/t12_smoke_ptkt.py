"""Zero-GPU smoke: can pusht/kitchen datasets + envs build?"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, ".")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

for task, dp in [("pusht_keypoint",
                  "/mnt/pfs/yuchen/data/mip/pusht/pusht_cchi_v7_replay.zarr.zip"),
                 ("kitchen_state",
                  "/mnt/pfs/yuchen/data/mip/kitchen/kitchen_demos_multitask.zip")]:
    try:
        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.abspath("examples/configs")):
            cfg = compose(config_name="main", overrides=[
                f"task={task}", "network=chiunet",
                f"+task.dataset_path={dp}", "log.wandb_mode=disabled"])
        OmegaConf.set_struct(cfg, False)
        from mip.datasets.robomimic_dataset import make_dataset
        ds = make_dataset(cfg.task)
        print(f"SMOKE {task} dataset OK n={len(ds)}")
        from examples.train_robomimic import make_vec_env
        env = make_vec_env(cfg.task, seed=1)
        print(f"SMOKE {task} env OK")
    except Exception as e:
        print(f"SMOKE {task} FAIL {type(e).__name__}: {str(e)[:150]}")
print("SMOKE_DONE")

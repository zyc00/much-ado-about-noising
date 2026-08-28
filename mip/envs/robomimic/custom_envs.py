"""Custom robosuite environments used by scripted NFL witness datasets.

Importing this module registers the classes with robosuite (registration
happens via the environment metaclass at class-definition time), which is all
`EnvUtils.create_env_from_metadata` needs to build them from an hdf5's
env_args. Keep this import cheap and side-effect-free beyond registration.
"""
import os

from robosuite.environments.manipulation.lift import Lift

CUBE_HALF = float(os.environ.get("CUBE_HALF", "0.033"))


class BigCubeLift(Lift):
    """Lift with a large, gripper-width cube (uniform size, no randomization).

    Cube half-extent (default 0.033 m -> 66 mm cube vs the Panda gripper's
    ~80 mm opening): task difficulty is lateral grasp precision, with no wrist
    rotation required anywhere in the task. The size comes from the dataset's
    env_args (`cube_half` kwarg) so collected data is self-describing; the
    CUBE_HALF env var is the collection-time default.
    """

    def __init__(self, *args, cube_half=None, **kwargs):
        self._cube_half = float(cube_half) if cube_half is not None else CUBE_HALF
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        h = self._cube_half
        for g in self.cube.get_obj().findall(".//geom"):
            g.set("size", f"{h} {h} {h}")

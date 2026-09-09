"""Separate NEW_EMBODIMENT registration per task/process; never modify built-ins."""
from common import TASKS, LANGUAGE_KEY, action_keys, state_keys


def register(task):
    from gr00t.configs.data.embodiment_configs import register_modality_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import (ModalityConfig, ActionConfig, ActionRepresentation,
                                 ActionType, ActionFormat)
    keys = action_keys(task)
    config = {
        'video': ModalityConfig(delta_indices=[0], modality_keys=TASKS[task]['cameras']),
        'state': ModalityConfig(delta_indices=[0], modality_keys=state_keys(task)),
        'action': ModalityConfig(
            delta_indices=list(range(8)), modality_keys=keys,
            # Rot6D is already encoded. ABSOLUTE/DEFAULT prevents a second
            # state subtraction, scaling of controller deltas, or rotation conversion.
            action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
                            for _ in keys]),
        'language': ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }
    register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
    return config

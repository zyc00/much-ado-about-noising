"""Serve a fine-tuned RoboMimic checkpoint using native GR00T policy inference."""
import argparse
from common import TASKS, action_keys, state_keys


def wrap_policy(policy):
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    class AutocastSimPolicy(Gr00tSimPolicyWrapper):
        def _get_action(self, observation, options=None):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return super()._get_action(observation, options)

    return AutocastSimPolicy(policy)


if __name__ == '__main__':
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True, help='Fine-tuned checkpoint, not the bare base model')
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5555)
    a = p.parse_args()
    policy = Gr00tPolicy(embodiment_tag='NEW_EMBODIMENT', model_path=a.model_path, device='cuda')
    mods = policy.modality_configs
    assert mods['action'].modality_keys == action_keys(a.task), 'Checkpoint/task mismatch'
    assert mods['state'].modality_keys == state_keys(a.task), 'Checkpoint/task mismatch'
    assert mods['video'].modality_keys == TASKS[a.task]['cameras'], 'Checkpoint/task mismatch'
    with PolicyServer(wrap_policy(policy), host=a.host, port=a.port) as server:
        server.run()

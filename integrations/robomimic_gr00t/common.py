"""Shared, reversible RoboMimic <-> GR00T mapping (no simulator/GR00T imports)."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

TASKS = {
    'tool_hang': dict(env_name='ToolHang', arms=1,
                     cameras=['sideview', 'robot0_eye_in_hand'], horizon=700,
                     instruction='Assemble the frame and hang the tool on it.'),
    'transport_ph': dict(env_name='TwoArmTransport', arms=2,
                        cameras=['shouldercamera0', 'shouldercamera1',
                                 'robot0_eye_in_hand', 'robot1_eye_in_hand'], horizon=700,
                        instruction='Use both arms to transfer the payload into the target bin.'),
}
LANGUAGE_KEY = 'annotation.human.task_description'


def action_keys(task):
    return [key for arm in range(TASKS[task]['arms'])
            for key in (f'robot{arm}_pose', f'robot{arm}_gripper')]


def state_keys(task):
    return [f'robot{arm}_{field}' for arm in range(TASKS[task]['arms'])
            for field in ('eef_pos', 'eef_quat', 'gripper_qpos')]


def encode_actions(raw, task):
    """Absolute controller [xyz, rotvec, grip] -> [xyz, first two R rows, grip]."""
    raw = np.asarray(raw)
    arms = TASKS[task]['arms']
    if raw.shape[-1] != 7 * arms or not np.isfinite(raw).all():
        raise ValueError(f'Expected finite {7 * arms}-D controller actions, got {raw.shape}')
    blocks = raw.reshape(-1, arms, 7)
    matrix = Rotation.from_rotvec(blocks[..., 3:6].reshape(-1, 3)).as_matrix()
    rot6 = matrix[:, :2, :].reshape(-1, arms, 6)
    result = np.concatenate((blocks[..., :3], rot6, blocks[..., 6:7]), axis=-1)
    return result.reshape(*raw.shape[:-1], 10 * arms).astype(np.float32)


def decode_actions(encoded, task):
    """Inverse mapping; reject degenerate rotations instead of silently issuing NaNs."""
    encoded = np.asarray(encoded, dtype=np.float64)
    arms = TASKS[task]['arms']
    if encoded.shape[-1] != 10 * arms or not np.isfinite(encoded).all():
        raise ValueError(f'Expected finite {10 * arms}-D model actions, got {encoded.shape}')
    blocks = encoded.reshape(-1, arms, 10)
    a, b = blocks[..., 3:6], blocks[..., 6:9]
    norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
    if (norm_a < 1e-8).any():
        raise ValueError('Degenerate first rotation-6D basis vector')
    a = a / norm_a
    b = b - (a * b).sum(axis=-1, keepdims=True) * a
    norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
    if (norm_b < 1e-8).any():
        raise ValueError('Degenerate second rotation-6D basis vector')
    b /= norm_b
    matrix = np.stack((a, b, np.cross(a, b)), axis=-2)
    rotvec = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec()
    rotvec = rotvec.reshape(-1, arms, 3)
    raw = np.concatenate((blocks[..., :3], rotvec, blocks[..., 9:10]), axis=-1)
    return raw.reshape(*encoded.shape[:-1], 7 * arms)


def pack_state(obs, task):
    values = [np.asarray(obs[k], dtype=np.float32) for k in state_keys(task)]
    result = np.concatenate(values, axis=-1)
    if result.shape[-1] != 9 * TASKS[task]['arms'] or not np.isfinite(result).all():
        raise ValueError('Invalid proprioceptive observation')
    return result


def policy_observation(obs, task):
    """Native GR00T sim wrapper wire format: batch=1, observation history=1."""
    output = {f'state.{k}': np.asarray(obs[k], np.float32)[None, None]
              for k in state_keys(task)}
    for camera in TASKS[task]['cameras']:
        pixels = np.asarray(obs[f'{camera}_image'])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError(f'{camera}: expected unprocessed HWC uint8 RGB')
        # EnvRobosuite already flips MuJoCo's vertical direction to dataset convention.
        output[f'video.{camera}'] = pixels[None, None]
    output[LANGUAGE_KEY] = [TASKS[task]['instruction']]
    return output


def unpack_policy_actions(actions, task):
    chunks = [np.asarray(actions[f'action.{k}']) for k in action_keys(task)]
    if any(x.ndim != 3 or x.shape[0] != 1 for x in chunks):
        raise ValueError('Expected action groups shaped (1, horizon, dimensions)')
    return decode_actions(np.concatenate(chunks, axis=-1)[0], task)


def absolute_env_metadata(metadata, task):
    """image_abs files can retain DELTA metadata: explicitly fix only controller mode."""
    metadata = deepcopy(metadata)
    if metadata['env_name'] != TASKS[task]['env_name']:
        raise ValueError(f'Task/env mismatch: {task}, {metadata["env_name"]}')
    kwargs = metadata['env_kwargs']
    count = 0

    def visit(item):
        nonlocal count
        if isinstance(item, dict):
            if item.get('type') == 'OSC_POSE':
                item['control_delta'] = False
                item['input_type'] = 'absolute'
                item['input_ref_frame'] = 'world'
                count += 1
            else:
                for value in item.values():
                    visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(kwargs['controller_configs'])
    if count == 0:
        raise ValueError('No OSC_POSE controller found')
    kwargs['camera_names'] = TASKS[task]['cameras']
    kwargs['ignore_done'] = True
    return metadata


def modality_layout(task):
    state, action = {}, {}
    i = 0
    for key in state_keys(task):
        n = 4 if key.endswith('quat') else 2 if key.endswith('qpos') else 3
        state[key] = dict(start=i, end=i+n)
        i += n
    i = 0
    for key in action_keys(task):
        n = 1 if key.endswith('gripper') else 9
        action[key] = dict(start=i, end=i+n)
        i += n
    return dict(state=state, action=action,
                video={k: dict(original_key=f'observation.images.{k}')
                       for k in TASKS[task]['cameras']},
                annotation={'human.task_description': {}})

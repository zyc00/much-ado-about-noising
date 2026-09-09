"""Convert selected complete PH demos to standalone GR00T LeRobot v2 datasets.

Requires an explicitly supplied absolute-action HDF5 file. No action semantics
are inferred from its often stale env_args. Output creation refuses overwrites.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import TASKS, LANGUAGE_KEY, encode_actions, pack_state, modality_layout


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_video(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'threads': '2'}
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def convert(source, output, task, num_demos=None, seed=42, filter_key='train', demo_ids=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(source, 'r') as f:
        meta = json.loads(f['data'].attrs['env_args'])
        if meta['env_name'] != TASKS[task]['env_name']:
            raise ValueError('Source task does not match requested task')
        available = sorted(f['data'], key=lambda x: int(x.split('_')[-1]))
        if filter_key != 'all':
            path = f'mask/{filter_key}'
            if path not in f:
                raise ValueError(f'Missing {path}; explicitly use --filter-key all if intended')
            allowed = {x.decode() for x in f[path][:]}
            available = [x for x in available if x in allowed]
        if demo_ids is not None:
            selected = list(demo_ids)
            if len(set(selected)) != len(selected) or not set(selected) <= set(available):
                raise ValueError('Demo IDs are duplicated or outside requested source split')
        else:
            count = len(available) if num_demos is None else num_demos
            if count < 1 or count > len(available):
                raise ValueError(f'Cannot select {count} from {len(available)} demos')
            # Prefixes of this permutation give nested 10/20/50-demo subsets.
            selected = np.random.default_rng(seed).permutation(available)[:count].tolist()
        fps = int(meta['env_kwargs']['control_freq'])
        cameras = TASKS[task]['cameras']
        for name in selected:
            d = f['data'][name]
            if d['actions'].shape[1] != 7 * TASKS[task]['arms']:
                raise ValueError('Expected raw controller actions, not pre-encoded Rot6D')
            for camera in cameras:
                key = f'{camera}_image'
                if key not in d['obs']:
                    raise ValueError(f'{name}: missing {key}; regenerate cameras, never zero-fill')
                im = d['obs'][key]
                if im.dtype != np.uint8 or len(im.shape) != 4 or im.shape[-1] != 3:
                    raise ValueError(f'{name}/{key}: expected THWC uint8')
                if im.shape[0] != len(d['actions']):
                    raise ValueError('Image/action length mismatch')
        output.mkdir(parents=True)
        (output / 'meta').mkdir()
        (output / 'data/chunk-000').mkdir(parents=True)
        dump_json(output / 'meta/modality.json', modality_layout(task))
        dump_json(output / 'meta/robomimic_env.json', meta)
        instruction = TASKS[task]['instruction']
        (output / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': instruction})+'\n')
        episodes, provenance = [], []
        total = 0
        for index, name in enumerate(selected):
            d = f['data'][name]
            raw = d['actions'][:]
            action = encode_actions(raw, task)
            state = pack_state(d['obs'], task)
            n = len(raw)
            table = pa.table({
                'observation.state': pa.array(state.tolist(), type=pa.list_(pa.float32())),
                'action': pa.array(action.tolist(), type=pa.list_(pa.float32())),
                'timestamp': np.arange(n, dtype=np.float64)/fps,
                'frame_index': np.arange(n, dtype=np.int64),
                'episode_index': np.full(n, index, np.int64),
                'index': np.arange(total, total+n, dtype=np.int64),
                'task_index': np.zeros(n, np.int64), LANGUAGE_KEY: np.zeros(n, np.int64),
                'next.done': np.arange(n) == n-1,
                'next.reward': d['rewards'][:] if 'rewards' in d else np.zeros(n),
            })
            pq.write_table(table, output / f'data/chunk-000/episode_{index:06d}.parquet')
            for camera in cameras:
                write_video(output / f'videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4',
                            d['obs'][f'{camera}_image'][:], fps)
            episodes.append({'episode_index': index, 'tasks': [instruction], 'length': n})
            provenance.append({'episode_index': index, 'source_demo': name, 'length': n,
                               'raw_action_sha256': hashlib.sha256(raw.tobytes()).hexdigest()})
            total += n
            print(f'CONVERTED {task} {index+1}/{len(selected)} {name} {n}', flush=True)
        features = {}
        for camera in cameras:
            shape = list(f['data'][selected[0]]['obs'][f'{camera}_image'].shape[1:])
            features[f'observation.images.{camera}'] = {
                'dtype': 'video', 'shape': shape, 'names': ['height', 'width', 'channel'],
                'info': {'video.fps': fps, 'video.height': shape[0], 'video.width': shape[1],
                         'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                         'video.channels': 3, 'video.is_depth_map': False, 'has_audio': False}}
        for key, width in [('action', action.shape[1]), ('observation.state', state.shape[1])]:
            features[key] = {'dtype': 'float32', 'shape': [width], 'names': None}
        for key in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index', LANGUAGE_KEY]:
            features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
        dump_json(output / 'meta/info.json', {
            'codebase_version': 'v2.0', 'robot_type': 'robomimic_panda',
            'total_episodes': len(selected), 'total_frames': total, 'total_tasks': 1,
            'total_videos': len(selected)*len(cameras), 'total_chunks': 1, 'chunks_size': 1000,
            'fps': fps, 'splits': {'train': f'0:{len(selected)}'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': features})
        (output / 'meta/episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in episodes))
        dump_json(output / 'meta/provenance.json', {
            'source_hdf5': str(source), 'source_size_bytes': source.stat().st_size,
            'task': task, 'source_filter': filter_key, 'selection_seed': seed,
            'source_action_mode': 'absolute', 'encoding': 'xyz + rotation matrix first two rows + gripper',
            'image_transform': 'none; stored RoboMimic HWC RGB convention',
            'normalization': 'GR00T computes stats from this selected dataset only',
            'episodes': provenance})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Absolute-action RoboMimic image HDF5')
    p.add_argument('--output', required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--action-mode', choices=['absolute'], required=True,
                   help='Explicit assertion: never inferred from filename or stale metadata')
    p.add_argument('--num-demos', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--filter-key', default='train')
    p.add_argument('--demo-ids', nargs='+')
    a = p.parse_args()
    convert(a.source, a.output, a.task, a.num_demos, a.seed, a.filter_key, a.demo_ids)

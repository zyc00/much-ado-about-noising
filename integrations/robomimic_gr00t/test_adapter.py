import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common import (TASKS, encode_actions, decode_actions, modality_layout,
                    absolute_env_metadata, pack_state, policy_observation,
                    unpack_policy_actions)
from convert import convert


@pytest.mark.parametrize('task', TASKS)
def test_roundtrip(task):
    arms = TASKS[task]['arms']
    raw = np.random.default_rng(8).normal(size=(2, 8, 7*arms))
    out = decode_actions(encode_actions(raw, task), task)
    for arm in range(arms):
        i = arm*7
        np.testing.assert_allclose(out[..., i:i+3], raw[..., i:i+3], atol=1e-6)
        np.testing.assert_allclose(out[..., i+6], raw[..., i+6], atol=1e-6)
        a=Rotation.from_rotvec(raw[..., i+3:i+6].reshape(-1,3)).as_matrix()
        b=Rotation.from_rotvec(out[..., i+3:i+6].reshape(-1,3)).as_matrix()
        np.testing.assert_allclose(a,b,atol=1e-6)
    encoded = encode_actions(raw[:1], task)
    layout=modality_layout(task)
    wire={f'action.{k}': encoded[...,v['start']:v['end']] for k,v in layout['action'].items()}
    np.testing.assert_allclose(unpack_policy_actions(wire,task),out[0],atol=1e-6)


def test_degenerate_rotation():
    with pytest.raises(ValueError,match='Degenerate'):
        decode_actions(np.zeros(10),'tool_hang')


def test_controller_is_explicit_and_nonmutating():
    original={'env_name':'TwoArmTransport','env_kwargs':{'controller_configs':{
        'type':'BASIC','body_parts':{'right':{'type':'OSC_POSE','control_delta':True}}}}}
    converted=absolute_env_metadata(original,'transport_ph')
    assert original['env_kwargs']['controller_configs']['body_parts']['right']['control_delta']
    assert converted['env_kwargs']['controller_configs']['body_parts']['right']['input_type']=='absolute'
    assert len(converted['env_kwargs']['camera_names'])==4


def synthetic(path, task, missing_camera=False):
    spec=TASKS[task]
    with h5py.File(path,'w') as f:
        root=f.create_group('data')
        root.attrs['env_args']=json.dumps({'env_name':spec['env_name'],'env_kwargs':{'control_freq':20}})
        f.create_dataset('mask/train',data=np.array([b'demo_0',b'demo_2']))
        for demo in range(3):
            g=root.create_group(f'demo_{demo}')
            g.create_dataset('actions',data=np.zeros((10,7*spec['arms'])))
            obs=g.create_group('obs')
            for arm in range(spec['arms']):
                obs.create_dataset(f'robot{arm}_eef_pos',data=np.zeros((10,3)))
                obs.create_dataset(f'robot{arm}_eef_quat',data=np.tile([0,0,0,1],(10,1)))
                obs.create_dataset(f'robot{arm}_gripper_qpos',data=np.zeros((10,2)))
            for cam in spec['cameras'][1:] if missing_camera else spec['cameras']:
                obs.create_dataset(cam+'_image',data=np.zeros((10,16,16,3),np.uint8))


@pytest.mark.parametrize('task',TASKS)
def test_conversion_split_metadata_and_video(tmp_path,task):
    import av
    import pyarrow.parquet as pq
    source=tmp_path/'source.hdf5'; synthetic(source,task)
    out=convert(source,tmp_path/'dataset',task,num_demos=1)
    provenance=json.loads((out/'meta/provenance.json').read_text())
    assert provenance['episodes'][0]['source_demo'] in ['demo_0','demo_2']
    table=pq.read_table(out/'data/chunk-000/episode_000000.parquet')
    assert table.num_rows==10
    for video in (out/'videos').rglob('*.mp4'):
        with av.open(str(video)) as c: assert len(list(c.decode(video=0)))==10
    with pytest.raises(FileExistsError): convert(source,out,task)


def test_missing_camera_fails_before_output(tmp_path):
    source=tmp_path/'source.hdf5'; synthetic(source,'transport_ph',missing_camera=True)
    with pytest.raises(ValueError,match='missing'):
        convert(source,tmp_path/'dataset','transport_ph')
    assert not (tmp_path/'dataset').exists()


def test_images_no_double_flip():
    obs={'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0,0,0,1]),
         'robot0_gripper_qpos':np.zeros(2)}
    for c in TASKS['tool_hang']['cameras']:
        obs[c+'_image']=np.arange(48,dtype=np.uint8).reshape(4,4,3)
    wire=policy_observation(obs,'tool_hang')
    np.testing.assert_array_equal(wire['video.sideview'][0,0],obs['sideview_image'])
    assert pack_state(obs,'tool_hang').shape==(9,)


def test_wire_serialization_and_pickle_rejection():
    import msgpack
    from client import encode, decode
    x={'video.camera':np.arange(48,dtype=np.uint8).reshape(1,1,4,4,3),
       'state.pose':np.arange(9,dtype=np.float32).reshape(1,1,9),
       'instruction':['Hang the tool.']}
    y=msgpack.unpackb(msgpack.packb(x,default=encode),object_hook=decode,raw=False)
    np.testing.assert_array_equal(x['video.camera'],y['video.camera'])
    np.testing.assert_array_equal(x['state.pose'],y['state.pose'])
    with pytest.raises(TypeError): encode(np.array([object()],dtype=object))
    with pytest.raises(ValueError): decode({b'nd':True,b'kind':b'O',b'data':b'untrusted'})


def test_normalization_override_survives_pretrained_loader(monkeypatch,tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace as S
    from launch import install_pipeline_overrides

    class Processor:
        use_percentiles=True
        state_action_processor=S(use_percentiles=True)
        statistics={'new_embodiment':{}}
        def set_statistics(self, stats, override=False):
            assert override and not self.use_percentiles
            assert not self.state_action_processor.use_percentiles
            self.recomputed=True

    class Pipeline:
        def __init__(self):
            self.config=S(model=S(use_percentiles=False))
            self.save_cfg_dir=tmp_path
        def _create_model(self):
            return S(config=S(use_percentiles=True,to_filtered_json=lambda:'{}'))
        def _create_dataset(self):
            self.processor=Processor()
            return 'dataset',None

    @contextmanager
    def rank0(**kw):
        yield True

    monkeypatch.setitem(sys.modules,'gr00t.model.gr00t_n1d7.setup',S(Gr00tN1d7Pipeline=Pipeline))
    monkeypatch.setitem(sys.modules,'gr00t.utils.dist_utils',S(run_or_wait_on_rank0=rank0))
    install_pipeline_overrides()
    pipe=Pipeline()
    assert not pipe._create_model().config.use_percentiles
    assert pipe._create_dataset()==('dataset',None)
    assert pipe.processor.recomputed
    assert json.loads((tmp_path/'robomimic_processor_overrides.json').read_text())['use_percentiles'] is False

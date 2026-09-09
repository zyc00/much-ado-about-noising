"""Real pretrained-model forward/backward and native policy inference, no optimizer steps."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from configuration import register
from common import TASKS, LANGUAGE_KEY, action_keys, policy_observation, unpack_policy_actions


def main():
    from launch import install_pipeline_overrides
    install_pipeline_overrides()
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from gr00t.policy.policy import BasePolicy
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from serve import wrap_policy

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base-model', default='/mnt/pfs/yuchen/groot/model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--serve-port', type=int, help='Keep last task (Transport) real policy running for a wire smoke test')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    model = None
    results = []
    tag = EmbodimentTag.NEW_EMBODIMENT
    for task in TASKS:
        # Only this test reuses a process for two separate custom embodiments.
        MODALITY_CONFIGS.pop(tag.value, None)
        modalities = register(task)
        cfg = get_default_config()
        cfg.data.datasets = [SingleDatasetConfig(
            dataset_paths=[str(args.root / f'{task}_smoke')], embodiment_tag=tag.value)]
        cfg.data.modality_configs = {tag.value: modalities}
        cfg.data.shard_size = 64
        cfg.data.episode_sampling_rate = 1.0
        cfg.data.num_shards_per_epoch = 1
        cfg.training.start_from_checkpoint = args.base_model
        cfg.training.num_gpus = 1
        cfg.model.loss_type = 'hetero_t'
        cfg.model.ht_mvt = True
        cfg.model.ht_df = 4 * 8 * 10 * TASKS[task]['arms']
        cfg.model.state_dropout_prob = 0.0
        cfg.model.use_percentiles = False
        directory = args.output / task
        directory.mkdir()
        pipeline = Gr00tN1d7Pipeline(cfg, directory)
        if model is None:
            # Use the same loading/initialization path as launch_finetune.
            model = pipeline._create_model().cuda()
        pipeline.model = model
        pipeline._create_dataset(directory)
        processor = pipeline.processor
        processor.eval()
        processor.save_pretrained(directory / 'processor')
        loader = LeRobotEpisodeLoader(args.root / f'{task}_smoke', modalities)
        data = extract_step_data(loader[0], 0, modalities, tag, allow_padding=False)
        feature = processor([{'type': MessageType.EPISODE_STEP.value, 'content': data}])
        batch = processor.collator([feature])
        mask = batch['inputs']['action_mask'].bool()
        expected = 8 * 10 * TASKS[task]['arms']
        assert int(mask.sum()) == expected, (mask.shape, mask.sum(), expected)
        normalized = batch['inputs']['action'].float().numpy()
        decoded = processor.decode_action(normalized, tag, {k:v[None] for k,v in data.states.items()})
        true_actions = np.concatenate([data.actions[k] for k in action_keys(task)], axis=-1)
        decoded_actions = np.concatenate([decoded[k][0] for k in action_keys(task)], axis=-1)
        norm_error = float(np.abs(true_actions-decoded_actions).max())
        assert norm_error < 1e-5, norm_error
        record = {'task': task, 'valid_action_elements': expected,
                  'normalization_roundtrip_error': norm_error, 'objectives': {}}
        for objective in ('flow', 'mse', 'hetero_t'):
            model.config.loss_type = objective
            model.action_head.config.loss_type = objective
            model.config.ht_df = cfg.model.ht_df
            model.train()
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**batch)['loss']
            assert torch.isfinite(loss).all(), objective
            loss.backward()
            gradients = [p.grad for p in model.action_head.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients), objective
            record['objectives'][objective] = {'loss': float(loss.detach()),
                                               'finite_gradient_tensors': len(gradients)}
            model.zero_grad(set_to_none=True)
            print('FORWARD_BACKWARD ' + json.dumps(record['objectives'][objective]), flush=True)
        model.eval()
        # Reuse the already loaded real model/processor; the native inference
        # implementation and strict checks are unchanged. No synthetic outputs.
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        BasePolicy.__init__(policy, strict=True)
        policy.model = model
        policy.processor = processor
        policy.embodiment_tag = tag
        policy.modality_configs = modalities
        policy.collate_fn = processor.collator
        policy.language_key = LANGUAGE_KEY
        wrapper = wrap_policy(policy)
        obs = {k: v[0] for k,v in data.states.items()}
        obs.update({f'{k}_image': v[0] for k,v in data.images.items()})
        record['inference'] = {}
        for objective in ('flow', 'mse', 'hetero_t'):
            model.config.loss_type = objective
            model.action_head.config.loss_type = objective
            actions, _ = wrapper.get_action(policy_observation(obs, task))
            raw = unpack_policy_actions(actions, task)
            assert raw.shape == (8, 7*TASKS[task]['arms']) and np.isfinite(raw).all()
            record['inference'][objective] = list(raw.shape)
        results.append(record)
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print('TASK_SMOKE_PASS ' + json.dumps(record), flush=True)
    if args.serve_port:
        from gr00t.policy.server_client import PolicyServer
        with PolicyServer(wrapper, host='0.0.0.0', port=args.serve_port) as server:
            server.run()


if __name__ == '__main__':
    main()

"""Standard policy server, with explicit per-continuation RNG seeding."""
import argparse
import random
import numpy as np
import torch
from huggingface_hub import snapshot_download
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer


class SeededPolicy(Gr00tSimPolicyWrapper):
    def reset(self, options=None):
        if options and 'seed' in options:
            seed = int(options['seed'])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return self.policy.reset(options)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--embodiment-tag', required=True)
    p.add_argument('--port', type=int, required=True)
    args = p.parse_args()
    # Resolve the identical cached processor locally; avoids a tokenizer
    # metadata request that ignores HF_HUB_OFFLINE in this Transformers build.
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    original_builder = processing.build_processor
    def cached_processor(name, kwargs):
        local = snapshot_download(name, local_files_only=True) if not name.startswith('/') else name
        return original_builder(local, kwargs)
    processing.build_processor = cached_processor
    policy = SeededPolicy(Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
                                    model_path=args.checkpoint, device='cuda', strict=True))
    with PolicyServer(policy, host='127.0.0.1', port=args.port) as server:
        server.run()

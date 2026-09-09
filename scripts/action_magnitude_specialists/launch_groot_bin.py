#!/usr/bin/env python3
"""Environment-guarded GR00T launcher that filters precomputed timestep bins."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np

# The established GR00T processor path performs a Hugging Face model-metadata
# lookup while patching the tokenizer, even though all weights are local.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
import gr00t.model.gr00t_n1d7.setup as gr00t_setup


_BACKBONE_CACHE = Path(
    "/mnt/pfs/yuchen/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/"
    "snapshots/9ce19a195e423419c349abfc86fd07178b230561"
)
_ORIGINAL_BUILD_PROCESSOR = processing.build_processor
processing.build_processor = lambda name, kwargs: _ORIGINAL_BUILD_PROCESSOR(
    str(_BACKBONE_CACHE), {**kwargs, "local_files_only": True}
)

# The retained WidowX warm start was saved after the heteroscedastic head had
# been added, whereas these specialists instantiate the established MSE head.
# Hugging Face correctly reports the four scale-head tensors as unused.  The
# GR00T pipeline treats every unused tensor as fatal, so narrowly allow only
# those tensors; all shared policy weights are still loaded strictly, and any
# other missing, unexpected, or mismatched key remains an error.
_ORIGINAL_FROM_PRETRAINED = gr00t_setup.AutoModel.from_pretrained


class _AutoModelWithMSEWarmStart:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        result = _ORIGINAL_FROM_PRETRAINED(*args, **kwargs)
        if not kwargs.get("output_loading_info", False):
            return result
        model, loading_info = result
        unexpected = loading_info.get("unexpected_keys", [])
        ignored = [key for key in unexpected if ".sigma_decoder." in key]
        if ignored:
            loading_info["unexpected_keys"] = [
                key for key in unexpected if key not in ignored
            ]
            print(
                f"[action-mag] ignored {len(ignored)} unused sigma-decoder "
                "tensors when loading an MSE model",
                flush=True,
            )
        return model, loading_info


gr00t_setup.AutoModel = _AutoModelWithMSEWarmStart

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset


_ORIGINAL_SHARD_DATASET = ShardedSingleStepDataset.shard_dataset


def _filtered_shard_dataset(self: ShardedSingleStepDataset) -> None:
    _ORIGINAL_SHARD_DATASET(self)
    bin_index = int(os.environ["ACTION_MAG_BIN"])
    split = os.environ.get("ACTION_MAG_SPLIT", "train")
    manifest_dir = Path(os.environ["ACTION_MAG_MANIFEST_DIR"])
    manifest_path = manifest_dir / f"{Path(self.dataset_path).name}.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing action-magnitude manifest: {manifest_path}")
    with np.load(manifest_path) as manifest:
        episode_ids = manifest["episode_ids"]
        offsets = manifest["offsets"]
        bins = manifest["bins"]
        is_train = manifest["is_train"]
    loader_ids = np.asarray(
        [row["episode_index"] for row in self.episode_loader.episodes_metadata], dtype=np.int64
    )
    if not np.array_equal(loader_ids, episode_ids):
        raise ValueError(f"Episode order mismatch for {manifest_path}")
    want_train = split == "train"
    filtered_shards = []
    filtered_lengths = []
    for shard in self.sharded_episodes:
        filtered = []
        for episode_ordinal, step_indices in shard:
            start, end = int(offsets[episode_ordinal]), int(offsets[episode_ordinal + 1])
            episode_bins = bins[start:end]
            episode_split = is_train[start:end]
            keep = (episode_bins[step_indices] == bin_index) & (
                episode_split[step_indices] == want_train
            )
            selected = step_indices[keep]
            if len(selected):
                filtered.append((episode_ordinal, selected))
        if filtered:
            filtered_shards.append(filtered)
            filtered_lengths.append(sum(len(indices) for _, indices in filtered))
    if not filtered_shards:
        raise ValueError(f"No samples for bin={bin_index}, split={split}, dataset={self.dataset_path}")
    self.sharded_episodes = filtered_shards
    self.shard_lengths = np.asarray(filtered_lengths, dtype=np.int64)
    print(
        f"[action-mag] dataset={Path(self.dataset_path).name} bin={bin_index} split={split} "
        f"samples={int(self.shard_lengths.sum())} shards={len(self.shard_lengths)}",
        flush=True,
    )


def main() -> None:
    ShardedSingleStepDataset.shard_dataset = _filtered_shard_dataset
    runpy.run_path(
        "/mnt/pfs/yuchen/groot/Isaac-GR00T/gr00t/experiment/launch_finetune.py",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()

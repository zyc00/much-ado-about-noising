#!/usr/bin/env python3
"""Environment-guarded LeRobot launcher that maps samples to one action bin."""

from __future__ import annotations

import os
import runpy

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


_ORIGINAL_INIT = LeRobotDataset.__init__
_ORIGINAL_LEN = LeRobotDataset.__len__
_ORIGINAL_GETITEM = LeRobotDataset.__getitem__
_ORIGINAL_GET_RAW_ITEM = LeRobotDataset.get_raw_item


def _init(self, *args, **kwargs):
    _ORIGINAL_INIT(self, *args, **kwargs)
    manifest_path = os.environ["ACTION_MAG_MANIFEST"]
    bin_index = int(os.environ["ACTION_MAG_BIN"])
    split = os.environ.get("ACTION_MAG_SPLIT", "train")
    with np.load(manifest_path) as manifest:
        bins = manifest["bins"]
        is_train = manifest["is_train"]
        indices = manifest["indices"]
    if len(bins) != self.num_frames or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"Manifest/data mismatch: {manifest_path}")
    want_train = split == "train"
    self._action_mag_index_map = np.flatnonzero(
        (bins == bin_index) & (is_train == want_train)
    ).astype(np.int64)
    if not len(self._action_mag_index_map):
        raise ValueError(f"No samples for bin={bin_index}, split={split}")
    print(
        f"[action-mag] bin={bin_index} split={split} samples={len(self._action_mag_index_map)}",
        flush=True,
    )


def _len(self):
    mapping = getattr(self, "_action_mag_index_map", None)
    return len(mapping) if mapping is not None else _ORIGINAL_LEN(self)


def _map_index(self, idx: int) -> int:
    mapping = getattr(self, "_action_mag_index_map", None)
    if mapping is None:
        return idx
    if idx < 0:
        idx += len(mapping)
    return int(mapping[idx])


def _getitem(self, idx):
    if isinstance(idx, slice):
        return [self[item_idx] for item_idx in range(*idx.indices(len(self)))]
    return _ORIGINAL_GETITEM(self, _map_index(self, idx))


def _get_raw_item(self, idx):
    return _ORIGINAL_GET_RAW_ITEM(self, _map_index(self, idx))


def main() -> None:
    LeRobotDataset.__init__ = _init
    LeRobotDataset.__len__ = _len
    LeRobotDataset.__getitem__ = _getitem
    LeRobotDataset.get_raw_item = _get_raw_item
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")


if __name__ == "__main__":
    main()

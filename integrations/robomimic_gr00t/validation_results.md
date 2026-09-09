# Integration validation — 2026-09-07

No few-shot training or optimizer updates were performed. These are
interface tests, not learned-policy benchmark results.

Local validation: **10 tests passed**, plus Python compilation and shell
syntax checks. Two missing local RPC dependencies (`msgpack`,
`msgpack-numpy`) were added; no existing package was upgraded.

## Real GR00T model checks

Base weights: `/mnt/pfs/yuchen/groot/model` (GR00T N1.7).
Native training-pipeline loader, real two-episode datasets, one input batch.

| Task | Valid chunk elements | Normalization round-trip max error | Flow/MSE/HT backward | Inference shape |
|---|---:|---:|---|---|
| Tool-Hang | 80 (8×10) | 3.06e-8 | All finite | 8×7 controller actions |
| Transport PH | 160 (8×20) | 1.50e-8 | All finite | 8×14 controller actions |

All three objectives produced finite losses and gradients. 537 action-head
gradient tensors were checked for Flow/MSE and 541 for HT. Native strict
policy inference and action decoding passed for all six task/objective pairs.
See [raw results](results/model_smoke.json).

An actual GR00T server was then queried from the separate RoboMimic Python
environment: Transport executed two four-action chunks (8 steps total),
including the second query conditioned on the new simulator observation.
This completed without interface errors. The unadapted model did not
complete the task in this deliberately truncated smoke rollout; that is
**not** a measured benchmark performance result.
See [wire rollout](results/transport_wire_smoke.json).

## Expert replay and controller mapping

Validation episodes were fixed to demo_0/demo_1, not selected by success.
Rotation-6D encode/decode error was at most 5.96e-8. Initial proprioception
matched exactly on both tasks. Paired one-step tests restored XML, simulator
state, reset seed, and controller state before each branch.

| Task | Largest paired one-step state difference | Converted replay successes | Original float64 successes |
|---|---:|---:|---:|
| Tool-Hang | 1.25e-6 | 1/2 | 1/2 |
| Transport PH | 9.23e-7 | 2/2 | 2/2 |

Tool-Hang's successful episode **differs** between the two representations.
Directly casting its original actions to float32 also changes the result:
both float32 raw replays failed, although original float64 demo_1 succeeded.
Thus complete contact-rich trajectories are sensitive to numeric precision;
the mapping passes local transition tests but does not preserve every
full-trajectory success. This is retained as a diagnostic, not hidden or
counted as a trained policy result.
See [Tool-Hang controls](results/tool_hang_replay.json) and
[Transport controls](results/transport_ph_replay.json).

For Tool-Hang, the first rendered frames match original dataset images
pixel-for-pixel. For Transport, mean absolute pixel differences are 1.85–3.30
on a 0–255 scale; flipped alternatives differ by roughly 35–81. All four
cameras have the correct orientation. MP4 encoding is lossy; online raw
images are compared against source HDF5 frames, not compressed videos.

## Prepared data

Cluster root: `/mnt/pfs/yuchen/groot/robomimic_adapter_20260907`.

- `tool_hang_smoke/`, `transport_ph_smoke/`: two fixed replay episodes each;
  diagnostic only, selected with explicit `--filter-key all`.
- `tool_hang_train20_seed42/`, `transport_ph_train20_seed42/`: 20 complete
  demos each, drawn only from the 180-demo official train split. Source IDs
  and action hashes are in each `meta/provenance.json`. The 20 validation
  demos are excluded. These datasets have not been used for training.
  Tool-Hang has 9,526 frames / 40 videos; Transport has 9,450 frames /
  80 videos. GR00T statistics have been generated separately for both subsets.
- `model_smoke_v5/summary.json`: passing model checks. Earlier v1–v4
  directories retain failed diagnostic attempts for audit, not results.
- `tool_hang_replay_final.json`, `transport_ph_replay_v3.json`: final
  controller comparisons, also copied to this directory's `results/`.
- `transport_wire_smoke.json`: real client/server simulator smoke result.

## Issues found and handled

1. The old Transport image file has only one camera. The adapter uses the
   regenerated four-camera absolute-action file and rejects missing views.
2. Absolute-action HDF5 metadata still advertises delta control. Controller
   mode is explicitly set to world-frame absolute in the evaluator.
3. The installed GR00T processor loader silently ignores `use_percentiles`
   overrides. A process-local shim honors min/max normalization and saves
   the corrected processor; shared training sources are untouched.
4. State-only simulator resets do not reset controller internals. Paired
   transition tests now reset both, avoiding false action-mapping failures.
5. The simulator environment lacks Transformers. A small protocol-compatible
   RPC client avoids importing the heavyweight GR00T policy package there.

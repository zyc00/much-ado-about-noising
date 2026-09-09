# RoboMimic → GR00T N1.7

Independent adapter for **Tool-Hang PH** and **Transport PH**. No edits to
shared GR00T model, data, or training sources are required. This connects
data conversion, custom-embodiment training, inference, and simulator rollout;
it does **not** establish few-shot policy performance.

## Mapping and scope

| | Tool-Hang | Transport PH |
|---|---|---|
| Cameras | sideview, robot0 wrist | two shoulders, two wrists |
| Proprioception | 9 | 18 |
| GR00T action dimensions | 10 | 20 |
| Controller action dimensions | 7 | 14 |
| Prediction horizon | 8 | 8 |
| Default execution horizon | 4 | 4 |

Proprioception is EEF position (3), quaternion xyzw (4), and gripper qpos (2)
per arm. No privileged object state enters the model. Input images retain the
source resolution (84×84 in the verified files) and undergo GR00T's normal
image processing. No extra flip is applied: EnvRobosuite already converts
MuJoCo images into the stored dataset convention.

**Only absolute-action source datasets are supported in this first version.**
The converter requires `--action-mode absolute`; it never guesses from ranges
or metadata. Stored raw `[xyz, rotation-vector, gripper]` actions are reversibly
encoded as `[xyz, first two rows of rotation matrix, gripper]`, matching the
standard row-based Rot6D representation. The evaluator reconstructs rotations
and explicitly selects world-frame absolute OSC_POSE control. Positions and
rotations must not be clipped to [-1,1]; only predicted gripper commands are
clipped to the controller's [-1,1] range. Gripper sign is unchanged.

Each task uses `NEW_EMBODIMENT` in **a separate training process/checkpoint**.
Do not mix these two incompatible layouts under that tag in one job. Shared
GR00T weights are loaded from the base checkpoint; the custom embodiment
mapping is not a claim that this exact task/action interface was pretrained.

Normalization uses **selected-training-demo min/max**, not full-dataset or
validation statistics. `launch.py` works around a verified upstream issue:
the installed processor's `from_pretrained` ignores the `use_percentiles`
override. It sets the requested normalization and recomputes parameters in
this process only, then the standard trainer saves the corrected processor.

## Environments and cluster paths

Cluster root: `/mnt/pfs/yuchen/groot/robomimic_adapter_20260907`.
Code copy: `code/`. All generated data/results stay below this root.

- GR00T code: `/mnt/pfs/yuchen/groot/Isaac-GR00T` and its `.venv/bin/python`.
- Conversion/simulator Python:
  `/mnt/pfs/yuchen/code/much-ado-about-noising/.venv/bin/python`.
- Simulator-side RPC additionally needs `msgpack`, `msgpack-numpy`, and
  `pyzmq`. The existing cluster MIP environment has pyzmq; copies of the two
  missing modules from the GR00T Python 3.12 environment are isolated in
  `rpc_deps/`. No shared cluster environment was modified.
- GR00T is not imported in the simulator process. The lightweight client
  uses its existing ZeroMQ/msgpack_numpy protocol and rejects object arrays.

Verified source files:

```
/mnt/pfs/yuchen/data/mip/robomimic/tool_hang/ph/image_abs.hdf5
/mnt/pfs/yuchen/code/much-ado-about-noising/data/transport_ph_image4_abs.hdf5
```

Do not use the older `data/mip/robomimic/transport/ph/image_abs.hdf5`: that
copy contains only one camera. Missing views are errors, never zero-filled.

## Convert a subset

Run with the conversion/simulator Python:

```bash
python convert.py \
  --source /path/to/image_abs.hdf5 \
  --output /new/path/tool_hang_train20_seed42 \
  --task tool_hang --action-mode absolute \
  --num-demos 20 --seed 42 --filter-key train
```

Use `--task transport_ph` and the four-camera absolute dataset for Transport.
The default split is `mask/train` (180 demos); `mask/valid` is not used.
Selection is by complete episode. The same seed yields nested 10/20/50-demo
subsets. Each subset is a standalone dataset with its own future statistics.
Omit `--num-demos` to convert the entire requested split. Existing output
directories are refused. `meta/provenance.json` preserves source demo IDs,
raw-action hashes, split/seed, and action/image conventions.

## Train (not launched during integration)

```bash
NUM_GPUS=1 MAX_STEPS=2000 GLOBAL_BATCH_SIZE=64 \
  bash train.sh tool_hang /path/to/dataset /new/path/run flow
```

Replace `flow` with `mse` or `hetero_t`. The same initialization, modules,
batch, steps, images, and normalization are used. Additional GR00T CLI flags
can follow the objective. The skeleton uses 5% warmup, LR 1e-4, and saves
every 250 steps; these are **untuned starter settings**, not reported results.

HT explicitly enables the joint multivariate loss. Its default `nu=4d` is
320 for Tool-Hang (8×10 active dimensions) and 640 for Transport (8×20),
including the two extra Rot6D rows and gripper channels. Set `HT_NU` or pass
explicit loss arguments to override. No nu staircase is silently applied.
The default sigma initialization is GR00T's default and has not been tuned
for these tasks. Flow/MSE/HT comparison jobs have not been launched.

## Serve and evaluate

In the GR00T Python environment, with the GR00T root on `PYTHONPATH`:

```bash
python serve.py --task transport_ph --model-path /path/to/run/checkpoint-2000 --port 5555
```

In the simulator environment:

```bash
PYTHONPATH=/path/to/adapter/code:/path/to/rpc_deps \
  python evaluate.py --mode policy --dataset /path/to/converted/dataset \
  --host 127.0.0.1 --port 5555 --episodes 50 --seed-start 10000 \
  --output /new/path/rollouts.json
```

Use the same simulator seeds and execution horizon across objectives.
The default 700-step horizon follows the repository's task configs.
`--max-steps 8` is available **only for an interface smoke test** and marks
the output `smoke_only: true`; it is not a benchmark success rate.
The serving wrapper uses native GR00T validation/decoding under bfloat16
autocast and checks that the checkpoint's modalities match the selected task.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q test_adapter.py
```

The tests cover reversible single-/dual-arm mapping, missing-view failures,
episode split integrity, video lengths, normalization layout, image direction,
and RPC serialization/object-array rejection.

`smoke_gr00t.py` loads real pretrained weights via the training pipeline,
computes Flow/MSE/HT forward and backward passes on both tasks, checks finite
gradients and the exact action masks, validates normalization round trips,
and calls native policy inference. It performs **zero optimizer updates**.
`--serve-port 5588` keeps the last task's model alive for an end-to-end wire
test; stop that test server after use.

`evaluate.py --mode replay --source ... --dataset ... --demo-ids demo_0 demo_1`
compares converted actions against original float64 and float32 controls.
One-step comparisons reset controller internals, XML, simulator state, and
the reset seed; resetting only qpos/qvel is not a valid paired controller test.
Mapping tolerances and full-trajectory success agreement are reported
separately. Tool-Hang contact trajectories can change success under float32
rounding: do not present these smoke replay counts as a policy success rate.

See `validation_results.md` for measured integration results and artifact paths.

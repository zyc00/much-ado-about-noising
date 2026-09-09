# Bridge / Fractal coverage expansion (2026-09-07)

Requested change: include dataset-wide Bridge coverage, add Fractal, and use dataset names only. Do not replace general MSE with a zero model or HT predictions.

## Checkpoints and sample frame

- Bridge: `/mnt/pfs/yuchen/groot/ft_wxmse/checkpoint-20000`; dataset `bridge_orig_lerobot` (53,192 episodes).
- Fractal: `/mnt/pfs/yuchen/groot/ft_fr_mse/checkpoint-20000`; dataset `fractal_lerobot` (87,212 episodes).
- Exclude episodes shorter than the eight-step action chunk. No action-magnitude filtering, no policy-held-out claim, no re-training, no recomputed model normalizer.
- Include every first-instruction group after stripping/lowercasing, including an unlabeled group in dataset-wide statistics. No semantic merging.
- Fixed seed 20260907, up to 12 uniformly sampled episodes per instruction group, one random complete-chunk start per available progress decile. Every eligible instruction group is sampled. Short episodes may lack late progress bins; those remain missing.
- Bridge: 19,746 groups including unlabeled, 25,043 sampled episodes, 207,498 states.
- Fractal: 579 groups including unlabeled, 6,353 sampled episodes, 54,093 states.
- Save original target and prediction for all sampled states, along with episode/task/progress and inverse episode-selection weights. These weights recover the eligible episode population for dataset-wide progress summaries, not a uniform frame-weighted census.
- Multi-demonstration task panels include nonempty instruction groups with 12 sampled episodes: 230 Bridge groups and 468 Fractal groups. All sparse groups remain in the saved dataset-wide statistics; do not describe the task-panel subset as the whole dataset. This distinction differs from GR1/LIBERO's canonical task identities.
- Continuous action channels 0:6, eight valid future steps, original checkpoint normalization. Gripper excluded. Batched deterministic inference in eval mode with bf16 autocast.

## Execution / outputs

- Pod `yuchen-bridge-fractal-scale-0907`, 4 GPUs (2 per dataset). Existing training pods untouched.
- Cluster root `/mnt/pfs/yuchen/crossdataset_general_mse_20260907/`.
- Each dataset: `sampling_manifest.json`, `rankXX_partXXXX.npz`, and completion markers.
- Two-episode smoke checks passed for both checkpoints before full probing.
- `scripts/summarize_bridge_fractal_scale.py` requires all completion markers, checks sample count, duplicate IDs and checkpoint provenance, then writes `expanded_summary.json` and `all_instruction_groups.json` on the cluster.
- `scripts/cluster/finish_bridge_fractal_scale.sh` waits for completion, downloads the summary and builds `all_datasets_progress_lines.png/pdf` locally. It never creates a completed-looking figure from partial probes.
- Existing raw data and the three-row figure remain available; labels on the existing figure are now RoboCasa-GR1, LIBERO, Bridge. The old Bridge row remains explicitly three tasks until the expanded output is ready.

## Claim boundaries / review

The expansion covers the entire eligible instruction sampling frame, not every state. Task panels require repeated demonstrations and their coverage is reported separately. Stage curves and action/residual correlations are descriptive fitting diagnostics; neither establishes irreducible noise, contact causality, or policy-held-out generalization. All qualifying task curves are retained and five are highlighted by a fixed random seed, not chosen by effect strength.

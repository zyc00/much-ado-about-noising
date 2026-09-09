# Action-magnitude specialist probe

This probe asks whether the irreducible continuous-action residual scale changes
with the magnitude of the requested motion. It trains five MSE specialists for
each of three existing policy/data stacks:

- GR00T N1.7 on GR1/RoboCasa;
- pi0.5 on LIBERO;
- GR00T N1.7 on Bridge/WidowX.

## Controlled protocol

For each observation, the binning statistic is the RMS of the policy-normalized
continuous action chunk. GR1 uses its 29 configured arm, hand-joint, and waist
channels. LIBERO and WidowX use the first six continuous channels and exclude the
binary gripper. Thresholds are fit only on training episodes. GR1 and LIBERO use
within-task quintiles; WidowX uses global quintiles because its 19,974 free-form
language annotations are too sparse to serve as semantic task IDs.

Every tenth episode within each task is held out deterministically. A specialist
is trained only on one training quintile and will be evaluated only on held-out
episodes assigned with the training thresholds. Thus, the target comparison is
the held-out continuous residual RMS across Q1--Q5, not the training loss.

All runs retain the corresponding established architecture, normalization,
augmentation, batch size, optimizer, and data staging conventions. Each uses a
fresh 5k-step cosine schedule and saves only the final checkpoint:

- GR1: 60k MSE checkpoint, global batch 512, LR 1e-4;
- LIBERO: 30k pi0.5 MSE checkpoint, global batch 32, LR 2.5e-5, and a
  167-step warmup (the original 1k/30k warmup ratio scaled to 5k steps);
- WidowX: earliest retained prefit checkpoint (10k MSE + 4k Hetero-t), global
  batch 1024, LR 1e-4. All specialist updates themselves use plain MSE.

The WidowX initialization difference must be disclosed if these results enter
the paper. It does not alter the specialist objective, but means this stack is a
prefit-then-MSE diagnostic rather than a warm start from a pure-MSE checkpoint.
When this checkpoint is loaded into the MSE architecture, the launcher ignores
only its four unused `sigma_decoder` tensors. Loading remains strict for every
shared policy tensor and for all other missing, unexpected, or mismatched keys.

Cluster outputs live under
`/mnt/pfs/yuchen/action_mag_specialists_20260905/`. The three persistent workers
are pinned to nodes 29, 11, and 18 and run the fifteen jobs sequentially in
balanced queues.

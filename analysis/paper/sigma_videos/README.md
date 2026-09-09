# GR1 HT rollouts with the predicted sigma (2026-09-07)

Videos of successful closed-loop rollouts of the HT c=2 GR1 policy (`ft_gr1c2/checkpoint-60000`, 51.5% on the 24-task
harness) with the head's predicted scale sigma drawn underneath, one task per video, five tasks.

## How the data were produced (cluster)

- `run_gr1_sigma_videos.sh` (scratchpad copy; PFS `groot/run_gr1_sigma_videos.sh`): official harness, `rollout_policy.py`
  with `--n-envs 1 --n-action-steps 8 --max-episode-steps 720 --seed 1234` (basket rerun: `--seed 4321`, 8 episodes),
  `--video-dir` on PFS, and the client trajectory dump (`GR00T_TRAJ_DUMP`). The policy server is restarted per task.
- Server-side sigma dump: `gr00t_n1d7.py` patch behind `GROOT_SIGMA_DUMP=<jsonl>` (backup `.bak_presigdump`). At every
  policy call it additionally evaluates the sigma decoder (inference normally skips it) and writes, per batch element,
  `sigma` = mean over the 8 executed steps x 29 valid channels of softplus(s_raw + sbias) + 1e-3 (exactly the
  training-loss scalar), `per_step` (8 means over channels), `per_dim` (29 means over steps), and `groups`
  (arm = channels 0-13, hand = 14-25, waist = 26-28; order left_arm, right_arm, left_hand, right_hand, waist).
  sbias = -1.088 (checkpoint config). Units: normalized action space.
- Alignment: one policy call = 8 env steps; the recorded video holds one frame per 2 env steps (4 frames per call, checked:
  52 calls <-> 208 frames). Episode boundaries and success flags come from the client dump (`done`, `success`).

## Files

- `<task>_ep<k>.mp4`: first successful episode of the task. Top: recorded composite (two ego views, cropped to rows 40-212).
  Middle: predicted sigma (red, training definition), arm / hand / waist channel means, moving cursor; shaded spans =
  left (blue) / right (green) hand more closed than its mid-range (normalized hand joint state). Bottom: right-hand
  closure and arm joint speed as motion-stage cues. 20 fps, 640 x 720.
- `<task>_ep<k>.png`: six-keyframe contact sheet with the same sigma curve (paper-ready).
- `summary.json`: per task: episode index, number of calls/frames, sigma min/max/first, peak step, channel-group means,
  successes among the rolled-out episodes.
- `raw/<task>/`: `videos/*.mp4` (one per episode, `_s1` = success), `sigma.jsonl`, `traj_slim.npz` (client dump without
  the image arrays), `client.log`. Checksums verified against the PFS copies.
- Renderer: `../scripts_gr1_mechanism/make_sigma_videos.py <raw dir> <out dir> [--fps 20] [--episode first_success|k]`.

Rollout successes (single env, 4 episodes unless noted): cutting board -> pan 1/4, tray -> pot 4/4, placemat -> basket 4/8
(seed 4321), bottle -> cabinet + close 3/4, can -> drawer + close 2/4. These are small-n readings of the same policy whose
24 x 20 harness numbers are 0.90 / 0.85 / 0.85 / 0.65 / 0.55.

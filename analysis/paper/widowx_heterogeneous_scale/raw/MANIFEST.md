# Raw probe data behind fig_widowx_heterogeneous_scale (and student_t_tails / longtail_motivation)

Recorded 2026-09-06. Cluster copy: `/mnt/pfs/yuchen/widowx_general_scale_20260906/raw`. Machine-readable: `manifest.json`.

Selection: three most frequent nonempty lowercase instructions among episodes with >= 24 frames -> sweep into pile, open the drawer, close the drawer; 48 episodes per task, 12 evenly spaced chunk starts per episode, seed 20260906; 1728 states in total. Each state's target/prediction/residual is an 8 x 6 array (8 steps x 6 continuous channels (x, y, z, roll, pitch, yaw), gripper excluded; normalized action units). The three files families share episode, task_id and step exactly.

| file | states | checkpoint | objective | bytes | sha256 |
|---|---|---|---|---|---|
| flow_rank0.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000 | flow | 1,416,638 | fe459472f352e401… |
| flow_rank1.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000 | flow | 1,417,677 | d21338724bfa9246… |
| flow_rank2.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000 | flow | 1,416,688 | 738471b39fa33394… |
| flow_rank3.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxflow/checkpoint-20000 | flow | 1,416,096 | c819adf62547f41e… |
| hg_rank0.npz | 1728 | /mnt/pfs/yuchen/groot/ft_wxpurehg/checkpoint-9000 | hg | 679,258 | 74f547c60694cd6d… |
| mse_rank0.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxmse/checkpoint-20000 | mse | 3,462,522 | 39700e60b3abbba3… |
| mse_rank1.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxmse/checkpoint-20000 | mse | 3,463,837 | 063bb5b604969161… |
| mse_rank2.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxmse/checkpoint-20000 | mse | 3,462,437 | 2cbaf644e16c6cc1… |
| mse_rank3.npz | 432 | /mnt/pfs/yuchen/groot/ft_wxmse/checkpoint-20000 | mse | 3,461,555 | bf80724f74e3bdf2… |

Fields:

- `flow_rank0.npz`: episode [432] int64, task_id [432] int64, step [432] int64, length [432] int64, progress [432] float64, target [432, 8, 6] float32, prediction [432, 8, 6] float32, residual [432, 8, 6] float32, samples [432, 16, 8, 6] float32, spread [432] float64
- `flow_rank0.npz`: episode [432] int64, task_id [432] int64, step [432] int64, length [432] int64, progress [432] float64, target [432, 8, 6] float32, prediction [432, 8, 6] float32, residual [432, 8, 6] float32, samples [432, 16, 8, 6] float32, spread [432] float64
- `hg_rank0.npz`: episode [1728] int64, task_id [1728] int64, step [1728] int64, length [1728] int64, progress [1728] float64, target [1728, 8, 6] float32, prediction [1728, 8, 6] float32, residual [1728, 8, 6] float32, sigma [1728] float64, m [1728] float64, d [1728] float64

Definitions: `residual` = prediction - target. MSE: `state` (proprioception, 8-dim padded) and `embedding` (mean backbone feature) are kept for the
scale-fitting panel. Flow: `samples` = 16 draws of the 4-step sampler at the same observation, `prediction` = their mean, `spread` = RMS of
(samples - mean). HG: `sigma` = the loss's per-state scalar scale (masked chunk-mean softplus(s_raw + ht_sbias) + 1e-3 over all valid dims),
`m` = mean squared residual over all valid dims (incl. gripper), `d` = number of valid dims (56).

Generators: scripts/probe_widowx_general_scale.py (cluster job scripts/cluster/widowx_general_scale.yaml, 4 ranks); scripts/probe_widowx_hg_scale.py (single rank, 2026-09-06).
Consumers: scripts/plot_widowx_heterogeneous_scale.py (v1), plot_widowx_heterogeneous_scale_v2.py (v5 figure), plot_student_t_tails.py, plot_longtail_motivation_v2.py, plot_longtail_motivation_v3.py (the recommended long-tail figure).

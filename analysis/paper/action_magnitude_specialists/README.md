# Split-and-fit action-scale motivation figure

The figure is generated from episode-disjoint held-out predictions of MSE
specialists trained on action-magnitude quintiles. Regenerate on a machine that
can access the cluster result directory:

```bash
python scripts/action_magnitude_specialists/plot_split_fit_figure.py \
  --results /mnt/pfs/yuchen/action_mag_specialists_20260905/results \
  --output analysis/paper/action_magnitude_specialists
```

Panel (a) uses empirical centered residual-coordinate histograms and Gaussian
curves whose scales are fitted from those held-out residuals. Panel (b) reports
the uncentered residual RMS and task/episode-cluster bootstrap intervals. The
current preview contains all five PI0.5/LIBERO specialists, GR1 Q1/Q2/Q5, and
WidowX Q1/Q3/Q5. The remaining GR1 and WidowX middle quintiles are added
automatically as their result files become available.

# Fractal: nu 224 to 64 with scale-gradient compensation

Run: `/mnt/pfs/yuchen/groot/ft_fr_nu224to64_hgcomp_20260909`  
Pod: `yuchen-fr-nu224to64-hgcomp-0909`

The run starts fresh from the original pretrained GR00T model and uses the
standard Fractal HT recipe and native softplus scale. Updates 1--7000 use
fixed `nu=224`. Updates 7001--12000 use a smooth log-linear decay from 224 to
64; updates 12001--20000 hold `nu=64`.

After the decay begins, a detached-mean Gaussian term restores exactly the
part of the direct scale gradient removed by lowering nu. Let
`q=||a-mu||^2/sigma^2`, and detach the following coefficient:

```
lambda(q,nu) = 224/(224+q) - nu/(nu+q).
```

The auxiliary is `lambda * [q_detached_mu/(2d) + log(sigma)]`. Therefore

```
d(L_HT_nu + L_aux)/d log(sigma)
  = 224/(224+q) * (1-q/d),
```

which is exactly the scale gradient of the fixed-nu=224 Student-t loss at the
current prediction and scale. The auxiliary has no direct mean gradient.
Thus compensation ramps naturally from zero and cannot overshoot the nu=224
reference direct scale gradient. This guarantee does not claim identical Adam
updates or identical parameter trajectories, since shared features also
receive the changed mean-loss gradient.

The endpoint evaluation matches the old nu=224 result protocol: seed 1234,
six Fractal tasks, 100 rollouts per task, 5 vector environments, NAS=1, and
300 maximum steps. Exact source, unit tests, reference config, schedule, and
training diagnostics are retained with the run.

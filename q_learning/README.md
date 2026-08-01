# MIP/HT objectives vs FFN on Q-learning (toy MDP from geyang/ffn)

Question: is the spectral-bias failure of neural value approximation (Yang,
Ajay, Agrawal, ICLR 2022) the same failure mode we found on human ToolHang
BC data — large-residual (heavy-tailed) target noise dominating MSE
gradients — and do the objectives that fixed BC (hetero-Student-t NLL,
hetero-Gauss NLL, two-step MIP regression) also help value fitting and
fitted value iteration, compared to their Fourier-feature fix?

Testbed: their RandMDP (1-D state, 2 actions, `option='fixed'`, 100-state
discretization), tabular VI ground truth Q*, their 4x400 MLP / LFF(b)
architecture, RMSprop lr 1e-4, full-batch, target sync every epoch —
all from `ffn/toy_mdp/value_iteration_lff.py`. Metric: RMSE(Q, Q*) vs
epochs (4000).

Arms = arch x objective: arch in {mlp, lff1, lff5}; objective in
{mse, huber (their default), ht (Student-t NLL, nu=2, learned sigma head),
hg (Gaussian NLL, learned sigma head), mip (scalar two-step anchor
regression, t_two_step=0.9)}. 8 seeds per cell.

Experiments:
- E1 `sup_clean`: supervised fit of Q*. Pure spectral bias, zero noise.
- E2 `sup_gnoise` / `sup_tnoise`: supervised fit of Q* + FIXED label noise
  (Gaussian / Student-t df=2, scale 0.25). The BC analogy (noisy demos).
  Scored against CLEAN Q*.
- E3 `nfq`: their fitted-VI loop (bootstrap targets). Structured target
  error, no injected noise.
- E4 `nfq_tnoise`: fitted VI + per-epoch heavy-tailed reward noise
  (Student-t df=2, scale 0.25). The "large loss noise" regime in RL form.

Pre-registered predictions (falsifiable):
1. E1: LFF >> MLP (their result); ht/hg/mip do NOT beat mse (no noise to
   absorb -> if they DO, the benefit is not noise-side).
2. E2 tnoise: ht/hg >> mse; mse chases outliers. LFF5+mse WORSE than
   mlp+mse (local interpolation fits noise). If LFF5+ht best overall,
   the two fixes are complementary (spectral + robustness), same as our
   human-data "curriculum" story.
3. E2 gnoise: hg ~ ht; smaller gap to mse (Gaussian noise is MSE-matched;
   only variance reduction at stake).
4. E3: their claim (underfit, not noise) predicts LFF wins and ht/hg/mip
   change little. Our mechanism allows a bounded-influence win only if
   bootstrap error is heavy-tailed across states.
5. E4: ht/hg stabilize; mse degrades most; mip partial (anchor absorbs
   target-side noise); FFN alone does not fix it (bandwidth is not
   robustness).

Files:
- `ffn/` — clone of github.com/geyang/ffn (their code, unmodified)
- `qlib.py` — shared lib: their MDP/VI/LFF/loop + our objective ports
- `run_toy.py` — one (exp, arch, loss, seed) cell -> JSON line
- `run_all.py` — full sweep, parallel; writes `results.jsonl` + table

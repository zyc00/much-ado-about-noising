# Detach audit: gradients received by shared features

## Scope and protocol

- Fractal original hold-7k annealing run, checkpoints 10000 and 12000 (native
  nu 128 and 32, respectively).
- Same first 128 distinct-episode samples from the previous 1024-example
  `mixed_0` protocol. This is a diagnostic subset of training demonstrations,
  not a policy-held-out evaluation or success-rate comparison.
- Capture the exact `model_output` tensor shared by `action_decoder` and
  `sigma_decoder`; assert the two decoder inputs are the same tensor object.
- Upstream forward is train-mode BF16 with fixed input processing/dropout.
  Replay both actual nonlinear MLP decoders in FP32, with TF32 disabled, on
  a detached FP32 copy of those features, now requiring gradients.
- Sigma uses the actual masked mean of softplus(raw + sbias) plus 0.001.
  All training channels, including gripper, are retained; effective d = 56.
- Different nu values use the SAME features, heads, targets, and sigma.
  No parameter or optimizer updates; this is not separately trained nu arms.
- Each gradient is of one example's per-coordinate-mean NLL. Norms flatten
  that example's entire shared feature tensor. We report the mean of these
  per-example norms, not the norm of summed parameter gradients. There is
  no division by the diagnostic batch size.
- Reconstructed BF16 native loss matches actual loss. FP32 replay changes
  predictions slightly: maximum microbatch RMS differences are 0.000878
  (10k) and 0.000808 (12k), in normalized action units.

## Results

| Checkpoint | nu | Mean ||g_mu||, detach sigma | Mean ||g_sigma||, detach mu | Median per-example ||g_sigma|| / ||g_mu|| |
|---|---:|---:|---:|---:|
| 10k | 224 | 0.112205 | 0.019866 | 0.146241 |
| 10k | 7 | 0.120388 | 0.003157 | 0.020311 |
| 12k | 224 | 0.121290 | 0.027875 | 0.181263 |
| 12k | 7 | 0.126049 | 0.004240 | 0.025175 |

The mean scale-path norm drops to 15.9% and 15.2%, while the mean action-path
norm rises by 7.3% and 3.9%. Median branch cosine is 0.0629 and 0.0560;
negative cosines occur in 25.8% and 21.1% of samples. These figures do not
indicate uniformly opposing paths. Each sample's branch cosine is invariant
to nu at fixed features and predictions (up to numerical error).

All nu settings, stacked feature-gradient norms, and combined-gradient
statistics are in checkpoint_10000.json and checkpoint_12000.json.
Per-example norms, inner products, q and sigma are in the accompanying NPZ.

## Exact gradient identities

Let r = mu - a, q = ||r||^2 / sigma^2. For per-dimension NLL,

    g_mu_x(HT) = (nu+d)/(nu+q) * g_mu_x(HG)
    g_sigma_x(HT) = nu/(nu+q) * g_sigma_x(HG).

Here HG is the Gaussian objective evaluated at the SAME heads/features,
not a separately trained HG model. Consequently, for nonzero branch norms,

    ||g_sigma_x(HT)|| / ||g_mu_x(HT)||
      = nu/(nu+d) * ||g_sigma_x(HG)|| / ||g_mu_x(HG)||.

At d=56, changing nu from 224 to 7 multiplies EVERY per-example branch norm
ratio by (7/63)/(224/280) = 1/7.2. This exact factor applies to the
per-example ratios (and their median), not generally to a ratio of norms
averaged across samples or to shared-parameter gradient norms.

For the user's linear example mu=W1*x, sigma=W2*x>0:

    g_mu_x = (nu+d)/(d*sigma^2*(nu+q)) * W1^T*r
    g_sigma_x = nu*(d-q)/(d*sigma*(nu+q)) * W2^T.

For actual nonlinear heads, replace W1 and W2 with their local Jacobians;
the two HG scaling identities remain exact. Autograd verified both
identities and g_full = g_mu_x + g_sigma_x in every measured microbatch.

## Interpretation limits

Small nu changes the balance of the two objectives' paths into shared
features, not just their common gradient magnitude. This does not by itself
prove scale underfitting, determine an optimal gradient ratio, or explain a
success-rate difference. Backpropagation into upstream shared parameters,
sample cancellation, and Adam can change the resulting update geometry.

Source: scripts/probe_nu_shared_feature_gradients.py.
Cluster raw directory:
/mnt/pfs/yuchen/nu_recipe_debug_20260908/shared_feature_gradients

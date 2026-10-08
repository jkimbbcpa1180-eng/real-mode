# Phase extrapolator v3: model-choice fix

Changes since the first v3 review. License: CC0 1.0 Universal (public domain).

## What changed and why

1. **Model choice uses a validation horizon.** The physics/polynomial choice is now made at
   `min(horizon, validation horizon)`. Before, at +5 ms the backtest could only start at
   origins in the first half of the record, so it was judging "physics fitted on half the
   data", which is not the full-record fit actually used. At those origins the fit has not seen
   the exponential yet. The validation horizon is the longest one whose backtest still starts at
   >= 95% of the record (0.40 ms for the demo).
   *Assumed setting:* `VALIDATION_FRACTION = 0.95`, picked on tuning seeds 0-49 from
   {0.75, 0.90, 0.95}. It was then checked on held-out seeds 100-149.
2. **Reliability checks still use the actual horizon.** The v2 checks (horizon guard,
   state outside backtest range, error trend, RMSE vs range) run on the backtest at the
   requested horizon. A long-horizon forecast that can't be backtested there stays
   UNRELIABLE, and a `validation_note` says the band at that horizon comes from the model.
3. **Envelope band when the polynomial is chosen.** If a physics fit passes its checks but
   the polynomial is chosen, the band is the envelope of the polynomial's empirical band and
   the physics band. The two models disagree, and that disagreement is real uncertainty. The
   polynomial-only band is kept as `polynomial_only_interval`. Beyond the backtestable horizon,
   the model-only band is the envelope of the usable forecasts.
4. **Stricter residual-structure check.** Residuals are now flagged when the lag-1
   autocorrelation exceeds `3/sqrt(n)` (about 3 standard errors for white noise; 0.30 at
   n = 100). The old cutoff was 0.5. This catches more off-model fits
   (demo + 0.3 sin(2000t): physics flagged in 16/20 noisy seeds, up from 5/20).
   *Assumed setting.*

## Results (chosen model's band vs noise-free truth, sigma = 0.01)

| Seeds | Horizon | Before: median err / coverage | After: median err / coverage |
|---|---|---|---|
| tuning 0-49 | +0.5 ms | 0.0099 / 78% | 0.0083 / 96% |
| tuning 0-49 | +5 ms | 1.539 / 12% | 0.283 / 90% |
| tuning 0-49 | +20 ms | 397 / no band | 264 / 80% (model-only, unvalidated) |
| held-out 100-149 | +0.5 ms | 0.0106 / 72% | 0.0112 / 86% |
| held-out 100-149 | +5 ms | 1.544 / 8% | 0.812 / 94% |
| held-out 100-149 | +20 ms | 397 / no band | 397 / 94% (49 of 50 runs had a band) |

All +5 ms and +20 ms outputs are still marked UNRELIABLE, as before.

## Limits

- At +5 ms the 90% is reached partly through wide envelope bands (median width 2.4 on
  tuning seeds, 4.0 on held-out seeds). The record cannot tell the two models apart:
  in-sample BIC between physics and a global polynomial is a coin flip (26/50).
- +0.5 ms is over-covered on tuning seeds (96%) and slightly under-covered on held-out
  seeds (86%; 82% when physics was chosen). 50 seeds give about +/-4 points of sampling error.
- +20 ms is beyond what the data can backtest. Its bands are model-only and enormous.

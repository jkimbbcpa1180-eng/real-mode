# REAL MODE: Temporal Phase Extrapolator

Forecasts a fast signal a short time ahead from high-frame-rate samples, and reports how much
to trust each forecast. License: CC0 1.0 Universal (public domain). Copy, modify and use freely.

## Files

- `phase_extrapolator.py`: the original v1.
- `phase_extrapolator_v2.py`: fixes v1. It scales time, picks the window and polynomial degree by
  walk-forward backtest, reports a backtest error and a 90% interval, and flags UNRELIABLE output.
- `phase_extrapolator_v3.py`: fits the physics model `a*sin(wt) + c*cos(wt) + b*exp(lam*t) [+ d]`,
  propagates parameter uncertainty, and uses physics only when the data back it up. Otherwise it
  falls back to v2's polynomial. It needs `phase_extrapolator_v2.py` in the same folder, and numpy only.
- `test_phase_extrapolator.py` (21 tests), `test_phase_extrapolator_v3.py` (26 tests).
  Run `python3 -m unittest -v test_phase_extrapolator test_phase_extrapolator_v3`.
- `demo_output_v3.txt`: demo run. `v3_changes.md`: the model-choice fix, with full results.
- `phase_extrapolator.patch`: v1 to v2 patch.

## v3 results (50 noisy runs, sigma 0.01; chosen model's 90% band vs the noise-free truth)

| Runs | Horizon | Median error | Band holds the truth |
|---|---|---|---|
| tuning (seeds 0-49) | +0.5 ms | 0.0083 | 96% |
| tuning (seeds 0-49) | +5 ms | 0.283 | 90% |
| held-out (seeds 100-149) | +0.5 ms | 0.0112 | 86% |
| held-out (seeds 100-149) | +5 ms | 0.812 | 94% |

On the clean demo signal, v3 recovers the true parameters (omega 200, lambda 300, B = e^-3)
to about 1e-12 and predicts +20 ms exactly.

## Limits (read before using)

- **The model form is assumed.** If the real system isn't one sinusoid plus one exponential,
  the fit can look fine inside the record and still extrapolate badly. The residual checks catch
  some off-model signals (a second frequency, a straight line, pure noise), but not all.
  An added quadratic term is absorbed without a flag.
- **The +5 ms coverage comes partly from wide bands.** A 10 ms record can't reliably tell the
  physics model from a polynomial (an in-sample BIC comparison is a coin flip, 26 of 50). When
  the polynomial is chosen, the band spans both models. Median band width at +5 ms is 2.4 to 4.0.
  A sharp, well-calibrated +5 ms band is not achievable from this record length.
- **+0.5 ms coverage varies by run set**: 96% on tuning runs, 86% on held-out runs (82% when
  physics is chosen). 50 runs carry about +/-4 points of sampling error.
- **+20 ms is beyond what the data can backtest.** Its bands are model-only and enormous
  (median width about 1,800 to 3,700). Treat its coverage numbers as meaningless.
  Every +5 ms and +20 ms output is marked UNRELIABLE.
- **Bands are first-order (delta method)**, not a bootstrap. They can under-cover when the
  forecast is strongly nonlinear.
- **Thresholds are assumed settings, not physics**: validation fraction 0.95 (chosen on the
  tuning runs only), residual autocorrelation cutoff 3/sqrt(n), runs-test z >= -3, R^2 >= 0.99,
  paired-test z > 1.645, 70% minimum backtest coverage, |lambda * span| <= 15.
- The demo signal is synthetic. Results on real sensor data are untested.

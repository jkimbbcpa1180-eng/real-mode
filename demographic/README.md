# Korea demographic scenario simulator: v2.0 review and v2.1

Credit: humanity. Public domain (CC0-1.0). Python 3.9+, standard library only
(the chart used matplotlib).

- `korea_demographic_realmode.py`: the original v2.0, unchanged (9/9 tests pass).
- `korea_demographic_v2_1.py`: fixed and recalibrated v2.1 with policy scenarios (14/14 tests).
- `v2_0_to_v2_1.patch`: exact diff (applies byte-exact).
- `demo_output_v2_0.txt`, `demo_output_v2_1.txt`, `population_baseline_vs_all_combined.png`.

Run: `python3 korea_demographic_v2_1.py --test` or
`python3 korea_demographic_v2_1.py --years 100 --scenario all_combined_central`.

## What the review found in v2.0
The core accounting (exposure, the births/deaths/migration balance, the national
TFR, the logit fertility, the policy ramp) is correct. The demo inputs were not:
its death curve implies a life expectancy of about 72.5 years (real: 83.7), giving
about 1.41M deaths a year against about 363k actual in 2025. That, not low
fertility alone, drove its collapse to 6.5M by 2126. Other fixes: misspelled or
missing policy keys were silently ignored or crashed; there was no bound on
immigration inputs; sexes had identical mortality; the sex ratio at birth was 50/50;
Seoul and the rest of Korea had the same TFR; the median age was not interpolated.

## v2.1 demo calibration (approximate, not a forecast)
Fitted to 2025 Statistics Korea totals: 51.7M people, TFR 0.80 (preliminary),
254.5k births, 363.4k deaths, 65+ share 20.7%, life expectancy 86.6 (F) / 80.8 (M),
105.8 boys per 100 girls, Seoul TFR about 0.60. Year 1 of the model: births 261k,
deaths 381k, 65+ 20.8%. The five-year age bands are approximations; real use
should load the single-year KOSIS tables. Net immigration of +80k a year is a guess.

## Policy scenarios (effect sizes are GUESSES)
Each scenario has low / central / high long-run TFR effects: homes_for_families
(+5/12/25%), work_hours (+3/8/15%), end_hagwon (+3/10/20%),
equal_pay_subcontractors (+2/6/12%), free_care (+5/12/25%), and all_combined
(which likely overstates the total, since some policies overlap). These are
informed by, not taken from, OECD "Korea's Unborn Future" (2025) and family-policy
studies.

| Scenario | TFR in 2056 | Population 2056 | Population 2126 |
|---|---|---|---|
| baseline (TFR held at 0.80) | 0.80 | 41.3M | 13.2M |
| all_combined low | 0.95 | 42.0M | 15.6M |
| all_combined central | 1.23 | 43.3M | 20.7M |
| all_combined high | 1.74 | 45.6M | 33.1M |

No scenario reaches replacement fertility (2.1). These policies can slow the
decline a lot, but on these assumptions they don't reverse it on their own.

## Limits
Deterministic expected counts, no uncertainty intervals, no feedback from the
economy, constant life tables, and uncalibrated policy effects. Not a backtested
forecast.

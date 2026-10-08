# REAL MODE dispatch v2: results summary (2026-10-08 KST)

**All camera data is EXAMPLE/SIMULATED. There is no real camera feed. Every forecast-error size (sigma) and
every weather/demand distribution below is an ASSUMPTION, not a measurement.**

Files: `dispatch_v2.py` (engine, camera interfaces, Monte Carlo), `test_dispatch_v2.py` (19 tests, all pass),
`run_v2_report.py` (v1 vs v2 + 100k run), `mc_results_v2.json` (full results). Python 3.9+ stdlib, seed 20261008.

## v1 vs v2 at the original demo point (850 W/m² clear-sky, cloud 0.15)
| Wind | v1 solar | v1 wind | v1 total | v2 solar | v2 wind | v2 total |
|---|---|---|---|---|---|---|
| 4.8 m/s | 6.792 | 0.000 (all 4 cut by 0.25 MW floor) | 6.792 | 4.351 | 0.395 | 4.746 |
| 9.0 m/s | 6.792 | 2.748 (WTG_02 cut) | 9.540 | 4.351 | 3.895 | 8.247 |
| 12.0 m/s | 6.792 | 8.138 | 14.930 | 4.351 | 8.610 | 12.961 |

- v2 wind is higher: correct cubic curve, no fixed floor. At 4.8 m/s WTG_01 and WTG_04 make 0.196 MW each.
- v2 solar is lower on purpose: 25% commercial module x 0.82 performance ratio instead of a 32% lab cell.
- Wake search: switching WTG_02 off would gain only 0.039 MW (4.8 m/s) and 0.030 MW (9 m/s) in the model,
  under the 0.05 MW switching threshold (ASSUMED), so all turbines stay on. WTG_02 sits only 350 m (2.9 rotor
  diameters) behind WTG_01, which is why it loses so much wind.

## Monte Carlo: 100,000 draws, without vs with cameras (SIMULATED)
Same "true" weather and demand in both runs; only the forecast errors differ. pct = plain percentile.

| Metric (MW, MWh per 1-h interval) | Without cameras | With cameras (SIMULATED) |
|---|---|---|
| Delivered: mean / pct10 / pct50 / pct90 | 8.095 / 4.226 / 6.685 / 14.230 | 8.114 / 4.226 / 6.684 / 14.315 |
| Solar: mean / pct10 / pct50 / pct90 | 4.040 / 3.278 / 4.280 / 4.401 | same (same true weather) |
| Wind: mean / pct10 / pct50 / pct90 | 4.441 / 0.000 / 2.607 / 12.441 | 4.441 / 0.000 / 2.607 / 12.440 |
| Grid-curtailed MWh: mean / pct90 | 0.385 / 1.812 | 0.366 / 1.673 |
| RMSE available power (total) | 1.735 | 1.655 (-4.6%) |
| RMSE solar | 0.518 | 0.421 (-18.8%) |
| RMSE wind | 1.669 | 1.606 (-3.8%) |
| RMSE demand | 0.502 | 0.407 (-18.8%) |
| RMSE delivered (after grid limit) | 1.456 | 1.418 (-2.6%) |
| Grid-limit breach share | 1.45% | 1.35% |

Reading it: cameras help solar and demand forecasts the most. Total error falls only ~5% because wind-speed
error dominates and cameras do not measure wind speed (kept equal in both runs). A tighter forecast lets the
operator keep a smaller safety margin, so curtailment drops about 5%. These are model results from
assumed inputs, not field measurements.

## Verified vs assumed
Verified (checked 2026-10-08):
- Kasten–Czeplak G(N)/G(0) = 1 − 0.75(N/8)^3.4, defined against clear-sky GHI (Kasten & Czeplak 1980, Solar Energy 24:177).
- Katic sum-of-squares wake combination (Katic et al. 1986, "A Simple Model for Cluster Efficiency", DTU Orbit).
- Commercial tandem modules ~25% (Oxford PV, pv magazine 2026-06-18); certified lab tandem modules 29.4–31.4%,
  cell record 35.5% (LONGi, 2026-07-14). So 32% is labeled LAB/TARGET.
- PV performance ratio: US federal fleet median 0.79, 0.85 achievable (US DOE 2022); NREL fleet median soiling 2–3%.
- Yaw loss ~cos(γ)^p with p ~1.4–3 (WES 2024; NREL forum, FLORIS default 1.88). Lidar yaw correction gave
  +1.8% AEP for a 7° error (NRG/First Wind) and +2.4% for a 7.5° offset (NREL).
- All-sky imagers: GHI nowcast RMSE 6.9–18.1%, skill ~0.2 vs persistence (IEA PVPS Task 16 benchmark,
  Logothetis et al. 2022); blended satellite+imager+persistence skill 0.24 (Nouri et al., DLR 2025).
- Flock's stated default plate-read retention: 7 days (Flock LPR policy page).

Assumed (not measured): feathered Ct 0.05; 0.05 MW wake-switch threshold; Weibull c=9 m/s k=2; direction
spread 15°; cloud Beta(0.8,1.6); soiling Beta(2,98); true yaw error sigma 6°; icing 1%; demand 6±1 MW;
2,000±500 vehicles/h; EV share 10%, 15% stop to charge, 40 kWh/session; export cap 8 MW; all forecast sigmas
(wind 1.2 m/s, direction 8°, cloud 0.20 vs 0.16 with imager, soiling 0.5 pt, yaw 1.5°, icing detection 95%,
traffic 100 vs 500 vehicles/h, base demand 0.4 MW). The 0.16/0.20 cloud ratio was picked to sit near the
~0.2 published skill, but that skill is for GHI, so mapping it to cloud fraction is itself an assumption.

## Camera privacy rules (traffic counts)
Counts only, one integer per bin of at least 15 minutes. Plates, images and per-vehicle records are never sent
or stored by this system. Used only for energy forecasting, never joined with other data or shared for
enforcement. Prefer counting-only devices, because LPR products keep plate reads themselves.

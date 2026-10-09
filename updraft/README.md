<!-- SPDX-License-Identifier: CC0-1.0 -->
# Mountain Updraft v4.2: physics fixes to v4.1

`mountain_updraft_v4_2.py` is a fixed version of v4.1. It uses only the standard library and is licensed CC0 1.0. It keeps the v4.1 layout, function names and the optional FastAPI server (`--serve`). v4.1 itself is unchanged. `v4_2.patch` is the `diff -u` between the two versions.

```
python3 mountain_updraft_v4_2.py          # benchmark
python3 -m unittest test_mountain_updraft_v4_2   # 13 tests
python3 make_demo_v4_2.py                 # regenerates demo_output_v4_2.txt
```

## What changed

1. **Coupled draft.** In v4.1, the full buoyant head drove the velocity, and 70 % of that same head was also taken by the turbine, so the pressure was counted twice. In v4.2, (1−x)·head = (K_in + f·L/D + K_esp + K_exit)·ρv²/2. The defaults are f = 0.010, L = H/sin 30°, K_exit = 1 and K_esp = 0.5, and all are assumptions. Because dT = Q_th/(ṁ·cp), the turbine power ≈ η·x·gH·Q_th/(cp·T). This does not depend on the velocity, to first order. The ceiling g·H/(cp·T) is about 2.2 % for H = 650 m. Losses reduce the flow, not the power. The remaining second-order effect is a few percent.
2. **Turbine efficiency.** The default is 0.80. This is a ducted, pressure-staged turbine, not the open-rotor 0.44 used in v4.1, which is kept as a sensitivity case. Source: Gannon & von Backström, "Solar chimney turbine characteristics", *Solar Energy* 76 (2004). They measured 85–90 % total-to-total and 77–80 % total-to-static on a 0.72 m model, and report 80 % as the usual assumption (Schlaich 1995). This is a model-scale figure. A full-scale efficiency is not verified.
3. **Sounding.** The buoyancy integral is ∫g(ρ_a − ρ_p)dz along the conduit. The plume cools at the dry-adiabatic rate. The ambient is interpolated from the sounding and uses virtual temperature from the dewpoint (Bolton). The stable-layer penalty is reported against a neutral atmosphere.
4. **ESP.**
   - Charge is field (Pauthenier) plus diffusion (White). Adding the two is an approximation. The White diffusion term is corrected: v4.1 left the particle radius (and one electron charge) out of the log argument, overstating diffusion charge about 10x for PM2.5.
   - Cunningham and Deutsch-Anderson are kept.
   - New outputs: SCA, PM2.5 removed in kg/day, corona power and ESP pressure drop.
     - The ambient PM2.5 of 35 µg/m³ is only an example.
     - The corona current density of 0.3 mA/m² is an assumption.
   - The theoretical migration velocity is about 0.5 m/s. Real industrial effective velocities are much lower, so a derate of 0.25 (an assumption) is applied, and the theoretical value is also reported.
   - `clean_air_km3_day` is replaced by three outputs: air treated, PM2.5 removed and air vented above the inversion.
5. **Grid.**
   - The synthetic-inertia cooldown now uses an injected `now_s` and per-instance state. No wall clock is used.
   - A BESS is added (assumed 2 MWh / 1.5 MW, SoC 10–95 %, 90 % round-trip). Export can exceed generation only when the BESS can supply the difference.
   - The droop math is unchanged.
6. **Status.** The hard-coded probability is removed, and the status is no longer always "OPTIMAL". It is now NOMINAL, DEGRADED or FAULT, based on the listed computed checks.
7. **Scale reality.** New outputs: thermal MW, the chimney limit, net electric MW, PV on the same land (20 % modules, assumed) and daily energy for an example 12 h half-sine day.
8. **Stress.** The centrifugal root stress for a uniform blade is ρω²(R² − r_hub²)/2, with r_hub = R − blade length.

## Results at the demo inputs (780 W/m², demo sounding)

| | v4.1 | v4.2 |
|---|---|---|
| dT | 14.36 K | 29.17 K |
| velocity / flow | 25.24 m/s / 11,417 m³/s | 12.42 m/s / 5,619 m³/s |
| electrical | 1.269 MW (η 0.44, double-counted head) | 1.872 MW (η 0.80); 1.030 MW at η 0.44 |
| net after ESP corona | n/a | 1.734 MW (0.58 % of sunlight on the collector) |
| stable-layer penalty | n/a | 9.0 % |
| ESP capture | 20.1 % | 18.4 % effective (55.6 % theoretical); SCA 1.6 s/m |
| PM2.5 removed | n/a | ~3 kg/day at 35 µg/m³ (example) |
| air treated | "0.198 km³/day clean air" | 0.486 km³/day treated (and vented above the inversion) |
| status | OPTIMAL (hard-coded) | DEGRADED (SCA far below the industrial range of ≥ 20 s/m) |

Flow steps (`demo_output_v4_2.txt`):
- Exit loss only, neutral atmosphere: 7,404 m³/s. The hand check gives ~7,640; the ~3 % gap comes from air density falling with height in the integrated head.
- Adding friction and the ESP loss: 5,798 m³/s.
- Adding the stable sounding: 5,619 m³/s.

## Caveats

- **Physics.** This is a 1-D lumped model. The collector is a single energy balance with no heat loss by default; `collector_loss_w_m2k` is a sensitivity input, and U = 5 cuts power by about 13 %. Loss coefficients, the BESS size, the ESP derate, the corona current and the PV efficiency are assumptions, not design values.
- **Power ceiling.** The system cannot beat g·H/(cp·T) ≈ 2.2 % of the collected heat. The same land with PV gives about 60 MW peak against about 1.7 MW net here: 460 MWh/day vs 12.5 MWh/day for the example day.
- **ESP.** Air cleaning is small at this plate area. A meaningful SCA (≥ 20 s/m) at about 5,600 m³/s would need over 100,000 m² of plate.
- **City-scale context (rough, order-of-magnitude only).** 0.5 km³/day is a small fraction of the air over a city of roughly 150 km², because a mixing layer a few hundred metres deep holds tens of km³. The plant's ~1.7 MW is negligible next to a city's electricity demand, which is in the hundreds of MW. These figures are not sourced and are not a siting study.

- **Redaction:** the original v4.1's target-region line named a specific city; the published copy says Gyeonggi-do only, to keep personal location out of the repo.

<!-- SPDX-License-Identifier: CC0-1.0 -->
# GNSS/IMU error budget v2, with a drone v3 hook (fixes by Grok)

`gnss_budget_v2.py` is a fixed version of `gnss_imu_budget.py` (v1). The v1 file is unchanged, and `v2.patch` is `diff -u` from v1 (verified to reproduce v2 byte-exact). The script needs only numpy and is CC0.

```
python3 gnss_budget_v2.py                 # budget tables
python3 -m unittest test_gnss_budget_v2   # 10 tests
python3 mc_gnss_v2.py heldout             # drone MC (needs ../drone/v3 and its v2 parent), ~21 min on 8 cores
python3 make_demo_v2.py                   # regenerates demo_output_v2.txt
```

## Fixes

1. **Crash.** `compute_budget(imu=None)` now creates `IMUConfig()`. v1 used `field(default_factory=...)` as a plain function default, which crashed the demo.
2. **SBAS counted once.** SBAS is now one corrected-residual budget, with the same structure as DO-229 (σ² = σ_flt² + σ_UIRE² + σ_air² + σ_tropo²): the SBAS orbit/clock residual replaces the broadcast terms, and the grid iono replaces Klobuchar. v1 also added a 0.71 m "sbas_geo" term on top. At v1's defaults SBAS still came out better than L1 alone, because v1's broadcast orbit/clock values are large, but the extra term was double counting. The SBAS numbers here are assumptions, not DO-229 values.
3. **Slow vs fast split.** Every source is split into a slow Gauss-Markov part with a per-source τ (orbit/clock, iono, tropo, PPP ambiguity, half of multipath) and fast white noise (receiver, the other half of multipath). The slow parts are combined into one bias σ with a variance-weighted τ, which preserves the area under the autocorrelation. `drone_gps_model()` returns per-axis horizontal and vertical bias σ, τ and white σ: the drone v2/v3 GPS model (Gauss-Markov) plus white noise. v2 "standard GPS" was 0.5 m with τ 30 s, isotropic, with no white noise.
4. **IMU drift.** The √(sample rate) factor is gone, because the density is already per √Hz. Accel noise now gives n·t^1.5/√3, and the bias b·t²/2 is kept. Gyro tilt (g·θ) is added as g·b_g·t³/6 + g·ARW·t^2.5/√20. This is an unaided bound: real attitude filters level against gravity. It can be switched off with `include_tilt=False`.
5. **Definitions.**
   - pr·HDOP is the horizontal radial RMS (DRMS). Per-axis = DRMS/√2, and CEP50 = 1.1774·per-axis = 0.83·DRMS. v1 put 1.177 on the DRMS, which is about 41 % too large.
   - Vertical 0.6745σ is kept.
   - The iono DOP-inflation hack is removed (DOP is geometry). The iono slant is now the Klobuchar/IS-GPS-200 obliquity factor.
   - Receiver noise is the textbook DLL thermal-noise formula (coherent early-late), about 0.37 m at 45 dB-Hz with a 1 Hz loop and a 0.1-chip correlator. v1's formula was a constant at its defaults.
6. **Iono-free code noise (new).** The L1/L5 iono-free combination amplifies code noise and multipath by about 2.35 (L5 noise assumed at 0.5× L1). In a quiet sky this makes L1+L5 slightly worse than L1 alone in suburban multipath. Carrier smoothing, which is not modelled here, would reduce this.
7. **PPP convergence.** PPP is modelled as two cases. "Not converged" has a float-ambiguity error of about 0.4 m (pseudorange) with τ 900 s and code-level noise. "Converged" runs on carrier phase, at the cm level.

## Sources (checked)

- **GPS broadcast SISRE:** about 0.7 m RMS (Montenbruck et al. 2015, GPS Solutions 19:321) and about 0.6 m for 2017 (Montenbruck et al. 2018, Adv. Space Res.). v1's orbit 2.0 m + clock 1.5 m is therefore about 4× conservative. It is kept as the default, and `broadcast_scale=0.24` gives the measured level (sensitivity rows in the demo).
- **PPP float convergence, GPS-only:** about 20–40 min to 10 cm horizontal (e.g. Remote Sensing 16(8):1434, 2024). Multi-GNSS or ambiguity fixing brings it to about 5–20 min.
- **Not used or not re-verified:** DO-229 is used only for the budget structure. ICAO Annex 10 and NRLMSISE-00 (from the v1 list) were not re-checked. Every τ, the SBAS residuals, PPP real-time residuals and the multipath split are assumptions.

## v1 vs v2: fused horizontal error (2 s outage, suburban, metres)

| configuration | v1 quiet | v2 quiet | v1 storm | v2 storm |
|---|---|---|---|---|
| L1 bare | 3.75 | 3.63 | 14.06 | 13.41 |
| L1 + SBAS | 2.12 | 1.66 | 7.09 | 6.66 |
| L1 + L5 | 3.45 | 3.98 | 3.45 | 3.99 |
| L1 + L5 + SBAS | 1.99 | 2.69 | 2.00 | 2.69 |
| L1 + L5 + PPP (converged) | 1.70 | 0.83 (GNSS-only 0.10) | 1.71 | 0.85 (0.22) |
| L1 + L5 + PPP (not converged) | – | 2.66 | – | – |

Notes on the table:
- v2 is DRMS.
- In v1 the 2 s IMU term was about 1.2 m (the √200 bug).
- In v2 the precise configurations are dominated by v1's kept "process-noise" allowance (0.5·q·t²/√3 ≈ 0.58 m per axis at 2 s).

## Drone v3: SLAM-only recon, no markers and no pad

Setup:
- Held-out seed 20264008, never used before. 12 scenes per template, one run. There was no tuning pass; a 1-scene plumbing check ran on seed 20263008.
- The GPS error is the budget's per-axis Gauss-Markov bias (horizontal and vertical) plus white noise.
- The SLAM is given the matched bias σ and τ.

Each cell gives median coverage / p5 / % of scenes ≥ 0.90 / median point error (m):

| mode | bias h/v (m), τ | lawnmower | perimeter_orbit | poi_close_pass |
|---|---|---|---|---|
| quiet L1 | 2.43/5.16, 852 s | 0.00/0.00/0/4.85 | 0.02/0.00/0/1.50 | 0.00/0.00/0/2.99 |
| quiet L1+SBAS | 0.84/1.79, 391 s | 0.00/0.00/8/1.69 | 0.13/0.01/0/0.53 | 0.27/0.00/0/0.56 |
| quiet L1+L5 | 2.40/5.10, 709 s | 0.00/0.00/0/4.68 | 0.02/0.00/0/1.45 | 0.00/0.00/0/3.08 |
| quiet L1+L5+SBAS | 1.21/2.56, 44 s | 0.00/0.00/0/1.14 | 0.21/0.01/0/0.71 | 0.29/0.00/0/1.14 |
| **quiet L1+L5+PPP converged** | 0.068/0.145, 813 s | **0.986/0.919/92/0.154** | **0.996/0.948/100/0.048** | **1.000/0.948/100/0.048** |
| quiet L1+L5+PPP not converged | 1.18/2.50, 104 s | 0.00/0.00/0/1.68 | 0.10/0.01/0/0.77 | 0.19/0.00/0/1.17 |
| RTK (v2 reference, 0.03 m, 30 s) | 0.03/0.03, 30 s | 0.979/0.965/100/0.031 | 0.999/0.995/100/0.026 | 1.000/1.000/100/0.026 |
| storm L1+SBAS | 4.64/9.84, 155 s | 0.00/0.00/0/7.74 | 0.01/0.00/0/4.07 | 0.00/0.00/0/6.55 |
| storm L1+L5+SBAS | 1.21/2.57, 45 s | 0.00/0.00/0/1.14 | 0.10/0.01/0/0.71 | 0.29/0.00/0/1.16 |
| storm L1+L5+PPP converged | 0.151/0.321, 265 s | 0.703/0.004/33/0.281 | 0.967/0.719/83/0.095 | 0.944/0.854/75/0.088 |

**Answer.** Without markers, the only configuration that reaches ≥ 90 % median coverage on all three templates is L1+L5 with converged PPP in quiet conditions (and the RTK reference). In the storm, converged PPP still passes perimeter_orbit and poi_close_pass, but lawnmower drops to 0.70. Its vertical bias of 0.32 m exceeds the 0.30 m coverage tolerance on flat ground. PPP that has not converged (the first ~20–40 min, GPS-only) behaves like SBAS, so PPP only helps a drone if the receiver has converged before take-off. SBAS and L5, alone or together, stay at the metre level, far from the 0.30 m tolerance.

## Caveats

- **Model scope.** This is an order-of-magnitude budget at one elevation (45°), one fixed DOP and one multipath environment. The variance-weighted τ is an approximation for several Gauss-Markov processes.
- **Assumed v1 values.** v1's broadcast orbit/clock values are about 4× above measured SISRE. Even at measured SISRE, L1-type configurations stay above 1 m per axis (demo sensitivity), so the answer does not change.
- **Iono-free noise.** The amplification, with no carrier smoothing, penalises L1+L5 and SBAS+L5. Their slow multipath is above 1 m either way.
- **Drone wrapper.** The v3 wrapper subclasses `Flight` and the SLAM in memory; the drone v3 files are not modified. The coverage tolerance (0.30 m) and the scenes are the v2/v3 ones.

- **Redaction:** the published copies of the original and v2 name Gyeonggi-do/Korea instead of a specific city and coordinates.

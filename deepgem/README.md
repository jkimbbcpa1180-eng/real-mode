<!-- SPDX-License-Identifier: CC0-1.0 -->
# DeepGem x STAR Drone — v7.4 (honest rebuild of v7.3.1)

By Grok (Grok Bot), fixing the original v7.3.1 design. `deepgem_star_v7_3_1.py` is the original, unchanged; `v7_4.patch` turns it into `deepgem_star_v7_4.py` (`patch deepgem_star_v7_3_1.py v7_4.patch`).

License: CC0 1.0 (public domain dedication). Standard library only, Python 3.9+.

v7.4 keeps v7.3.1's structure, function and class names, CLI flags, the
solar/wind chassis skin and the four flight regimes. What it changes is the
physics and the evaluation that v7.3.1 got wrong or never measured. It is a
**desk model and teaching artifact**. It is not production hardware and it is
not a flight controller. Only the power budget of the decision "brain" is
modeled. Flight/propulsion power is not modeled anywhere.

## Run

```
python3 deepgem_star_v7_4.py                 # default run (seed 101, 2400 steps)
python3 deepgem_star_v7_4.py --json          # same, as JSON
python3 deepgem_star_v7_4.py --burst-mw 1100 --burst-ms 40   # v7.3.1-sized burst
python3 -m unittest test_deepgem_star_v7_4 -v               # 59 tests
python3 make_demo_v7_4.py > demo_output_v7_4.txt            # sweeps + table (~30 s)
```

Kept flags: `--methane-kg --isotope --isotope-mg --carbon-loop --steps --seed --json`.
New flags: `--burst-mw --burst-ms`.

## v7.3.1 vs v7.4 (default runs, seed 101, 2400 steps)

Both columns come from actually running each file (see `demo_output_v7_4.txt`, section 6).

| Metric | v7.3.1 | v7.4 |
|---|---|---|
| Graphite H(1273)−H(298), kJ/mol (JANAF 17.922) | 13.521 | 17.922 |
| Net process heat, MJ/kg CH4 | 5.2198 (feed-based) | 6.2767 (products-based) |
| ΔH_rxn at T, MJ/kg CH4 (JANAF 91.809 kJ/mol) | not computed | 5.724 |
| SOFC efficiency | 0.65 (unlabeled) | 0.60 (ASSUMED, swept) |
| Electrical efficiency, % of CH4 LHV | 32.0429 | 28.2433 |
| Ni-63 μ/ρ in source, cm²/g (measured 1480) | 25.0 | 1480.0 |
| Source thickness, µm | 14.2248 (diamond density) | 5.6129 (nickel density) |
| Self-absorption factor | 0.940025 | 0.135053 |
| Ni-63 purity / specific activity, Ci/g | 1.0 implicit / 56.1351 | 0.18 / 10.1043 |
| Pair-creation energy, eV | 15.816 (Klein formula) | 13.1 (measured) |
| Voc, V | 0.6486 | 0.8537 |
| Betavoltaic output, µW | 3.18373 | 0.087592 |
| Beta fraction of harvest, % | 6.015e-05 | 1.6543e-06 |
| Harvest actually storable, % | not reported | 5.698e-05 |
| Inference burst, µJ | 44000 (1.1 W × 40 ms) | 6.0 (42 MACs) |
| ESR Ω / leakage W | 0.05 / 1.2e-7 | 5.0 / 3.0e-6 (+ DC-DC 0.85) |
| Brownouts | 296 (12.33%) | 0 (0.00%) |
| Schmitt transitions (default) | 0 (unreachable) | 0 (reachable; 365 at 44 mJ) |
| Reflex threshold / rate | 0.35 fixed / 100.00% | 0.2143 (1.5/K) / 5.33% |
| Label rule | int(7S)%7 hash | argmax W·(x−0.5), learnable |
| Drone accuracy (prequential) | 13.74% | 55.87% |
| Server accuracy (prequential) | 14.32% | 53.01% |
| Drone accuracy (held-out) | not measured | 67.37% |
| Server accuracy (held-out) | not measured | 52.21% |
| Bayes ceiling | ~14.3% (hash = chance) | 87.14% |
| Drone Brier (prequential) | 0.865806 | 0.643768 |
| Temperature | nudged toward top-prob 0.45 | fit on held-out NLL: 0.56 |
| Drone ECE before → after | not measured | 0.2386 → 0.0726 |

Chance accuracy is 1/7 = 14.29%. v7.3.1's classifiers score at chance because its label
is a hash of the features, which can't be learned.

## What changed and why

**Module 1: pyrolysis + SOFC (ground plant).**
- Graphite enthalpy now comes from the NIST-JANAF graphite table, interpolated, with
  H(298.15) = 0. v7.3.1's "graphite Shomate" set gives H(298) = −7.05 kJ/mol on its own.
  Even used as a delta from 298 K it is 24.6% low at 1273 K.
- H2 uses the correct NIST Shomate range (the 1000–2500 K set above 1000 K).
- The first-law balance is Q = ΔH298 + (products 298→T) − recuperated. It reports
  ΔH_rxn(T) = 91.8 kJ/mol, which matches JANAF 91.809.
- The SOFC efficiency is labeled ASSUMED and swept:
  - 0.45 → 21.18%
  - 0.50 → 23.54%
  - 0.55 → 25.89%
  - 0.60 → 28.24%
  - 0.65 → 30.60% electrical.

**Module 2: diamond betavoltaic.**
- Source thickness uses nickel density (8.908 g/cm³), not diamond density.
- New purity parameter. The default 0.18 gives about 10 Ci/g, matching commercial ">10 Ci/g" Ni-63.
- Uses the measured Ni-63 attenuation in nickel (1480 cm²/g). v7.3.1's 25 cm²/g overstated
  escaping power by about 7×.
- Half-space emission: one face radiates into one converter (factor 0.5).
- Pair energy is the measured 13.1 eV.
- Diode I0 = 2.7e-20 A/cm². With this value, a prototype-like cell (Isc 1.27 µA, 0.25 cm²)
  gives Voc 1.0196 V, against the published 1.02 V.
- Cross-check: this model's saturated escaping power for pure Ni-63 is 1.9584 µW/cm² per face,
  against a published Monte Carlo 3.28. The ratio is 0.597, so the model is conservative.
- Cell efficiency is 4.97%, in line with the 5–6% reported for the prototype.
- The carbon loop is now an explicit, documented no-op. Fossil CH4 carbon contains essentially
  no C-14, so the fraction is tracked for bookkeeping only.

**Module 3: storage.**
- Realistic ESR and leakage, a DC-DC efficiency of 0.85, and clipping when the cap is full.
  There are three presets: `ideal_low_esr` (v7.3.1's values), `typical`, and `coin_47mF`
  (a real 47 mF datasheet: ESR ≤120 Ω, 14 mA peak).
- The Schmitt hysteresis now really engages. A burst refused for lack of energy trips safe mode,
  and the brain re-arms only at 2.65 V. In v7.3.1 the lowest post-burst voltage was 2.001 V,
  so the guard never fired.
- The burst is parameterized and sized to the actual 42 MACs (default 12 mW × 0.5 ms = 6 µJ).
  `burst_feasible()` checks both stored energy and peak current. Burst sweep:

  | Burst, µJ | Feasible | Brownout rate | Halt rate | Schmitt transitions |
  |---|---|---|---|---|
  | 6 | yes | 0 | 0 | 0 |
  | 500 | yes | 0 | 0 | 0 |
  | 4,400 | yes | 0 | 0 | 0 |
  | 44,000 | yes | 7.62% | 13.33% | 365 |
  | 80,000 | **no** (needs ~0.19 J vs 0.17 J usable) | — | — | — |

  On `coin_47mF`, 44 mJ is infeasible because 0.4151 A exceeds the 14 mA rated peak.

**Module 4: classifier.**
- The label rule is documented and learnable: argmax of a fixed random linear map
  (seed 20240607) on the drone-visible features, with 15% uniform label noise.
- The Bayes ceiling is stated exactly: 1 − 0.15 + 0.15/7 = 87.14%.
- Held-out evaluation uses frozen models on fresh samples.
- Post-hoc temperature scaling is fit by NLL on the first half of the held-out set, and ECE
  (15 bins) is reported on the second half.
- The reflex threshold is relative to chance: 1.5/K = 0.2143.
- Results across seeds 101–105 (2400 training steps each):
  - Drone held-out: 67.93% ± 0.72%.
  - Server held-out: 54.21% ± 4.26%.
  - Drone ECE: 0.2492 → 0.0875 after scaling.
  - Server ECE: 0.1677 → 0.0866 after scaling.
- Accuracy vs training length (seed 101): the gap to the ceiling comes from limited data.

  | Training steps | Drone | Server |
  |---|---|---|
  | 2,400 | 67.37% | 52.21% |
  | 9,600 | 72.05% | 66.76% |
  | 24,000 | 77.99% | 70.86% |

## Assumed settings (not measured; change them to explore)

| Setting | Default | Note |
|---|---|---|
| SOFC electrical efficiency | 0.60 | swept 0.45–0.65 in the demo |
| Equilibrium conversion applied | 1.0 (off) | 98.21% is the equilibrium value at 1273 K |
| Ni-63 isotopic purity | 0.18 | about 10 Ci/g |
| Diode I0 | 2.7e-20 A/cm² | tuned to a published 1.02 V prototype |
| Charge collection | 0.90 | |
| C-14 μ/ρ | 812 cm²/g | Gleason formula extrapolated below its fit range |
| ESR / leakage / DC-DC | 5 Ω / 3 µW / 0.85 | the "typical" preset |
| Capacitance, thresholds | 0.05 F; v_low 2.0 V, re-arm 2.65 V, max 3.3 V | |
| Inference burst | 12 mW × 0.5 ms = 6 µJ | 42 MACs + softmax |
| Label weights / noise | gauss(0, 1.5), seed 20240607 / 0.15 | energy-reserve weight 0 |
| Reflex margin | 1.5 × (1/K) | |

## Honest limits

- **Flight power is not modeled.** The only load is the decision brain, so "0 brownouts" says
  nothing about whether a drone can fly on this skin.
- The simulator makes one decision per 360 s step. `TACTICAL_DT_S = 30` is used only in the
  beta-only night check.
- **Beta cannot carry the brain at night.** It delivers 0.074 µW usable, against 0.2 µW for
  the brain plus 3 µW leakage, a shortfall of 42.98×. Night survival comes from stored solar
  and perched wind. Beta is 1.65e-06% of the harvest.
- The cap is full most of the time. 3,888,515.78 J was clipped, and only 5.698e-05% of the
  harvest could actually be stored. The solar total is a gross figure, not usable energy.
- The betavoltaic model is about 0.6× the published saturation value because it ignores
  spectral hardening. I0 is fitted, not derived.
- Module 1 is a ground plant with a lumped balance: no kinetics, no catalyst, no carbon
  handling energy. Its electricity does not reach the drone.
- Labels are synthetic. The accuracies show the model can learn a learnable rule. They are
  not evidence of real-world target recognition.
- The server model lags the drone because it trains only on connected steps (29% link drops).
  Its seed-to-seed spread is larger (± 4.26%).
- **The source needs a license.** The default 25 mg source at purity 0.18 is 9.346 GBq
  (252.6 mCi). That is 25,261× the US exempt quantity (10 µCi, 10 CFR 30.71) and 93.5× the
  IAEA general exemption level (1e8 Bq). Flying it adds crash-dispersal risk. Korean rules
  were not checked.
- Wind totals differ from v7.3.1 (86,661.61 J) because random-number consumption changed.
  Solar (4,488,065.83 J) is identical.

## References (values used above)

- NIST Chemistry WebBook Shomate coefficients for CH4 and H2 (Chase 1998). NIST-JANAF tables
  for graphite, CH4, H2, H2O and CO2 (copies in `refs/`).
- ENSDF Ni-63 evaluation (NDS 196, 2024). NNDC C-14 data.
- Ni-63 attenuation in nickel, and the Monte Carlo saturation value of 3.28 µW/cm²:
  Belghachi et al., arXiv:1903.09098 (Table 1 cites Schweitzer 1952).
- Diamond pair-creation energy of about 13.1 eV: phys. stat. sol. (a), doi 10.1002/pssa.201600195.
- Ni-63 diamond betavoltaic prototype (Voc 1.02 V, Isc 1.27 µA, 0.25 cm², efficiency 5–6%):
  Bormashov et al., 2018.
- Abracon ADCV-S05R5SA473W supercapacitor datasheet (`refs/`).
- 10 CFR 30.71 Schedule B (`refs/`). IAEA GSR Part 3 exemption levels; IAEA RS-G-1.9
  D-values (`refs/`).

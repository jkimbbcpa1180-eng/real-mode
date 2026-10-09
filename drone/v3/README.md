<!-- SPDX-License-Identifier: CC0-1.0 -->
# drone v3

v3 builds on v2 (`../`). It imports the v2 modules and does not change them; `test_hygiene_v3`
checks the sha256 of the v2 files. Everything here is simulation. All sensor, power and
wind numbers are ASSUMED (typical values), not measured.

## v3 section: results (held-out seeds, frozen settings)

Each feature had one tuning pass on its tuning seed. The settings were then frozen and
one held-out run was done on a fresh seed. Exception: the swarm tuning table in `demo_output_v3.txt` is from v3.0 (one-sided
avoidance) and was not rerun for v3.1. The numbers below are from the held-out runs
(`mc_*_heldout_v3.json`); the tuning numbers are in `demo_output_v3.txt`.

| # | Feature | v2 baseline | v3 (held-out) | Verdict |
|---|---------|-------------|---------------|---------|
| 6 | Offline safe return (270 runs, wind 0/3/6 m/s) | 14.1 % success, landing err median 70.9 m, 3.0 % collisions | 95.9 % success, median 0.48 m, p90 1.85 m, max 265 m, 0 collisions | Large gain. Long tail at 6 m/s. |
| 4 | Look-closer re-pass (12 scenes x 3 templates) | coverage (std GPS) perimeter 0.621, poi 0.79, lawnmower 0.519 | 0.667 / 0.828 / 0.594 | Small gain. Repeating the whole template does better per minute. |
| 1 | Swarm, perimeter, n = 1 -> 4 (v3.1 reciprocal avoidance) | 295 s, coverage 0.604 | n2 193 s / 0.668, n4 142 s / 0.679; 0 contacts at every n, closest 1.06 m | Faster, no contacts. Many near misses at n4 (see failures). |
| 1 | Swarm, lawnmower, n = 1 -> 4 (v3.1) | 234 s, coverage 0.376 | n2 170 s / 0.533, n4 237 s / 0.512; 0 contacts, closest 1.03 m | n4 is **not** faster than n1 |
| 1 | Swarm on a fresh seed 20265001 (v3.1, frozen) | perimeter n1 228 s / 0.562, lawnmower n1 193 s / 0.485 | perimeter n4 116 s / 0.709, lawnmower n4 165 s / 0.488; 0 contacts at every n, closest 1.05 m | Confirms 0 contacts; avoidance off: 41 contact ticks (perimeter n4) |
| 1 | Relay (in flight, perimeter n4) | cloud coverage 0.329 at 60 s, 0.538 at 90 s (no relay) | 0.426 / 0.640 | Earlier data on the ground |
| 5 | Change detection, RTK | precision 0.50, recall 0.18 | precision 0.53, recall 0.18; 0.25 false alarms per no-change pair | Works for objects of 1 m or more (recall 1.0); objects of 0.5 m or less are never found |
| 5 | Change detection, standard GPS | precision 0.04, 17.5 false alarms per pair | precision 0.04, 19.1 false alarms per pair | **Not usable.** Alignment does not help. |
| 3 | Lidar + sonar, perimeter, standard GPS | coverage 0.64, endurance 43.8 min | coverage 0.744, endurance 24.7 min | +10 pts coverage, -44 % endurance. No gain with RTK or on lawnmower. |
| 2 | Thermal hot spots (RTK, 10 m standoff) | no sensor | vents recall 0.85, people 0.74, precision 0.73 | Glass reflections cause most false alarms. Indoor people are never seen. |

Seeds (tuning / held-out): safe return 20263006 / 20264006, re-pass 20263004 / 20264004,
swarm 20263001 / 20264001 (+ fresh 20265001 for v3.1), change 20263005 / 20264005, lidar 20263003 / 20264003,
thermal 20263002 / 20264002. The development scenes used seeds 1, 5, 9, 42, 123 and 321.
The demo uses seed 7.

## Failure cases (all seen in the runs above)

- **Safe return long tail.** 9 of 270 held-out runs landed 93–265 m away. All had 6 m/s
  wind, link loss at 30 % or 60 % of the mission, and the pad was never seen. The
  dead-reckoning error from wind and gusts grew larger than the search footprint, and the
  search itself was flown on the drifting estimate. Without an absolute reference (GNSS,
  a beacon, or a known landmark) this cannot be fixed; the search only bounds it. Sonar
  SLAM fixes did not help on held-out (95.9 % with SLAM vs 96.3 % without).
- **Swarm v3.0 (one-sided avoidance) had contacts; fixed in v3.1, re-tested.** v3.0 let the
  lower-priority drone hold and climb while the other never yielded: on held-out perimeter
  n4, 2 of 6 scenes had contacts (15 ticks, closest 0.06 m; kept in
  `mc_swarm_heldout_v3_0_onesided.json`). v3.1 (`reciprocal_avoidance`) splits the needed
  velocity change half/half between both drones (ORCA-like), drops the part of the
  nominal velocity that drives into the conflict, keeps right when exactly head-on, and
  splits vertically inside 3 m. It was developed on dev seed 123, then frozen and run
  once on the held-out seed and once on a never-used seed (20265001): **0 contacts at every
  n in both**; closest approach 1.03–1.06 m (n4). Without avoidance the fresh seed had 41
  contact ticks (perimeter n4).
- **But near misses (< 2 m) are high at n4**: 754 episodes on held-out lawnmower, 60 on
  held-out perimeter, about 320 on fresh perimeter and 238 on fresh lawnmower. The pairs
  chatter at the 3 m inner boundary, and the 4 m trigger on GPS estimates leaves only
  about 1 m of margin to truth. The cost is time: held-out lawnmower n4 took 236.8 s, no
  faster than n1 (234.1 s); n3 took 185.9 s. On fresh, lawnmower n4 (165 s) is slower
  than n3 (143 s). Perimeter n4 is still the fastest (142 s held-out, 116 s fresh). The
  avoidance uses a truth-clearance proxy for local obstacle sensing (ASSUMED) so it never
  pushes a drone within 1 m of a surface.
- **Re-pass can lower coverage under standard GPS.** On held-out, 5 of 36 runs dropped by
  more than 0.02 (worst -0.43); repeating the template dropped 5 of 36 (worst -0.33). The
  new passes are registered with a GPS error that has drifted since the first pass, so
  free-space carving removes cells that were correct. The demo scene shows this (0.722 ->
  0.545). No drops with RTK.
- **Re-pass no-fly.** The held-out run had 1 no-fly violation (a transit grazing the slab).
  It was fixed after the held-out run (`NOFLY_MARGIN_M` = 2 m) and checked only on that
  one case. That fix has no fresh held-out run.
- **Swarm alignment** made the observable-direction error worse on held-out perimeter
  (n2 0.16 -> 0.27 m, n4 0.38 -> 0.45 m). It helped on lawnmower (0.28 -> 0.15 m). Facades
  seen from one side constrain only 1–2 directions.
- **Change detection with standard GPS:** the GPS error drifts within a mission (time
  constant 30 s, sigma 0.5 m), so a single rigid offset cannot align two maps. Lawnmower
  maps also constrain only z.
- **Thermal:** at a 5 m standoff, recall for people is 0.2. The camera looks level at the
  facade, and people standing 1.5–6 m from the wall fall under or behind the field of view.
  At 40 m, vent recall falls to 0.29 because a 0.3–0.6 m vent fills less than a pixel.

## Physical limits

- Dead reckoning without an absolute reference drifts with the wind and gust error;
  calibrating the air data only slows the drift.
- Sonar range and beam width limit map detail. Under standard GPS, map error is set by the
  GPS error (median about 0.25–0.4 m), whichever sensor is used.
- A lidar on a 1.45 kg drone costs 0.6 kg and 10 W. With hover power scaling as m^1.5,
  that is about 109 W instead of 62 W, so endurance falls from 44 to 25 min. A 16-channel,
  ±15° lidar sees almost nothing directly below, so it does not help a lawnmower pattern.
  Dark or wet surfaces (dropout drawn U(0.05, 0.8)) reduce returns.
- LWIR does not pass through walls or glass. Glass reflects (emissivity about 0.1), so it
  shows the sky as cold or a warm object as a false hot spot. A 32x24 array dilutes
  targets smaller than a pixel footprint (55° / 32 px is about 3 cm/m of range).
- Simplifications: the thermal MC uses a kinematic orbit with GPS-like pose error, not the
  flight loop, and gets depth from the true surface + 0.1 m noise. The lidar is
  subsampled (24 azimuths x 16 channels per tick, max 120 points). Lidar-in-SLAM is not
  evaluated.

## Files and running

- Modules:
  - `common_v3.py`: shared setup and helpers.
  - `safe_return_v3.py`: feature 6.
  - `mission_v3.py`: `Flight` and `run_mission_v3` with the `repass`, `swarm_n` and `lidar` toggles.
  - `repass_v3.py`: feature 4.
  - `swarm_v3.py`: feature 1.
  - `change_v3.py`: feature 5.
  - `sensors_v3.py`: ray casting and the lidar (feature 3).
  - `thermal_v3.py`: feature 2.
- Monte Carlo runs: `python3 mc_<feature>_v3.py tuning|heldout`.
- Tests: `python3 -m unittest`. These include the flight-loop isolation test: flight outputs
  are bit-identical with the heavy mission work on or off.
- Demo and renders: `python3 make_demo_v3.py` writes `demo_output_v3.txt`,
  `out/safe_return_paths.png` and `out/swarm_merged_map.png`.
- Paste parts: `python3 make_parts_v3.py` writes `parts/v3_partN.md` and checks that they
  rebuild the sources byte-exact.

License: CC0 1.0 Universal (public domain).

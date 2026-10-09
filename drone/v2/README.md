# drone_core_v2

A fixed version of `drone_core_v1.py`. It addresses every issue in `review_v1.md` and keeps
v1's structure and names where they still make sense. Not pushed anywhere.

License: CC0 1.0 Universal (public domain). Dependencies: numpy only.

## Files
| File | What it is |
|---|---|
| `drone_core_v2.py` | The fixed core, now with the heading filter (self-tests + demo: `python3 drone_core_v2.py`) |
| `drone_uplink_v2.py` | Live telemetry stream, incremental cloud map (deltas + assembler), link profiles/failover, final map export |
| `drone_slam_v2.py` | Sonar SLAM (translation-only): keyframes, planar normals, point-to-plane registration with degeneracy handling, loop closure, least-squares back-end with the GPS error model, map re-integration |
| `drone_recon_v2.py` | Recon layer: range-dependent point covariance, 3-D evidence grid, PLY/PNG export, mission templates, coverage metric, magnetic disturbance log |
| `test_drone_core_v2.py`, `test_drone_uplink_v2.py`, `test_drone_recon_v2.py`, `test_drone_slam_v2.py` | Tests: `python3 -m unittest test_drone_core_v2 test_drone_uplink_v2 test_drone_recon_v2 test_drone_slam_v2` |
| `mc_v2.py` | Closed-loop Monte Carlo, v1 vs v2 + heading scenarios: `python3 mc_v2.py 300` (writes `mc_results_v2.json`) |
| `mc_recon_v2.py` | Mission-template Monte Carlo (coverage, accuracy, cloud uplink): `python3 mc_recon_v2.py 12` (writes `mc_recon_results_v2.json`) |
| `mc_slam_v2.py` | SLAM Monte Carlo, before/after on the same missions: `python3 mc_slam_v2.py final` (writes `mc_slam_results_v2.json`, log `mc_slam_run.log`) |
| `mc_table.py` | Prints the v1/v2 comparison table from `mc_results_v2.json` |
| `demo_output_v2.txt` | Self-tests, v1's assertions on v2, the v2 demo, v1 vs v2, heading/uplink/recon demos (`make_demo_v2.py`) |
| `recon_out/` | Demo exports: `recon_perimeter_orbit.ply`, `recon_poi_close_pass.ply` (x y z sigma rgb), `recon_render.png`; SLAM demo `recon_slam_before.ply/.png`, `recon_slam_after.ply/.png` |
| `v1_tests_on_v2.py` | v1's 11 self-test assertions run as written against v2 |
| `v2_part*.md`, `uplink_v2_part*.md`, `recon_v2_part*.md`, `slam_v2_part*.md` | Paste parts (`make_parts_v2.py`; joined, each set rebuilds its file exactly) |

## What changed, per review item
**Safety-critical**
- **S1 voxel gate.** The drone-distance gate is gone. The ranger now gates echoes on its own evidence: SNR,
  amplitude, range window, bearing jitter (azimuth scaled by cos el), and agreement in range between echoes
  (max spread 0.15 m; the earliest consistent cluster wins over late multipath).
- **S2 pose pulled by echoes.** `update_voxel` is removed. Echoes from unknown obstacles go into an `ObstacleMap`
  (points with sensor covariance, merged and expired). The pose is corrected only by ranges to **known** landmarks
  (`DragConsistentKF.update_landmark_range`, with H = -(l-p)^T/|l-p| and a chi-square gate). The Jacobian is
  checked numerically in the tests.
- **S3 IR dependency.** Acoustic groups are processed with or without IR. IR is paired by bearing (≤10°) and time
  (≤50 ms) and is used only as a cross-check.
- **S4 motor throttling.** Flight-critical loads (motors, avionics) and the sonar are never scaled. Only optional
  loads (IR) are shed. `mission_action` is RETURN_HOME at ≤25% charge and LAND at ≤15% (or when the horizon's
  critical energy exceeds what is left above the reserve). The budget is sum(power·duty)·horizon. The allocator
  doesn't drain or floor the battery: `BatteryState.consume()` records real or simulated energy and can go to 0.
- **S5 t_cpa.** For an obstacle at r relative to the drone, t_cpa = (r·v)/|v|², clamped ≥ 0, and
  d_cpa = |r − v·t_cpa|. This equals the reference −(r·v_rel)/|v_rel|² with v_rel = −v. The guidance also
  reports time to contact and stopping distance (v·t_react + v²/(2·a_plan)). Each obstacle limits the
  velocity toward it to the speed that can still stop within the free distance. The drone brakes when the
  closing speed exceeds that limit, using the deceleration needed (at least a_plan, at most a_emergency).
  There is also a speed limit tied to sensor range: an obstacle first seen at 15 m must still be stoppable,
  which gives 9.36 m/s. The repulsive "flare" field is removed.
- **S6 stale guidance position.** Guidance uses the latency-extrapolated position. Echoes are anchored at the
  filter position at their reception time and then expressed relative to "now".

**Correctness**
- **C1 tests.** The tests are honest (below), and the closed-loop Monte Carlo replaces the old open-loop one.
- **C2 trust.** Sigmas are reported (`pos_sigma_m`, `vel_sigma_mps`). `kinematic_tp = 1/(1+σp/1 m)` has no floor
  and is monotone: 0.91 at 0.1 m, 0.50 at 1 m, 0.25 at 3 m. It's a readable summary, not a calibrated
  probability. System trust is halved, and speed is limited to 2 m/s, when there has been no sonar data for
  more than 0.35 s.
- **C3 IR bias.** The bias corrects IR only. Its gate is |IR − acoustic| ≤ 0.15 m + 3σ (a larger residual is
  treated as a wrong pairing).
- **C4 penetrated_barrier.** Removed. v2 reports `acoustic_minus_ir_m` and `acoustic_ir_consistent` (3σ) instead.
- **C5 zero sigma.** Range σ is at least 0.02 m and bearing σ at least 2° (assumed sensor floors). Single echoes
  are flagged.
- **C6 Q.** F = expm(A·dt) and Q is the exact Van Loan discretization of white jerk (PSD q) through the drag
  model. The extra white-acceleration term is removed. With drag 0 it matches the standard white-jerk Q to 1e-9.
- **C7 sound speed.** Cramer (1993). It matches the reference values to 0.01 m/s (343.99, 347.30, 351.47 m/s).
- **C8 timing.**
  - Round-trip motion compensation: r = (c − v·u)·tof/2. The test error is under 5 mm, versus 5 cm uncorrected
    at 10 m/s.
  - Late telemetry packets are counted and not fused, and the filter clock never goes backwards.
  - Joseph-form update with solve().
  - IMU acceleration is now used as a measurement.
- **C9 power_tp.** Equals clip((soc − 15%)/(25% − 15%), 0, 1), so it drops as charge approaches the reserve.

**Cosmetic.**
- Docstrings now match the code. The IR gate no longer has unused arguments or the speed factor.
- The pulse interval docstring is accurate, and the interval is floored at the round-trip time at max range
  (88 ms at 15 m). v1 allowed 12.5 ms pings.
- The demo comment is fixed.
- Ultrasonic absorption is stated as an assumed, unverified ~1.3 dB/m at 40 kHz. It is informational only,
  because the measured SNR already includes absorption.

## Heading: magnetometer compass (new)
Heading has its own small filter, `HeadingEstimator`, with state [yaw, gyro-z bias]. Roll and pitch come from the
telemetry attitude. There is no full 3-D attitude estimator (see limits). Frame: ENU, body x forward / y left / z up,
yaw counter-clockwise from East, ZYX Euler (R = Rz(yaw) Ry(pitch) Rx(roll)). The compass heading is reported as
(90° − yaw) mod 360.
- **Predict.** The Euler yaw rate is yaw_dot = (q sinφ + (r − b) cosφ)/cosθ. The bias is a random walk. Q is built
  from the gyro ARW and the bias random walk.
- **Magnetometer update.**
  - Calibration: B_cal = soft_iron · (raw − hard_iron − k_motor · I_motor). The motor-current hook is k_motor,
    which defaults to 0 until it is bench-measured.
  - Tilt compensation: m = Ry(pitch) Rx(roll) B_cal; yaw = (π/2 − declination) − atan2(m_y, m_x); dip = atan2(−m_z, |m_xy|).
  - Gates, in order: |B| within 15% of the site field, dip within 10° of the site dip, then a chi-square
    innovation gate (6.63, 1 dof, 99%) on the innovation wrapped to (−π, π].
  - A rejected reading is counted by reason and not fused, so the yaw sigma keeps growing on gyro alone.
    Nothing ever fakes confidence.
- **Noise.**
  - White terms (sensor noise divided by the horizontal field, tilt error × tan(dip)) go into R.
  - The calibration floor (1°) doesn't average down. It is added to the reported sigma and to the gate, not to R.
  - Anomaly evidence: a |B| or dip deviation from the site model inflates R by (ΔB/B_h)² + Δdip².
- **Re-acquire.**
  - Before the first magnetic lock, 20 consecutive innovation rejections inflate the yaw variance by y². This
    covers a wrong initial yaw.
  - After lock this is not done, because a steady plausible distortion would drag the heading.
- **Landmarks.**
  - Bearings to known landmarks give yaw = θ_world − β_level. R = σ² + tilt² + (cross-track position variance)/distance².
  - They are applied before the magnetometer in each tick.
  - Six consecutive landmark rejections inflate the yaw variance and fuse that bearing. A magnetic lock on a
    distorted field is then the likelier culprit.
- **Tick output.** A top-level `heading` dict carries:
  - `yaw_enu_deg`, `compass_heading_deg`, `heading_sigma_deg`, `gyro_z_bias_dps`
  - `mag_status`, `mag_ok`, accepted/rejected counts (field / dip / innovation), `mag_field_ut`, `mag_dip_deg`
  - landmark accepted/rejected counts and re-acquire counters
- **No gyro in the telemetry.** The original v2 behaviour is kept and no heading sigma is claimed. That is why the
  original v1/v2 Monte Carlo rows are unchanged apart from the map merge gate (below).

## Heading fused into the drone's mapping and safety (new)
1. **Rays and covariance.**
   - Echo bearings are rotated with the fused heading.
   - Each obstacle point's covariance gains j jᵀ σψ², with j = z × (r·u). This gives a lateral sigma of about range × heading sigma.
   - The map merge gate is now Mahalanobis (chi-square 3 dof, 99%; points closer than 0.3 m always merge). A large
     lateral sigma therefore no longer merges points that are well separated in range. This slightly changed the
     v2 min-clearance numbers in the table below. Collisions are unchanged.
2. **Landmarks keep heading honest** while the magnetometer is rejected (tests and MC below).
3. **Safety.**
   - Heading sigma above 10° (ASSUMED) limits speed to 2 m/s, like the blind-sonar rule.
   - Trust is multiplied by heading_tp = 1/(1 + σψ/10°).
4. **Magnetic anomaly map: not built.**
   - `MagDisturbanceLog` (in `drone_recon_v2.py`) logs |B| or dip departures along the flight path as
     "possible wiring/steel nearby (field disturbance at drone position)".
   - It makes no claim about anything behind a wall and is not used for navigation.
   - It was not validated.

## Uplink: live stream and incremental cloud map (new, `drone_uplink_v2.py`)
The uplink sends only the drone's own data. There is no real network code and no interception of anyone else's signals.
- **Live stream.**
  - One compact JSON line per message: `{"crc": crc32(canonical body), "body": {v, seq, t, kind, pos, pos_sig, vel,
    yaw, yaw_sig, mag_ok, act, mis, near, blind, hdg_unc, soc, why}}`.
  - Periodic states are rate-limited (5 Hz, token bucket at 2000 B/s; ASSUMED).
  - During an outage a bounded state queue (300) drops the oldest states. Safety and action-change events go to a
    separate queue (2000) and are sent first.
  - `on_tick()` is O(1) and never raises.
- **Incremental cloud map (primary).**
  - During flight, `MapSync` sends deltas: the grid cells whose quantised value changed, plus new points. Each
    delta is versioned, sequence-numbered, CRC32-checked and zlib-compressed.
  - `CloudMapAssembler` (an in-memory fake) rebuilds the map. It is idempotent and order-tolerant: the highest
    version per cell wins, and points are id-keyed.
  - Each link has a bounded backlog (2 MB; ASSUMED). When it overflows, the oldest deltas are dropped and a
    snapshot resync goes out when a link is up. The snapshot has its own queue and is never dropped.
  - At mission end the cloud and drone map hashes are compared. A snapshot reconciles any difference.
  - The drone never has to come back to report.
- **Final map export.**
  - The `ObstacleMap` (points, heading-inflated covariances, timestamps) plus a flight summary is exported as
    canonical JSON with a SHA-256 checksum.
  - It is split into chunks with per-chunk SHA-256, so an interrupted upload resumes from the chunks the receiver
    already has.
- **Links.**
  - WiFi, LTE and satellite (Starlink-like) profiles. Their bandwidth, latency, outage statistics, WiFi range and
    power are ASSUMED placeholders, not measurements or vendor specs.
  - `LinkManager` picks the preferred link that is up for each message and fails over when it goes down.
  - Real WiFi range is limited (the WiFi-only rows below show it). Cellular/LTE or satellite is needed beyond it.
- **Satellite terminal power and mass.**
  - The terminal is an optional power load, `satellite_terminal` = 50 W (ASSUMED; "tens of watts").
  - It is shed like any optional load at the return-home threshold.
  - A satellite terminal is roughly a kilogram and draws tens of watts. It needs a larger airframe than the
    55 W-hover model assumed here.
- **Not modelled.**
  - **Satellite as positioning.** Satellite internet is not modelled as a positioning source, and there is no GPS
    improvement. Starlink provides no official positioning service. Signal-of-opportunity positioning is
    experimental research and is out of scope.
  - **Encryption and authentication.** Encryption and authentication of the real link are out of scope and must
    be added before real use. CRC32 catches accidental corruption only; it is not tamper protection.
- **Never blocks the flight loop.**
  - `process_flight_tick` never calls the uplink. `MapSync.collect/pump` run in the uplink worker.
  - In CPython the uplink worker step (collect + pump for four link configurations) took up to 0.77 s in the recon MC. A busy thread competes for
    the GIL, so on a real vehicle run it in a separate process.

## Recon: 3-D map, mission templates, coverage (new, `drone_recon_v2.py`)
- **Closer means more detail.**
  - Point covariance = range noise along the ray + (range × bearing σ)² across it + (range × heading σ)² about
    vertical + pose covariance. This is the same model as the core's obstacle points.
  - A sparse 3-D log-odds evidence grid (0.25 m cells). Free space is ray-cast along each beam up to just before
    the hit. The hit's evidence is spread over a disc of radius 2 × lateral σ, perpendicular to the ray.
  - Far hits are therefore smeared over many cells and close hits land in one.
  - Each cell keeps the smallest lateral σ that hit it, its "effective detail".
  - A beam without an echo adds no free-space evidence, because a soft absorber can't be ruled out.
- **Export.**
  - `write_ply` writes an ASCII PLY with x y z sigma rgb (green = small σ).
  - `render_png` uses matplotlib if it is installed, and otherwise skips with a note. matplotlib was installed
    on the build box for the render; the core stays numpy-only.
- **Mission templates.** Each one generates waypoints plus a sonar look direction from an area/box, a standoff and
  the beam half-angle. Swaths overlap by 30%.
  - `lawnmower`: nadir sweep over roofs and open ground. Lane = 2·standoff·tan(15°)·0.7. The standoff is reduced so
    the ground at the beam edge stays within 90% of sonar range.
  - `perimeter_orbit`: rounded-corner loops around a building footprint, looking at the nearest facade. Levels are
    spaced by the swath.
  - `poi_close_pass`: circles at a short standoff around a point of interest.
  - `interior_room_sweep`: authorized interior mapping, described next.
- **Authorized interior mapping.**
  - The drone flies inside a room and maps its walls, floor and ceiling with the same sonar grid. That includes
    pans across each corner, plus floor and ceiling lanes.
  - It refuses to plan unless `mission_config["interior_authorized"] is True`, which means the operator confirms
    the owner's permission.
  - It is not through-wall sensing. Airborne ultrasound reflects almost entirely at walls and gives surfaces only.
- **Coverage metric (sim only).**
  - Coverage is the fraction of true surface samples (0.25 m spacing) on the template's target surfaces that have
    a reconstructed point within 0.3 m (ASSUMED tolerance).
  - Points are the hits whose grid cell ended up occupied.
  - Accuracy is the distance from each such point to the nearest true surface. Error along a surface isn't counted.

## Assumed settings (not measured)
| Setting | Value |
|---|---|
| Range noise floor / bearing floor | 0.02 m / 2° |
| Max range spread within a group | 0.15 m |
| IR pairing window | 10°, 50 ms |
| Planning deceleration a_plan / emergency cap | 4.0 / 5.0 m/s² |
| Reaction time (ping 0.1 + latency 0.05 + tick 0.05 + actuator 0.1) | 0.30 s |
| Drone radius / standoff | 0.25 m / 1.0 m |
| Blind after / blind speed limit | 0.35 s / 2 m/s |
| Map: in-view timeout / out-of-view memory / merge gate / drift | 0.5 s / 3 s / 0.3 m / 0.2 m/s |
| Sonar half-FOV (map expiry only) / max range | 15° / 15 m |
| Jerk PSD q | 25 m²/s⁵ (set before the Monte Carlo, not tuned on it) |
| IMU accel sigma default | 0.3 m/s² |
| Kinematic trust reference sigma | 1.0 m |
| RETURN_HOME / LAND charge | 25% / 15% |
| Max plausible IR bias | 0.15 m |
| Ultrasonic absorption | 1.3 dB/m (unverified, informational) |
| Back-off inside standoff | 0.5 m/s per m, max 0.5 m/s |

**Heading / magnetometer (all ASSUMED)**

| Setting | Value |
|---|---|
| Gyro ARW / bias random walk / initial bias sigma | 3e-4 rad/s/√Hz / 2e-5 rad/s/√s / 0.02 rad/s |
| Initial heading sigma (telemetry yaw at start) | 30° |
| Site declination / field / dip (`MagCalibration`) | −8° / 50 µT / 53° (set per site) |
| Field gate / dip gate / innovation gate | ±15% / ±10° / chi-square 6.63 (1 dof, 99%) |
| Mag noise / tilt sigma / calibration floor | 0.5 µT / 0.5° / 1° (floor added to reported sigma, not R) |
| Hard iron / soft iron / motor coupling defaults | 0 / identity / 0 µT/A (bench calibration required) |
| Re-acquire after (mag, before first lock) / (landmarks) | 20 / 6 consecutive rejections |
| Heading-uncertain threshold / speed limit / trust reference | 10° / 2 m/s / 10° |

**Uplink (all ASSUMED placeholders, not measured, not vendor specs)**

| Setting | Value |
|---|---|
| Live stream budget / burst / state rate | 2000 B/s / 4000 B / 5 Hz |
| State queue / event queue / sends per flush | 300 / 2000 / 50 |
| Map delta: collect period / entries per message / backlog per link | 1 s / 400 / 2 MB |
| Quantisation (sync + hash) | log-odds 0.1, cell sigma 1 cm, points 1 mm |
| WiFi | 1 MB/s, 5 ms, outages mean up 120 s / down 1 s, range 100 m from the station, 1 W |
| LTE | 250 kB/s, 60 ms, mean up 60 s / down 4 s, 3 W |
| Satellite (Starlink-like) | 100 kB/s uplink, 50 ms, mean up 90 s / down 3 s, 50 W terminal (`SAT_TERMINAL_W`) |

**Recon (all ASSUMED)**

| Setting | Value |
|---|---|
| Grid cell / hit / miss / clamps / occupied threshold | 0.25 m / +0.85 / −0.40 / [−2, 3.5] / 0.8 |
| Hit footprint radius | 2 × lateral sigma, between 0.125 m and 1.0 m |
| Coverage tolerance | 0.3 m |
| Swath overlap | 30% |
| Lawnmower ground-range margin / minimum roof clearance | 90% of sonar range / 2 m |
| Magnetic disturbance log thresholds | ±10% field or ±5° dip |

**SLAM (`drone_slam_v2.py`, all ASSUMED; two values tuned on the tuning seed only)**

| Setting | Value |
|---|---|
| Keyframe length | 1 s |
| GPS error model used as the prior | Gauss–Markov, σ 0.5 m, τ 30 s (standard); σ 0.03 m for RTK |
| Normals | 12 nearest neighbours within 1.5 m; planar if thickness ≤ 0.08 m, λmin ≤ 0.1 λmid, λmid ≥ 0.1 λmax |
| ICP | 10 iterations, nearest-neighbour gate 1.5 → 0.5 m, point-to-plane gate 0.35 m, normals agree within 25°, Cauchy-like weights |
| Minimum matches (tuned) | 15 (was 25) |
| Plane roughness | 0.03 m |
| Effective independent matches per registration (tuned) | 15 (was 30) |
| Observable direction | eigenvalue ≥ 5% of the largest and implied σ ≤ 0.10 m |
| Consistency gate vs prediction | chi-square 16.27 (3 dof, 99.9%) |
| Scan / reference point caps, bounding-box margin | 250 / 6000 points, 2 m |

## Tests
102 tests, all passing (run with `python3 -m unittest test_drone_core_v2 test_drone_uplink_v2 test_drone_recon_v2 test_drone_slam_v2`):
60 core, 17 uplink, 16 recon and 9 SLAM. They include 22 subTest cases (16 closed-loop wall stops and 6 battery levels).

`test_drone_core_v2` (60):
- **TestGeometry** (4): quaternion, sound_speed_vs_cramer_reference, t_cpa_matches_analytic, t_cpa_simple_cases
- **TestHeading** (14): angle_wrap_across_180, converges_from_wrong_initial_yaw, distorted_field_rejected, euler_round_trip, gyro_bias_drift_corrected, hard_soft_iron_calibration, motor_current_compensation, no_gyro_keeps_v2_behaviour, plausible_distortion_after_lock_does_not_drag_heading, reported_sigma_includes_calibration_floor, sigma_grows_while_mag_rejected, stale_telemetry_does_not_update_heading, tick_reports_heading_and_rays_use_estimate, tilt_compensation_20deg
- **TestHeadingInPipeline** (6): field_deviation_inflates_mag_noise, heading_sigma_inflates_obstacle_covariance, heading_uncertainty_slows_and_lowers_trust, landmark_bearing_tilt_and_wrap, landmark_bearings_keep_heading_when_mag_rejected, landmarks_override_a_plausible_distorted_mag_lock
- **TestIR** (4): ir_bias_applies_to_ir_only, ir_bias_converges, ir_paired_by_bearing_and_time, no_penetrated_barrier_claim
- **TestObstaclesAndPose** (9): echo_round_trip_motion_compensation, far_obstacles_not_gated_by_drone_distance, guidance_uses_latency_extrapolated_position, landmark_jacobian_numeric, landmark_range_corrects_pose, landmark_via_tick, map_expires_and_merges, pose_not_pulled_by_obstacle_echoes, still_sees_obstacles_without_ir
- **TestPower** (4): budget_is_power_times_horizon_and_no_floor, hover_never_throttled, low_battery_actions, low_battery_in_tick
- **TestRanger** (5): bearing_jitter_rejected, low_snr_and_amplitude_rejected, multipath_rejected, single_echo_uses_noise_floor, two_late_one_early
- **TestStopsBeforeWall** (6): brake_trigger_at_stopping_distance, constrain_velocity_does_not_redirect_into_slide, speed_limited_by_sensor_range, stopping_distance_formula, stops_before_pole, stops_from_various_speeds_and_angles
- **TestTrustAndFilter** (5): Q_white_jerk_and_drag, late_telemetry_not_fused, pulse_interval_never_below_round_trip, system_trust_drops_when_blind, trust_monotone_over_gps_range
- **TestV1SelfTestsAndHygiene** (3): demo_runs, no_personal_names_and_cc0 (all new files included), v1_self_tests_ported

`test_drone_uplink_v2` (17):
- **TestStream** (7): crc_detects_corruption, file_transport, never_blocks_flight_loop, queue_keeps_safety_events_during_outage, rate_limit_and_bandwidth, round_trip, transport_errors_do_not_propagate
- **TestMapUpload** (3): chunked_resume, corrupted_chunk_rejected, export_round_trip_and_checksum
- **TestIncrementalCloudMap** (4): deltas_rebuild_identical_map, out_of_order_and_duplicate_deltas, outage_then_snapshot_resync, corrupted_map_frame_rejected
- **TestLinks** (3): link_failover, bandwidth_budget, satellite_terminal_is_optional_power_load

`test_drone_recon_v2` (16):
- **TestRangeResolution** (2): point_covariance_grows_with_range, core_obstacle_point_covariance_grows_with_range
- **TestEvidenceGrid** (3): free_space_carved_and_hit_occupied, no_hit_adds_no_evidence, closer_gives_finer_detail
- **TestExport** (2): ply_round_trip, png_render_or_honest_skip
- **TestMissionTemplates** (5): lawnmower_swaths_overlap, lawnmower_keeps_ground_in_sonar_range, perimeter_orbit_standoff_and_levels, poi_close_pass_looks_at_poi, interior_mapping_requires_authorization
- **TestSceneAndCoverage** (2): raycast_box_and_room, coverage_metric
- **TestMagDisturbanceLog** (2): logs_only_disturbances_with_honest_label, core_reports_field_for_the_log

`test_drone_slam_v2` (9):
- **TestRegistration** (5): recovers_known_offset (3 planes, offset 0.3/−0.2/0.15 m recovered within 3 cm),
  single_flat_wall_constrains_only_its_normal (along-track and vertical offsets left untouched),
  flat_wall_slam_adds_no_along_track_information (< 0.1% of the normal's information), too_few_points_is_rejected,
  line_of_points_is_not_planar
- **TestLoopClosure** (2): loop_closure_reduces_drift, rebuilt_map_is_on_the_true_surfaces
  - In loop_closure_reduces_drift, a steady drift of 0.012/−0.008/0.004 m/s over two rings around a 6×6×4 m box
    leaves a horizontal shape error (rms, common offset removed) of 0.052 m with loop closure, 0.175 m with
    odometry-only registration, and 0.207 m uncorrected.
- **TestIndependenceFromFlightLoop** (2): core_does_not_import_slam, flight_outputs_identical_with_slam_running

v1's 11 self-test assertions run as written against v2:
```
PASS  1 c_sound 25C/60% in (345,350)
PASS  2 zero-latency identity (tp > 0.4)  -- failed in v1 (trust clipped to 0.40); passes now
PASS  3 positive latency monotone
PASS  4 set_environment raises c
PASS  5 attitude round trip
PASS  6 multi-echo median
PASS  7 IR bias converges
N/A   8 voxel update moves position toward echo  -- INVERTED by design: update_voxel removed; v2 test asserts the pose does NOT move
N/A   9 gate rejects 1e6 m voxel  -- REPLACED: no drone-distance gate; v2 test asserts a 1e6 m echo is rejected by the ranger envelope and a valid 12 m echo is kept
N/A   10 power budget throttles motors  -- INVERTED by design: v2 test asserts motors granted 1.0, action LAND, battery not floored
FAIL  11 pulse interval shrinks (base 0.05 s)  -- CHANGED: 0.05 s is below the 88 ms round trip at 15 m, so v2 floors it there; v2 test uses base 0.2 s (shrinks) and asserts the floor
```

## Demo, same inputs as v1
| Field | v1 | v2 |
|---|---|---|
| action | TRACKING_NOMINAL | COLLISION_LIKELY_MAX_BRAKE |
| obstacle points kept | 0 (both gated out) | 2 |
| closest obstacle | none | 0.826 m from the latency-compensated position |
| time to CPA | none | 0.066 s |
| command accel (m/s²) | [0, 0, 0] | [-4.64, -1.86, -0.06] (capped at 5) |
| position used by guidance | [10.0, 5.0, -2.5] (42 ms stale) | [10.501, 5.034, -2.508] |
| kinematic trust | 0.543 | 0.977 (pos sigma 0.024 m) |
| penetrated_barrier | (no points) | field removed; acoustic−IR disagreement 0.267 m on the first echo (flagged inconsistent), 0.011 m on the second |
| motors_hover granted | 0.8 | 0.8 (never scaled) |

At 12 m/s, with obstacles 1.4–2.7 m away, v1 reported nominal tracking. v2 brakes as hard as it can and
flags that a collision is likely, because the obstacle is already inside the stopping distance. Full output:
`demo_output_v2.txt`. That file also contains:
- the heading demo (clean field vs steel nearby, with one landmark)
- a 60 s live stream with a 20 s outage
- the recon demo (perimeter orbit + POI pass on tuning scene 0, RTK-like)
- the SLAM demo (perimeter orbit, ordinary GPS, tuning scene 0): coverage 0.622 before → 0.886 after, navigation
  error 0.698 → 0.468 m, remaining datum offset 0.449 m, 0.976 with the datum removed (diagnostic)
- the interior authorization refusal

The recon demo writes `recon_out/recon_perimeter_orbit.ply`, `recon_out/recon_poi_close_pass.ply` and
`recon_out/recon_render.png`. The SLAM demo writes `recon_out/recon_slam_before.ply/.png` and
`recon_out/recon_slam_after.ply/.png`.

## Closed-loop Monte Carlo
Setup (`mc_v2.py`, seed 20261008):
- A point-mass drone with true linear drag (0.18/1.45 1/s), a first-order actuator (0.1 s) and a 5 m/s² limit
  flies toward a wall (70%) or a 0.3 m pole (30%) placed on its path 18–30 m ahead.
- Speed is 2–12 m/s and heading is ±30° from the wall normal.
- GPS σ is 0.5, 1 or 2 m (white). Velocity σ is 0.1 m/s and IMU σ is 0.3 m/s². Telemetry is 50 ms late.
- Sonar runs at 10 Hz with a 15° half-beam, 15 m range and 1.5 cm range noise. Echoes carry exact round-trip
  timing from the moving drone. Each echo has a 10% multipath chance, each ping a 5% chance of a spurious early
  low-SNR echo, and each ping a 10% chance of dropping out.
- IR (bias 3 cm) is supplied whenever the sonar has a hit. This is generous to v1, which drops sonar without IR.
- Both versions close the loop through the same controller: a velocity P-loop toward v0·heading (v2 uses its own
  `safe_velocity_mps`). Each version's `command_accel` is applied while its action is a brake or flare.
- Collision: the true clearance (distance minus 0.25 m body radius) reaches ≤ 0.
- Consistency is the 3-D position NEES of the "now" estimate against its reported covariance.

seed 20261008, 300 flights per variant, total runtime 585.2 s

| variant | collisions | min clearance median / p5 / min (m) | pos err median / p95 (m) | NEES mean (ideal 3) | NEES<=7.81 (ideal 0.95) | runtime s |
|---|---|---|---|---|---|---|
| v1 | 100.0% | -0.148 / -0.384 / -0.479 | 0.251 / 1.29 | 41.94 | 0.941 | 12.9 |
| v1_gate_off | 87.3% | -0.107 / -0.328 / -0.459 | 91.793 / 591.617 | 644982.52 | 0.292 | 19.2 |
| v2 | 0.0% | 0.977 / 0.745 / 0.44 | 0.172 / 0.722 | 3.03 | 0.947 | 44.2 |
| v2_no_ir | 0.0% | 0.977 / 0.745 / 0.44 | 0.172 / 0.722 | 3.03 | 0.947 | 42.7 |
| v2_gps_correlated_tau5s | 0.0% | 0.97 / 0.598 / 0.253 | 1.2 / 3.956 | 170.52 | 0.035 | 43.3 |
| holdout_seed_v1 | 100.0% | -0.15 / -0.452 / -0.539 | 0.264 / 1.259 | 50.16 | 0.939 | 6.1 |
| holdout_seed_v2 | 0.0% | 0.976 / 0.814 / 0.471 | 0.177 / 0.738 | 3.0 | 0.959 | 23.7 |

By GPS sigma (collisions; pos err median; NEES mean; NEES<=7.81):
- v1: 0.5 m: 100%, 0.146, 5.06, 0.974 | 1.0 m: 100%, 0.235, 17.04, 0.956 | 2.0 m: 100%, 0.525, 99.29, 0.896
- v1_gate_off: 0.5 m: 95%, 72.572, 603391.22, 0.327 | 1.0 m: 83%, 99.175, 662021.47, 0.277 | 2.0 m: 84%, 102.182, 664750.57, 0.276
- v2: 0.5 m: 0%, 0.099, 3.09, 0.955 | 1.0 m: 0%, 0.164, 2.92, 0.945 | 2.0 m: 0%, 0.33, 3.07, 0.942
- v2_no_ir: 0.5 m: 0%, 0.099, 3.09, 0.955 | 1.0 m: 0%, 0.164, 2.92, 0.945 | 2.0 m: 0%, 0.33, 3.07, 0.942
- v2_gps_correlated_tau5s: 0.5 m: 0%, 0.699, 144.36, 0.03 | 1.0 m: 0%, 1.283, 184.63, 0.029 | 2.0 m: 0%, 2.48, 181.81, 0.046
- holdout_seed_v1: 0.5 m: 100%, 0.139, 5.21, 0.978 | 1.0 m: 100%, 0.263, 18.1, 0.93 | 2.0 m: 100%, 0.493, 112.17, 0.914
- holdout_seed_v2: 0.5 m: 0%, 0.101, 3.08, 0.956 | 1.0 m: 0%, 0.171, 3.14, 0.947 | 2.0 m: 0%, 0.319, 2.83, 0.97

By speed and obstacle (collision rate):
- v1: v0<=9 100% (n=215), v0>9 100% (n=85), wall 100%, pole 100%
- v1_gate_off: v0<=9 83% (n=215), v0>9 98% (n=85), wall 94%, pole 67%
- v2: v0<=9 0% (n=215), v0>9 0% (n=85), wall 0%, pole 0%
- v2_no_ir: v0<=9 0% (n=215), v0>9 0% (n=85), wall 0%, pole 0%
- v2_gps_correlated_tau5s: v0<=9 0% (n=215), v0>9 0% (n=85), wall 0%, pole 0%
- holdout_seed_v1: v0<=9 100% (n=97), v0>9 100% (n=53), wall 100%, pole 100%
- holdout_seed_v2: v0<=9 0% (n=97), v0>9 0% (n=53), wall 0%, pole 0%

With 0 collisions in 450 v2 flights (300 + 150 held-out), the 95% upper bound on the collision rate in this sim is about 0.7% (rule of three).


What the table shows:
- **v1 collides in every flight.** Its distance gate keeps a return only when the obstacle is within a few
  filter sigmas of the drone (~1–2 m), which is too late to stop even at 2 m/s.
- **`v1_gate_off`** is a diagnostic only: v1 with the gate disabled. Obstacles then reach its pose filter
  (S2), and the position error grows to tens or hundreds of metres. Fixing the gate alone isn't enough.
- **v2 has no collisions,** including at 9–12 m/s, where the sensor-range speed limit applies. It stops
  about 1 m from the obstacle (the standoff).
- **Compared with the pre-heading run:** the v1 rows are identical. The v2 rows differ only in min clearance
  (v2 min 0.315 → 0.44 m; correlated-GPS min 0.302 → 0.253 m, NEES 171.03 → 170.52) because of the new
  Mahalanobis map merge gate. Collision counts didn't change.
- **v2 with no IR gives identical results.** IR is only a cross-check in v2. In this sim it adds no safety.
- **v2's NEES is close to ideal with white GPS noise.**
- **With time-correlated GPS (Gauss–Markov, τ = 5 s) v2 is badly overconfident** (see limits). Collisions
  stay at 0% because avoidance uses relative sonar geometry, not absolute position.


### Heading scenarios (v2 only; main seed 300 flights, held-out seed 150 flights, kept separate)
Same flights as above, plus the following (all ASSUMED).
- **Gyro:** z bias U(±1°/s); initial yaw error N(0, 5°), and the core is told 5°.
- **Attitude source:** roll/pitch error 0.5°. True tilt comes from the thrust acceleration.
- **Magnetometer:** hard iron N(0, 15 µT), calibrated to 0.3 µT; soft iron I + N(0, 0.03), calibrated to 0.005;
  motor coupling N(0, 0.3 µT/A) with a 10% compensation error; noise 0.5 µT.
- **Modes:**
  - `mag_off`: gyro only.
  - `mag_on`: magnetometer with the calibration errors above.
  - `mag_bursts`: random body-frame bursts of 10–40 µT lasting 0.5–3 s, starting with 1% chance per tick.
  - `mag_anomaly`: a world-fixed indoor-like field of 20 µT with a 10 m wavelength, random phases. It is present
    from the start, so it passes the |B|/dip gates much of the time.
  - `mag_anomaly_landmarks`: the same anomaly plus three known landmarks on the obstacle plane (−6, 0, +6 m
    lateral). Bearings at 5 Hz, 0.5° noise, ±60° field of view, occlusion not modelled.
- **Map error metrics:** the distance from each active map point to the true surface. "World" is as stored.
  "Relative" is re-anchored at the true drone position, which is what guidance uses.

| seed / mode | heading err median / p95 / max (deg) | final err median / p95 | heading NEES mean (ideal 1) | NEES≤3.84 (ideal 0.95) | mag accepted / rejected | landmark bearings acc / rej | ticks heading-uncertain | collisions | min clearance median / p5 / min (m) | map pt err world median / p95 (m) | map pt err relative median / p95 (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main / mag_off | 4.253 / 12.193 / 18.366 | 5.704 / 15.135 | 0.604 | 0.99 | 0 / 0 | 0 / 0 | 0.279 | 0.00% | 0.928 / 0.56 / 0.295 | 0.108 / 0.812 | 0.048 / 0.755 |
| main / mag_on | 0.675 / 2.083 / 4.854 | 0.576 / 1.902 | 0.925 | 0.955 | 62274 / 13 | 0 / 0 | 0.0 | 0.00% | 0.975 / 0.72 / 0.206 | 0.089 / 0.547 | 0.024 / 0.358 |
| main / mag_bursts | 0.712 / 2.16 / 28.352 | 0.668 / 1.969 | 1.005 | 0.958 | 47701 / 14586 | 0 / 0 | 0.0 | 0.00% | 0.975 / 0.598 / 0.304 | 0.088 / 0.535 | 0.024 / 0.346 |
| main / mag_anomaly | 9.834 / 43.83 / 77.679 | 18.628 / 55.534 | 44.38 | 0.493 | 8313 / 52487 | 0 / 0 | 0.032 | 4.67% | 0.899 / 0.018 / -0.126 | 0.186 / 2.417 | 0.145 / 2.381 |
| main / mag_anomaly_landmarks | 0.795 / 6.708 / 55.84 | 2.429 / 13.981 | 4.395 | 0.794 | 6528 / 55759 | 22238 / 108 | 0.001 | 0.00% | 0.975 / 0.706 / 0.326 | 0.085 / 0.56 | 0.026 / 0.367 |
| holdout / mag_off | 3.9 / 11.681 / 21.339 | 4.962 / 13.891 | 0.567 | 0.99 | 0 / 0 | 0 / 0 | 0.271 | 0.00% | 0.904 / 0.498 / 0.153 | 0.108 / 0.756 | 0.054 / 0.762 |
| holdout / mag_on | 0.792 / 2.167 / 12.722 | 0.675 / 1.895 | 1.106 | 0.95 | 30763 / 3 | 0 / 0 | 0.0 | 0.00% | 0.972 / 0.654 / 0.384 | 0.096 / 0.478 | 0.026 / 0.34 |
| holdout / mag_bursts | 0.896 / 2.557 / 39.118 | 0.8 / 2.26 | 1.335 | 0.935 | 23048 / 7718 | 0 / 0 | 0.0 | 0.00% | 0.967 / 0.699 / 0.409 | 0.097 / 0.525 | 0.028 / 0.374 |
| holdout / mag_anomaly | 11.013 / 45.54 / 76.233 | 22.159 / 58.44 | 53.558 | 0.48 | 3933 / 26748 | 0 / 0 | 0.049 | 0.67% | 0.873 / 0.261 / -0.043 | 0.179 / 1.924 | 0.136 / 1.888 |
| holdout / mag_anomaly_landmarks | 0.804 / 6.199 / 49.552 | 2.067 / 11.092 | 5.369 | 0.767 | 2584 / 28182 | 11071 / 39 | 0.0 | 0.00% | 0.975 / 0.764 / 0.567 | 0.093 / 0.472 | 0.026 / 0.35 |

Reference without heading estimation (yaw known): v2 map point error world 0.086 / 0.54 m,
relative 0.022 / 0.333 m (main seed); held-out world
0.094 / 0.466 m, relative 0.023 / 0.306 m. Heading MC runtime: total MC 585.2 s.

**Does heading affect collisions? Yes, when the heading is wrong and the filter doesn't know it.**
- `mag_anomaly` collided in 14 of 300 main-seed flights (4.67%) and 1 of 150 held-out flights (0.67%). The cause
  is a smooth distortion with plausible |B|/dip that is present at lock. It rotates echo bearings, so the wall
  appears off-path; relative map error p95 is about 2 m.
- With known landmarks, the same anomaly gave 0 collisions in both seeds.
- `mag_off`, `mag_on` and `mag_bursts` had 0 collisions. Their heading error is small, or (gyro-only) honestly
  reported, which triggers the slow-down 27–28% of the time.
- **Heading NEES.**
  - `mag_on` and `mag_bursts` are close to ideal: 0.93–1.34.
  - `mag_off` is conservative: about 0.6.
  - `mag_anomaly` is badly overconfident: 44–54.
  - `mag_anomaly_landmarks` is still overconfident: 4.4–5.4. Late in the flight the landmarks leave the ±60° field
    of view and the magnetometer drags the heading slowly within its gate (max error 50–56°).
- **Design changes made during this work (disclosed).** I changed the design after 10–12-flight smoke tests on
  main-seed scenarios:
  - mag re-acquire only before the first lock
  - calibration floor moved out of R into the reported sigma
  - landmarks applied before the magnetometer, with landmark re-acquire
  - anomaly-evidence R inflation
  - The held-out seed was not used for any of these. I also tried a "doubt" inflation of the reported sigma after
    repeated rejections. It turned 1–3 s bursts into 51° sigmas and a marginal wall contact in a smoke test, so
    it was removed.


### Recon mission templates (`mc_recon_v2.py`; tuning seed 20261009, 12 scenes; held-out seed 20261010, 12 scenes; runtime 698.3 s on 8 processes)
**Rerun with a sim fix (09:27–09:39 KST).** The telemetry velocity used to be the commanded 2 m/s even on ticks
where the drone stopped short at a waypoint, so the reported velocity disagreed with the true motion by 0.33–0.5
m/s on average in perimeter/POI missions. The core trusts velocity, so its position drifted (median error
1.1–1.5 m in perimeter/POI before the fix). The velocity is now the true displacement per tick, and every number below comes from
the rerun. Standard-GPS coverage went up, for example perimeter 0.349 → 0.605 on the tuning seed;
RTK and interior barely changed.

**Setup**
- **Scenes:** a building box (8–16 m × 8–16 m × 4–9 m), a point-of-interest column (0.8–1.6 m wide, 3–6 m tall)
  6–10 m away, and open ground. A separate room (5–10 × 4–8 × 2.6–3.5 m) is used for the authorized interior mode.
- **Flight:** the drone follows each template's waypoints at 2 m/s. drone_core_v2 estimates pose and heading every
  0.1 s from noisy telemetry.
  - GPS-like position error is Gauss–Markov with τ 30 s: 0.5 m (`gps_standard`) or 0.03 m (`gps_rtk`).
  - Indoors, an assumed positioning source gives 0.1 m with τ 10 s.
  - Gyro bias U(±1°/s), magnetometer on.
- **Sonar:** an assumed gimballed head with 5×5 jittered beams inside ±15°, at 10 Hz. Range 15 m, range noise
  1.5 cm, bearing noise 1°. 10% ping dropout, 10% per-beam miss, 5% multipath.
- **Standoffs:** lawnmower 6 m (reduced automatically to keep the ground in range), perimeter 4 m, POI 2 m,
  interior 1.2 m.
- **Targets:** lawnmower = roofs + open ground; perimeter = facades; POI = column sides; interior = walls, floor
  and ceiling.

| seed | template | position source | coverage median / p5 / min | missions ≥ 90% | point err median / p95 (m, medians over missions) | points within 0.3 m | effective detail (m) | nav pos err median (m) | heading err median (deg) | mission s | min distance to surfaces (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | lawnmower | gps_standard | 0.558 / 0.28 / 0.251 | 8% | 0.25 / 0.645 | 0.609 | 0.237 | 0.713 | 0.221 | 165.4 | 4.1 |
| main | lawnmower | gps_rtk | 0.976 / 0.959 / 0.953 | 100% | 0.032 / 0.108 | 1.0 | 0.229 | 0.05 | 0.221 | 165.4 | 4.1 |
| main | perimeter_orbit | gps_standard | 0.605 / 0.457 / 0.456 | 0% | 0.303 / 0.79 | 0.496 | 0.122 | 0.725 | 0.198 | 224.9 | 0.243 |
| main | perimeter_orbit | gps_rtk | 0.999 / 0.995 / 0.994 | 100% | 0.029 / 0.093 | 1.0 | 0.101 | 0.051 | 0.198 | 224.9 | 0.243 |
| main | poi_close_pass | gps_standard | 0.722 / 0.374 / 0.37 | 0% | 0.274 / 0.779 | 0.538 | 0.088 | 0.664 | 0.141 | 79.2 | 0.875 |
| main | poi_close_pass | gps_rtk | 1.0 / 0.999 / 0.998 | 100% | 0.03 / 0.085 | 1.0 | 0.056 | 0.053 | 0.141 | 79.2 | 0.875 |
| main | interior_room_authorized | indoor_pos | 0.993 / 0.983 / 0.978 | 100% | 0.072 / 0.18 | 0.998 | 0.043 | 0.162 | 0.145 | 138.1 | 1.2 |
| holdout | lawnmower | gps_standard | 0.386 / 0.157 / 0.155 | 0% | 0.412 / 0.801 | 0.412 | 0.232 | 0.695 | 0.127 | 176.7 | 4.393 |
| holdout | lawnmower | gps_rtk | 0.985 / 0.975 / 0.967 | 100% | 0.035 / 0.109 | 1.0 | 0.222 | 0.049 | 0.127 | 176.7 | 4.393 |
| holdout | perimeter_orbit | gps_standard | 0.591 / 0.448 / 0.409 | 0% | 0.302 / 0.875 | 0.496 | 0.122 | 0.811 | 0.315 | 189.9 | 0.492 |
| holdout | perimeter_orbit | gps_rtk | 0.998 / 0.994 / 0.994 | 100% | 0.029 / 0.097 | 1.0 | 0.101 | 0.054 | 0.315 | 189.9 | 0.492 |
| holdout | poi_close_pass | gps_standard | 0.684 / 0.402 / 0.363 | 33% | 0.337 / 0.703 | 0.443 | 0.089 | 0.694 | 0.165 | 67.0 | 0.875 |
| holdout | poi_close_pass | gps_rtk | 1.0 / 1.0 / 1.0 | 100% | 0.031 / 0.088 | 1.0 | 0.056 | 0.052 | 0.165 | 67.0 | 0.875 |
| holdout | interior_room_authorized | indoor_pos | 0.991 / 0.987 / 0.986 | 100% | 0.062 / 0.181 | 0.998 | 0.043 | 0.157 | 0.113 | 160.8 | 1.2 |

**Is the 90% goal reached (no SLAM)?**
- **With RTK-like positioning (0.03 m): yes, in every mission, on both seeds.** The worst single mission was
  0.953. Points had a median error of 0.029–0.035 m and a p95 of 0.085–0.109 m.
- **Indoors, the authorized interior mode reached ≥ 90% in every mission (worst 0.978)**, with the assumed 0.1 m
  indoor positioning source.
- **With standard GPS (0.5 m, τ 30 s): no.** Coverage medians were 0.386–0.722. Between 0% and 33% of missions
  reached 90% (33% only for held-out POI).
  - The pose estimate is off by 0.66–0.81 m (median), so surfaces are shifted by more than the 0.3 m tolerance.
  - The SLAM section below shows what map-based correction adds.
- **Closer means more detail.** The median effective detail (best lateral sigma per occupied cell) was:
  - lawnmower (4–6 m above the roofs, farther from the ground): 0.22–0.24 m
  - perimeter (4 m): 0.10–0.12 m
  - POI (2 m): 0.056–0.089 m
  - interior (1.2 m): 0.043 m
- **Disclosed design changes after the first full run.** The first run (before the velocity fix too) used the
  same seeds. It gave:
  - lawnmower RTK: 83% of missions ≥ 90% on both seeds (worst 0.21 tuning, 0.585 held-out)
  - interior: 0% ≥ 90% (median 0.856 tuning, 0.873 held-out)
  - Diagnosis on tuning-seed scenes only:
    - Lawnmower: the ground was beyond sonar range under tall roofs.
    - Interior: strips next to the room corners were never looked at.
  - Fixes: a range-limited lawnmower standoff, and interior corner pans.
  - I had seen that run's held-out numbers, so seed 20261010 is not fully blind for these two templates. The SLAM
    evaluation therefore uses a fresh seed.

**Cloud map during flight (incremental deltas; uplink is independent of the GPS mode, so RTK and interior rows are shown).**
"In cloud" means the cloud holds the identical quantised value at mission end, before any reconciliation.

| seed | template | links | occupied cells in cloud median / min | all cells | points | deltas kB/s | sent kB/s | deltas dropped | resyncs | failovers | hash equal before reconcile | hash equal after reconcile | radio energy per mission (Wh, ASSUMED power) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | lawnmower | wifi_only | 0.623 / 0.0 | 0.674 | 0.697 | 34.04 | 23.59 | 9347 | 0 | 0 | 0.083 | 1.0 | wifi 0.046 |
| main | lawnmower | lte_only | 0.991 / 0.949 | 0.994 | 0.996 | 34.04 | 33.85 | 0 | 0 | 0 | 0.0 | 1.0 | lte 0.138 |
| main | lawnmower | satellite_only | 0.981 / 0.973 | 0.987 | 0.994 | 34.04 | 33.83 | 0 | 0 | 0 | 0.0 | 1.0 | satellite 2.298 |
| main | lawnmower | wifi+lte+satellite | 0.988 / 0.973 | 0.992 | 0.995 | 34.04 | 33.83 | 0 | 0 | 48 | 0.083 | 1.0 | wifi 0.046, lte 0.138, satellite 2.298 |
| main | perimeter_orbit | wifi_only | 0.991 / 0.0 | 0.997 | 0.998 | 7.94 | 7.88 | 103 | 0 | 0 | 0.0 | 1.0 | wifi 0.062 |
| main | perimeter_orbit | lte_only | 0.993 / 0.976 | 0.998 | 0.999 | 7.94 | 7.93 | 0 | 0 | 0 | 0.0 | 1.0 | lte 0.187 |
| main | perimeter_orbit | satellite_only | 0.992 / 0.976 | 0.998 | 0.999 | 7.94 | 7.93 | 0 | 0 | 0 | 0.0 | 1.0 | satellite 3.124 |
| main | perimeter_orbit | wifi+lte+satellite | 0.993 / 0.976 | 0.998 | 0.999 | 7.94 | 7.93 | 0 | 0 | 54 | 0.0 | 1.0 | wifi 0.062, lte 0.187, satellite 3.124 |
| main | poi_close_pass | wifi_only | 0.442 / 0.0 | 0.495 | 0.496 | 4.99 | 2.17 | 0 | 0 | 0 | 0.0 | 1.0 | wifi 0.022 |
| main | poi_close_pass | lte_only | 0.946 / 0.885 | 0.996 | 0.994 | 4.99 | 4.97 | 0 | 0 | 0 | 0.0 | 1.0 | lte 0.066 |
| main | poi_close_pass | satellite_only | 0.946 / 0.885 | 0.996 | 0.994 | 4.99 | 4.97 | 0 | 0 | 0 | 0.0 | 1.0 | satellite 1.1 |
| main | poi_close_pass | wifi+lte+satellite | 0.946 / 0.885 | 0.996 | 0.994 | 4.99 | 4.97 | 0 | 0 | 14 | 0.0 | 1.0 | wifi 0.022, lte 0.066, satellite 1.1 |
| main | interior_room_authorized | wifi_only | 0.494 / 0.0 | 0.496 | 0.495 | 3.37 | 1.5 | 0 | 0 | 0 | 0.0 | 1.0 | wifi 0.038 |
| main | interior_room_authorized | lte_only | 0.992 / 0.91 | 0.995 | 0.996 | 3.37 | 3.3 | 0 | 0 | 0 | 0.0 | 1.0 | lte 0.115 |
| main | interior_room_authorized | satellite_only | 0.993 / 0.988 | 0.997 | 0.996 | 3.37 | 3.36 | 0 | 0 | 0 | 0.0 | 1.0 | satellite 1.918 |
| main | interior_room_authorized | wifi+lte+satellite | 0.993 / 0.988 | 0.997 | 0.996 | 3.37 | 3.36 | 0 | 0 | 39 | 0.0 | 1.0 | wifi 0.038, lte 0.115, satellite 1.918 |
| holdout | lawnmower | wifi_only | 0.933 / 0.0 | 0.963 | 0.983 | 32.25 | 25.57 | 9078 | 1 | 0 | 0.083 | 1.0 | wifi 0.049 |
| holdout | lawnmower | lte_only | 0.986 / 0.978 | 0.991 | 0.997 | 32.25 | 32.17 | 0 | 0 | 0 | 0.083 | 1.0 | lte 0.147 |
| holdout | lawnmower | satellite_only | 0.986 / 0.978 | 0.991 | 0.997 | 32.25 | 32.15 | 0 | 0 | 0 | 0.0 | 1.0 | satellite 2.454 |
| holdout | lawnmower | wifi+lte+satellite | 0.986 / 0.978 | 0.991 | 0.997 | 32.25 | 32.15 | 0 | 0 | 41 | 0.083 | 1.0 | wifi 0.049, lte 0.147, satellite 2.454 |
| holdout | perimeter_orbit | wifi_only | 0.883 / 0.0 | 0.935 | 0.943 | 7.92 | 7.56 | 0 | 0 | 0 | 0.083 | 1.0 | wifi 0.053 |
| holdout | perimeter_orbit | lte_only | 0.991 / 0.969 | 0.997 | 0.997 | 7.92 | 7.9 | 0 | 0 | 0 | 0.083 | 1.0 | lte 0.158 |
| holdout | perimeter_orbit | satellite_only | 0.994 / 0.969 | 0.998 | 0.998 | 7.92 | 7.91 | 0 | 0 | 0 | 0.083 | 1.0 | satellite 2.638 |
| holdout | perimeter_orbit | wifi+lte+satellite | 0.994 / 0.969 | 0.998 | 0.998 | 7.92 | 7.91 | 0 | 0 | 33 | 0.083 | 1.0 | wifi 0.053, lte 0.158, satellite 2.638 |
| holdout | poi_close_pass | wifi_only | 0.9 / 0.0 | 0.988 | 0.988 | 4.83 | 4.54 | 0 | 0 | 0 | 0.0 | 1.0 | wifi 0.019 |
| holdout | poi_close_pass | lte_only | 0.931 / 0.793 | 0.995 | 0.993 | 4.83 | 4.79 | 0 | 0 | 0 | 0.0 | 1.0 | lte 0.056 |
| holdout | poi_close_pass | satellite_only | 0.937 / 0.867 | 0.995 | 0.994 | 4.83 | 4.79 | 0 | 0 | 0 | 0.0 | 1.0 | satellite 0.931 |
| holdout | poi_close_pass | wifi+lte+satellite | 0.946 / 0.895 | 0.995 | 0.995 | 4.83 | 4.79 | 0 | 0 | 6 | 0.0 | 1.0 | wifi 0.019, lte 0.056, satellite 0.931 |
| holdout | interior_room_authorized | wifi_only | 0.493 / 0.0 | 0.492 | 0.491 | 3.24 | 1.59 | 0 | 0 | 0 | 0.0 | 1.0 | wifi 0.045 |
| holdout | interior_room_authorized | lte_only | 0.996 / 0.987 | 0.997 | 0.998 | 3.24 | 3.22 | 0 | 0 | 0 | 0.25 | 1.0 | lte 0.134 |
| holdout | interior_room_authorized | satellite_only | 0.996 / 0.987 | 0.997 | 0.998 | 3.24 | 3.22 | 0 | 0 | 0 | 0.25 | 1.0 | satellite 2.233 |
| holdout | interior_room_authorized | wifi+lte+satellite | 0.996 / 0.987 | 0.997 | 0.998 | 3.24 | 3.22 | 0 | 0 | 28 | 0.25 | 1.0 | wifi 0.045, lte 0.134, satellite 2.233 |

**What the uplink rows show**
- **LTE, satellite or combined links:** 93.1–99.6% (median) of the occupied map cells were already in the cloud
  when the mission ended. The rest is mostly the last second of deltas plus outages near the end.
- **WiFi only:** the station is 60–150 m away with a 100 m assumed range. Missions with the station out of range
  delivered 0% (min 0.0), and the medians range 0.442–0.991 depending on geometry. This is the honest case for
  needing LTE or satellite.
- **Failover:** the combined config fails over between links. There were 6–54 failovers per 12 missions, and
  nothing was dropped.
- **Hash match:** after the final reconciliation snapshot, the cloud hash equalled the drone hash in 100% of
  missions.
- **Satellite energy:** the assumed 50 W terminal used 0.93–3.12 Wh per mission. Arithmetic on the assumed
  numbers: +50 W on the assumed 60.5 W critical load (hover 55 + avionics 2 + sonar 3.5) is 83% more draw. Endurance
  would scale by about 0.55 unless the battery and airframe grow.
- **Delta bandwidth:** 3.2–34.0 kB/s, compressed. The lawnmower is the highest.

### Sonar SLAM (`mc_slam_v2.py`, `drone_slam_v2.py`; tuning seed 20261009, fresh held-out seed 20262009, 12 scenes each; runtime 791.6 s on 8 processes)
**What it does.** Translation-only SLAM on the mapping worker.
- **Keyframes:** one per second of sonar pings.
- **Normals:** local PCA normals; only planar points are used.
- **Registration:** point-to-plane ICP of each keyframe against all earlier keyframes. Revisits of earlier
  passes give loop-closure constraints.
  - Matches must agree in normal direction.
  - Degenerate directions are dropped: a single wall only constrains its normal.
- **Back-end:** a linear least-squares back-end over keyframe position corrections. The prior is the GPS error
  model (Gauss–Markov, zero mean).
- **Outputs:** an online estimate per keyframe, a smoothed estimate at mission end, and a map re-integrated with
  the smoothed poses.
- **Not in the flight loop.** It never runs inside `process_flight_tick` and does not change the flight estimate.
  A test checks that flight outputs are identical with SLAM running.

**Before/after is paired.** Each mission produces the no-SLAM map and the SLAM maps from the same noise.
- **Tuning (tuning seed only):** `ICP_MIN_MATCHES` 25 → 15 and `SLAM_N_EFF` 30 → 15.
  - These were chosen from tuning-seed runs of: the start settings; min matches 15; min matches 15 with 8
    neighbours; the same with a 0.8 m neighbour radius; and min matches 15 with N_eff 15.
  - The neighbour variants were worse for perimeter shape.
  - Two early runs did not apply the neighbour settings because of a default-argument bug. They were discarded.
  - Before those runs, the normal-agreement gate and the relative "not a line" test were added. They came from
    synthetic unit-test scenes, not from MC seeds.
- **Fresh seed:** 20262009 was run once, at the end, with the settings frozen.
- **Other modes:** `gps_rtk` is a no-regression check. `gps_tau90` uses a true GPS correlation time of 90 s while
  SLAM still assumes 30 s (model mismatch).

Columns: coverage before is median / p5 / missions ≥ 90%; coverage after is the smoothed SLAM map, median / p5 /
missions ≥ 90%; online is the map built during flight; "datum removed" is a diagnostic that uses the truth (see
below).

| set | template | GPS | coverage before | coverage after SLAM | online | datum removed (diagnostic) / ≥ 90% | point err median / p95 (m) before → after | nav err median (m) before → after | NEES mean / frac ≤ 7.81 (smoothed) | core filter NEES (median of mission means) | remaining datum error median (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tuning | lawnmower | gps_standard | 0.558 / 0.28 / 8% | 0.941 / 0.469 / 67% | 0.842 | 0.968 / 100% | 0.25 / 0.645 → 0.114 / 0.2 | 0.713 → 0.603 (online 0.639) | 2.475 / 0.984 (online 2.538 / 0.985) | 118.504 | 0.399 |
| tuning | perimeter_orbit | gps_standard | 0.605 / 0.457 / 0% | 0.737 / 0.48 / 17% | 0.63 | 0.933 / 58% | 0.303 / 0.79 → 0.213 / 0.487 | 0.725 → 0.568 (online 0.644) | 3.588 / 0.911 (online 3.02 / 0.948) | 130.818 | 0.38 |
| tuning | poi_close_pass | gps_standard | 0.722 / 0.374 / 0% | 0.795 / 0.46 / 17% | 0.764 | 0.977 / 75% | 0.274 / 0.779 → 0.26 / 0.596 | 0.664 → 0.602 (online 0.604) | 3.981 / 0.869 (online 3.306 / 0.899) | 120.238 | 0.465 |
| tuning | lawnmower | gps_rtk | 0.976 / 0.959 / 100% | 0.976 / 0.963 / 100% | 0.982 | 0.976 / 100% | 0.032 / 0.108 → 0.031 / 0.106 | 0.05 → 0.048 (online 0.051) | 4.331 / 0.863 (online 4.528 / 0.85) | 15.745 | 0.023 |
| tuning | perimeter_orbit | gps_rtk | 0.999 / 0.995 / 100% | 0.999 / 0.995 / 100% | 0.999 | 0.999 / 100% | 0.029 / 0.093 → 0.022 / 0.084 | 0.051 → 0.041 (online 0.046) | 3.622 / 0.906 (online 3.692 / 0.891) | 14.89 | 0.022 |
| tuning | poi_close_pass | gps_rtk | 1.0 / 0.999 / 100% | 1.0 / 0.997 / 100% | 1.0 | 1.0 / 100% | 0.03 / 0.085 → 0.026 / 0.077 | 0.053 → 0.048 (online 0.048) | 4.575 / 0.836 (online 4.122 / 0.87) | 16.84 | 0.027 |
| tuning | lawnmower | gps_tau90 | 0.601 / 0.249 / 17% | 0.932 / 0.001 / 67% | 0.829 | 0.974 / 100% | 0.243 / 0.585 → 0.157 / 0.233 | 0.621 → 0.576 (online 0.583) | 3.456 / 0.937 (online 3.336 / 0.879) | 118.699 | 0.563 |
| tuning | perimeter_orbit | gps_tau90 | 0.599 / 0.371 / 0% | 0.592 / 0.239 / 8% | 0.636 | 0.956 / 83% | 0.304 / 0.745 → 0.262 / 0.524 | 0.744 → 0.598 (online 0.656) | 4.599 / 0.853 (online 3.382 / 0.936) | 128.11 | 0.549 |
| tuning | poi_close_pass | gps_tau90 | 0.679 / 0.287 / 8% | 0.702 / 0.321 / 17% | 0.629 | 0.992 / 83% | 0.275 / 0.67 → 0.291 / 0.53 | 0.705 → 0.65 (online 0.68) | 4.862 / 0.833 (online 3.913 / 0.889) | 105.73 | 0.617 |
| fresh | lawnmower | gps_standard | 0.436 / 0.272 / 0% | 0.936 / 0.105 / 67% | 0.742 | 0.972 / 100% | 0.331 / 0.683 → 0.17 / 0.248 | 0.795 → 0.671 (online 0.663) | 3.63 / 0.911 (online 3.462 / 0.915) | 160.368 | 0.465 |
| fresh | perimeter_orbit | gps_standard | 0.621 / 0.37 / 0% | 0.66 / 0.417 / 33% | 0.686 | 0.831 / 33% | 0.282 / 0.765 → 0.248 / 0.667 | 0.671 → 0.562 (online 0.61) | 4.054 / 0.844 (online 3.097 / 0.912) | 104.833 | 0.336 |
| fresh | poi_close_pass | gps_standard | 0.724 / 0.546 / 0% | 0.805 / 0.599 / 17% | 0.742 | 0.975 / 92% | 0.226 / 0.652 → 0.196 / 0.623 | 0.685 → 0.611 (online 0.605) | 2.552 / 0.979 (online 2.206 / 0.996) | 98.216 | 0.498 |
| fresh | lawnmower | gps_rtk | 0.98 / 0.966 / 100% | 0.98 / 0.968 / 100% | 0.983 | 0.98 / 100% | 0.033 / 0.106 → 0.032 / 0.104 | 0.055 → 0.052 (online 0.056) | 5.273 / 0.787 (online 5.357 / 0.788) | 19.65 | 0.027 |
| fresh | perimeter_orbit | gps_rtk | 0.998 / 0.995 / 100% | 0.998 / 0.995 / 100% | 0.998 | 0.998 / 100% | 0.028 / 0.101 → 0.022 / 0.09 | 0.047 → 0.039 (online 0.046) | 3.289 / 0.924 (online 3.388 / 0.919) | 12.617 | 0.017 |
| fresh | poi_close_pass | gps_rtk | 1.0 / 0.994 / 100% | 1.0 / 0.997 / 100% | 1.0 | 1.0 / 100% | 0.029 / 0.087 → 0.025 / 0.075 | 0.052 → 0.045 (online 0.047) | 3.485 / 0.934 (online 3.129 / 0.954) | 15.008 | 0.028 |
| fresh | lawnmower | gps_tau90 | 0.529 / 0.052 / 0% | 0.829 / 0.0 / 50% | 0.623 | 0.973 / 100% | 0.228 / 0.67 → 0.224 / 0.316 | 0.857 → 0.828 (online 0.844) | 4.952 / 0.807 (online 4.331 / 0.866) | 167.478 | 0.688 |
| fresh | perimeter_orbit | gps_tau90 | 0.614 / 0.242 / 0% | 0.641 / 0.063 / 8% | 0.625 | 0.965 / 75% | 0.29 / 0.838 → 0.24 / 0.532 | 0.675 → 0.622 (online 0.751) | 5.017 / 0.825 (online 3.715 / 0.897) | 104.722 | 0.489 |
| fresh | poi_close_pass | gps_tau90 | 0.696 / 0.483 / 0% | 0.708 / 0.54 / 33% | 0.694 | 1.0 / 100% | 0.234 / 0.678 → 0.249 / 0.568 | 0.686 → 0.636 (online 0.637) | 2.78 / 0.997 (online 2.593 / 1.0) | 94.076 | 0.624 |

**Fresh held-out, standard GPS, per mission (before → after SLAM):**
- `lawnmower`: 0.351→0.631, 0.33→0.189, 0.427→0.963, 0.445→0.934, 0.78→0.916, 0.777→0.969, 0.41→0.341, 0.477→0.965, 0.724→0.966, 0.394→0.938, 0.202→0.002, 0.587→0.963
- `perimeter_orbit`: 0.466→0.391, 0.603→0.727, 0.663→0.913, 0.602→0.439, 0.788→0.908, 0.469→0.556, 0.748→0.924, 0.684→0.996, 0.699→0.676, 0.639→0.637, 0.526→0.643, 0.251→0.503
- `poi_close_pass`: 0.562→0.587, 0.872→0.917, 0.667→0.788, 0.669→0.809, 0.797→0.832, 0.731→0.801, 0.78→0.984, 0.819→0.843, 0.58→0.705, 0.717→0.839, 0.756→0.737, 0.525→0.608

**Results on standard GPS**
- **Is the 90% goal reached with standard GPS? No.**
  - On the fresh held-out set, SLAM raised median coverage:
    - lawnmower: 0.436 → 0.936 (0% → 67% of missions ≥ 90%)
    - perimeter: 0.621 → 0.66 (0% → 33%)
    - POI: 0.724 → 0.805 (0% → 17%)
  - The p5 stays low (0.105 / 0.417 / 0.599).
  - On the tuning seed: lawnmower 0.558 → 0.941 (67%), perimeter 0.605 → 0.737 (17%), POI 0.722 → 0.795 (17%).
- **Why it stops short: a constant GPS datum offset can't be removed without a reference.**
  - SLAM fixes the map's shape: sonar registration constrains positions relative to the map.
  - Nothing in this sim tells it where the whole map sits in the world except GPS averaged over the mission.
  - With τ 30 s and 60–225 s missions, that average is only good to about 0.23–0.37 m per axis (smoothed datum
    sigma, from the solver).
  - The remaining common offset had a median of 0.34–0.50 m on the fresh set. Whole surfaces then shift past the
    0.3 m tolerance, for example fresh lawnmower mission 10: vertical datum error 0.475 m, coverage 0.002.
  - With that offset removed (a diagnostic that uses the truth, not achievable in the field) the fresh medians are
    0.972 / 0.831 / 0.975.
  - Reaching 90% in the true frame needs an absolute reference. Options: RTK/PPK, surveyed ground control points,
    a surveyed take-off point, or accepting a map frame tied to the take-off point.
- **Unobservable box dimensions (perimeter, POI).**
  - A facade only constrains its own normal. Opposite faces are never seen together, so the box's width and length
    along the direction of travel are not observable from translation-only sonar SLAM; only the GPS prior holds them.
  - That is why perimeter shape errors remain even with the datum removed (fresh median 0.831, only 33% ≥ 90%).
  - Failure cases on the fresh set:
    - Perimeter missions 0 and 3: 0.466 → 0.391 and 0.602 → 0.439. With the datum removed, 0.464 and 0.301: SLAM
      bent the shape.
    - Lawnmower missions 1, 6 and 10: 0.33 → 0.189, 0.41 → 0.341, 0.202 → 0.002. The shape was fine (datum removed
      ≥ 0.94); the vertical datum offset was the cause.
  - POI registrations are often rejected for too few planar matches (425 of 796 keyframes on the fresh set): the
    column is narrow.
- **Lawnmower only constrains height.** It sees flat roofs and ground. Every registration had exactly one
  observable direction (vertical), so horizontal position is not corrected at all. That matches the targets:
  horizontal surfaces only care about height.
- **Navigation error (median, smoothed):** fresh 0.795 → 0.671, 0.671 → 0.562, 0.685 → 0.611 m. The online
  (in-flight) estimate lands in between (fresh 0.663 / 0.61 / 0.605 m). It can only anchor to the GPS seen so
  far, and its map covers less (fresh medians 0.742 / 0.686 / 0.742).
- **Consistency.**
  - The core filter treats correlated GPS as white noise: mission-mean NEES 98–160, 0.3–2.7% of samples ≤ 7.81.
  - SLAM-corrected keyframe positions: smoothed NEES 2.48–4.05 on standard GPS (ideal 3), 84–98% ≤ 7.81. That is
    slightly overconfident on perimeter/POI.
  - Under the τ 90 s mismatch: NEES 2.78–5.02, and the remaining datum error grows to median 0.49–0.69 m.
- **RTK:** no regression. Coverage was unchanged (fresh 0.98 / 0.998 / 1.0, 100% ≥ 90%). Navigation error
  improved slightly (0.055 → 0.052, 0.047 → 0.039, 0.052 → 0.045 m), but NEES is 3.3–5.3 (overconfident on the
  lawnmower).
- **Effective detail is coarser after SLAM** (0.26–0.44 m vs 0.09–0.24 m). The map now carries each keyframe's
  honest position uncertainty relative to the map datum, where the core's covariance was far too small.

**Runtime (CPython, one process per mission, standard GPS)**

| set / template / GPS | keyframes | registrations | loop-closure constraints | rejected (few matches) | rejected (inconsistent) | observable dims 0/1/2/3 | keyframe step ms median / p95 max / max | online map integration per keyframe, max ms | end smoothing + map rebuild s median / max |
|---|---|---|---|---|---|---|---|---|---|
| tuning | lawnmower | gps_standard | 1996 | 1984 | 18436 | 0 | 0 | [0, 1982, 2, 0] | 17.681 / 33.1 / 62.3 | 282.4 | 10.642 / 13.46 |
| tuning | perimeter_orbit | gps_standard | 2574 | 2274 | 7915 | 288 | 0 | [0, 1961, 312, 1] | 12.714 / 29.4 / 47.0 | 89.8 | 9.597 / 13.29 |
| tuning | poi_close_pass | gps_standard | 946 | 506 | 1624 | 428 | 0 | [0, 355, 149, 2] | 6.292 / 11.1 / 14.0 | 62.7 | 2.718 / 3.72 |
| fresh | lawnmower | gps_standard | 2189 | 2177 | 19575 | 0 | 0 | [0, 2177, 0, 0] | 16.329 / 31.9 / 37.8 | 287.4 | 11.367 / 15.62 |
| fresh | perimeter_orbit | gps_standard | 2331 | 2088 | 7954 | 231 | 0 | [0, 1790, 297, 1] | 12.183 / 32.6 / 41.0 | 64.0 | 8.309 / 12.94 |
| fresh | poi_close_pass | gps_standard | 796 | 359 | 930 | 425 | 0 | [0, 258, 100, 1] | 5.065 / 11.0 / 14.0 | 77.7 | 1.972 / 3.92 |

The keyframe step (registration + solve) runs once per second, with a worst case of tens of ms in these runs.
- **Dense solve.** The solve is dense, with cost growing with the cube of the number of keyframes; at the 600 s
  mission cap that is 600 keyframes. Missions here are ≤ 240 s.
- **Map re-integration.** Re-integrating the map at mission end takes up to about 16 s in CPython (post-flight).

## Honest limits
- **Correlated GPS.** The filter assumes white GPS noise. Real GPS error is correlated over seconds. In that case
  v2's reported position sigma is several times too small (NEES ≈ 171 against an ideal 3). The fix
  would be GPS-bias states or decimating/inflating GPS updates. It isn't done here, so treat `pos_sigma_m` as
  optimistic on real GPS.
- **What the sim leaves out.** No aerodynamics beyond linear drag, no rotor wash, no attitude dynamics
  (yaw = heading, level flight), no wind. The sonar model is idealized: no beam pattern, no specular misses on
  oblique or smooth surfaces, no soft absorbers, and one return per ping (the nearest in-beam point). There is
  one static obstacle per flight, no moving obstacles, no IR physics, and no sensor clock skew.
- **Bearing is assumed.** The sonar is assumed to report bearing within the beam. A single-element
  transducer doesn't; it would need an array or several sensors.
- **Obstacles are points.** Walls are handled as sets of points. With the 15° forward beam, obstacles beside
  the drone are remembered for only 3 s (assumed). After that, a stopped drone can creep past a pole (min
  clearance in the sim 0.32 m, below the 1 m standoff but no contact).
- **Settings are assumed.** All the values in the table above are, including the braking capability. If the
  real vehicle brakes harder or softer than 4–5 m/s², the speed limit and standoff behaviour change accordingly.
- **Trust scores aren't calibrated.** `kinematic_tp`, `map_tp` and `system_tp` are readable summaries, not
  probabilities. Use the sigmas.
- **Power logic is threshold-based.** RETURN_HOME and LAND use fixed charge thresholds. There's no
  energy-to-home estimate, and the battery model is linear Wh, ignoring voltage sag and temperature.
- **Absorption is unverified.** The ultrasonic absorption coefficient wasn't checked against ISO 9613-1.
- **Heading.**
  - **Filter structure.** There is no full 3-D attitude estimator: roll/pitch come from telemetry, only the
    gyro-z bias is estimated, and the pose and heading filters are separate (the landmark bearing R accounts for
    position sigma conservatively).
  - **Indoor steel and rebar.** These produce smooth distortions with plausible |B| and dip. The magnetometer
    alone cannot detect them, and the reported sigma is then overconfident (NEES 44–54 in the anomaly MC). Use
    known landmarks, and treat a magnetometer indoors as unreliable.
  - **Landmark identity is assumed correct.** A misidentified landmark that is rejected 6 times in a row would
    re-acquire to it.
- **Magnetic sensing.** External magnetic sensing detects currents or steel only very close to the conductor. The
  disturbance log says "possible wiring/steel nearby" at the drone's position and nothing more. No magnetic
  anomaly map was built or validated.
- **No through-wall sensing.** Airborne ultrasound reflects almost entirely at walls and gives surfaces only, not
  interiors. Through-wall imaging is out of scope. The interior mode maps rooms only by flying inside them, and
  it must be used only with the owner's permission (`interior_authorized` flag).
- **Recon sim.**
  - **Sensor and motion model.** Kinematic path (no dynamics or wind); an assumed gimballed 5×5 multibeam sonar
    head (a random sample of a dense fan); fixed per-beam miss and multipath probabilities; no beam pattern and
    no specular loss on oblique faces.
  - **Scenes.** Axis-aligned boxes only.
  - **SLAM fixes shape, not the datum.** `drone_slam_v2` corrects the map's shape, but a constant GPS datum
    offset can't be removed without a reference (RTK/PPK, ground control points, a surveyed take-off point).
    With standard GPS the 90% goal at 0.3 m in the true frame is not reached (see the SLAM section).
  - **SLAM's own limits.** It is translation-only (heading from the magnetometer filter). It assumes the GPS error
    model; under a τ mismatch NEES rises to about 5.
    - Unobservable: box width/length along the direction of travel, and horizontal position over flat ground.
    - Narrow POI columns often give too few planar matches.
    - On 2 of 12 fresh perimeter missions it made the shape worse.
    - No point-to-plane ambiguity handling for repetitive structure beyond the gates.
    - Real sonar (specular misses, beam pattern) would give fewer planar points than this sim.
  - **Coverage metric.** It counts a sample as covered if any kept point lies within 0.3 m. Error along a surface
    is not counted.
- **Uplink.**
  - **Link profiles** are placeholders (not measured). Outages are random episodes plus a WiFi range circle; there
    is no terrain or occlusion and no real radio behaviour.
  - **Reconciliation** sends a full snapshot, where a real system would send only the missing entries.
  - **Not secure.** CRC32 is not tamper-proof. No encryption or authentication: both must be added before real use.
  - **Python worker.** The map-sync worker step took up to 0.77 s in CPython in the recon MC. Run it in a separate
    process on a vehicle.
- **Satellite.** It is a data link only. It provides no positioning and no GPS improvement here. Starlink offers no
  official positioning service, and signal-of-opportunity positioning is experimental research (out of scope). A
  terminal adds roughly a kilogram and tens of watts, so it needs a larger airframe.

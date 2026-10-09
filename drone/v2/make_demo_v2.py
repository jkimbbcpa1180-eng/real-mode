"""Writes demo_output_v2.txt: v2 self-tests, v2 demo, v1-vs-v2 on the same demo inputs,
heading, uplink, incremental cloud map and recon demos (writes recon_out/*.ply and .png). CC0."""
import contextlib
import io
import json
import math
import os
import time

import numpy as np

import drone_recon_v2 as rc
import drone_uplink_v2 as up
import mc_recon_v2 as mr

import drone_core_v2 as v2
import run_v1
import v1_tests_on_v2

KEYS = [("action", lambda o: o["flight_guidance"]["action"]),
        ("closest_obstacle_m", lambda o: o["flight_guidance"]["closest_obstacle_m"]),
        ("time_to_cpa_s", lambda o: o["flight_guidance"]["time_to_cpa_s"]),
        ("command_accel_mps2", lambda o: o["flight_guidance"]["command_accel_mps2"]),
        ("points kept", lambda o: len(o["spatial_cloud_3d"])),
        ("kinematic_tp", lambda o: o["predictive_telemetry"]["kinematic_tp"]),
        ("system_tp", lambda o: o["flight_guidance"]["system_tp"]),
        ("pos used by guidance", lambda o: o["predictive_telemetry"].get("post_fusion_pos",
                                                                        o["predictive_telemetry"]["extrapolated_pos"])),
        ("penetrated_barrier", lambda o: ([p.get("penetrated_barrier") for p in o["spatial_cloud_3d"]]
                                          if "kinematic_tp" in o["predictive_telemetry"] and "pos_sigma_m" not in o["predictive_telemetry"]
                                          else "field removed; acoustic_minus_ir_m reported")),
        ("motors_hover granted", lambda o: o["power_management"]["granted_fractions"]["motors_hover"])]

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print("=== python3 drone_core_v2.py: self-tests ===")
    v2._self_test()
    print("\n=== v1's 11 self-test assertions run literally against v2 ===")
    for name, r, note in v1_tests_on_v2.run():
        print(f"{r.split(' ')[0]:5s} {name}" + (f"  -- {note}" if note else ""))
    print("\n=== v2 demo (same inputs as the v1 demo; IR returns given bearings) ===")
    out2 = v2.demo()
    print(json.dumps(out2, indent=2))
    out1 = run_v1.demo()
    print("\n=== Same demo, v1 vs v2 (key fields) ===")
    print("Scenario: 12 m/s, echoes at ~2.7 m and ~1.4 m, telemetry 42 ms old.")
    print(f"{'field':28s} | {'v1':40s} | v2")
    for name, f in KEYS:
        print(f"{name:28s} | {str(f(out1)):40s} | {f(out2)}")
    print("\nv1 gated out both echoes (drone-distance gate) and reported TRACKING_NOMINAL at 12 m/s with an")
    print("obstacle ~1 m ahead of the extrapolated position. v2 keeps both, anchors them at the reception-time")
    print("pose, measures from the latency-compensated position and commands maximum braking (the obstacle")
    print("is already inside the stopping distance, so it also flags COLLISION_LIKELY).")

    # ------------------------------------------------------------------ heading
    print("\n=== Heading: gyro (1 deg/s bias) + magnetometer + one known landmark, 20 s ===")
    cal = v2.MagCalibration()
    D, I, F = cal.declination_rad, cal.dip_rad, cal.field_ut
    bw = F * np.array([math.cos(I) * math.sin(D), math.cos(I) * math.cos(D), -math.sin(I)])
    yaw = math.radians(30.0)
    R = v2.Attitude.from_euler(0, 0, yaw).to_matrix()
    lm = np.array([40.0, 25.0, 0.0])
    rng = np.random.default_rng(1)
    for label, steel in (("clean field", np.zeros(3)), ("steel nearby (|B| +60%)", 30.0 * bw / np.linalg.norm(bw))):
        core = v2.AutonomousDroneCore(mag_calibration=cal, landmarks={"L": lm},
                                      heading_init_sigma_rad=math.radians(10.0))
        for k in range(400):
            tm = v2.AirframeState(0.05 * k, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                                  attitude=v2.Attitude.from_euler(0, 0, yaw + math.radians(8.0)),
                                  pos_sigma_m=np.full(3, 0.05))
            tm.gyro_rps = np.array([0, 0, math.radians(1.0)]) + rng.normal(0, 1e-3, 3)
            tm.mag_body_ut = R.T @ (bw + steel) + rng.normal(0, 0.5, 3)
            az = v2.wrap_pi(math.atan2(lm[1], lm[0]) - yaw) + rng.normal(0, math.radians(0.5))
            out = core.process_flight_tick(tm, 0.0, [], landmark_bearings=[("L", az, 0.0, math.radians(0.5))]
                                           if k % 4 == 0 else None)
        h = out["heading"]
        print(f"{label:26s} true yaw 30.0 deg (start error 8 deg) -> yaw {h['yaw_enu_deg']} deg, sigma "
              f"{h['heading_sigma_deg']} deg, mag accepted/rejected {h['mag_accepted']}/{h['mag_rejected']}, "
              f"landmark bearings accepted {h['landmark_bearings_accepted']}, gyro bias est "
              f"{h['gyro_z_bias_dps']} deg/s, |B| {h['mag_field_ut']} uT")
    log = rc.MagDisturbanceLog(F, I)
    log.observe(out["predictive_telemetry"]["now_s"], np.zeros(3), h["mag_field_ut"], math.radians(h["mag_dip_deg"]))
    print("magnetic disturbance log (flight path only):", json.dumps(log.events))

    # ------------------------------------------------------------------ live stream with an outage
    print("\n=== Live stream: 60 s at 20 Hz, link down from t=20 s to t=40 s ===")
    tr = up.InMemoryTransport()
    um = up.UplinkManager(tr)
    core = v2.AutonomousDroneCore()
    worst = 0.0
    for k in range(1200):
        t = 0.05 * k
        tr.up = not (20.0 <= t < 40.0)
        tm = v2.AirframeState(t, np.array([2.0 * t, 0, 5.0]), np.array([2.0, 0, 0]), np.zeros(3), np.zeros(3),
                              pos_sigma_m=np.full(3, 0.5))
        out = core.process_flight_tick(tm, 0.0, [], desired_velocity_mps=np.array([2.0, 0, 0]))
        if k in (300, 700):                       # two safety events, one during the outage
            out["flight_guidance"]["action"] = "COLLISION_AVOIDANCE_BRAKE"
        um.on_tick(out, t)
        um.flush(t)
    decoded = [up.decode_message(m) for m in tr.sent]
    ev = [m for m in decoded if m["kind"] == "event"]
    st = um.stats
    print(f"built {st['built']} msgs, sent {st['sent']} ({st['sent_bytes']} B, {st['sent_bytes'] / 60:.0f} B/s avg), "
          f"dropped non-critical states {st['dropped_state']}, dropped events {st['dropped_event']}, "
          f"max on_tick {st['max_on_tick_s'] * 1e3:.3f} ms")
    print(f"events delivered: {len(ev)} (reasons: {sorted(set(m['why'] for m in ev))}); "
          f"sequence numbers unique: {len(set(m['seq'] for m in decoded)) == len(decoded)} (events are sent "
          f"before queued states, so arrival order is not seq order; the receiver sorts by seq); still queued "
          f"at end: {um.queued()}")
    print("one state message:", tr.sent[-1].decode().strip()[:300])

    # ------------------------------------------------------------------ recon missions + cloud map
    os.makedirs("recon_out", exist_ok=True)
    sc = mr.make_scenes(1, np.random.default_rng(mr.SEED))[0]
    print("\n=== Recon: perimeter orbit (4 m standoff) + POI close pass (2 m), RTK-like GPS, main-seed scene 0 ===")
    allP, allS, paths = [], [], []
    for tpl in ("perimeter_orbit", "poi_close_pass"):
        r = mr.run_mission(sc, tpl, "gps_rtk", record=True)
        P, S = r["_map"].filtered_points()
        allP.append(P); allS.append(S); paths.append(r["_path"])
        n = rc.write_ply(f"recon_out/recon_{tpl}.ply", P, S, comment=f"{tpl}, sim scene 0, gps_rtk")
        print(f"{tpl}: {r['mission_s']} s, {r['n_points']} hits, {n} points kept (occupied cells) -> "
              f"recon_out/recon_{tpl}.ply; coverage of target within {rc.COVERAGE_TOL_M} m: {r['coverage']:.3f}; "
              f"point error median/p95 {r['acc_median']:.3f}/{r['acc_p95']:.3f} m; effective detail "
              f"(median best lateral sigma of occupied cells) {r['effective_detail_m']:.3f} m")
        for name, u in r["uplink"].items():
            print(f"   uplink {name:20s}: occupied cells in cloud at mission end {u['frac_occupied']:.3f}, "
                  f"points {u['frac_points']:.3f}, deltas {u['delta_Bps'] / 1000:.1f} kB/s, resyncs {u['resyncs']}, "
                  f"failovers {u['failovers']}, link up {u['link_up_frac']}, hash match after reconcile "
                  f"{u['hash_match_after_reconcile']}")
    res = rc.render_png("recon_out/recon_render.png", np.concatenate(allP), np.concatenate(allS),
                        boxes=[sc["building"], sc["poi"]], path_xyz=np.concatenate(paths),
                        title="Reconstructed points (colour = sigma), truth boxes black, flight path blue")
    print("render:", "recon_out/recon_render.png" if res["written"] else "skipped", "-", res["note"])

    print("\n=== Sonar SLAM: perimeter orbit, ordinary GPS (0.5 m, tau 30 s), main-seed scene 0 ===")
    r = mr.run_mission(sc, "perimeter_orbit", "gps_standard", links=(), record=True, slam=True)
    sl = r["slam"]
    for tag, rm, cov, acc_m, acc_p, nav in (
            ("before (no SLAM)", r["_map"], r["coverage"], r["acc_median"], r["acc_p95"], r["pos_err_median"]),
            ("after (SLAM, smoothed)", sl["_map"], sl["smoothed"]["coverage"], sl["smoothed"]["acc_median"],
             sl["smoothed"]["acc_p95"], sl["nav_err_smoothed_median"])):
        P, Sg = rm.filtered_points()
        fn = "recon_out/recon_slam_" + ("before" if tag.startswith("before") else "after")
        rc.write_ply(fn + ".ply", P, Sg, comment=f"perimeter_orbit, sim scene 0, gps_standard, {tag}")
        rr = rc.render_png(fn + ".png", P, Sg, boxes=[sc["building"]], path_xyz=r["_path"],
                           title=f"Perimeter orbit, ordinary GPS, {tag}: coverage {cov:.3f}")
        print(f"{tag:24s}: coverage {cov:.3f}, point error median/p95 {acc_m:.3f}/{acc_p:.3f} m, "
              f"navigation error median {nav:.3f} m -> {fn}.ply" + (f", {fn}.png" if rr["written"] else ""))
    print(f"SLAM: {sl['keyframes']} keyframes, {sl['registrations']} registrations, {sl['loop_factors']} "
          f"loop-closure constraints, observable dims per registration (0/1/2/3) {sl['n_obs_hist']}; "
          f"NEES mean online {sl['nees_online_mean']:.2f}, smoothed {sl['nees_smoothed_mean']:.2f} (3 = consistent; "
          f"core filter alone {sl['core_nees_mean']:.1f}); remaining common offset of the map (datum) "
          f"{sl['datum_err_m']:.3f} m, its 1-sigma per axis {sl['datum_sigma_m']} m; with that offset removed "
          f"(diagnostic, uses truth) coverage {sl['smoothed_datum_removed']['coverage']:.3f}; keyframe step "
          f"median/max {sl['step_ms_median']:.1f}/{sl['step_ms_max']:.1f} ms (worker, not the flight loop)")

    print("\n=== Interior mapping is refused without explicit authorization ===")
    try:
        rc.interior_room_sweep([0, 0, 0], [6, 5, 3], 1.2, math.radians(15), {})
    except rc.AuthorizationError as e:
        print("refused:", e)
    wps = rc.interior_room_sweep([0, 0, 0], [6, 5, 3], 1.2, math.radians(15), {"interior_authorized": True})
    print(f"with interior_authorized=True: {len(wps)} waypoints, all inside the room")
s = buf.getvalue()
print(s)
open("demo_output_v2.txt", "w").write(s)

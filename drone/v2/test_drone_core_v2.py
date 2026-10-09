"""
Tests for drone_core_v2.py. Run: python3 -m unittest test_drone_core_v2 -v
License: CC0 1.0 Universal (public domain).
"""
import math
import pathlib
import re
import unittest

import numpy as np

import drone_core_v2 as v2
from drone_core_v2 import (AcousticEchoGroup, AcousticRanger, AcousticSensorReturn, AirframeState,
                           Attitude, AutonomousDroneCore, BatteryState, DragConsistentKF,
                           InfraredReturn)

HERE = pathlib.Path(__file__).resolve().parent


def telem(t=0.0, p=(0, 0, 0), v=(0, 0, 0), sig=0.5, att=None):
    return AirframeState(t, np.array(p, float), np.array(v, float), np.zeros(3), np.zeros(3),
                         attitude=att or Attitude(), pos_sigma_m=np.full(3, sig))


def group(range_m, c, az=0.0, t=0.0, n=3, snr=25.0):
    return AcousticEchoGroup([AcousticSensorReturn(az, 0.0, 2 * range_m / c, 0.9, t, snr) for _ in range(n)])


class TestStopsBeforeWall(unittest.TestCase):
    """Closed loop through the same simulator as mc_v2 (truth drag dynamics, noisy sensors)."""

    def test_stops_from_various_speeds_and_angles(self):
        import mc_v2
        for v0 in (3.0, 6.0, 9.0, 12.0):
            for hdg in (0.0, 20.0):
                for gps in (0.5, 2.0):
                    sc = dict(id=0, kind="wall", v0=v0, heading=math.radians(hdg), dist=22.0,
                              pole_offset=0.0, gps_sigma=gps, temp_c=20.0, rh=50.0, seed=7)
                    r = mc_v2.run_flight(sc, "v2", record=True)
                    with self.subTest(v0=v0, hdg=hdg, gps=gps):
                        self.assertFalse(r["collided"])
                        self.assertGreater(r["min_clearance"], 0.2)
                        self.assertLess(r["trace"][-1]["speed"], 0.8)

    def test_stops_before_pole(self):
        import mc_v2
        sc = dict(id=0, kind="pole", v0=8.0, heading=0.1, dist=22.0, pole_offset=0.1,
                  gps_sigma=1.0, temp_c=20.0, rh=50.0, seed=11)
        r = mc_v2.run_flight(sc, "v2")
        self.assertFalse(r["collided"])

    def test_stopping_distance_formula(self):
        core = AutonomousDroneCore()
        v = 8.0
        self.assertAlmostEqual(core.stopping_distance_m(v), v * core.t_react + v * v / (2 * core.a_plan))
        vs = core.safe_speed_for_distance(10.0)
        self.assertAlmostEqual(core.stopping_distance_m(vs), 10.0, places=9)

    def test_brake_trigger_at_stopping_distance(self):
        core = AutonomousDroneCore()
        v = np.array([8.0, 0, 0])
        stop = core.stopping_distance_m(8.0)
        near = core.drone_radius + core.standoff + stop - 1.0
        far = core.drone_radius + core.standoff + stop + 1.0
        g_near = core._predictive_avoidance(v, [np.array([near, 0, 0])])
        g_far = core._predictive_avoidance(v, [np.array([far, 0, 0])])
        self.assertIn("BRAKE", g_near["action"])
        self.assertLess(g_near["a_cmd"][0], -core.a_plan + 1e-9)
        self.assertNotIn("BRAKE", g_far["action"])
        # moving away from an obstacle: no braking
        g_away = core._predictive_avoidance(-v, [np.array([near, 0, 0])])
        self.assertNotIn("BRAKE", g_away["action"])

    def test_speed_limited_by_sensor_range(self):
        core = AutonomousDroneCore()
        vmax = core.max_safe_speed()
        free = core.ranger.max_range_m - core.drone_radius - core.standoff
        self.assertAlmostEqual(core.stopping_distance_m(vmax), free, places=9)
        sv = core.constrain_velocity(np.array([20.0, 0, 0]), [], vmax)
        self.assertAlmostEqual(np.linalg.norm(sv), vmax)

    def test_constrain_velocity_does_not_redirect_into_slide(self):
        core = AutonomousDroneCore()
        u = np.array([math.cos(0.3), math.sin(0.3), 0.0])
        out = core.constrain_velocity(np.array([8.0, 0, 0]), [(u, 0.0)], 10.0)
        self.assertLess(np.linalg.norm(out), 1e-9)


class TestObstaclesAndPose(unittest.TestCase):
    def test_still_sees_obstacles_without_ir(self):
        core = AutonomousDroneCore()
        out = core.process_flight_tick(telem(), 0.0, [group(5.0, core.c_sound)])
        self.assertEqual(len(out["spatial_cloud_3d"]), 1)
        self.assertAlmostEqual(out["flight_guidance"]["closest_obstacle_m"], 5.0, delta=0.05)

    def test_pose_not_pulled_by_obstacle_echoes(self):
        a, b = AutonomousDroneCore(), AutonomousDroneCore()
        for k in range(20):
            t = 0.05 * k
            a.process_flight_tick(telem(t, sig=2.0), 0.0, [group(1.5, a.c_sound, t=t)])
            b.process_flight_tick(telem(t, sig=2.0), 0.0, [])
        np.testing.assert_allclose(a.kf.x, b.kf.x, atol=1e-12)
        np.testing.assert_allclose(a.kf.P, b.kf.P, atol=1e-12)

    def test_far_obstacles_not_gated_by_drone_distance(self):
        # v1 dropped echoes more than a few filter-sigmas from the drone.
        core = AutonomousDroneCore()
        out = core.process_flight_tick(telem(sig=0.05), 0.0, [group(12.0, core.c_sound)])
        self.assertEqual(len(out["spatial_cloud_3d"]), 1)

    def test_landmark_range_corrects_pose(self):
        lm = np.array([10.0, 0.0, 0.0])
        core = AutonomousDroneCore(landmarks={"L1": lm})
        core.compensate_latency(telem(sig=2.0, p=(1.5, 0, 0)), 0.0)   # truth is the origin
        err0 = abs(core.kf.x[0])
        for _ in range(5):
            core.kf.update_landmark_range(lm, 10.0, 0.05)
        self.assertLess(abs(core.kf.x[0]), 0.2 * err0)
        self.assertIsNone(core.kf.update_landmark_range(lm, 30.0, 0.05))   # gated outlier

    def test_landmark_jacobian_numeric(self):
        kf = DragConsistentKF(0.18, 1.45)
        kf.initialize(telem(p=(1.0, 2.0, 3.0)))
        lm = np.array([4.0, -1.0, 2.0])
        p = kf.x[0:3]
        H = -(lm - p) / np.linalg.norm(lm - p)
        eps = 1e-6
        num = [(np.linalg.norm(lm - (p + eps * e)) - np.linalg.norm(lm - (p - eps * e))) / (2 * eps)
               for e in np.eye(3)]
        np.testing.assert_allclose(H, num, atol=1e-8)

    def test_landmark_via_tick(self):
        core = AutonomousDroneCore(landmarks={"L1": [10.0, 0.0, 0.0]})
        core.process_flight_tick(telem(sig=2.0, p=(1.5, 0, 0)), 0.0, [])
        out = core.process_flight_tick(telem(0.05, sig=2.0, p=(1.5, 0, 0)), 0.0, [],
                                       landmark_ranges=[("L1", 10.0, 0.05), ("unknown", 3.0, 0.05)])
        self.assertEqual(out["predictive_telemetry"]["landmark_updates"], 1)
        self.assertLess(abs(out["predictive_telemetry"]["extrapolated_pos"][0]), 0.5)

    def test_echo_round_trip_motion_compensation(self):
        c = v2.sound_speed_cramer(20.0, 50.0)
        core = AutonomousDroneCore(temperature_c=20.0, humidity_pct=50.0)
        vx, d_recv = 10.0, 5.0                    # moving at 10 m/s toward a wall 5 m ahead at reception
        tof = 2 * d_recv / c
        for _ in range(20):                       # out path is from where we were at emission
            tof = (d_recv + vx * tof + d_recv) / c
        core.compensate_latency(AirframeState(0.0, np.zeros(3), np.array([vx, 0, 0]), np.zeros(3), np.zeros(3)), 0.0)
        est = core.ranger.estimate(AcousticEchoGroup([AcousticSensorReturn(0, 0, tof, 0.9, 0.0, 25)] * 3))
        vox = core.resolve_acousto_ir_voxel(est, Attitude(), 0.0)
        self.assertLess(abs(vox["acoustic_range_m"] - d_recv), 0.005)
        self.assertGreater(abs(vox["acoustic_range_raw_m"] - d_recv), 0.05)

    def test_guidance_uses_latency_extrapolated_position(self):
        core = AutonomousDroneCore()
        tau = 0.1
        tm = AirframeState(0.0, np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3), np.zeros(3),
                           pos_sigma_m=np.full(3, 0.05), vel_sigma_mps=np.full(3, 0.01))
        # obstacle at x = 8 m; echo received at now = 0.1 s, when the drone is at ~1 m
        g = AcousticEchoGroup([AcousticSensorReturn(0, 0, 2 * 7.0 / core.c_sound * (core.c_sound / (core.c_sound - 10.0)),
                                                    0.9, tau, 25)] * 3)
        out = core.process_flight_tick(tm, tau, [g])
        self.assertAlmostEqual(out["flight_guidance"]["closest_obstacle_m"], 7.0, delta=0.05)

    def test_map_expires_and_merges(self):
        m = v2.ObstacleMap(timeout_s=0.5)
        m.update(np.array([5.0, 0, 0]), np.eye(3) * 1e-4, 0.0)
        m.update(np.array([5.02, 0, 0]), np.eye(3) * 1e-4, 0.1)
        self.assertEqual(len(m.active(0.1)), 1)
        self.assertEqual(m.items[0].hits, 2)
        m.prune(0.7, in_view=lambda o: False)        # out of view: remembered (memory_s)
        self.assertEqual(len(m.active(0.7)), 1)
        m.prune(0.7, in_view=lambda o: True)         # in view and not re-seen: forgotten
        self.assertEqual(len(m.active(0.7)), 0)
        m.update(np.array([5.0, 0, 0]), np.eye(3) * 1e-4, 1.0)
        m.prune(1.0 + v2.MAP_MEMORY_S + 0.1, in_view=lambda o: False)
        self.assertEqual(len(m.active(5.0)), 0)


class TestPower(unittest.TestCase):
    def test_hover_never_throttled(self):
        for rem in (45.0, 12.0, 7.0, 3.0, 0.5, 0.0):
            core = AutonomousDroneCore(battery=BatteryState(45.0, rem))
            for h in (0.1, 1.0, 600.0):
                p = core.allocate_duty_cycle({"motors_hover": 1.0, "avionics": 1.0,
                                              "acoustic": 1.0, "infrared": 1.0}, h)
                with self.subTest(rem=rem, h=h):
                    self.assertEqual(p["granted_fractions"]["motors_hover"], 1.0)
                    self.assertEqual(p["granted_fractions"]["avionics"], 1.0)
                    self.assertEqual(p["granted_fractions"]["acoustic"], 1.0)
                    self.assertFalse(p["flight_critical_scaled"])

    def test_low_battery_actions(self):
        req = {"motors_hover": 1.0, "avionics": 1.0, "acoustic": 1.0, "infrared": 1.0}
        acts = {}
        for soc in (0.9, 0.30, 0.20, 0.10, 0.0):
            core = AutonomousDroneCore(battery=BatteryState(45.0, 45.0 * soc))
            p = core.allocate_duty_cycle(req, 0.1)
            acts[soc] = p
            self.assertEqual(core.battery.remaining_wh, 45.0 * soc)   # allocate does not drain
        self.assertEqual(acts[0.9]["mission_action"], "NOMINAL")
        self.assertEqual(acts[0.30]["mission_action"], "NOMINAL")
        self.assertEqual(acts[0.20]["mission_action"], "RETURN_HOME")
        self.assertIn("infrared", acts[0.20]["shed_loads"])
        self.assertEqual(acts[0.10]["mission_action"], "LAND")
        self.assertEqual(acts[0.0]["mission_action"], "LAND")
        # power trust reflects state (monotone in charge)
        tps = [acts[s]["power_tp"] for s in (0.0, 0.10, 0.20, 0.30, 0.9)]
        self.assertEqual(tps, sorted(tps))
        self.assertLess(acts[0.20]["power_tp"], 1.0)

    def test_budget_is_power_times_horizon_and_no_floor(self):
        core = AutonomousDroneCore()
        p = core.allocate_duty_cycle({"motors_hover": 0.8, "avionics": 1.0}, 2.0)
        expect = (55.0 * 0.8 + 2.0) * 2.0 / 3600.0
        self.assertAlmostEqual(p["energy_needed_wh"], round(expect, 5))
        b = BatteryState(45.0, 7.0)
        b.consume(5.0)
        self.assertAlmostEqual(b.remaining_wh, 2.0)        # below the 15% reserve: not floored
        b.consume(10.0)
        self.assertEqual(b.remaining_wh, 0.0)

    def test_low_battery_in_tick(self):
        core = AutonomousDroneCore(battery=BatteryState(45.0, 9.0))
        out = core.process_flight_tick(telem(), 0.0, [group(3.0, core.c_sound)],
                                       infrared_returns=[InfraredReturn(3.0)])
        self.assertEqual(out["flight_guidance"]["mission_action"], "RETURN_HOME")
        self.assertEqual(out["ranging_stats"]["n_ir_paired"], 0)    # IR shed
        self.assertEqual(len(out["spatial_cloud_3d"]), 1)           # sonar kept


class TestGeometry(unittest.TestCase):
    def test_t_cpa_matches_analytic(self):
        core = AutonomousDroneCore()
        rng = np.random.default_rng(1)
        for _ in range(200):
            r = rng.normal(0, 10, 3)
            v = rng.normal(0, 5, 3)
            g = core._predictive_avoidance(v, [r])
            # reference: relative velocity of the obstacle is -v
            vr = -v
            t_ref = max(0.0, -float(r @ vr) / float(vr @ vr))
            d_ref = float(np.linalg.norm(r + vr * t_ref))
            self.assertAlmostEqual(g["t_cpa"], t_ref, places=9)
            self.assertAlmostEqual(g["d_cpa"], d_ref, places=9)

    def test_t_cpa_simple_cases(self):
        core = AutonomousDroneCore()
        g = core._predictive_avoidance(np.array([2.0, 0, 0]), [np.array([10.0, 0, 0])])
        self.assertAlmostEqual(g["t_cpa"], 5.0)
        self.assertAlmostEqual(g["ttc"], (10.0 - core.drone_radius) / 2.0)
        g = core._predictive_avoidance(np.array([2.0, 0, 0]), [np.array([10.0, 3.0, 0])])
        self.assertAlmostEqual(g["t_cpa"], 5.0)
        self.assertAlmostEqual(g["d_cpa"], 3.0)
        g = core._predictive_avoidance(np.array([-2.0, 0, 0]), [np.array([10.0, 0, 0])])
        self.assertEqual(g["t_cpa"], 0.0)

    def test_quaternion(self):
        q = Attitude.from_yaw(math.pi / 2)
        np.testing.assert_allclose(q.rotate([1, 0, 0]), [0, 1, 0], atol=1e-12)
        rng = np.random.default_rng(2)
        for _ in range(50):
            w, x, y, z = rng.normal(size=4) * 3     # not unit: must be normalized
            qq = Attitude(w, x, y, z)
            vb = rng.normal(size=3)
            np.testing.assert_allclose(qq.rotate(vb), qq.to_matrix() @ vb, atol=1e-10)
            np.testing.assert_allclose(qq.inverse_rotate(qq.rotate(vb)), vb, atol=1e-10)
            R = qq.to_matrix()
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_sound_speed_vs_cramer_reference(self):
        ref = {(20, 50): 343.99, (25, 60): 347.30, (30, 100): 351.47,
               (20, 0): 343.36, (25, 0): 346.27, (30, 0): 349.15}
        for (t, rh), c in ref.items():
            self.assertAlmostEqual(v2.sound_speed_cramer(t, rh), c, delta=0.01)
        self.assertAlmostEqual(AutonomousDroneCore(temperature_c=25, humidity_pct=60).c_sound, 347.30, delta=0.01)


class TestRanger(unittest.TestCase):
    c = 343.0

    def e(self, r, snr=25.0, amp=0.9, az=0.0):
        return AcousticSensorReturn(az, 0.0, 2 * r / self.c, amp, 0.0, snr)

    def test_multipath_rejected(self):
        rg = AcousticRanger(self.c)
        est = rg.estimate(AcousticEchoGroup([self.e(4.0), self.e(4.01), self.e(9.5, snr=14, amp=0.4)]))
        self.assertAlmostEqual(est["range_m"], 4.005, delta=0.01)
        self.assertEqual(est["n_kept"], 2)

    def test_two_late_one_early(self):
        # median sits on the late (multipath) pair; the single early echo is kept only if consistent
        rg = AcousticRanger(self.c)
        est = rg.estimate(AcousticEchoGroup([self.e(3.0), self.e(6.0), self.e(6.02)]))
        self.assertAlmostEqual(est["range_m"], 6.01, delta=0.02)
        est = rg.estimate(AcousticEchoGroup([self.e(3.0), self.e(3.02), self.e(6.0)]))
        self.assertAlmostEqual(est["range_m"], 3.01, delta=0.02)

    def test_low_snr_and_amplitude_rejected(self):
        rg = AcousticRanger(self.c)
        self.assertIsNone(rg.estimate(AcousticEchoGroup([self.e(4.0, snr=5.0)])))
        self.assertIsNone(rg.estimate(AcousticEchoGroup([self.e(4.0, amp=0.05)])))
        self.assertIsNone(rg.estimate(AcousticEchoGroup([self.e(40.0)])))   # beyond max range

    def test_bearing_jitter_rejected(self):
        rg = AcousticRanger(self.c)
        est = rg.estimate(AcousticEchoGroup([self.e(4.0), self.e(4.0), self.e(4.0, az=math.radians(30))]))
        self.assertEqual(est["n_kept"], 2)

    def test_single_echo_uses_noise_floor(self):
        rg = AcousticRanger(self.c)
        est = rg.estimate(AcousticEchoGroup([self.e(4.0)]))
        self.assertTrue(est["single_echo"])
        self.assertEqual(est["range_sigma_m"], v2.RANGE_SIGMA_FLOOR_M)
        self.assertEqual(est["az_sigma_rad"], v2.ANGLE_SIGMA_FLOOR_RAD)


class TestTrustAndFilter(unittest.TestCase):
    def test_trust_monotone_over_gps_range(self):
        sig = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
        tps, sps = [], []
        for s in sig:
            core = AutonomousDroneCore()
            for k in range(10):
                _, _, tp = core.compensate_latency(telem(0.05 * k, sig=s), 0.0)
            tps.append(tp)
            sps.append(core.kf.sigmas()["pos_sigma_m"])
        self.assertTrue(all(a > b for a, b in zip(tps, tps[1:])), tps)
        self.assertTrue(all(a < b for a, b in zip(sps, sps[1:])), sps)
        self.assertGreater(tps[0] - tps[-1], 0.3)     # not saturated

    def test_system_trust_drops_when_blind(self):
        core = AutonomousDroneCore()
        out = None
        seen = core.process_flight_tick(telem(0.0), 0.0, [group(5.0, core.c_sound)])
        for k in range(1, 12):
            out = core.process_flight_tick(telem(0.05 * k), 0.0, [])
        self.assertTrue(out["flight_guidance"]["sonar_blind"])
        self.assertLess(out["flight_guidance"]["system_tp"], seen["flight_guidance"]["system_tp"])
        self.assertLessEqual(out["flight_guidance"]["speed_limit_mps"], v2.BLIND_SPEED_LIMIT_MPS)
        never = AutonomousDroneCore().process_flight_tick(telem(), 0.0, [])
        self.assertTrue(never["flight_guidance"]["sonar_blind"])

    def test_Q_white_jerk_and_drag(self):
        q, dt = 3.0, 0.1
        kf0 = DragConsistentKF(0.0, 1.0, jerk_psd=q)
        ref = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                            [dt**3 / 6, dt**2 / 2, dt]])
        np.testing.assert_allclose(kf0._Q(dt)[::3, ::3], ref, rtol=1e-9, atol=1e-15)
        kf = DragConsistentKF(0.18, 1.45, jerk_psd=q)
        Q = kf._Q(dt)
        self.assertGreaterEqual(np.linalg.eigvalsh(Q).min(), -1e-15)
        lam = 0.18 / 1.45
        F = kf._F(dt)
        self.assertAlmostEqual(F[3, 3], math.exp(-lam * dt), places=12)
        self.assertAlmostEqual(F[0, 3], (1 - math.exp(-lam * dt)) / lam, places=12)
        # Q with drag: velocity variance slightly below the no-drag value
        self.assertLess(Q[3, 3], ref[1, 1])

    def test_late_telemetry_not_fused(self):
        core = AutonomousDroneCore()
        core.compensate_latency(telem(1.0, p=(0, 0, 0)), 0.0)
        core.compensate_latency(telem(1.05, p=(0, 0, 0)), 0.0)
        x_before, t_before = core.kf.x.copy(), core._last_update_t
        core.compensate_latency(telem(0.9, p=(50, 0, 0)), 0.0)
        np.testing.assert_allclose(core.kf.x, x_before)
        self.assertEqual(core._last_update_t, t_before)
        self.assertEqual(core.stale_telemetry_dropped, 1)

    def test_pulse_interval_never_below_round_trip(self):
        core = AutonomousDroneCore()
        core.compensate_latency(telem(), 0.0)
        core.kf.P *= 1e4
        self.assertGreaterEqual(core.acoustic_pulse_interval_s(0.05), 2 * 15.0 / core.c_sound - 1e-12)


class TestIR(unittest.TestCase):
    def test_ir_bias_applies_to_ir_only(self):
        core = AutonomousDroneCore()
        core.ir_bias.bias_m = 0.10
        core.compensate_latency(telem(sig=0.05), 0.0)
        est = core.ranger.estimate(group(3.0, core.c_sound))
        vox = core.resolve_acousto_ir_voxel(est, Attitude(), 0.0, InfraredReturn(3.10))
        self.assertAlmostEqual(vox["acoustic_range_m"], 3.0, delta=0.005)
        self.assertAlmostEqual(vox["infrared_surface_corrected_m"], 3.0, delta=0.005)
        self.assertTrue(vox["acoustic_ir_consistent"])

    def test_no_penetrated_barrier_claim(self):
        out = v2.demo()
        for p in out["spatial_cloud_3d"]:
            self.assertNotIn("penetrated_barrier", p)

    def test_ir_paired_by_bearing_and_time(self):
        reps = [AcousticSensorReturn(math.radians(-30), 0, 0.01, 0.9, 1.0, 20),
                AcousticSensorReturn(math.radians(10), 0, 0.01, 0.9, 1.0, 20)]
        irs = [InfraredReturn(2.0, 1.0, azimuth_rad=math.radians(11))]
        pairs = AutonomousDroneCore._pair_ir(reps, irs)
        self.assertEqual(list(pairs), [1])
        late = [InfraredReturn(2.0, 1.5, azimuth_rad=math.radians(11))]
        self.assertEqual(AutonomousDroneCore._pair_ir(reps, late), {})

    def test_ir_bias_converges(self):
        est = v2.IRBiasEstimator(alpha0=0.2)
        for _ in range(200):
            est.update(2.0, 0.01, 2.08, 0.9)
        self.assertAlmostEqual(est.bias_m, 0.08, delta=0.01)
        before = est.bias_m
        est.update(2.0, 0.01, 3.0, 0.9)       # wrong pairing: ignored
        self.assertEqual(est.bias_m, before)


# ---------------------------------------------------------------------------
# Heading / magnetometer
# ---------------------------------------------------------------------------
def b_world(cal):
    """Earth field in ENU for the calibration's declination, dip and magnitude."""
    D, I, F = cal.declination_rad, cal.dip_rad, cal.field_ut
    return F * np.array([math.cos(I) * math.sin(D), math.cos(I) * math.cos(D), -math.sin(I)])


def mag_raw(cal, roll, pitch, yaw, S=None, h=None, k=None, current=0.0, dist=None, noise=0.0, rng=None):
    R = Attitude.from_euler(roll, pitch, yaw).to_matrix()
    b = R.T @ b_world(cal) + (np.zeros(3) if dist is None else dist)
    S = np.eye(3) if S is None else S
    out = S @ b + (np.zeros(3) if h is None else h) + (np.zeros(3) if k is None else k) * current
    if noise:
        out = out + rng.normal(0, noise, 3)
    return out


def run_heading(est, yaw_true, bias, n, dt=0.05, mag=True, rng=None, roll=0.0, pitch=0.0, dist=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(n):
        gyro = np.array([0.0, 0.0, bias]) + rng.normal(0, est.gyro_arw / math.sqrt(dt), 3)
        est.predict(dt, gyro, roll, pitch)
        if mag:
            est.update_mag(mag_raw(est.cal, roll, pitch, yaw_true, dist=dist, noise=0.5, rng=rng), roll, pitch)
    return est


class TestHeading(unittest.TestCase):
    def test_euler_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            r, p, y = rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-3.1, 3.1)
            q = Attitude.from_euler(r, p, y)
            np.testing.assert_allclose(q.to_euler(), (r, p, y), atol=1e-9)
            Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
            np.testing.assert_allclose(q.to_matrix(), Rz @ v2._rot_y(p) @ v2._rot_x(r), atol=1e-9)

    def test_converges_from_wrong_initial_yaw(self):
        yaw = math.radians(30.0)
        est = v2.HeadingEstimator()
        est.initialize(yaw + math.radians(90.0), math.pi, 0.0)      # wrong by 90 deg, honest sigma
        run_heading(est, yaw, 0.0, 40)
        self.assertLess(abs(math.degrees(v2.wrap_pi(est.yaw - yaw))), 2.0)
        # overconfident start (30 deg sigma, 90 deg error): gate rejects, then re-acquires
        est2 = v2.HeadingEstimator()
        est2.initialize(yaw + math.radians(90.0), math.radians(30.0), 0.0)
        run_heading(est2, yaw, 0.0, 80)
        self.assertGreaterEqual(est2.n_reacquire, 1)
        self.assertLess(abs(math.degrees(v2.wrap_pi(est2.yaw - yaw))), 2.0)

    def test_gyro_bias_drift_corrected(self):
        yaw, bias = math.radians(-40.0), math.radians(1.0)          # 1 deg/s gyro-z bias
        on = v2.HeadingEstimator()
        on.initialize(yaw, math.radians(2.0), 0.0)
        run_heading(on, yaw, bias, 1200)                              # 60 s at 20 Hz
        off = v2.HeadingEstimator()
        off.initialize(yaw, math.radians(2.0), 0.0)
        run_heading(off, yaw, bias, 1200, mag=False)
        err_on = abs(math.degrees(v2.wrap_pi(on.yaw - yaw)))
        err_off = abs(math.degrees(v2.wrap_pi(off.yaw - yaw)))
        self.assertLess(err_on, 1.5)
        self.assertGreater(err_off, 50.0)                            # ~60 deg of uncorrected drift
        self.assertAlmostEqual(math.degrees(on.x[1]), 1.0, delta=0.1)
        self.assertGreater(off.yaw_sigma, on.yaw_sigma)

    def test_distorted_field_rejected(self):
        est = v2.HeadingEstimator()
        est.initialize(0.0, math.radians(5.0), 0.0)
        x0, P0 = est.x.copy(), est.P.copy()
        steel = np.array([25.0, 15.0, -20.0])                         # |B| ~78 uT vs 50 expected
        out = est.update_mag(mag_raw(est.cal, 0, 0, 0.0, dist=steel), 0.0, 0.0)
        self.assertEqual(out["status"], "rejected_field")
        np.testing.assert_array_equal(est.x, x0)
        np.testing.assert_array_equal(est.P, P0)
        self.assertEqual((est.n_rejected, est.n_rejected_field, est.n_accepted), (1, 1, 0))
        # a smaller disturbance (|B| 14% low, inside the field tolerance) that still bends
        # the heading by tens of degrees is caught by the innovation gate instead
        out = est.update_mag(mag_raw(est.cal, 0, 0, 0.0, dist=np.array([20.0, -10.0, 5.0])), 0.0, 0.0)
        self.assertEqual(out["status"], "rejected_innovation")
        np.testing.assert_array_equal(est.x, x0)
        # same |B| but wrong dip (field rotated toward horizontal) -> dip gate
        cal = est.cal
        bw = b_world(cal)
        flat = np.array([0.0, 1.0, 0.0]) * np.linalg.norm(bw)
        out = est.update_mag(flat, 0.0, 0.0)
        self.assertEqual(out["status"], "rejected_dip")
        self.assertEqual(est.n_rejected_dip, 1)

    def test_motor_current_compensation(self):
        k = np.array([0.4, -0.3, 0.8])                                # uT per A
        yaw = math.radians(60.0)
        raw = mag_raw(v2.MagCalibration(), 0, 0, yaw, k=k, current=30.0)
        plain = v2.HeadingEstimator()
        plain.initialize(yaw, math.radians(5.0), 0.0)
        comp = v2.HeadingEstimator(v2.MagCalibration(motor_ut_per_a=k))
        comp.initialize(yaw, math.radians(5.0), 0.0)
        mh_plain = plain.mag_heading(plain.calibrate(raw, 30.0), 0, 0)
        mh_comp = comp.mag_heading(comp.calibrate(raw, 30.0), 0, 0)
        self.assertGreater(abs(math.degrees(v2.wrap_pi(mh_plain["yaw"] - yaw))), 10.0)
        self.assertLess(abs(math.degrees(v2.wrap_pi(mh_comp["yaw"] - yaw))), 1e-6)
        self.assertEqual(comp.update_mag(raw, 0, 0, 30.0)["status"], "accepted")

    def test_hard_soft_iron_calibration(self):
        rng = np.random.default_rng(5)
        S = np.eye(3) + rng.normal(0, 0.05, (3, 3))
        h = np.array([12.0, -7.0, 20.0])
        cal = v2.MagCalibration(hard_iron_ut=h, soft_iron=np.linalg.inv(S))
        est = v2.HeadingEstimator(cal)
        for yaw in np.radians([-170, -90, 0, 45, 135, 179]):
            mh = est.mag_heading(est.calibrate(mag_raw(cal, 0.1, -0.2, yaw, S=S, h=h)), 0.1, -0.2)
            self.assertLess(abs(v2.wrap_pi(mh["yaw"] - yaw)), 1e-9)
            self.assertAlmostEqual(mh["field_ut"], cal.field_ut, places=6)
            self.assertAlmostEqual(mh["dip_rad"], cal.dip_rad, places=9)

    def test_tilt_compensation_20deg(self):
        cal = v2.MagCalibration()
        est = v2.HeadingEstimator(cal)
        r, p = math.radians(20.0), math.radians(-20.0)
        worst_untilted = 0.0
        for yaw in np.radians(np.arange(-180, 180, 15)):
            b = mag_raw(cal, r, p, yaw)
            mh = est.mag_heading(b, r, p)
            self.assertLess(abs(v2.wrap_pi(mh["yaw"] - yaw)), 1e-9)
            worst_untilted = max(worst_untilted, abs(v2.wrap_pi(est.mag_heading(b, 0.0, 0.0)["yaw"] - yaw)))
        self.assertGreater(math.degrees(worst_untilted), 20.0)       # without tilt compensation

    def test_angle_wrap_across_180(self):
        est = v2.HeadingEstimator()
        est.initialize(math.radians(179.5), math.radians(3.0), 0.0)
        out = est.update_mag(mag_raw(est.cal, 0, 0, math.radians(-179.5)), 0.0, 0.0)
        self.assertEqual(out["status"], "accepted")
        self.assertAlmostEqual(out["innovation_deg"], 1.0, delta=0.05)
        self.assertTrue(-math.pi < est.yaw <= math.pi)
        self.assertLess(abs(math.degrees(v2.wrap_pi(est.yaw - math.radians(-179.5)))), 1.0)
        # gyro integration also wraps
        est2 = v2.HeadingEstimator()
        est2.initialize(math.radians(179.0), 0.01, 0.0)
        est2.predict(1.0, np.array([0.0, 0.0, math.radians(2.0)]), 0.0, 0.0)
        self.assertAlmostEqual(math.degrees(est2.yaw), -179.0, delta=1e-6)
        self.assertEqual(v2.wrap_pi(math.pi), math.pi)
        self.assertEqual(v2.wrap_pi(-math.pi), math.pi)

    def test_sigma_grows_while_mag_rejected(self):
        yaw = 0.3
        est = v2.HeadingEstimator()
        est.initialize(yaw, math.radians(5.0), 0.0)
        run_heading(est, yaw, math.radians(0.5), 200)
        sig = [est.yaw_sigma]
        acc0 = est.n_accepted
        steel = np.array([0.0, 0.0, 40.0])
        rng = np.random.default_rng(1)
        for _ in range(10):                       # 10 x 1 s of distorted readings
            run_heading(est, yaw, math.radians(0.5), 20, rng=rng, dist=steel)
            sig.append(est.yaw_sigma)
        self.assertEqual(est.n_accepted, acc0)
        self.assertEqual(est.n_rejected, 200)
        self.assertTrue(all(b > a for a, b in zip(sig, sig[1:])), sig)
        self.assertGreater(sig[-1], 1.5 * sig[0])

    def test_plausible_distortion_after_lock_does_not_drag_heading(self):
        # a distortion that rotates the field by 30 deg keeps |B| and dip plausible; after a
        # magnetic lock it must be rejected (innovation gate) without re-acquiring onto it
        yaw = math.radians(10.0)
        est = v2.HeadingEstimator()
        est.initialize(yaw, math.radians(5.0), 0.0)
        run_heading(est, yaw, 0.0, 100)
        acc0 = est.n_accepted
        rng = np.random.default_rng(2)
        for _ in range(200):                                          # 10 s
            est.predict(0.05, rng.normal(0, est.gyro_arw / math.sqrt(0.05), 3), 0.0, 0.0)
            raw = mag_raw(est.cal, 0, 0, yaw + math.radians(30.0), noise=0.5, rng=rng)
            self.assertEqual(est.update_mag(raw, 0.0, 0.0)["status"], "rejected_innovation")
        self.assertEqual(est.n_accepted, acc0)
        self.assertEqual(est.n_reacquire, 0)
        self.assertLess(abs(math.degrees(v2.wrap_pi(est.yaw - yaw))), 2.0)

    def test_reported_sigma_includes_calibration_floor(self):
        est = v2.HeadingEstimator()
        est.initialize(0.2, math.radians(5.0), 0.0)
        run_heading(est, 0.2, 0.0, 400)
        _, sig = est.yaw_at(0.0)
        self.assertGreaterEqual(sig, est.cal.heading_floor_rad)
        self.assertLess(est.yaw_sigma, est.cal.heading_floor_rad)    # filter sigma alone would claim less

    def test_tick_reports_heading_and_rays_use_estimate(self):
        cal = v2.MagCalibration()
        core = AutonomousDroneCore(mag_calibration=cal, heading_init_sigma_rad=math.pi)
        yaw_true = math.radians(90.0)                     # facing north (+y)
        wrong = Attitude.from_yaw(0.0)                    # telemetry yaw says east
        out = None
        for k in range(40):
            t = 0.05 * k
            tm = telem(t, sig=0.05, att=wrong)
            tm.gyro_rps = np.zeros(3)
            tm.mag_body_ut = mag_raw(cal, 0, 0, yaw_true)
            g = AcousticEchoGroup([AcousticSensorReturn(0, 0, 2 * 5.0 / core.c_sound, 0.9, t, 25)] * 3)
            out = core.process_flight_tick(tm, 0.0, [g])
        h = out["heading"]
        for key in ("compass_heading_deg", "heading_sigma_deg", "mag_accepted", "mag_rejected", "mag_ok"):
            self.assertIn(key, h)
        self.assertTrue(h["mag_ok"])
        self.assertAlmostEqual(h["yaw_enu_deg"], 90.0, delta=1.0)
        self.assertAlmostEqual(h["compass_heading_deg"], 0.0 if h["compass_heading_deg"] < 180 else 360.0, delta=1.0)
        pt = out["spatial_cloud_3d"][0]["voxel_pos_enu_m"]
        self.assertAlmostEqual(pt[1], 5.0, delta=0.15)     # obstacle placed north, not east
        self.assertLess(abs(pt[0]), 0.15)

    def test_no_gyro_keeps_v2_behaviour(self):
        out = AutonomousDroneCore().process_flight_tick(telem(), 0.0, [])
        self.assertEqual(out["heading"]["heading_source"], "telemetry_attitude")
        self.assertIsNone(out["heading"]["heading_sigma_deg"])
        self.assertFalse(out["heading"]["mag_ok"])

    def test_stale_telemetry_does_not_update_heading(self):
        cal = v2.MagCalibration()
        core = AutonomousDroneCore(mag_calibration=cal)
        tm = telem(1.0)
        tm.gyro_rps = np.zeros(3)
        tm.mag_body_ut = mag_raw(cal, 0, 0, 0.0)
        core.process_flight_tick(tm, 0.0, [])
        x0 = core.heading.x.copy()
        late = telem(0.5)
        late.gyro_rps = np.array([0, 0, 1.0])
        late.mag_body_ut = mag_raw(cal, 0, 0, 1.0)
        out = core.process_flight_tick(late, 0.0, [])
        np.testing.assert_array_equal(core.heading.x, x0)
        self.assertEqual(out["heading"]["mag_status"], "no_reading")


class TestHeadingInPipeline(unittest.TestCase):
    """The fused heading feeds the obstacle map, landmark checks, trust and speed limits."""

    @staticmethod
    def tick(core, t, yaw_true, bias=0.0, mag=None, bearings=None, groups=(), att_yaw=None, rng=None):
        tm = telem(t, sig=0.05, att=Attitude.from_yaw(yaw_true if att_yaw is None else att_yaw))
        g = rng.normal(0, v2.GYRO_ARW_RAD_S_SQRT_HZ / math.sqrt(0.05), 3) if rng is not None else np.zeros(3)
        tm.gyro_rps = np.array([0.0, 0.0, bias]) + g
        tm.mag_body_ut = mag
        return core.process_flight_tick(tm, 0.0, list(groups), landmark_bearings=bearings)

    def test_heading_sigma_inflates_obstacle_covariance(self):
        core = AutonomousDroneCore()
        core.compensate_latency(telem(sig=0.05), 0.0)
        est = core.ranger.estimate(group(10.0, core.c_sound))
        a = core.resolve_acousto_ir_voxel(est, Attitude(), 0.0, None, None)
        b = core.resolve_acousto_ir_voxel(est, Attitude(), 0.0, None, math.radians(5.0))
        self.assertEqual(a["lateral_sigma_from_heading_m"], 0.0)
        self.assertAlmostEqual(b["lateral_sigma_from_heading_m"], 10.0 * math.radians(5.0), delta=0.01)
        # via the tick: gyro-only heading with 20 deg sigma -> lateral sigma ~ 10 m * 0.35 rad
        core2 = AutonomousDroneCore(heading_init_sigma_rad=math.radians(20.0))
        out = self.tick(core2, 0.0, 0.0, groups=[group(10.0, core2.c_sound)])
        lat = out["spatial_cloud_3d"][0]["lateral_sigma_from_heading_m"]
        self.assertAlmostEqual(lat, 10.0 * math.radians(20.0), delta=0.05)
        ob = core2.map.items[0]
        self.assertGreater(math.sqrt(ob.cov[1, 1]), 3.0)              # y is the lateral axis here

    def test_landmark_bearings_keep_heading_when_mag_rejected(self):
        yaw, bias = math.radians(20.0), math.radians(1.0)
        lm = np.array([30.0, 12.0, 0.0])
        cal = v2.MagCalibration()
        steel = mag_raw(cal, 0, 0, yaw, dist=np.array([30.0, 0.0, 0.0]))     # |B| out of tolerance
        results = {}
        for use_lm in (False, True):
            core = AutonomousDroneCore(mag_calibration=cal, landmarks={"L": lm},
                                       heading_init_sigma_rad=math.radians(2.0))
            rng = np.random.default_rng(4)
            out = None
            for k in range(600):                                       # 30 s
                bearings = None
                if use_lm:
                    az = v2.wrap_pi(math.atan2(lm[1], lm[0]) - yaw) + rng.normal(0, math.radians(0.5))
                    bearings = [("L", az, 0.0, math.radians(0.5))]
                out = self.tick(core, 0.05 * k, yaw, bias, mag=steel, bearings=bearings, rng=rng)
            results[use_lm] = out["heading"]
        err_lm = abs(results[True]["yaw_enu_deg"] - 20.0)
        err_no = abs(results[False]["yaw_enu_deg"] - 20.0)
        self.assertEqual(results[True]["mag_accepted"], 0)
        self.assertEqual(results[True]["mag_rejected"], 600)
        self.assertGreater(results[True]["landmark_bearings_accepted"], 500)
        self.assertLess(err_lm, 1.0)
        self.assertGreater(err_no, 20.0)
        self.assertLess(results[True]["heading_sigma_deg"], results[False]["heading_sigma_deg"])

    def test_landmarks_override_a_plausible_distorted_mag_lock(self):
        """A distortion with plausible |B| and dip present from the start makes the
        magnetometer lock ~25 deg off; known-landmark bearings (applied before the mag
        each tick) re-acquire after LM_REACQUIRE_AFTER rejections and win."""
        yaw = math.radians(20.0)
        lm = np.array([30.0, 12.0, 0.0])
        cal = v2.MagCalibration()
        bw = b_world(cal)
        # rotate the horizontal field by 25 deg: same |B| and dip, wrong heading
        a = math.radians(25.0)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        R = Attitude.from_euler(0, 0, yaw).to_matrix()
        bad = R.T @ (Rz @ bw)
        res = {}
        for use_lm in (False, True):
            core = AutonomousDroneCore(mag_calibration=cal, landmarks={"L": lm},
                                       heading_init_sigma_rad=math.radians(30.0))
            rng = np.random.default_rng(5)
            out = None
            for k in range(400):                                       # 20 s
                bearings = None
                if use_lm and k >= 100:                                # landmark appears after 5 s
                    az = v2.wrap_pi(math.atan2(lm[1], lm[0]) - yaw) + rng.normal(0, math.radians(0.5))
                    bearings = [("L", az, 0.0, math.radians(0.5))]
                out = self.tick(core, 0.05 * k, yaw, 0.0, mag=bad, bearings=bearings, rng=rng)
            res[use_lm] = (out["heading"], core.heading.n_lm_reacquire)
        err_no = abs(v2.wrap_pi(math.radians(res[False][0]["yaw_enu_deg"]) - yaw))
        err_lm = abs(v2.wrap_pi(math.radians(res[True][0]["yaw_enu_deg"]) - yaw))
        self.assertGreater(math.degrees(err_no), 20.0)                 # mag alone: locked wrong (limit)
        self.assertLess(math.degrees(err_lm), 2.0)
        self.assertGreaterEqual(res[True][1], 1)

    def test_field_deviation_inflates_mag_noise(self):
        cal = v2.MagCalibration()
        hd = v2.HeadingEstimator(cal)
        hd.initialize(0.0, math.radians(1.0), 0.0)
        bw = b_world(cal)
        clean = hd.update_mag(bw.copy(), 0.0, 0.0)
        hd2 = v2.HeadingEstimator(cal)
        hd2.initialize(0.0, math.radians(1.0), 0.0)
        r0 = hd2.P[0, 0]
        hd2.update_mag(bw * 1.10, 0.0, 0.0)                            # |B| 10% high, same direction
        hd3 = v2.HeadingEstimator(cal)
        hd3.initialize(0.0, math.radians(1.0), 0.0)
        hd3.update_mag(bw.copy(), 0.0, 0.0)
        self.assertEqual(clean["status"], "accepted")
        # less variance reduction when the field is disturbed (larger R)
        self.assertGreater(hd2.P[0, 0], hd3.P[0, 0])
        self.assertLess(hd2.P[0, 0], r0)

    def test_landmark_bearing_tilt_and_wrap(self):
        core = AutonomousDroneCore(landmarks={"L": [-50.0, 0.5, 0.0]},
                                   heading_init_sigma_rad=math.radians(3.0))
        yaw, roll, pitch = math.radians(178.0), math.radians(20.0), math.radians(-20.0)
        core.compensate_latency(telem(sig=0.05), 0.0)
        R = Attitude.from_euler(roll, pitch, yaw).to_matrix()
        d = np.array([-50.0, 0.5, 0.0])
        b = R.T @ (d / np.linalg.norm(d))
        az, el = math.atan2(b[1], b[0]), math.asin(b[2])
        tm = telem(0.0, sig=0.05, att=Attitude.from_euler(roll, pitch, math.radians(-178.0)))
        tm.gyro_rps = np.zeros(3)
        core2 = AutonomousDroneCore(landmarks={"L": [-50.0, 0.5, 0.0]},
                                    heading_init_sigma_rad=math.radians(5.0))
        out = core2.process_flight_tick(tm, 0.0, [], landmark_bearings=[("L", az, el, math.radians(0.2))])
        self.assertEqual(out["heading"]["landmark_bearing_status"], ["accepted"])
        e = v2.wrap_pi(math.radians(out["heading"]["yaw_enu_deg"]) - yaw)
        self.assertLess(abs(math.degrees(e)), 0.5)                  # pulled across +-180 to 178 deg
        # an inconsistent bearing (wrong landmark, 90 deg off) is gated
        out = core2.process_flight_tick(tm, 0.0, [], landmark_bearings=[("L", az + math.pi / 2, el, math.radians(0.2))])
        self.assertEqual(out["heading"]["landmark_bearing_status"], ["rejected_innovation"])

    def test_heading_uncertainty_slows_and_lowers_trust(self):
        good = AutonomousDroneCore(heading_init_sigma_rad=math.radians(2.0))
        bad = AutonomousDroneCore(heading_init_sigma_rad=math.radians(30.0))
        og = self.tick(good, 0.0, 0.0, groups=[group(14.0, good.c_sound)])
        ob = self.tick(bad, 0.0, 0.0, groups=[group(14.0, bad.c_sound)])
        self.assertFalse(og["flight_guidance"]["heading_uncertain"])
        self.assertTrue(ob["flight_guidance"]["heading_uncertain"])
        self.assertLessEqual(ob["flight_guidance"]["speed_limit_mps"], v2.HEADING_SLOW_SPEED_MPS)
        self.assertGreater(og["flight_guidance"]["speed_limit_mps"], v2.HEADING_SLOW_SPEED_MPS)
        self.assertLess(ob["flight_guidance"]["heading_tp"], og["flight_guidance"]["heading_tp"])
        self.assertLess(ob["flight_guidance"]["system_tp"], og["flight_guidance"]["system_tp"])
        # gyro-only heading sigma grows over time -> eventually crosses the threshold
        core = AutonomousDroneCore(heading_init_sigma_rad=math.radians(2.0))
        flags = [self.tick(core, 0.05 * k, 0.0)["flight_guidance"]["heading_uncertain"] for k in range(400)]
        self.assertFalse(flags[0])
        self.assertTrue(flags[-1])


class TestV1SelfTestsAndHygiene(unittest.TestCase):
    def test_v1_self_tests_ported(self):
        v2._self_test()     # v1's checks; the pose-pull, distance-gate and throttle ones are inverted

    def test_demo_runs(self):
        out = v2.demo()
        self.assertEqual(out["flight_guidance"]["mission_action"], "NOMINAL")
        self.assertIn("BRAKE", out["flight_guidance"]["action"])

    def test_no_personal_names_and_cc0(self):
        words = ["jo" + "hn", r"\b" + "ki" + "m" + r"\b", "an" + "san", "jki" + "mbb"]
        pat = re.compile("|".join(words), re.I)
        for name in ("drone_core_v2.py", "test_drone_core_v2.py", "mc_v2.py", "README.md",
                     "drone_uplink_v2.py", "test_drone_uplink_v2.py", "drone_recon_v2.py",
                     "test_drone_recon_v2.py", "mc_recon_v2.py", "make_demo_v2.py", "make_parts_v2.py",
                     "drone_slam_v2.py", "test_drone_slam_v2.py", "mc_slam_v2.py"):
            f = HERE / name
            if f.exists():
                txt = f.read_text()
                self.assertIsNone(pat.search(txt), name)
                if name.endswith(".py"):
                    self.assertIn("CC0", txt)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: CC0-1.0
"""Flight-loop isolation and mission-runner tests. Run: python3 -m unittest. CC0 1.0."""
import time
import unittest

from common_v3 import np
import mc_recon_v2 as MR
import mission_v3 as MV


def _short_mission():
    sc = MR.make_scenes(1, np.random.default_rng(123))[0]
    wps, scene, targets = MR.plan(sc, "poi_close_pass")
    return sc, wps[:150], scene, targets


class TestFlightLoopIsolation(unittest.TestCase):
    def test_heavy_work_does_not_change_flight_outputs(self):
        sc, wps, scene, _ = _short_mission()
        a = MV.Flight(sc, scene, MR.GPS_MODES["gps_standard"], 5, wps[0].pos)
        a.fly(wps, log_flight=True, map_on=False)
        b = MV.Flight(sc, scene, MR.GPS_MODES["gps_standard"], 5, wps[0].pos)

        def heavy(fl, hits, p_hat, P, sig_h, dyaw):      # mission-computer load + tries to touch state
            np.linalg.svd(np.random.default_rng(0).normal(size=(120, 120)))
            _ = fl.core.kf.x.copy()
        b.hooks.append(heavy)
        b.fly(wps, log_flight=True, map_on=True)
        self.assertEqual(len(a.flight_log), len(b.flight_log))
        for (va, aa, xa), (vb, ab, xb) in zip(a.flight_log, b.flight_log):
            self.assertTrue(np.array_equal(va, vb))
            self.assertTrue(np.array_equal(aa, ab))
            self.assertTrue(np.array_equal(xa, xb))
        self.assertGreater(len(b.rmap.points), 0)
        self.assertEqual(len(a.rmap.points), 0)

    def test_flight_tick_time_unchanged_by_map_size(self):
        sc, wps, scene, _ = _short_mission()
        f = MV.Flight(sc, scene, MR.GPS_MODES["gps_standard"], 6, wps[0].pos)
        f.fly(wps)
        import drone_core_v2 as v2
        tm = v2.AirframeState(timestamp_s=f.t + 0.1, pos_m=f.p, vel_mps=np.zeros(3), acc_mps2=np.zeros(3),
                              jerk_mps3=np.zeros(3))
        t0 = time.perf_counter()
        f.core.process_flight_tick(tm, 0.0, [], horizon_s=0.1)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 0.05)        # the core never iterates over the recon map


class TestMissionRunner(unittest.TestCase):
    def test_toggles(self):
        sc = MR.make_scenes(1, np.random.default_rng(321))[0]
        off = MV.run_mission_v3(sc, "poi_close_pass", "gps_rtk")
        on = MV.run_mission_v3(sc, "poi_close_pass", "gps_rtk", repass=True)
        self.assertNotIn("after_repass", off)
        self.assertEqual(off["first_pass"]["coverage"], on["first_pass"]["coverage"])   # same first pass
        self.assertGreaterEqual(on["after_repass"]["coverage"], on["first_pass"]["coverage"] - 0.02)
        li = MV.run_mission_v3(sc, "poi_close_pass", "gps_rtk", lidar=True)
        self.assertGreater(li["first_pass_wh"], off["first_pass_wh"] * 1.3)   # lidar mass + power cost
        self.assertEqual(li["_flight"].t, off["_flight"].t)                    # same trajectory timing


if __name__ == "__main__":
    unittest.main()

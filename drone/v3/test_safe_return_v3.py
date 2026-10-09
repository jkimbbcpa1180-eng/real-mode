# SPDX-License-Identifier: CC0-1.0
"""Tests for safe_return_v3 (feature 6). Run: python3 -m unittest. CC0 1.0."""
import math
import unittest

import common_v3 as C
from common_v3 import np
import safe_return_v3 as SR


def _scene(dist=120.0, soc=0.35):
    sc = SR.make_scenarios(1, np.random.default_rng(7))[0]
    sc["home"] = np.array([dist, 0.0, SR.ALT_M])
    sc["mast"] = (np.array([60.0, 30.0, 0.0]), np.array([62.0, 32.0, 20.0]))   # off the line
    sc["soc0"] = soc
    return sc


class TestGeometryAndPlanning(unittest.TestCase):
    def test_segment_rect(self):
        lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        self.assertTrue(SR.seg_hits_rect(np.array([-5.0, 0.0]), np.array([5.0, 0.0]), lo, hi))
        self.assertFalse(SR.seg_hits_rect(np.array([-5.0, 2.0]), np.array([5.0, 2.0]), lo, hi))

    def test_path_goes_around_mapped_building(self):
        rects = [(np.array([-10.0, -10.0]), np.array([10.0, 10.0]))]
        wps, sp, e = SR.plan_path(np.array([-30.0, 0.0, 6.0]), np.array([30.0, 0.0, 6.0]), rects, np.zeros(3))
        self.assertGreaterEqual(len(wps), 3)
        lo, hi = rects[0][0] - SR.INFLATE_M + 0.01, rects[0][1] + SR.INFLATE_M - 0.01
        for a, b in zip(wps[:-1], wps[1:]):
            self.assertFalse(SR.seg_hits_rect(a, b, lo, hi))
        self.assertTrue(math.isfinite(e) and e > 0)

    def test_headwind_costs_more_and_flies_faster(self):
        d = np.array([1.0, 0.0, 0.0])
        va_h, j_h = SR.best_leg_speed(d, np.array([-6.0, 0.0, 0.0]))
        va_t, j_t = SR.best_leg_speed(d, np.array([6.0, 0.0, 0.0]))
        self.assertGreater(j_h, j_t)
        self.assertGreater(va_h, va_t)

    def test_battery_aware_path_choice(self):
        # two equal-length detours around a block; the wind makes one cheaper
        rects = [(np.array([-5.0, -5.0]), np.array([5.0, 5.0]))]
        s, g = np.array([-20.0, 0.0, 6.0]), np.array([20.0, 0.0, 6.0])
        wps_n, _, _ = SR.plan_path(s, g, rects, np.array([0.0, 6.0, 0.0]))
        wps_s, _, _ = SR.plan_path(s, g, rects, np.array([0.0, -6.0, 0.0]))
        self.assertNotEqual(np.sign(wps_n[1][1]), np.sign(wps_s[1][1]))

    def test_mapped_obstacles_clusters(self):
        a = np.random.default_rng(0).uniform([0, 0, 0], [5, 5, 5], (200, 3))
        b = a + np.array([40.0, 0, 0])
        self.assertEqual(len(SR.mapped_obstacles(np.vstack([a, b]))), 2)

    def test_slide_safety_blocks_approach(self):
        box = (np.array([0.0, -5.0, 0.0]), np.array([10.0, 5.0, 10.0]))
        v = SR.slide_safety(np.array([-1.0, 0.0, 6.0]), np.array([5.0, 0.0, 0.0]), [box])
        self.assertLessEqual(v[0], 0.0 + 1e-9)
        self.assertGreater(abs(v[1]), 0.5)


class TestMissions(unittest.TestCase):
    def test_v3_returns_in_calm_air(self):
        r = SR.run_return(_scene(), "v3", 0.6, 0.0, 11)
        self.assertEqual(r.outcome, "landed")
        self.assertTrue(r.success, r)
        self.assertFalse(r.collision)

    def test_v3_returns_in_wind_with_map_fixes(self):
        r = SR.run_return(_scene(), "v3", 0.6, 6.0, 12)
        self.assertEqual(r.outcome, "landed")
        self.assertFalse(r.collision)
        self.assertLess(r.landing_err_m, SR.SUCCESS_R_M)
        self.assertGreater(r.slam_fixes, 0)

    def test_low_battery_lands_in_place(self):
        r3 = SR.run_return(_scene(soc=0.04), "v3", 0.6, 0.0, 13)
        self.assertTrue(r3.landed_in_place)
        self.assertEqual(r3.outcome, "landed")
        r2 = SR.run_return(_scene(soc=0.14), "v2", 0.6, 0.0, 13)      # v2 rule: LAND at <= 15 %
        self.assertTrue(r2.landed_in_place)

    def test_v3_goes_home_where_v2_rule_lands(self):
        r3 = SR.run_return(_scene(soc=0.14), "v3", 0.6, 0.0, 14)
        self.assertFalse(r3.landed_in_place)
        self.assertEqual(r3.outcome, "landed")

    def test_deterministic(self):
        a = SR.run_return(_scene(), "v3", 0.3, 3.0, 5)
        b = SR.run_return(_scene(), "v3", 0.3, 3.0, 5)
        self.assertEqual(a.landing_err_m, b.landing_err_m)


if __name__ == "__main__":
    unittest.main()

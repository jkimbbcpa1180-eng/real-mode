# SPDX-License-Identifier: CC0-1.0
"""Tests for repass_v3 (feature 4). Run: python3 -m unittest. CC0 1.0."""
import math
import unittest

from common_v3 import np, R
import repass_v3 as RP

LO, HI = np.array([0.0, 0.0, 0.0]), np.array([6.0, 4.0, 3.0])


def _map(hole=True, spacing=0.2):
    """Synthetic map of a box's four facades, with a 1.5 m x 1.5 m hole on the +x face."""
    m = R.ReconMap()
    rng = np.random.default_rng(0)
    for name, n, fixed, axis in (("-x", np.array([-1.0, 0, 0]), 0.0, 1), ("+x", np.array([1.0, 0, 0]), 6.0, 1),
                                 ("-y", np.array([0, -1.0, 0]), 0.0, 0), ("+y", np.array([0, 1.0, 0]), 4.0, 0)):
        for t in np.arange(0.05, (HI[axis]) - 0.05, spacing):
            for z in np.arange(0.05, 3.0, spacing):
                q = np.zeros(3); q[axis] = t; q[1 - axis] = fixed; q[2] = z
                if hole and name == "+x" and 1.0 <= t <= 2.5 and 1.0 <= z <= 2.5:
                    continue
                o = q + 3.0 * n
                cov = np.eye(3) * 0.01 ** 2
                beams = [(-n, 3.0 + rng.normal(0, 0.005), cov)] * 3
                m.add_ping(o, beams, 0.01, 0.0)
    return m


class TestRepass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _map()

    def test_finds_the_hole_on_the_right_face(self):
        gaps, info = RP.find_gaps(self.m, "facades")
        self.assertEqual(info["structures"], 1)
        big = [g for g in gaps if len(g.samples) >= 6]
        self.assertTrue(big)
        g = max(big, key=lambda g: len(g.samples))
        self.assertEqual(g.face, "+x")
        c = g.center
        self.assertTrue(0.8 <= c[1] <= 2.7 and 0.8 <= c[2] <= 2.7, c)

    def test_complete_map_has_no_big_gaps(self):
        gaps, _ = RP.find_gaps(_map(hole=False), "facades")
        self.assertLess(sum(len(g.samples) for g in gaps), 6)

    def test_plan_keeps_clearance_and_looks_at_face(self):
        gaps, _ = RP.find_gaps(self.m, "facades")
        plan = RP.plan_repasses(gaps, self.m, np.array([12.0, 2.0, 2.0]), np.array([20.0, 2.0, 2.0]), 4.0,
                                math.radians(15), 40.0, 45.0, 62.0, 2.0)
        self.assertTrue(plan.waypoints)
        pts, _ = self.m.filtered_points()
        for wp in plan.waypoints:
            self.assertGreaterEqual(float(np.min(np.linalg.norm(pts - wp.pos, axis=1))), 1.2 - 1e-6)
            self.assertGreaterEqual(wp.pos[2], 0.5 - 1e-9)

    def test_nofly_is_never_entered(self):
        gaps, _ = RP.find_gaps(self.m, "facades")
        nofly = [(np.array([7.5, -5.0, -1.0]), np.array([10.0, 9.0, 50.0]))]
        plan = RP.plan_repasses(gaps, self.m, np.array([12.0, 2.0, 2.0]), np.array([20.0, 2.0, 2.0]), 4.0,
                                math.radians(15), 40.0, 45.0, 62.0, 2.0, nofly=nofly)
        self.assertGreater(plan.skipped["nofly"], 0)
        for wp in plan.waypoints:
            self.assertFalse(np.all(wp.pos >= nofly[0][0]) and np.all(wp.pos <= nofly[0][1]))

    def test_interior_needs_authorization(self):
        g = RP.GapCluster(np.array([-1.0, 0, 0]), np.array([[5.0, 2.0, 1.5], [5.0, 2.5, 1.5], [5.0, 2.0, 2.0], [5.0, 2.5, 2.0]]), "in")
        plan = RP.plan_repasses([g], self.m, np.array([12.0, 2.0, 2.0]), np.array([20.0, 2.0, 2.0]), 4.0,
                                math.radians(15), 40.0, 45.0, 62.0, 2.0)
        self.assertEqual(plan.skipped["interior"], 1)
        self.assertFalse(plan.waypoints)

    def test_battery_reserve_blocks_repass(self):
        gaps, _ = RP.find_gaps(self.m, "facades")
        plan = RP.plan_repasses(gaps, self.m, np.array([12.0, 2.0, 2.0]), np.array([20.0, 2.0, 2.0]), 4.0,
                                math.radians(15), 0.25 * 45.0 + 0.01, 45.0, 62.0, 2.0)
        self.assertFalse(plan.waypoints)
        self.assertGreater(plan.skipped["budget"], 0)

    def test_transit_goes_over_when_blocked(self):
        pts, _ = self.m.filtered_points()
        idx = RP.PointIndex(pts, RP.SAFETY_M)
        tr = RP.transit(np.array([-3.0, 2.0, 1.5]), np.array([9.0, 2.0, 1.5]), idx, 6.0, np.array([1.0, 0, 0]),
                        boxes=RP.structures(pts))
        self.assertGreaterEqual(max(w.pos[2] for w in tr), 6.0 - 1e-9)


if __name__ == "__main__":
    unittest.main()

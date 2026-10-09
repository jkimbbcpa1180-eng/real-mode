# SPDX-License-Identifier: CC0-1.0
"""Tests for the v3 ray caster, lidar (feature 3) and thermal camera (feature 2). License: CC0 1.0 Universal."""
import math
import unittest

import common_v3 as C
from common_v3 import np
import sensors_v3 as SE
import thermal_v3 as TH
import mission_v3 as MV

BOX = (np.array([10.0, -5, 0]), np.array([20.0, 5, 10]))
GB = (np.array([-50.0, -50, 0]), np.array([50.0, 50, 0]))


class TestRaycast(unittest.TestCase):
    def test_box_and_ground(self):
        o = np.array([0.0, 0, 5])
        U_ = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 0, -1.0], [0, 0, 1.0]])
        r, h = SE.raycast_many(o, U_, [BOX], 100.0, GB)
        self.assertAlmostEqual(r[0], 10.0, places=6); self.assertEqual(h[0], 0)
        self.assertEqual(h[1], SE.NOHIT)
        self.assertAlmostEqual(r[2], 5.0, places=6); self.assertEqual(h[2], SE.GROUND)
        self.assertEqual(h[3], SE.NOHIT)


class TestLidar(unittest.TestCase):
    def test_returns_on_box_within_noise(self):
        L = SE.LidarHead([BOX], GB, [0.0], 0.0)
        rng = np.random.default_rng(0)
        beams = L.scan(np.array([0.0, 0, 5]), np.array([1.0, 0, 0]), None, rng)
        self.assertGreater(len(beams), 10)
        for u, r, *_ in beams:
            p = np.array([0.0, 0, 5]) + r * u
            on_box = abs(p[0] - 10.0) < 0.2 and abs(p[1]) <= 5.3
            on_ground = abs(p[2]) < 0.2
            self.assertTrue(on_box or on_ground)

    def test_full_dropout_surface_invisible(self):
        L = SE.LidarHead([BOX], GB, [1.0], 1.0)
        self.assertEqual(L.scan(np.array([0.0, 0, 5]), np.array([1.0, 0, 0]), None, np.random.default_rng(0)), [])

    def test_lidar_costs_power(self):
        self.assertGreater(SE.LidarHead.extra_power_w, 0)
        f0 = MV.Flight.__new__(MV.Flight); f0.sensors = [MV.SonarHead()]
        f1 = MV.Flight.__new__(MV.Flight); f1.sensors = [MV.SonarHead(), SE.LidarHead([BOX], GB, [0.0])]
        self.assertGreater(f1.power_w(2.0), f0.power_w(2.0) + 10.0)


class TestThermal(unittest.TestCase):
    def world(self, glass_T):
        lo, hi = BOX
        return {"lo": lo, "hi": hi, "amb": 15.0,
                "vents": [{"ax": 0, "sg": -1, "c": np.array([10.0, 0, 5]), "r": 0.3, "T": 70.0}],
                "glass": [{"ax": 0, "sg": -1, "c": np.array([10.0, 3, 5]), "hw": 1.0, "hh": 0.75, "Tr": glass_T}],
                "people": [], "indoor": [np.array([15.0, 0, 1])], "boxes": [(lo, hi)]}

    def test_vent_hot_and_sky_glass_cold(self):
        w = self.world(TH.SKY_C)
        T, D, dep = TH.frame(w, np.array([2.0, 0, 5]), np.array([1.0, 0, 0]), np.random.default_rng(0), GB)
        self.assertGreater(T.max(), 40.0)
        self.assertLess(T.min(), 5.0)          # sky-reflecting glass reads cold

    def test_detects_vent_not_indoor_person(self):
        w = self.world(TH.SKY_C)
        sc = {"area": (np.array([0.0, -20, 0]), np.array([30.0, 20, 0]))}
        dets = TH.run(sc, w, 8.0, 0.03, np.random.default_rng(1))
        s = TH.score(dets, TH.truth(w))
        self.assertEqual(dict(s["hits"])["vent"], True)
        self.assertEqual(dict(s["hits"])["person_indoor"], False)


if __name__ == "__main__":
    unittest.main()

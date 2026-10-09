"""
Tests for drone_recon_v2.py. Run: python3 -m unittest test_drone_recon_v2 -v
License: CC0 1.0 Universal (public domain).
"""
import math
import os
import tempfile
import unittest

import numpy as np

import drone_core_v2 as v2
import drone_recon_v2 as rc

DEG = math.radians(1.0)


def wall_hits(m: rc.ReconMap, dist: float, n_pings: int = 40, seed: int = 0, sig_h: float = DEG):
    """Pings at a wall x = 10 from x = 10 - dist, beams spread over the same 1 m x 1 m patch."""
    rng = np.random.default_rng(seed)
    o = np.array([10.0 - dist, 0.0, 2.0])
    for k in range(n_pings):
        beams = []
        for _ in range(5):
            target = np.array([10.0, rng.uniform(-0.5, 0.5), 2.0 + rng.uniform(-0.5, 0.5)])
            d = target - o
            r = float(np.linalg.norm(d))
            u = d / r
            beams.append((u, r, rc.point_covariance(r, u, 0.015, DEG, sig_h)))
        m.add_ping(o, beams, 0.015, 0.1 * k)


class TestRangeResolution(unittest.TestCase):
    def test_point_covariance_grows_with_range(self):
        u = np.array([1.0, 0.0, 0.0])
        lat = [rc.lateral_sigma(rc.point_covariance(r, u, 0.015, DEG, DEG), u) for r in (1, 4, 12)]
        self.assertLess(lat[0], lat[1])
        self.assertLess(lat[1], lat[2])
        # range * sqrt(bearing^2 + heading^2) across the ray in the horizontal plane
        self.assertAlmostEqual(lat[2], 12 * math.sqrt(2) * DEG, places=6)
        cov = rc.point_covariance(12, u, 0.015, DEG, DEG)
        self.assertAlmostEqual(math.sqrt(cov[0, 0]), 0.015, places=9)   # along the ray: range noise

    def test_core_obstacle_point_covariance_grows_with_range(self):
        """The flight core's own obstacle points follow the same range-dependent model."""
        core = v2.AutonomousDroneCore()
        core.process_flight_tick(v2.AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)), 0.0, [])
        att = v2.Attitude.from_euler(0, 0, 0)
        sig = []
        for r in (2.0, 10.0):
            echo = v2.AcousticSensorReturn(0.0, 0.0, 2.0 * r / core.c_sound, 0.85, 0.0, 25.0)
            est = core.ranger.estimate(v2.AcousticEchoGroup([echo] * 3))
            self.assertIsNotNone(est)
            out = core.resolve_acousto_ir_voxel(est, att, 0.0, None, heading_sigma_rad=math.radians(2))
            sig.append(float(np.sqrt(np.linalg.eigvalsh(out["_cov"]).max())))
        self.assertLess(sig[0], sig[1])
        self.assertGreater(sig[1], 10.0 * math.radians(2))     # at least range * heading sigma


class TestEvidenceGrid(unittest.TestCase):
    def test_free_space_carved_and_hit_occupied(self):
        g = rc.EvidenceGrid(0.25)
        for _ in range(4):
            g.integrate_beam(np.array([0.1, 0.1, 0.1]), np.array([5.1, 0.1, 0.1]), 0.015, 0.02)
        self.assertLess(g.lo[g.key(np.array([2.0, 0.1, 0.1]))], 0.0)       # free along the beam
        self.assertIn(g.key(np.array([5.1, 0.1, 0.1])), set(g.occupied()))
        self.assertNotIn(g.key(np.array([2.0, 0.1, 0.1])), set(g.occupied()))

    def test_no_hit_adds_no_evidence(self):
        g = rc.EvidenceGrid()
        g.integrate_beam(np.zeros(3), None, 0.015, 0.02)
        self.assertEqual(len(g.lo), 0)

    def test_closer_gives_finer_detail(self):
        near, far = rc.ReconMap(), rc.ReconMap()
        wall_hits(near, 1.5)
        wall_hits(far, 12.0)
        dn, df = near.effective_detail_m(), far.effective_detail_m()
        self.assertIsNotNone(dn)
        self.assertLess(dn, df)
        # one ping each: the near hit concentrates evidence (higher peak log-odds)
        n1, f1 = rc.ReconMap(), rc.ReconMap()
        wall_hits(n1, 1.5, n_pings=1)
        wall_hits(f1, 12.0, n_pings=1)
        self.assertGreater(max(n1.grid.lo.values()), max(f1.grid.lo.values()))
        # far hits are smeared over more cells
        cn = sum(1 for k, v in near.grid.lo.items() if v > 0)
        cf = sum(1 for k, v in far.grid.lo.items() if v > 0)
        self.assertLess(cn, cf)


class TestExport(unittest.TestCase):
    def test_ply_round_trip(self):
        m = rc.ReconMap()
        wall_hits(m, 3.0, n_pings=10)
        P, S = m.filtered_points()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.ply")
            n = rc.write_ply(path, P, S)
            P2, S2 = rc.read_ply(path)
        self.assertEqual(n, len(P))
        self.assertTrue(np.allclose(P2, P, atol=1e-4))
        self.assertTrue(np.allclose(S2, S, atol=1e-4))

    def test_png_render_or_honest_skip(self):
        m = rc.ReconMap()
        wall_hits(m, 3.0, n_pings=10)
        P, S = m.filtered_points()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.png")
            res = rc.render_png(path, P, S, boxes=[(np.array([10, -1, 0.0]), np.array([11, 1, 4.0]))])
            if res["written"]:
                self.assertGreater(os.path.getsize(path), 1000)
            else:
                self.assertIn("matplotlib", res["note"])
                self.assertFalse(os.path.exists(path))


class TestMissionTemplates(unittest.TestCase):
    def test_lawnmower_swaths_overlap(self):
        wps = rc.lawnmower([0, 0, 0], [30, 20, 0], top_z=5.0, standoff_m=6.0,
                           beam_half_rad=math.radians(15), overlap=0.3)
        ys = sorted(set(round(float(w.pos[1]), 6) for w in wps))
        sw = rc.swath_width(6.0, math.radians(15))
        self.assertTrue(all(b - a <= sw * 0.7 + 1e-9 for a, b in zip(ys[:-1], ys[1:])))
        self.assertTrue(all(abs(w.pos[2] - 11.0) < 1e-9 for w in wps))
        self.assertLessEqual(ys[0], 0.0 + 1e-9); self.assertGreaterEqual(ys[-1], 20.0 - 1e-9)
        self.assertTrue(all(np.allclose(w.look_dir, [0, 0, -1]) for w in wps))

    def test_lawnmower_keeps_ground_in_sonar_range(self):
        wps = rc.lawnmower([0, 0, 0], [20, 20, 0], top_z=9.0, standoff_m=6.0,
                           beam_half_rad=math.radians(15), max_range_m=15.0)
        z = wps[0].pos[2]
        self.assertLessEqual(z / math.cos(math.radians(15)), 0.9 * 15.0 + 1e-9)   # beam-edge ground range
        self.assertGreaterEqual(z - 9.0, rc.LAWNMOWER_MIN_ROOF_CLEARANCE_M)
        with self.assertRaises(ValueError):
            rc.lawnmower([0, 0, 0], [20, 20, 0], top_z=13.0, standoff_m=6.0,
                         beam_half_rad=math.radians(15), max_range_m=15.0)

    def test_perimeter_orbit_standoff_and_levels(self):
        lo, hi = np.array([0.0, 0.0, 0.0]), np.array([10.0, 6.0, 8.0])
        wps = rc.perimeter_orbit(lo, hi, 4.0, math.radians(15), overlap=0.3)
        for w in wps:
            q = np.array([np.clip(w.pos[0], lo[0], hi[0]), np.clip(w.pos[1], lo[1], hi[1])])
            self.assertAlmostEqual(float(np.linalg.norm(w.pos[:2] - q)), 4.0, delta=0.15)
            self.assertGreater(float(w.look_dir @ np.array([q[0] - w.pos[0], q[1] - w.pos[1], 0])), 0)
        zs = sorted(set(round(float(w.pos[2]), 6) for w in wps))
        sw = rc.swath_width(4.0, math.radians(15)) * 0.7
        self.assertTrue(all(b - a <= sw + 1e-9 for a, b in zip(zs[:-1], zs[1:])))
        self.assertLessEqual(zs[0] - 0.5 * sw, 0.5 + 1e-5)
        self.assertGreaterEqual(zs[-1] + 0.5 * sw, 8.0 - 1e-5)

    def test_poi_close_pass_looks_at_poi(self):
        wps = rc.poi_close_pass([5, 5, 0], 0.5, 4.0, 2.0, math.radians(15))
        for w in wps:
            d = np.array([5.0 - w.pos[0], 5.0 - w.pos[1], 0.0])
            self.assertAlmostEqual(float(np.linalg.norm(d)), 0.5 * math.sqrt(2) + 2.0, places=6)
            self.assertAlmostEqual(float(w.look_dir @ d / np.linalg.norm(d)), 1.0, places=6)

    def test_interior_mapping_requires_authorization(self):
        lo, hi = np.zeros(3), np.array([6.0, 5.0, 3.0])
        with self.assertRaises(rc.AuthorizationError):
            rc.interior_room_sweep(lo, hi, 1.2, math.radians(15), {})
        with self.assertRaises(rc.AuthorizationError):
            rc.interior_room_sweep(lo, hi, 1.2, math.radians(15), {"interior_authorized": "yes"})
        wps = rc.interior_room_sweep(lo, hi, 1.2, math.radians(15), {"interior_authorized": True})
        self.assertGreater(len(wps), 10)
        for w in wps:   # stays inside the room
            self.assertTrue(np.all(w.pos > lo) and np.all(w.pos < hi))


class TestSceneAndCoverage(unittest.TestCase):
    def test_raycast_box_and_room(self):
        sc = rc.Scene([(np.array([5.0, -1, 0]), np.array([6.0, 1, 3]))], np.zeros(3), np.ones(3) * 10, ground=True)
        self.assertAlmostEqual(sc.raycast(np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), 15), 5.0)
        self.assertAlmostEqual(sc.raycast(np.array([0.0, 0, 1]), np.array([0.0, 0, -1.0]), 15), 1.0)
        self.assertIsNone(sc.raycast(np.array([0.0, 0, 1]), np.array([-1.0, 0, 0]), 15))
        room = rc.Scene([], np.zeros(3), np.ones(3), False, rooms=[(np.zeros(3), np.array([6.0, 5, 3]))])
        self.assertAlmostEqual(room.raycast(np.array([3.0, 2.5, 1.5]), np.array([0, 1.0, 0]), 15), 2.5)

    def test_coverage_metric(self):
        lo, hi = np.array([0.0, 0, 0]), np.array([4.0, 4, 3])
        S = rc.box_surface_samples(lo, hi, 0.25, "sides")
        self.assertAlmostEqual(rc.coverage(S, S + 0.05), 1.0)
        self.assertEqual(rc.coverage(S, np.zeros((0, 3))), 0.0)
        # half the facades observed
        half = S[S[:, 0] < 2.0]
        self.assertAlmostEqual(rc.coverage(S, half), float(np.mean(S[:, 0] < 2.0 + 0.3)), delta=0.02)
        # points 0.5 m off every surface (outside tolerance) cover nothing on that face
        off = S[np.isclose(S[:, 0], 0.0)] - np.array([0.5, 0, 0])
        face = S[np.isclose(S[:, 0], 0.0)]
        self.assertEqual(rc.coverage(face, off, 0.3), 0.0)
        sc = rc.Scene([(lo, hi)], lo - 5, hi + 5, ground=False)
        self.assertTrue(np.allclose(sc.surface_distance(off), 0.5))


class TestMagDisturbanceLog(unittest.TestCase):
    def test_logs_only_disturbances_with_honest_label(self):
        log = rc.MagDisturbanceLog(50.0, math.radians(53.0))
        self.assertFalse(log.observe(0.0, np.zeros(3), 50.5, math.radians(53.5)))
        self.assertTrue(log.observe(1.0, np.array([1.0, 2, 3]), 62.0, math.radians(53.0)))
        self.assertTrue(log.observe(2.0, np.zeros(3), 50.0, math.radians(61.0)))
        self.assertFalse(log.observe(3.0, np.zeros(3), None, None))
        self.assertEqual(len(log.events), 2)
        self.assertIn("possible wiring/steel nearby", log.events[0]["label"])
        self.assertIn("drone position", log.events[0]["label"])

    def test_core_reports_field_for_the_log(self):
        core = v2.AutonomousDroneCore()
        tm = v2.AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        tm.gyro_rps = np.zeros(3)
        cal = core.heading.cal
        D, I, F = cal.declination_rad, cal.dip_rad, cal.field_ut
        tm.mag_body_ut = F * np.array([math.cos(I) * math.sin(D), math.cos(I) * math.cos(D), -math.sin(I)])
        h = core.process_flight_tick(tm, 0.0, [])["heading"]
        self.assertAlmostEqual(h["mag_field_ut"], 50.0, places=2)
        self.assertAlmostEqual(h["mag_dip_deg"], 53.0, places=2)


if __name__ == "__main__":
    unittest.main()

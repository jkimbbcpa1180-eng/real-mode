"""
Tests for drone_slam_v2 (run: python3 -m unittest test_drone_slam_v2).
License: CC0 1.0 Universal (public domain).
"""
import math
import unittest

import numpy as np

import drone_core_v2 as v2
import drone_recon_v2 as R
import drone_slam_v2 as S

BOX = (np.array([0.0, 0.0, 0.0]), np.array([6.0, 6.0, 4.0]))


def _planes(rng, n, which=("x", "y", "z"), size=4.0):
    pts = []
    for _ in range(n):
        f = which[rng.integers(len(which))]
        a, b = rng.uniform(0, size, 2)
        pts.append({"x": [0.0, a, b], "y": [a, 0.0, b], "z": [a, b, 0.0]}[f])
    return np.array(pts)


def _cov(n, s=0.01):
    return np.tile(np.eye(3) * s * s, (n, 1, 1))


def _fan(look, rng, n=4, half=math.radians(15)):
    look = look / np.linalg.norm(look)
    up = np.array([0.0, 0.0, 1.0]) if abs(look[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, look); e1 /= np.linalg.norm(e1)
    e2 = np.cross(look, e1)
    out = []
    for a in np.linspace(-half, half, n):
        for b in np.linspace(-half, half, n):
            d = look + math.tan(a + rng.uniform(-0.05, 0.05)) * e1 + math.tan(b + rng.uniform(-0.05, 0.05)) * e2
            out.append(d / np.linalg.norm(d))
    return out


def _sim_orbit(drift, seed=0, standoff=3.0, levels=None):
    """Truth orbit around BOX; the 'core' position is truth + drift(t). Yields pings."""
    rng = np.random.default_rng(seed)
    scene = R.Scene([BOX], BOX[0] - 6, BOX[1] + 6, True)
    wps = R.perimeter_orbit(BOX[0], BOX[1], standoff, math.radians(15), 0.3)
    if levels is not None:
        zs = sorted({round(float(w.pos[2]), 6) for w in wps})[:levels]
        wps = [w for w in wps if round(float(w.pos[2]), 6) in zs]
    p, t, i = wps[0].pos.copy(), 0.0, 0
    out = []
    while i < len(wps):
        t += 0.1
        d = wps[i].pos - p
        dist = float(np.linalg.norm(d))
        if dist <= 0.2:
            p = wps[i].pos.copy(); i += 1
        else:
            p = p + d / dist * 0.2
        beams = []
        for u in _fan(wps[min(i, len(wps) - 1)].look_dir, rng):
            r = scene.raycast(p, u, 15.0)
            if r is None:
                continue
            r_m = r + rng.normal(0, 0.015)
            um = u + rng.normal(0, math.radians(1.0), 3); um /= np.linalg.norm(um)
            beams.append((um, r_m, R.point_covariance(r_m, um, 0.015, math.radians(1.0), None, None)))
        c = np.asarray(drift(t), float)
        out.append((t, p + c, beams, c))
    return out, scene


def _run_slam(pings, **kw):
    sl = S.SonarSLAM(**kw)
    for t, o, beams, _ in pings:
        if beams:
            sl.add_ping(t, o, beams)
    sl.finish()
    T = np.array([p[0] for p in pings]); Ct = np.array([p[3] for p in pings])
    c_true = np.array([Ct[(T >= k.t0) & (T < k.t1)].mean(0) for k in sl.kfs])
    return sl, c_true


class TestRegistration(unittest.TestCase):
    def test_recovers_known_offset(self):
        rng = np.random.default_rng(3)
        M = _planes(rng, 600)
        N, pl = S.local_normals(M, M)
        off = np.array([0.3, -0.2, 0.15])
        Sc = M[::3] + off + rng.normal(0, 0.01, (200, 3))
        r = S.register_translation(Sc, _cov(len(Sc)), M[pl], N[pl], _cov(int(pl.sum())), np.zeros(3))
        self.assertTrue(r["ok"])
        self.assertEqual(r["n_obs"], 3)
        self.assertLess(np.linalg.norm(r["c"] - off), 0.03)

    def test_single_flat_wall_constrains_only_its_normal(self):
        """A single flat wall seen along-track: only the wall normal (x) is observable; the
        along-track (y) and vertical (z) offsets must NOT be 'corrected'."""
        rng = np.random.default_rng(4)
        M = _planes(rng, 500, which=("x",), size=8.0)
        N, pl = S.local_normals(M, M)
        self.assertGreater(pl.mean(), 0.9)
        off = np.array([0.2, 0.6, 0.3])
        Sc = M[::2] + off + rng.normal(0, 0.01, (250, 3))
        c0 = np.array([0.0, 0.05, -0.02])
        r = S.register_translation(Sc, _cov(len(Sc)), M[pl], N[pl], _cov(int(pl.sum())), c0)
        self.assertTrue(r["ok"])
        self.assertEqual(r["n_obs"], 1)
        self.assertGreater(abs(r["V_obs"][0, 0]), 0.99)
        self.assertAlmostEqual(r["c"][0], 0.2, delta=0.03)
        self.assertLess(np.linalg.norm(r["c"][1:] - c0[1:]), 1e-3)     # unchanged

    def test_flat_wall_slam_adds_no_along_track_information(self):
        rng = np.random.default_rng(5)
        sl = S.SonarSLAM(kf_dt_s=1.0)
        # drone flies along a wall at x=0 (looking -x); along-track core error grows in y
        for i in range(60):
            t = 0.1 * (i + 1)
            p = np.array([3.0, 0.2 * i, 2.0])
            c = np.array([0.0, 0.05 * t, 0.0])
            beams = []
            for u in _fan(np.array([-1.0, 0.0, 0.0]), rng, n=5):
                r = 3.0 / -u[0]
                beams.append((u, r + rng.normal(0, 0.01), R.point_covariance(r, u, 0.015, math.radians(1.0))))
            sl.add_ping(t, p + c, beams)
        sl.finish()
        self.assertGreater(len(sl.factors), 0)
        for k, j, A, b in sl.factors:
            ev, V = np.linalg.eigh(A)
            strong = V[:, ev > 1e-6 * max(ev[-1], 1e-12)]
            self.assertEqual(strong.shape[1], 1)
            self.assertGreater(abs(strong[0, 0]), 0.99)            # only the wall normal
            # along-track information only from the estimated normal's tilt: < 0.1% of the normal's
            self.assertLess(abs(A[1, 1]), 1e-3 * A[0, 0])

    def test_too_few_points_is_rejected(self):
        rng = np.random.default_rng(6)
        M = _planes(rng, 400)
        N, pl = S.local_normals(M, M)
        r = S.register_translation(M[:5], _cov(5), M[pl], N[pl], _cov(int(pl.sum())), np.zeros(3))
        self.assertFalse(r["ok"])

    def test_line_of_points_is_not_planar(self):
        Q = np.column_stack([np.linspace(0, 3, 30), np.zeros(30), np.zeros(30)])
        _, pl = S.local_normals(Q, Q)
        self.assertFalse(pl.any())


class TestLoopClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # steady drift of the core position (0.9 m over the mission), as a slow GPS error would do
        drift = lambda t: np.array([0.012 * t, -0.008 * t, 0.004 * t])  # noqa: E731
        cls.pings, cls.scene = _sim_orbit(drift, seed=11, levels=2)

    def _shape_rms(self, sl, c_true, use_smoothed=True):
        C = sl.smoothed()[0] if use_smoothed else np.array(sl.est_online)
        e = (C - c_true)[:, :2]                 # facades: horizontal error matters
        e = e - e.mean(0)                       # common offset removed: shape (relative) error
        return float(np.sqrt((e ** 2).sum(1).mean()))

    def test_loop_closure_reduces_drift(self):
        sl_lc, ct = _run_slam(self.pings, loop_closure=True)
        sl_od, ct2 = _run_slam(self.pings, loop_closure=False)
        e_lc, e_od = self._shape_rms(sl_lc, ct), self._shape_rms(sl_od, ct2)
        self.assertGreater(sl_lc.stats["loop_factors"], 0)
        self.assertEqual(sl_od.stats["loop_factors"], 0)
        self.assertLess(e_lc, 0.7 * e_od, (e_lc, e_od))
        self.assertLess(e_lc, 0.10)
        # and the corrected positions are better than the raw core ones (shape)
        raw = ct[:, :2] - ct[:, :2].mean(0)
        self.assertLess(e_lc, 0.5 * float(np.sqrt((raw ** 2).sum(1).mean())))

    def test_rebuilt_map_is_on_the_true_surfaces(self):
        sl, ct = _run_slam(self.pings)
        full = sl.smoothed_full()
        rm = sl.rebuild_map(R.ReconMap(), 0.015, full)
        pts, _ = rm.filtered_points()
        datum = (ct - full["c"]).mean(0)
        acc = self.scene.surface_distance(pts - datum)    # shape accuracy (datum removed)
        self.assertLess(float(np.median(acc)), 0.08)
        self.assertEqual(rm.datum_cov.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(rm.datum_cov) > 0))


class TestIndependenceFromFlightLoop(unittest.TestCase):
    def test_core_does_not_import_slam(self):
        with open(v2.__file__) as f:
            src = f.read()
        self.assertNotIn("drone_slam_v2", src)

    def test_flight_outputs_identical_with_slam_running(self):
        def run(with_slam):
            core = v2.AutonomousDroneCore()
            sl = S.SonarSLAM() if with_slam else None
            rng = np.random.default_rng(0)
            outs = []
            for i in range(30):
                t = 0.1 * (i + 1)
                tm = v2.AirframeState(timestamp_s=t, pos_m=np.array([0.2 * i, 0.0, 2.0]) + rng.normal(0, 0.3, 3),
                                      vel_mps=np.array([2.0, 0.0, 0.0]), acc_mps2=np.zeros(3), jerk_mps3=np.zeros(3),
                                      attitude=v2.Attitude.from_yaw(0.0), pos_sigma_m=np.full(3, 0.5),
                                      vel_sigma_mps=np.full(3, 0.1))
                o = core.process_flight_tick(tm, 0.0, [], horizon_s=0.1, desired_velocity_mps=np.array([2.0, 0, 0]))
                fg = o["flight_guidance"]
                outs.append(np.concatenate([core.kf.x[:3], fg["safe_velocity_mps"], fg["command_accel_mps2"]]))
                if sl is not None:
                    u = np.array([0.0, 1.0, 0.0])
                    sl.add_ping(t, core.kf.x[:3].copy(), [(u, 3.0, R.point_covariance(3.0, u, 0.015, 0.02))])
            return np.array(outs)
        np.testing.assert_array_equal(run(False), run(True))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: CC0-1.0
"""Tests for the ground-reference mode (ground_ref_v3). License: CC0 1.0 Universal."""
import math
import unittest

import common_v3 as C
from common_v3 import np
import ground_ref_v3 as G


def synthetic(moved=None, offset=1.0, seed=0, n_markers=4, T=120.0, false_at=None):
    """GM error c(t) (0.5 m, 30 s); empty pings (no sonar constraints); every 10 s two
    neighbouring markers are in view together for 4 s (redundancy is what makes a wrong
    marker detectable). Marker `moved` reads `offset` m off (wrong / moved marker)."""
    rng = np.random.default_rng(seed)
    sl = G.AnchoredSLAM(gps_sigma_m=0.5, gps_tau_s=30.0)
    phi = math.exp(-0.1 / 30.0)
    c = rng.normal(0, 0.5, 3)
    truth = []
    for i in range(int(T / 0.1)):
        t = i * 0.1
        c = phi * c + math.sqrt(1 - phi * phi) * 0.5 * rng.normal(size=3)
        sl.add_ping(t, np.zeros(3), [])
        truth.append((len(sl.kfs), c.copy()))
        slot = int(t // 10.0)
        if (t % 10.0) < 4.0:
            for m in (slot % n_markers, (slot + 1) % n_markers):
                if rng.random() < 0.3:
                    z = c + rng.normal(0, 0.05, 3)
                    if m == moved:
                        z = z + np.array([offset, 0.0, 0.0])
                    if false_at is not None and abs(t - false_at) < 0.05:
                        z = z + np.array([0.0, 2.0, 0.0])          # one false match (alone in its keyframe)
                    sl.queue_detection(m, z, 0.05 ** 2, 0.0005)
    sl.finish()
    K = len(sl.kfs)
    ct = np.zeros((K, 3)); n = np.zeros(K)
    for k, cc in truth:
        if k < K:
            ct[k] += cc; n[k] += 1
    ct /= np.maximum(n, 1)[:, None]
    return sl, ct


class TestGroundRef(unittest.TestCase):
    def test_gm_window(self):
        a, v = G.gm_window(0.5, 30.0, np.zeros(5))
        self.assertAlmostEqual(a, 1.0); self.assertAlmostEqual(v, 0.0, places=9)
        a, v = G.gm_window(0.5, 30.0, np.array([30.0]))
        self.assertAlmostEqual(a, math.exp(-1.0)); self.assertAlmostEqual(v, 0.25 * (1 - math.exp(-2.0)), places=9)

    def test_good_markers_kept_and_help(self):
        sl, ct = synthetic(moved=None)
        self.assertEqual(sl.reject_bad_markers(), [])
        full = sl.smoothed_full()
        e = np.linalg.norm(full["c"] - ct, axis=1)
        self.assertLess(float(np.median(e)), 0.15)           # vs ~0.5 m GPS error without anchors

    def test_moved_marker_rejected(self):
        for seed in range(5):
            sl, ct = synthetic(moved=2, offset=1.0, seed=seed)
            self.assertEqual(sl.reject_bad_markers(), [2], seed)
            self.assertFalse(any(f["src"] == ("marker", 2) for f in sl.anchors))

    def test_single_false_match_dropped_marker_kept(self):
        sl, ct = synthetic(moved=None, seed=3, false_at=60.0)
        self.assertEqual(sl.reject_bad_markers(), [])
        full = sl.smoothed_full()
        self.assertLess(float(np.median(np.linalg.norm(full["c"] - ct, axis=1))), 0.15)

    def test_place_markers_on_ground_along_path(self):
        class W:
            def __init__(self, p): self.pos = np.array(p, float)
        wps = [W([0, 0, 10]), W([40, 0, 10]), W([40, 40, 10])]
        M = G.place_markers(wps, 5, np.random.default_rng(1))
        self.assertEqual(M.shape, (5, 3))
        self.assertTrue(np.all(M[:, 2] == 0.0))
        self.assertTrue(np.all(np.minimum(np.abs(M[:, 1]), np.abs(M[:, 0] - 40)) <= 1.0 + 1e-9))


if __name__ == "__main__":
    unittest.main()

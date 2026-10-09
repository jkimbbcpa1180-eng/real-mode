# SPDX-License-Identifier: CC0-1.0
"""Tests for swarm_v3 (feature 1). Run: python3 -m unittest. CC0 1.0."""
import unittest

from common_v3 import np, R, U
import mc_recon_v2 as MR
import swarm_v3 as SW

SC = MR.make_scenes(1, np.random.default_rng(42))[0]


class TestSplitAndDeconfliction(unittest.TestCase):
    def test_lawnmower_strips_cover_area_on_layers(self):
        tasks = SW.split_lawnmower(SC, 3)
        a_lo, a_hi = SC["area"]
        xs = np.concatenate([[w.pos[0] for w in t] for t in tasks])
        self.assertLessEqual(xs.min(), a_lo[0] + 1.0)
        self.assertGreaterEqual(xs.max(), a_hi[0] - 1.0)
        z = [t[0].pos[2] for t in tasks]
        self.assertAlmostEqual(z[1] - z[0], SW.LAYER_M)

    def test_perimeter_sectors_cover_ring(self):
        full = R.perimeter_orbit(*SC["building"], MR.STANDOFF["perimeter_orbit"], MR.BEAM_HALF, MR.OVERLAP)
        tasks = SW.split_perimeter(SC, 4)
        self.assertGreaterEqual(sum(len(t) for t in tasks), len(full) - 4)

    def test_launch_transit_is_above_structures(self):
        tasks = SW.split_perimeter(SC, 2)
        pad = SW.pads(SC, 2)[0]
        top = max(SC["building"][1][2], SC["poi"][1][2])
        w = SW.with_launch(pad, tasks[0], top + 3.0)
        scene = R.Scene([SC["building"], SC["poi"]], *SC["area"], True)
        d = scene.surface_distance(np.array([q.pos for q in w[:len(w) - len(tasks[0])]]))
        self.assertGreater(float(d[1:].min()), 0.4)


class TestRelay(unittest.TestCase):
    def test_multihop_route_and_backpressure(self):
        st = np.array([0.0, 0.0, 2.0])
        c = SW.SwarmComms(3, st, 1, relay=True)
        for ls in c.links:          # no outages for the test
            for l in ls:
                l._next_switch = 1e9
        pos = [np.array([80.0, 0, 5]), np.array([125.0, 0, 5]), np.array([170.0, 0, 5])]
        c.step(0.1, pos)
        self.assertTrue(c.direct_up(0))
        self.assertFalse(c.direct_up(2))
        self.assertEqual(c._routes[2], 1)              # 2 -> 1 -> 0 -> station
        ok = [c.send_from(2, b"2|x") for _ in range(SW.RELAY_QUEUE_MSGS + 5)]
        self.assertEqual(sum(ok), min(SW.RELAY_QUEUE_MSGS, SW.MESH_MSGS_PER_TICK))
        self.assertLessEqual(len(c.relay_q[1]), SW.RELAY_QUEUE_MSGS)
        c2 = SW.SwarmComms(3, st, 1, relay=False)
        c2.step(0.1, pos)
        self.assertIsNone(c2._routes[2])


class TestAlignment(unittest.TestCase):
    def test_recovers_offset_in_observable_directions(self):
        rng = np.random.default_rng(3)
        lo, hi = np.array([0.0, 0.0, 0.0]), np.array([8.0, 6.0, 4.0])
        ref = R.box_surface_samples(lo, hi, 0.2, "sides") + rng.normal(0, 0.02, (1, 3))
        off = np.array([0.4, -0.3, 0.0])
        pts = R.box_surface_samples(lo, hi, 0.2, "sides") + off + rng.normal(0, 0.02, (1, 3))
        a = SW.align_to(ref, pts, rng)
        self.assertTrue(a["ok"])
        self.assertLess(np.linalg.norm((a["c"] - off)[:2]), 0.1)


class TestReciprocalAvoidance(unittest.TestCase):
    def _fly(self, offsets, dz=0.0):
        """Two point drones on a head-on course, nominal 2 m/s, avoidance from (offset) estimates."""
        P = [np.array([0.0, 0.0, 10.0]), np.array([30.0, 0.0, 10.0 + dz])]
        G = [np.array([30.0, 0.0, 10.0]), np.array([0.0, 0.0, 10.0 + dz])]
        V = [np.zeros(3), np.zeros(3)]
        dmin, dev, dt = np.inf, [0.0, 0.0], 0.1
        for _ in range(400):
            A = SW.reciprocal_avoidance([P[k] + offsets[k] for k in range(2)], V, [True, True])
            for k in range(2):
                g = G[k] - P[k]
                nom = g / max(np.linalg.norm(g), 1e-9) * min(2.0, np.linalg.norm(g) / dt)
                na = float(np.linalg.norm(A[k]))
                if na > 1e-9 and float(nom @ A[k]) < 0:          # same projection as mission_v3.Flight
                    nom = nom - float(nom @ A[k]) / na ** 2 * A[k]
                v = nom + A[k]
                s = np.linalg.norm(v)
                v = v * min(1.0, 3.0 / max(s, 1e-9))
                V[k] = v
                P[k] = P[k] + v * dt
                dev[k] = max(dev[k], float(np.hypot(P[k][1], P[k][2] - G[k][2])))
            dmin = min(dmin, float(np.linalg.norm(P[0] - P[1])))
        return dmin, dev, P, G

    def test_head_on_both_deviate_and_keep_distance(self):
        dmin, dev, P, G = self._fly([np.zeros(3), np.zeros(3)])
        self.assertGreaterEqual(dmin, SW.NEAR_MISS_M)        # safety distance
        self.assertGreater(dev[0], 0.5)                       # BOTH drones yield
        self.assertGreater(dev[1], 0.5)
        self.assertLess(abs(dev[0] - dev[1]), 0.2)            # and share it equally
        for k in range(2):
            self.assertLess(np.linalg.norm(P[k] - G[k]), 0.5)  # and still reach their goals

    def test_head_on_offset_layers_no_deadlock(self):
        dmin, dev, P, G = self._fly([np.zeros(3), np.zeros(3)], dz=2.4)
        self.assertGreaterEqual(dmin, SW.NEAR_MISS_M)
        for k in range(2):
            self.assertLess(np.linalg.norm(P[k] - G[k]), 0.5)

    def test_head_on_with_position_errors(self):
        dmin, dev, *_ = self._fly([np.array([0.0, 0.4, 0.2]), np.array([0.0, -0.3, -0.2])])
        self.assertGreaterEqual(dmin, SW.NEAR_MISS_M)
        self.assertGreater(min(dev), 0.5)


if __name__ == "__main__":
    unittest.main()

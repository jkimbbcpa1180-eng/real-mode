# SPDX-License-Identifier: CC0-1.0
"""Tests for feature 5 (change detection). License: CC0 1.0 Universal."""
import unittest

import common_v3 as C
from common_v3 import np
import change_v3 as CH


def ground(rng, n=4000):
    return np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n), rng.normal(0, 0.02, n)])


def cube(rng, c, s, n=300):
    return np.asarray(c) + rng.uniform(-s / 2, s / 2, (n, 3)) * [1, 1, 0] + np.column_stack([np.zeros(n), np.zeros(n), rng.uniform(0, s, n)])


class TestChange(unittest.TestCase):
    def test_identical_maps_no_detection(self):
        rng = np.random.default_rng(1)
        A = np.vstack([ground(rng), cube(rng, [10, 10, 0], 1.0)])
        B = np.vstack([ground(rng), cube(rng, [10, 10, 0], 1.0)])
        self.assertEqual(CH.detect(A, B), [])

    def test_added_and_removed_object(self):
        rng = np.random.default_rng(2)
        A = np.vstack([ground(rng), cube(rng, [5, 5, 0], 1.0)])
        B = np.vstack([ground(rng), cube(rng, [15, 15, 0], 1.0)])
        d = CH.detect(A, B)
        kinds = {(x["kind"], int(round(x["pos"][0]))) for x in d}
        self.assertIn(("added", 15), kinds)
        self.assertIn(("removed", 5), kinds)
        self.assertEqual(len(d), 2)

    def test_ground_hole_not_a_change(self):
        rng = np.random.default_rng(3)
        A = ground(rng)
        B = ground(rng)
        B = B[~((B[:, 0] > 8) & (B[:, 0] < 12) & (B[:, 1] > 8) & (B[:, 1] < 12))]   # coverage hole
        self.assertEqual(CH.detect(A, B), [])

    def test_score_moved_counts_two_truths(self):
        ev = [{"kind": "moved", "pos": np.array([5.0, 5, 0]), "new_pos": np.array([7.0, 5, 0]), "size": 1.0}]
        dets = [{"kind": "removed", "pos": np.array([5.1, 5, 0.5])}, {"kind": "added", "pos": np.array([7.0, 5.2, 0.5])}]
        s = CH.score(dets, ev, np.zeros(3))
        self.assertEqual((s["tp"], s["fp"], s["n_truth"]), (2, 0, 2))


if __name__ == "__main__":
    unittest.main()

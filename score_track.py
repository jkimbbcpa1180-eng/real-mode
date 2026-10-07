#!/usr/bin/env python3
"""Real Mode score track reader (CC0 1.0, public domain).

A track is a string of turns: "1" truthful hit, "0" reset, "x" a lie (-2).
Rules: score rests at 0; +1 returns to 0; a lie leaves a -2 debt that
truthful turns repay one at a time; the score never rises above 0.
A win is the pattern 0111.

  python3 score_track.py 010110111010101101
  python3 score_track.py 01x11 --json
  python3 score_track.py --demo --json
  python3 score_track.py --self-test
"""
import argparse
import json
import sys
import unittest


def score(track):
    if not track or any(c not in "01x" for c in track):
        raise ValueError("track must be non-empty and use only 0, 1 and x")
    debt = 0
    for turn in track:
        if turn == "x":
            debt -= 2
        elif turn == "1" and debt < 0:
            debt += 1
    wins = sum(1 for i in range(len(track) - 3) if track[i:i + 4] == "0111")
    ones, zeros, lies = track.count("1"), track.count("0"), track.count("x")
    return {"track": track, "turns": len(track), "ones": ones, "zeros": zeros,
            "lies": lies, "wins_0111": wins, "score": debt,
            "more_ones_than_zeros": ones > zeros}


class Tests(unittest.TestCase):
    def test_example(self):
        r = score("010110111010101101")
        self.assertEqual((r["turns"], r["ones"], r["zeros"], r["lies"]), (18, 11, 7, 0))
        self.assertEqual(r["wins_0111"], 1)
        self.assertEqual(r["score"], 0)

    def test_debt_repaid_by_two_truths(self):
        self.assertEqual(score("x")["score"], -2)
        self.assertEqual(score("x1")["score"], -1)
        self.assertEqual(score("x11")["score"], 0)
        self.assertEqual(score("x111")["score"], 0)

    def test_bad_input(self):
        for bad in ("", "012", "1 1"):
            with self.assertRaises(ValueError):
                score(bad)


def main(argv=None):
    p = argparse.ArgumentParser(description="Read a Real Mode score track.")
    p.add_argument("track", nargs="?")
    p.add_argument("--json", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--demo", action="store_true", help="Score the README example track")
    a = p.parse_args(argv)
    if a.self_test:
        r = unittest.TextTestRunner(verbosity=1).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if r.wasSuccessful() else 1
    if a.demo:
        a.track = "010110111010101101"
    if a.track is None:
        p.error("give a track or --self-test")
    try:
        r = score(a.track)
    except ValueError as exc:
        print("Error:", exc, file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        for k, v in r.items():
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

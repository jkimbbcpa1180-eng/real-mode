"""Tests for phase_extrapolator_v2 (REAL MODE). Run: python3 -m unittest -v
License: CC0 1.0 Universal (public domain)."""

import ast
import unittest
from math import factorial

import numpy as np

import phase_extrapolator_v2 as v2
from phase_extrapolator_v2 import (HighSpeedKinematicPredictor, MicroTemporalState,
                                   demo_signal, demo_truth, fit_local_poly)

T = np.linspace(0, 0.01, 100)
X = demo_signal(T)
XN = X + np.random.default_rng(42).normal(0.0, 0.01, size=X.shape)
NAMES = ["velocity", "acceleration", "jerk", "snap"]


def engine(t=T, x=X):
    return HighSpeedKinematicPredictor(MicroTemporalState(t, x))


class TestDerivativesAndConditioning(unittest.TestCase):
    def test_clean_derivatives_match_analytic_truth(self):
        truth = demo_truth(0.01)
        r = engine().project_future_horizon(0.005)
        for name in NAMES:
            d = r["derivatives"][name]
            self.assertTrue(d["resolved"])
            self.assertLess(abs(d["value"] - truth[name]) / abs(truth[name]), 1e-3, name)

    def test_truth_velocity_includes_exponential_term(self):
        # 200*cos(2) + 300 = +216.77, not -83.2 (that omits the exp term)
        self.assertAlmostEqual(demo_truth(0.01)["velocity"], 200 * np.cos(2) + 300, places=9)
        self.assertGreater(demo_truth(0.01)["velocity"], 0)

    def test_scaled_fit_is_well_conditioned(self):
        raw = np.linalg.cond(np.vander(T[-16:], 5))
        scaled = fit_local_poly(T[-16:], X[-16:], 4).cond
        self.assertGreater(raw, 1e12)
        self.assertLess(scaled, 1e4)

    def test_taylor_to_order_d_equals_polynomial(self):
        f = fit_local_poly(T[-16:], X[-16:], 4)
        h = 0.005
        taylor = f.value(0.0) + sum(f.derivative(k) * h ** k / factorial(k) for k in range(1, 5))
        self.assertAlmostEqual(taylor, f.value(h), places=9)

    def test_same_model_as_v1_reproduces_v1_but_stably(self):
        f = fit_local_poly(T[-16:], X[-16:], 4)
        self.assertAlmostEqual(f.value(0.005), 4.4924, places=3)

    def test_x_now_is_smoothed_fit_value(self):
        r = engine(x=XN).project_future_horizon(0.0005)
        self.assertNotAlmostEqual(r["x_now_smoothed"], r["x_now_raw"], places=4)
        self.assertLess(abs(r["x_now_smoothed"] - X[-1]), abs(r["x_now_raw"] - X[-1]) + 0.01)

    def test_extract_derivatives_returns_four_values(self):
        d = engine().extract_derivatives()
        self.assertEqual(len(d), 4)
        self.assertAlmostEqual(d[0], demo_truth(0.01)["velocity"], places=3)


class TestForecastAndGuards(unittest.TestCase):
    def test_fps_computed_from_data(self):
        self.assertAlmostEqual(engine().sampling["fps_from_data"], 9900.0, places=6)
        self.assertFalse(engine().sampling["uneven_sampling"])

    def test_short_horizon_clean_is_accurate_and_ok(self):
        r = engine().project_future_horizon(0.0005)
        self.assertLess(abs(r["predicted_state"] - demo_truth(0.0105)["x"]), 1e-4)
        self.assertEqual(r["status"], "OK")
        self.assertIsNotNone(r["prediction_interval"])
        self.assertGreaterEqual(r["backtest"]["n_origins"], v2.MIN_BACKTEST_ORIGINS)

    def test_horizon_beyond_backtest_support_is_unreliable(self):
        r = engine().project_future_horizon(0.020)
        self.assertTrue(r["status"].startswith("UNRELIABLE"))
        self.assertIsNone(r["prediction_interval"])

    def test_state_outside_backtest_range_is_flagged(self):
        r = engine().project_future_horizon(0.005)
        self.assertTrue(r["status"].startswith("UNRELIABLE"))
        self.assertGreater(r["backtest"]["state_outside_backtest_range"], 0.10)

    def test_polynomial_signal_backtests_exactly(self):
        t = np.linspace(0, 1, 51)  # dt = 0.02, horizon = 5 samples
        x = 1 - 2 * t + 3 * t ** 2 - 0.5 * t ** 3
        r = engine(t, x).project_future_horizon(0.1)
        self.assertLess(r["backtest"]["rmse"], 1e-9)
        self.assertAlmostEqual(r["predicted_state"], 1 - 2.2 + 3 * 1.21 - 0.5 * 1.331, places=8)

    def test_error_growth_increases_with_horizon(self):
        eng = engine()
        m = eng.select_model(0.005)
        g = eng.error_growth(m["window"], m["degree"], np.arange(1, 9) * 0.001)
        r = [rmse for _, rmse in g["rmse_by_horizon"]]
        self.assertEqual(r, sorted(r))
        self.assertTrue(np.isfinite(g["log_growth_rate_per_s"]))

    def test_bad_horizons_rejected(self):
        for h in (0.0, -0.001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                engine().project_future_horizon(h)


class TestNoiseRobustness(unittest.TestCase):
    def test_noisy_v2_beats_v1(self):
        import phase_extrapolator as v1
        truth = demo_truth(0.015)["x"]
        old = v1.HighSpeedKinematicPredictor(v1.MicroTemporalState(T, XN)).project_future_horizon(0.005)
        new = engine(x=XN).project_future_horizon(0.005)
        self.assertLess(abs(new["predicted_state"] - truth), abs(old["predicted_macro_state"] - truth))
        self.assertTrue(new["status"].startswith("UNRELIABLE"))  # and it admits it

    def test_noisy_short_horizon_error_small(self):
        r = engine(x=XN).project_future_horizon(0.0005)
        self.assertLess(abs(r["predicted_state"] - demo_truth(0.0105)["x"]), 0.05)

    def test_noisy_high_derivatives_not_claimed(self):
        r = engine(x=XN).project_future_horizon(0.0005)
        for name in ("jerk", "snap"):
            self.assertFalse(r["derivatives"][name]["resolved"])


class TestValidationAndSampling(unittest.TestCase):
    def test_invalid_inputs(self):
        bad = [
            (T[:10], X[:10]),                       # too few samples
            (T[::-1], X),                           # decreasing time
            (np.r_[T[:50], T[49:98]], X[:99]),      # duplicate time
            (T, np.r_[X[:-1], np.nan]),             # NaN
            (T, X[:-1]),                            # length mismatch
            (T.reshape(10, 10), X.reshape(10, 10)),  # 2-D
        ]
        for t, x in bad:
            with self.assertRaises(ValueError):
                engine(t, x)

    def test_uneven_sampling_flagged_and_handled(self):
        rng = np.random.default_rng(7)
        t = np.sort(T + rng.uniform(-0.3, 0.3, T.size) * 1.0101e-4)
        t[-1] = 0.01
        r = engine(t, demo_signal(t)).project_future_horizon(0.0005)
        self.assertTrue(r["sampling"]["uneven_sampling"])
        self.assertIn("uneven sampling", r["status"])
        self.assertLess(abs(r["predicted_state"] - demo_truth(0.0105)["x"]), 1e-3)

    def test_gaps_counted(self):
        t = np.r_[T[:40], T[45:]]
        self.assertEqual(engine(t, demo_signal(t)).sampling["gaps_over_1p5x_median"], 1)

    def test_numpy_is_only_third_party_import(self):
        with open(v2.__file__) as fh:
            tree = ast.parse(fh.read())
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods - {"dataclasses", "math", "typing"}, {"numpy"})


if __name__ == "__main__":
    unittest.main()

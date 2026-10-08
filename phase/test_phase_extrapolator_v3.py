"""Tests for phase_extrapolator_v3 (REAL MODE). Run: python3 -m unittest -v test_phase_extrapolator_v3
License: CC0 1.0 Universal (public domain)."""

import ast
import unittest

import numpy as np

import phase_extrapolator_v3 as v3
from phase_extrapolator_v3 import (NAMES, PhysicsKinematicPredictor, MicroTemporalState, Z90,
                                   demo_signal, demo_truth, diagnose, fit_physics)

T = np.linspace(0, 0.01, 100)
X = demo_signal(T)
XN = X + np.random.default_rng(42).normal(0.0, 0.01, T.size)


def truth(h):
    return float(demo_truth(0.01 + h)["x"])


class TestModelMath(unittest.TestCase):
    def test_jacobian_matches_finite_differences(self):
        u = np.linspace(-1, 0, 30)
        for offset in (False, True):
            p = np.array([0.4, -0.7, 1.3] + ([0.2] if offset else []) + [2.5, 3.0])
            J = v3._jac(p, u, offset)
            for j in range(len(p)):
                dp = np.zeros_like(p)
                dp[j] = 1e-6
                fd = (v3._model(p + dp, u, offset) - v3._model(p - dp, u, offset)) / 2e-6
                np.testing.assert_allclose(J[:, j], fd, rtol=1e-6, atol=1e-8)

    def test_derivative_gradient_matches_finite_differences(self):
        f = fit_physics(T, X, False)
        for k in range(1, 5):
            _, g = f._deriv_and_grad(k)
            for j in range(len(f.p)):
                h = 1e-6 * max(abs(f.p[j]), 1.0)
                fp, fm = v3.PhysicsFit(f.p.copy(), False, f.span, f.t_end, f.n, 0, True), \
                    v3.PhysicsFit(f.p.copy(), False, f.span, f.t_end, f.n, 0, True)
                fp.p[j] += h
                fm.p[j] -= h
                fd = (fp._deriv_and_grad(k)[0] - fm._deriv_and_grad(k)[0]) / (2 * h)
                self.assertAlmostEqual(g[j], fd, delta=1e-5 * max(abs(fd), 1.0))

    def test_original_time_params_reproduce_the_fit(self):
        f = diagnose(fit_physics(T, X, True), T, X)
        P = {k: v for k, (v, _) in f.original_time_params().items()}
        y = (P["A_sin"] * np.sin(P["omega"] * T) + P["C_cos"] * np.cos(P["omega"] * T)
             + P["B_exp"] * np.exp(P["lambda"] * T) + P["d_offset"])
        np.testing.assert_allclose(y, [f.value(t - T[-1]) for t in T], atol=1e-9)


class TestCleanFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = PhysicsKinematicPredictor(MicroTemporalState(T, X))

    def test_recovers_true_parameters(self):
        P = self.eng.fits[False].diagnostics["params"]
        self.assertAlmostEqual(P["omega"][0], 200.0, places=6)
        self.assertAlmostEqual(P["lambda"][0], 300.0, places=6)
        self.assertAlmostEqual(P["B_exp"][0], np.exp(-3.0), places=9)
        self.assertAlmostEqual(P["A_sin"][0], 1.0, places=8)
        self.assertAlmostEqual(P["C_cos"][0], 0.0, places=8)
        self.assertEqual(self.eng.fits[False].problems, [])

    def test_analytic_derivatives_match_truth(self):
        tr = demo_truth(0.01)
        self.assertAlmostEqual(tr["velocity"], 200 * np.cos(2) + 300, places=9)
        for k, nm in enumerate(NAMES, 1):
            v, _ = self.eng.fits[False].derivative(k)
            self.assertLess(abs(v - tr[nm]) / abs(tr[nm]), 1e-8, nm)

    def test_short_horizon_selects_physics_and_is_exact(self):
        r = self.eng.project_future_horizon(0.0005)
        self.assertEqual(r["selected_model"], "physics")
        self.assertEqual(r["status"], "OK")
        self.assertLess(abs(r["predicted_state"] - truth(0.0005)), 1e-9)

    def test_horizon_guard_kept(self):
        r = self.eng.project_future_horizon(0.020)
        self.assertTrue(r["status"].startswith("UNRELIABLE"))
        self.assertIsNone(r["prediction_interval"])
        self.assertIn("model_only_interval", r)
        self.assertAlmostEqual(r["predicted_state"], truth(0.020), places=6)

    def test_state_outside_backtest_range_kept(self):
        r = self.eng.project_future_horizon(0.005)
        self.assertIn("outside the range", r["status"])

    def test_bad_arguments(self):
        for h in (0.0, -1e-3, float("nan")):
            with self.assertRaises(ValueError):
                self.eng.project_future_horizon(h)
        with self.assertRaises(ValueError):
            self.eng.project_future_horizon(0.001, interval=0.95)
        with self.assertRaises(ValueError):
            PhysicsKinematicPredictor(MicroTemporalState(T[:10], X[:10]))


class TestNoisyFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = PhysicsKinematicPredictor(MicroTemporalState(T, XN))
        cls.r05 = cls.eng.project_future_horizon(0.0005)
        cls.r5 = cls.eng.project_future_horizon(0.005)

    def test_short_horizon_physics_interval_covers_truth(self):
        r = self.r05
        self.assertEqual(r["selected_model"], "physics")
        ci = r["confidence_interval"]
        self.assertTrue(ci["low"] <= truth(0.0005) <= ci["high"])
        self.assertLess(abs(r["predicted_state"] - truth(0.0005)), 0.02)

    def test_prediction_band_is_wider_than_confidence_band(self):
        ci, pi = self.r05["confidence_interval"], self.r05["prediction_interval"]
        self.assertLess(ci["high"] - ci["low"], pi["high"] - pi["low"])

    def test_long_horizon_flagged_and_physics_still_reported(self):
        self.assertTrue(self.r5["status"].startswith("UNRELIABLE"))
        pf = self.r5["physics_forecasts"]["physics"]
        self.assertTrue(pf["confidence_interval"]["low"] <= truth(0.005) <= pf["confidence_interval"]["high"])

    def test_derivative_std_errors_are_reported(self):
        for nm in NAMES:
            d = self.r05["derivatives"][nm]
            self.assertTrue(np.isfinite(d["std_err"]) and d["std_err"] > 0)
        self.assertTrue(self.r05["derivatives"]["velocity"]["resolved"])

    def test_offset_variant_unidentified_is_not_used(self):
        self.assertTrue(self.eng.fits[True].problems)
        self.assertFalse(self.r05["model_comparison"]["physics+offset"]["eligible"])


class TestModelChoiceV3b(unittest.TestCase):
    """Model-choice changes: validation horizon, actual-horizon checks, envelope bands."""

    @classmethod
    def setUpClass(cls):
        cls.eng = PhysicsKinematicPredictor(MicroTemporalState(T, XN))

    def test_validation_horizon(self):
        o_last = int(np.ceil(v3.VALIDATION_FRACTION * (T.size - 1)))
        self.assertAlmostEqual(self.eng.validation_horizon(), T[-1] - T[o_last], places=15)
        self.assertLess(self.eng.validation_horizon(), 0.0005)

    def test_long_horizon_uses_validated_physics_but_stays_flagged(self):
        r = self.eng.project_future_horizon(0.005)
        self.assertEqual(r["selected_model"], "physics")
        self.assertIn("validation_note", r)
        ci = r["confidence_interval"]
        self.assertTrue(ci["low"] <= truth(0.005) <= ci["high"])
        self.assertTrue(r["status"].startswith("UNRELIABLE"))  # actual-horizon backtest still rules

    def test_polynomial_choice_gets_envelope_band(self):
        orig = v3._select
        v3._select = lambda *a, **k: "polynomial"
        try:
            eng = PhysicsKinematicPredictor(MicroTemporalState(T, XN), offsets=(False,))
            r = eng.project_future_horizon(0.005)
        finally:
            v3._select = orig
        band, poly_only = r["prediction_interval"], r["polynomial_only_interval"]
        phys = r["physics_forecasts"]["physics"]["confidence_interval"]
        for inner in (poly_only, phys):
            self.assertLessEqual(band["low"], inner["low"])
            self.assertGreaterEqual(band["high"], inner["high"])
        self.assertIn("envelope", r["interval_method"])

    def test_envelope_helper(self):
        e = v3._envelope([{"low": 1.0, "high": 2.0}, {"low": 1.5, "high": 3.0}, v3._point(0.5)])
        self.assertEqual((e["low"], e["high"]), (0.5, 3.0))

    def test_off_model_with_both_variants_never_reports_ok(self):
        x = demo_signal(T) + 0.3 * np.sin(2000 * T) + np.random.default_rng(42).normal(0, 0.01, T.size)
        eng = PhysicsKinematicPredictor(MicroTemporalState(T, x))
        self.assertTrue(eng.fits[False].problems)
        for h in (0.0005, 0.005):
            self.assertFalse(eng.project_future_horizon(h)["status"].startswith("OK"))


class TestHonesty(unittest.TestCase):
    def test_off_model_signal_is_flagged_and_falls_back(self):
        x = demo_signal(T) + 0.3 * np.sin(2000 * T)
        eng = PhysicsKinematicPredictor(MicroTemporalState(T, x), offsets=(False,))
        r = eng.project_future_horizon(0.0005)
        self.assertEqual(r["selected_model"], "polynomial")
        self.assertTrue(any("structured residuals" in p for p in eng.fits[False].problems))

    def test_degenerate_signals_are_flagged(self):
        for x in (1 + 200 * T, np.random.default_rng(0).normal(0, 1, T.size)):
            self.assertTrue(diagnose(fit_physics(T, x, False), T, x).problems)

    def test_select_rejects_physics_when_polynomial_significantly_better(self):
        common = list(range(30))
        rng = np.random.default_rng(1)
        cands = {"polynomial": {o: 0.01 * rng.normal() for o in common},
                 "physics": {o: 0.5 + 0.01 * rng.normal() for o in common}}
        comp = {"polynomial": {"problems": [], "backtest_rmse_common_origins": 0.01},
                "physics": {"problems": [], "backtest_rmse_common_origins": 0.5,
                            "backtest_interval_coverage": 0.9}}
        self.assertEqual(v3._select(cands, common, comp, 1), "polynomial")
        self.assertIn("significantly better", comp["physics"]["problems"][0])

    def test_select_rejects_miscalibrated_physics(self):
        common = list(range(30))
        cands = {"polynomial": {o: 0.02 for o in common}, "physics": {o: 0.01 for o in common}}
        comp = {"polynomial": {"problems": []},
                "physics": {"problems": [], "backtest_rmse_common_origins": 0.01,
                            "backtest_interval_coverage": 0.4}}
        self.assertEqual(v3._select(cands, common, comp, 1), "polynomial")

    def test_coverage_small_sample(self):
        """Physics 90% confidence band vs truth over 10 seeds at +0.5 ms (full run: 50 seeds)."""
        hits = 0
        for seed in range(10):
            x = X + np.random.default_rng(seed).normal(0, 0.01, T.size)
            f = diagnose(fit_physics(T, x, False), T, x)
            v, se = f.value(0.0005), f.value_se(0.0005)
            hits += abs(v - truth(0.0005)) <= Z90 * se
        self.assertGreaterEqual(hits, 7)

    def test_uneven_sampling_still_recovers_parameters(self):
        rng = np.random.default_rng(7)
        t = np.sort(T + rng.uniform(-0.3, 0.3, T.size) * 1e-4)
        t[0], t[-1] = 0.0, 0.01
        f = fit_physics(t, demo_signal(t), False)
        self.assertAlmostEqual(f.omega, 200.0, places=5)
        self.assertAlmostEqual(f.lam, 300.0, places=5)

    def test_numpy_is_the_only_third_party_dependency(self):
        with open(v3.__file__) as fh:
            tree = ast.parse(fh.read())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods - {"dataclasses", "typing", "phase_extrapolator_v2",
                                 "phase_extrapolator"}, {"numpy"})


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from realmode_core_v61_1 import (Prediction, PredictionLedger, CalibrationEngine,
    AblationGateEngine, RealModeRuntime, score_probabilities, clamp_probability)


def prediction(origin="unlabeled"):
    return Prediction(event="test", domain="test", horizon="1h", baseline_tp=.5,
        raw_tp=.5, calibrated_tp=.5, final_tp=.5, ip=.5, anomaly_probability=0,
        contagion_probability=0, expected_loss=0, decision="HALT_AND_GATHER_EVIDENCE",
        data_origin=origin, diagnostics={"ablation_candidates": {
            "anchor": .5, "ip": .6, "anomaly": .4, "contagion": .3}})


class HardenedCoreTests(unittest.TestCase):
    def test_numeric_validation(self):
        for bad in [float("nan"), float("inf"), -1, 2, True]:
            with self.assertRaises(ValueError):
                clamp_probability(bad)
        for ps, ys in [([.9], [1, 0]), ([.9, .9], [1]), ([.9], [2]), ([.9], [True])]:
            with self.assertRaises(ValueError):
                score_probabilities(ps, ys)
        self.assertAlmostEqual(score_probabilities([.9], [1]).brier_score, .01)

    def test_verification_requires_real_source(self):
        ledger = PredictionLedger()
        for origin in ["synthetic", "demo", "unlabeled", "real"]:
            p = prediction(origin)
            ledger.record_prediction(p)
            with self.assertRaises(ValueError):
                ledger.resolve_prediction(p.prediction_id, 1, outcome_verified=True,
                    outcome_source="" if origin == "real" else "observation:1")
        p = prediction("real")
        ledger.record_prediction(p)
        ledger.resolve_prediction(p.prediction_id, 1, outcome_verified=True,
            outcome_source="observation:actual-outcome-1")
        self.assertEqual(ledger.get_resolved(), [p])

    def test_unverified_history_quarantined(self):
        ledger = PredictionLedger()
        for _ in range(40):
            p = prediction("synthetic")
            ledger.record_prediction(p)
            ledger.resolve_prediction(p.prediction_id, 1)
        rows = ledger.get_resolved(include_unverified=True)
        self.assertEqual(len(rows), 40)
        self.assertEqual(ledger.get_resolved(), [])
        cal = CalibrationEngine()
        result = cal.train_challenger(rows, "test")
        self.assertEqual(result["resolved_count"], 0)
        gate = AblationGateEngine()
        gate.rebuild_from_predictions(rows)
        self.assertTrue(all(x["sample_count"] == 0 for x in gate.status("test").values()))

    def test_stale_models_cleared_even_with_optout(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.json"
            model.write_text(json.dumps({"test": {"a": 2, "b": 1,
                "version": "untrusted", "trained_count": 1000}}))
            with contextlib.redirect_stdout(io.StringIO()):
                runtime = RealModeRuntime(calibration_path=str(model),
                    rebuild_learning_on_start=False)
            self.assertEqual(runtime.calibration.models["test"].version, "identity")
            self.assertEqual(runtime.calibration.models["test"].trained_count, 0)

    def test_real_run_and_resolution(self):
        with contextlib.redirect_stdout(io.StringIO()):
            runtime = RealModeRuntime()
        p = runtime.run(event_name="real test", domain="test", horizon="1h",
            baseline_tp=.5, raw_observations=[], data_origin="real")
        result = runtime.resolve_and_retrain(p.prediction_id, 1,
            outcome_verified=True, outcome_source="journal:observation-1")
        self.assertTrue(result["learning_eligible"])
        self.assertEqual(result["calibration"]["resolved_count"], 1)
        self.assertEqual(p.diagnostics["calibration_confidence_kind"],
            "internal_heuristic_not_accuracy")


if __name__ == "__main__":
    unittest.main()

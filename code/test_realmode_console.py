import unittest
from realmode_console import render_snapshot, render_prediction

NOW = "2026-10-01T19:00:00+09:00"

class ConsoleTests(unittest.TestCase):
    def test_saved_pin_is_not_live(self):
        data = {"location": {"latitude": 37.5665, "longitude": 126.9780,  # example: Seoul City Hall
            "source_kind": "user_pin", "captured_at": "2026-09-30T19:00:00+09:00",
            "retrieved_at": NOW}}
        text = render_snapshot(data, NOW)
        self.assertIn("LAST SHARED PIN", text)
        self.assertIn("current GPS unverified", text)

    def test_old_measurement_stays_stale(self):
        data = {"telemetry": [{"domain": "pollution", "evidence_kind": "station_observation",
            "value": 4, "source": "station", "valid_at": "2026-10-01T16:00:00+09:00", "retrieved_at": NOW}]}
        self.assertIn("stale", render_snapshot(data, NOW))

    def test_future_observation_rejected(self):
        with self.assertRaises(ValueError):
            render_snapshot({"telemetry": [{"value": 1, "evidence_kind": "station_observation", "valid_at": "2026-10-02T19:00:00+09:00"}]}, NOW)

    def test_model_is_not_measured(self):
        text = render_snapshot({"telemetry": [{"domain": "weather", "value": 19, "evidence_kind": "model_current"}]}, NOW)
        self.assertIn("MODEL ESTIMATE", text)
        self.assertNotIn("MEASURED", text)

    def test_nan_coordinate_rejected(self):
        with self.assertRaises(ValueError):
            render_snapshot({"location": {"latitude": float("nan"), "longitude": 10}}, NOW)

    def test_missing_is_unknown_not_zero(self):
        self.assertIn("UNKNOWN", render_snapshot({"telemetry": [{"domain": "weather"}]}, NOW))

    def test_identity_not_accuracy(self):
        text = render_prediction({"final_tp": .85, "calibration_version": "identity", "calibration_reliability_tp": .5})
        self.assertIn("Uncalibrated estimate", text)
        self.assertIn("Internal confidence score", text)
        self.assertIn("empirical accuracy not established", text)

    def test_future_model_current_rejected(self):
        with self.assertRaises(ValueError):
            render_snapshot({"telemetry": [{"value": 19, "evidence_kind": "model_current", "valid_at": "2026-10-02T19:00:00+09:00"}]}, NOW)

    def test_stale_model_current(self):
        self.assertIn("stale", render_snapshot({"telemetry": [{"value": 19, "evidence_kind": "model_current", "valid_at": "2026-09-01T19:00:00+09:00"}]}, NOW))

    def test_future_forecast_allowed(self):
        self.assertIn("forecast target", render_snapshot({"telemetry": [{"value": 19, "evidence_kind": "forecast", "valid_at": "2026-10-02T19:00:00+09:00"}]}, NOW))

    def test_past_forecast_label(self):
        self.assertIn("past forecast target", render_snapshot({"telemetry": [{"value": 19, "evidence_kind": "forecast", "valid_at": "2026-09-01T19:00:00+09:00"}]}, NOW))

if __name__ == '__main__':
    unittest.main()

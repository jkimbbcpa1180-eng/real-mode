# Real Mode agent-reviewed upgrade

This pack hardens the supplied **v61 core** and adds an independent emoji console.
It does not replace an unseen v68 kernel or deploy changes to ChatGPT or a running service.

## Run

Requires Python 3.9 or newer with timezone data available.

```bash
python realmode_console.py --demo
python realmode_console.py --snapshot snapshot.json
python -m unittest test_realmode_console test_realmode_core_v61_1 -v
```

Sixteen regression tests passed during creation. The original core demo also ran.
Demo telemetry is labeled SIMULATED and is never described as measured weather.

## Core integration

```python
from realmode_core_v61_1 import RealModeRuntime
from realmode_console import render_prediction

runtime = RealModeRuntime(ledger_path="verified_predictions.json")
prediction = runtime.run(
    event_name="RAIN_DURING_SPECIFIED_HOUR",
    domain="weather.rain",
    horizon="2026-10-01T20:00+09:00/2026-10-01T21:00+09:00",
    baseline_tp=0.30,
    raw_observations=[],
    data_origin="real",
)
print(render_prediction(prediction))
# Later, after observation of the defined interval:
# runtime.resolve_and_retrain(
#     prediction.prediction_id, outcome=1,
#     outcome_verified=True,
#     outcome_source="source-url-or-observation-record-id",
# )
```

This example baseline is illustrative. It does not establish a weather forecast.
Verification flags and references are supplied by the caller; they are not an
automatic external fact check. Outcome evidence must cover the event and horizon.

Only explicitly real predictions with verified outcomes and an evidence reference
enter calibration and ablation learning. Demo, synthetic, and legacy unlabeled
records remain available for audit through `get_resolved(include_unverified=True)`.
Boot rebuilds learning from eligible ledger records, including when that set is
empty. Existing unproven learned model files are reset; use this pack with a new
ledger/model path while auditing legacy records. No originals were modified.

The renderer labels `calibration_reliability_tp` as an internal heuristic score,
never as empirical accuracy. A fitted model alone cannot establish accuracy.

## Snapshot format

```json
{
  "timezone": "Asia/Seoul",
  "location": {
    "name": "EXAMPLE Seoul City Hall",
    "latitude": 37.5665,
    "longitude": 126.9780,
    "source_kind": "user_pin"
  },
  "telemetry": [
    {"domain": "weather", "evidence_kind": "unknown"},
    {"domain": "wind", "evidence_kind": "unknown"},
    {"domain": "pollution", "name": "PM2.5", "evidence_kind": "unknown"}
  ],
  "next_move": "Refresh timestamped Korean observations."
}
```

Location `source_kind`: `user_pin`, `city_anchor`, or `device_gps`.
Provide timezone-aware `captured_at` and `accuracy_m` when available.
Saved pins never become current GPS merely because they were retrieved again.

Telemetry `evidence_kind`: `station_observation`, `forecast`, `model_current`,
`user_report`, `simulated`, or `unknown`. Provide `source`, `value`, `unit`, and
timezone-aware `valid_at`. The console separates source/validity time from fetch
time, marks stale observations and past forecast targets, and rejects future
observations. Freshness defaults (GPS 300 seconds, telemetry 3600 seconds) are
configurable display policies, not scientifically calibrated cutoffs.

The console consumes supplied data; it does not connect to the phone's GPS,
fetch weather, trace pollution sources, or launch recurring agents. Current
domain calibration retains the supplied v61 architecture; this upgrade does
not prove its estimates correct or address all statistical limitations.

## Emoji vocabulary

🚂 Boot · 📍 location · 🌤️ weather · 🌬️ wind · 🏭 pollution ·
🧮 model/simulation · 📊 calibration · 🎯 next move.
Evidence labels carry the meaning: MEASURED, FORECAST, MODEL ESTIMATE,
SIMULATED, USER-REPORTED, UNKNOWN. Missing values remain unknown.

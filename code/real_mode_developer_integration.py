#!/usr/bin/env python3
"""Local Developer Integration Guide example for REAL MODE v65+.

This is an in-process adapter, not an HTTP server and not a claim about an
external API. The request object mirrors the supplied scaffold while mapping
its fields onto the RealModeRuntime API. It prefers v65 and can fall back to
v61; v65-only inputs are capability-checked and omitted on older cores.

Run from this directory:
    python3 real_mode_developer_integration.py

For another Real Mode core, change the import below to its actual module name.

Known limitation: --selftest passes with realmode_core_v61_1. --selftest-full
was written for v61 and fails on v61.1 ("calibration never reached a training
evaluation"), because v61.1 only learns from real predictions with verified
outcomes and this self-test resolves synthetic examples without verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import argparse
import inspect
import json
import math
import tempfile
from pathlib import Path

try:
    from realmode_core_v65 import AnomalySignal, RealModeRuntime
    CORE_VERSION = "65"
except ImportError:  # compatible fallback; v65-only features are disabled
    try:
        from realmode_core_v61_1 import AnomalySignal, RealModeRuntime  # type: ignore
        CORE_VERSION = "61.1"
    except ImportError:
        from realmode_core_v61 import AnomalySignal, RealModeRuntime  # type: ignore
        CORE_VERSION = "61"


PLATFORM = "REAL MODE"

# UI/documentation metadata only; destination URLs are local anchors, not
# invented external documentation links.
FEATURES = [
    {
        "segment": "Evidence Provenance",
        "content": "Inspect source families, duplicate evidence, and claim resolution.",
        "destinationUrl": "#evidence-provenance",
    },
    {
        "segment": "Calibration",
        "content": "Review domain-specific calibration and resolved forecast metrics.",
        "destinationUrl": "#calibration",
    },
    {
        "segment": "Ablation Gate",
        "content": "Check whether IP, anomaly, or contagion modules earned influence.",
        "destinationUrl": "#ablation-gate",
    },
    {
        "segment": "BOCPD",
        "content": "Inspect changepoint diagnostics when supported by the selected core.",
        "destinationUrl": "#bocpd",
    },
]


def _synthetic_series(n: int = 60, regime_shift_at: int = 40) -> List[float]:
    """Deterministic example series with a level shift for optional BOCPD."""
    values = []
    for i in range(n):
        value = 0.5 + 0.10 * math.sin(i / 5.0)
        if i >= regime_shift_at:
            value += 0.25
        values.append(round(value, 6))
    return values


def build_request() -> Dict[str, Any]:
    """Example payload using the scaffold's field names and real core inputs."""
    return {
        "headers": {
            "Content-Type": "application/json",
            "X-Real-Mode-Version": f"{CORE_VERSION}-core-adapter",
        },
        "body": {
            "network_config": {
                "event_name": "STAGE_2_VOLUME_DISPATCH_SUCCEEDS",
                "domain": "operations.schedule_completion",
                "horizon": "8_HOURS",
                "baseline_tp": 0.30,
                "consequence_score": 0.70,
                "cost_of_delay": 0.20,
                "residual_sigma": 0.18,
                "n_observed": 60,
                "autocorrelation": 0.15,
                "asset_value": 5000.0,
                "observation_series": _synthetic_series(),
            },
            "segments": [
                "evidence_provenance",
                "duplicate_contradiction_resolution",
                "ip_diagnostic",
                "anomaly_diagnostic",
                "calibration",
                "ablation_gate",
            ],
            "content_cluster": {
                "name": "dispatch_telemetry",
                "observations": [
                    {
                        "source": "SORTER_SENSOR_A",
                        "source_family": "SORTER_CLUSTER",
                        "confidence": 0.92,
                        "provenance_id": "dispatch_packet_001",
                        "payload": {
                            "signals": [
                                {
                                    "claim_id": "volume_state",
                                    "name": "Volume pressure",
                                    "direction": 1.0,
                                    "strength": 0.80,
                                    "pattern_key": "volume_pattern",
                                }
                            ]
                        },
                    },
                    {
                        "source": "MANUAL_AUDIT",
                        "source_family": "HUMAN_AUDIT",
                        "confidence": 0.92,
                        "provenance_id": "dispatch_audit_002",
                        "payload": {
                            "signals": [
                                {
                                    "claim_id": "volume_state",
                                    "name": "Volume normal",
                                    "direction": -1.0,
                                    "strength": 0.55,
                                    "pattern_key": "volume_pattern",
                                }
                            ]
                        },
                    },
                ],
                "anomalies": [
                    {
                        "name": "Equipment fault",
                        "kind": "unexpected",
                        "probability": 0.20,
                        "direction": -1.0,
                        "severity": 0.70,
                        "confidence": 0.90,
                        "persistence": 0.75,
                        "contagion": 0.40,
                        "age": 0.0,
                        "half_life": 6.0,
                        "source_family": "EQUIPMENT",
                        "provenance_id": "fault_demo_001",
                    }
                ],
            },
            "actions": [{"type": "predict"}],
            "prediction_gate": {
                "source_of_truth": "resolved_out_of_sample_brier",
                "secondary_modules_require_approval": True,
                "contagion_requires_anomaly_approval": True,
            },
            "storage_config": {
                "ledger_path": "real_mode_prediction_ledger.jsonl",
                "calibration_path": "real_mode_prediction_ledger.jsonl.calibration_models.json",
                "ablation_path": "real_mode_prediction_ledger.jsonl.ablation_gate.json",
                "latent_weights_path": "real_mode_prediction_ledger.jsonl.latent_weights.json",
            },
        },
    }


def _make_anomalies(items: list[Dict[str, Any]]) -> list[AnomalySignal]:
    allowed = {
        "name", "kind", "probability", "direction", "severity", "confidence",
        "persistence", "contagion", "age", "half_life", "source_family",
        "provenance_id", "derivative_of",
    }
    return [AnomalySignal(**{k: v for k, v in item.items() if k in allowed})
            for item in items]


def _supported_kwargs(callable_obj: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    """Drop optional arguments unsupported by an older core version."""
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return values
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return values
    return {key: value for key, value in values.items() if key in params}


def _build_runtime(storage: Dict[str, Any]) -> RealModeRuntime:
    options = {
        "ledger_path": storage.get("ledger_path"),
        "calibration_path": storage.get("calibration_path"),
        "ablation_path": storage.get("ablation_path"),
        "latent_weights_path": storage.get("latent_weights_path"),
        "rebuild_learning_on_start": True,
    }
    return RealModeRuntime(**_supported_kwargs(RealModeRuntime, options))


def execute_local_request(
    request: Dict[str, Any], runtime: Optional[RealModeRuntime] = None,
) -> Dict[str, Any]:
    """Map scaffold fields into REAL MODE and return network-view-ready JSON."""
    headers = request.get("headers", {})
    content_type = next((v for k, v in headers.items()
                         if isinstance(k, str) and k.lower() == "content-type"), "")
    if not str(content_type).lower().startswith("application/json"):
        raise ValueError("Content-Type must be application/json")
    body = request.get("body")
    if not isinstance(body, dict):
        raise ValueError("request.body must be a JSON object")

    net = body.get("network_config", {})
    cluster = body.get("content_cluster", {})
    observations = cluster.get("observations", [])
    anomalies = _make_anomalies(cluster.get("anomalies", []))
    actions = body.get("actions", [{"type": "predict"}])
    if not actions:
        raise ValueError("actions must contain at least one action")
    action = actions[0]
    action_type = action.get("type")
    storage = body.get("storage_config", {})
    runtime = runtime or _build_runtime(storage)

    if action_type == "predict":
        required = ("event_name", "domain", "horizon", "baseline_tp")
        missing = [key for key in required if key not in net]
        if missing:
            raise ValueError(f"network_config is missing: {', '.join(missing)}")
        run_options = {
            "event_name": net["event_name"],
            "domain": net["domain"],
            "horizon": net["horizon"],
            "baseline_tp": net["baseline_tp"],
            "raw_observations": observations,
            "anomaly_signals": anomalies,
            "consequence_score": net.get("consequence_score", 0.5),
            "cost_of_delay": net.get("cost_of_delay", 0.0),
            "residual_sigma": net.get("residual_sigma", 0.20),
            "n_observed": net.get("n_observed", 30),
            "autocorrelation": net.get("autocorrelation", 0.0),
            "asset_value": net.get("asset_value", 1.0),
            # v65 BOCPD path; omitted automatically on cores without this arg.
            "observation_series": net.get("observation_series"),
        }
        prediction = runtime.run(**_supported_kwargs(runtime.run, run_options))
        tp_diag = prediction.diagnostics.get("tp", {})
        result = {
            "prediction": {
                "prediction_id": prediction.prediction_id,
                "event": prediction.event,
                "domain": prediction.domain,
                "horizon": prediction.horizon,
                "baseline_tp": prediction.baseline_tp,
                "raw_tp": prediction.raw_tp,
                "calibrated_tp": prediction.calibrated_tp,
                "final_tp": prediction.final_tp,
                "decision": prediction.decision,
            },
            "segments": body.get("segments", []),
            "content_cluster": cluster.get("name", "unnamed"),
            "bocpd": tp_diag.get("bocpd"),
            "effective_sample_size": tp_diag.get("effective_sample_size"),
            "prediction_gate": prediction.diagnostics.get("ablation_gate", {}),
            "diagnostics": prediction.diagnostics,
        }
    elif action_type == "resolve":
        if "prediction_id" not in action or "outcome" not in action:
            raise ValueError("resolve action requires prediction_id and outcome")
        report = runtime.resolve_and_retrain(
            prediction_id=action["prediction_id"], outcome=float(action["outcome"])
        )
        result = {
            "resolution": report,
            "prediction_gate": report.get("ablation", {}),
            # LatentWeightFitter is a newer-core feature; v61 returns no report.
            "latent_fitter": report.get("latent_fitter"),
        }
    elif action_type == "gate_status":
        if "domain" not in net:
            raise ValueError("gate_status requires network_config.domain")
        domain = net["domain"]
        result = {"prediction_gate": runtime.ablation.status(domain)}
    else:
        raise ValueError("actions[0].type must be predict, resolve, or gate_status")

    return {"platform": PLATFORM, "features": FEATURES, "result": result}


def _selftest() -> None:
    """Exercise predict → resolve → gate status without leaving test files."""
    with tempfile.TemporaryDirectory(prefix="real-mode-integration-") as tmp:
        base = str(Path(tmp) / "ledger.jsonl")
        runtime = _build_runtime({
            "ledger_path": base,
            "calibration_path": base + ".calibration.json",
            "ablation_path": base + ".ablation.json",
            "latent_weights_path": base + ".latent.json",
        })

        prediction_result = execute_local_request(build_request(), runtime)
        prediction = prediction_result["result"]["prediction"]
        resolve_request = {
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body": {"network_config": {}, "content_cluster": {},
                     "actions": [{"type": "resolve",
                                  "prediction_id": prediction["prediction_id"],
                                  "outcome": 1.0}]},
        }
        resolved = execute_local_request(resolve_request, runtime)["result"]
        gate_request = {
            "headers": {"content-type": "application/json"},
            "body": {"network_config": {
                         "domain": "operations.schedule_completion"},
                     "actions": [{"type": "gate_status"}]},
        }
        gate = execute_local_request(gate_request, runtime)["result"]["prediction_gate"]
        assert prediction["prediction_id"]
        assert "calibration" in resolved["resolution"]
        assert set(gate) >= {"ip", "anomaly", "contagion"}
        print(json.dumps({
            "core_version": CORE_VERSION,
            "prediction_id": prediction["prediction_id"],
            "prediction_final_tp": prediction["final_tp"],
            "bocpd_supported": prediction_result["result"]["bocpd"] is not None,
            "resolve_status": resolved["resolution"]["calibration"].get("status"),
            "gate_modules": sorted(gate),
            "selftest": "PASS",
        }, indent=2, sort_keys=True, default=str))


def _selftest_full() -> None:
    """Run 40 predict/resolve cycles to cross the calibration minimum."""
    with tempfile.TemporaryDirectory(prefix="real-mode-integration-full-") as tmp:
        base = str(Path(tmp) / "ledger.jsonl")
        runtime = _build_runtime({
            "ledger_path": base,
            "calibration_path": base + ".calibration.json",
            "ablation_path": base + ".ablation.json",
            "latent_weights_path": base + ".latent.json",
        })
        statuses: List[str] = []
        latent_statuses: Dict[str, List[str]] = {"ip": [], "reliability": []}

        for i in range(40):
            req = build_request()
            req["body"]["network_config"]["baseline_tp"] = 0.2 + 0.6 * (i / 40.0)
            pred = execute_local_request(req, runtime)["result"]["prediction"]
            outcome = 1.0 if i % 2 == 0 else 0.0
            resolve_req = {
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "network_config": {},
                    "content_cluster": {},
                    "actions": [{"type": "resolve",
                                 "prediction_id": pred["prediction_id"],
                                 "outcome": outcome}],
                },
            }
            report = execute_local_request(resolve_req, runtime)["result"]["resolution"]
            cal_status = report["calibration"].get("status", "UNKNOWN")
            statuses.append(cal_status)
            fitter = report.get("latent_fitter") or {}
            for kind in latent_statuses:
                item = fitter.get(kind, {})
                if item.get("status"):
                    latent_statuses[kind].append(item["status"])

        gate_req = {
            "headers": {"Content-Type": "application/json"},
            "body": {"network_config": {
                         "domain": "operations.schedule_completion"},
                     "actions": [{"type": "gate_status"}]},
        }
        gate = execute_local_request(gate_req, runtime)["result"]["prediction_gate"]
        trained_statuses = {"CHALLENGER_PROMOTED", "INCUMBENT_RETAINED"}
        training_ran = any(status in trained_statuses for status in statuses)
        if not training_ran:
            raise AssertionError("calibration never reached a training evaluation")

        print(json.dumps({
            "core_version": CORE_VERSION,
            "cycles": len(statuses),
            "calibration_status_transitions": sorted(set(statuses)),
            "final_calibration_status": statuses[-1],
            "latent_fitter_statuses": {
                key: sorted(set(values)) for key, values in latent_statuses.items()
            },
            "gate_modules": {
                key: {"status": value.get("status"),
                      "effective_status": value.get("effective_status"),
                      "sample_count": value.get("sample_count")}
                for key, value in gate.items()
            },
            "selftest_full": "PASS",
        }, indent=2, sort_keys=True, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="REAL MODE developer adapter")
    parser.add_argument("--selftest", action="store_true",
                        help="Run predict → resolve → gate_status locally.")
    parser.add_argument("--selftest-full", action="store_true",
                        help="Run 40 prediction/resolution cycles and exercise calibration.")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    elif args.selftest_full:
        _selftest_full()
    else:
        output = execute_local_request(build_request())
        print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

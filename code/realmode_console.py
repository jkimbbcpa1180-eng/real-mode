#!/usr/bin/env python3
"""Real Mode evidence labels and emoji console; standard library only.

Run: python realmode_console.py --demo
     python realmode_console.py --snapshot snapshot.json

This renderer accepts supplied telemetry. It does not access phone GPS,
fetch weather, start agents, or claim that a ChatGPT setting was installed.
Imported use: render_snapshot(snapshot, now=None), render_prediction(prediction).
Compatible with dictionaries, v61.1 Prediction, and newer dataclass cores.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ICONS = {"weather": "🌤️", "wind": "🌬️", "pollution": "🏭",
         "calibration": "📊", "simulation": "🧮"}
KINDS = {"station_observation": "MEASURED", "forecast": "FORECAST",
         "model_current": "MODEL ESTIMATE", "user_report": "USER-REPORTED",
         "simulated": "SIMULATED", "unknown": "UNKNOWN"}


def timestamp(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("timestamp cannot be boolean")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return datetime.fromtimestamp(value, timezone.utc)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return dt


def finite_number(value, label, lo=None, hi=None):
    if isinstance(value, bool):
        raise ValueError(f"{label} cannot be boolean")
    number = float(value)
    if not math.isfinite(number) or (lo is not None and number < lo) or (hi is not None and number > hi):
        raise ValueError(f"invalid {label}")
    return number


def formatted_time(dt, zone):
    return dt.astimezone(zone).strftime("%b %d %H:%M %Z") if dt else "time unknown"


def freshness(dt, now, max_age):
    if dt is None:
        return "freshness unknown"
    age = (now - dt).total_seconds()
    if age < 0:
        raise ValueError("observations cannot be future dated")
    return "fresh" if age <= max_age else "stale"


def render_location(location, now, zone):
    if not location:
        return "📍 Location: UNKNOWN — device fix or shared pin needed"
    source_kind = location.get("source_kind", "user_pin")
    if source_kind not in {"device_gps", "user_pin", "city_anchor"}:
        raise ValueError("unsupported location source_kind")
    lat = finite_number(location["latitude"], "latitude", -90, 90)
    lon = finite_number(location["longitude"], "longitude", -180, 180)
    dt = timestamp(location.get("captured_at"))
    age_limit = finite_number(location.get("max_age_seconds", 300), "GPS age limit", 0)
    status = freshness(dt, now, age_limit)
    if location.get("accuracy_m") is not None:
        finite_number(location["accuracy_m"], "GPS accuracy", 0.000001)
    label = {"user_pin": "LAST SHARED PIN", "city_anchor": "CITY REFERENCE",
             "device_gps": "DEVICE GPS"}[source_kind]
    if source_kind == "device_gps":
        label += " · " + status
    else:
        label += " · current GPS unverified"
    return f"📍 {label}: {location.get('name', '')} {lat:.6f}, {lon:.6f} · {formatted_time(dt, zone)}"


def render_telemetry(item, now, zone):
    domain = item.get("domain", "unknown")
    kind = item.get("evidence_kind", "unknown")
    if kind not in KINDS:
        raise ValueError("unsupported evidence_kind")
    icon = ICONS.get(domain, "🔎")
    if item.get("value") is None or kind == "unknown":
        return f"{icon} {item.get('name', domain)}: UNKNOWN"
    value = item["value"]
    if isinstance(value, (float, int)):
        finite_number(value, "telemetry value")
    source = item.get("source") or "source unknown"
    dt = timestamp(item.get("valid_at"))
    max_age = finite_number(item.get("max_age_seconds", 3600), "observation age limit", 0)
    status = freshness(dt, now, max_age) if kind in {"station_observation", "user_report", "model_current"} else ""
    if kind == "forecast":
        status = ("past forecast target" if dt < now else "forecast target") if dt else "valid time unknown"
    if kind == "simulated":
        status = "demo only"
    label = KINDS[kind]
    return f"{icon} {item.get('name', domain)}: {label} · {value} {item.get('unit', '')} · {status} · {formatted_time(dt, zone)} · {source}"


def render_prediction(prediction):
    p = asdict(prediction) if is_dataclass(prediction) else dict(prediction)
    estimate = finite_number(p["final_tp"], "final_tp", 0, 1)
    version = p.get("calibration_version", "identity")
    label = "Uncalibrated estimate" if version in {"identity", "provisional_identity"} else "Model estimate"
    origin = p.get("data_origin", "unlabeled")
    text = f"🧮 {label}: {estimate:.1%} · {p.get('domain', 'unknown domain')} · {p.get('horizon', 'unknown horizon')} · origin {origin}"
    if p.get("calibration_reliability_tp") is not None:
        internal = finite_number(p["calibration_reliability_tp"], "internal confidence", 0, 1)
        text += f"\n📊 Internal confidence score: {internal:.3f} · empirical accuracy not established here"
    return text


def render_snapshot(snapshot, now=None):
    now = timestamp(now) if now is not None else datetime.now(timezone.utc)
    zone = ZoneInfo(snapshot.get("timezone", "Asia/Seoul"))
    lines = ["🚂 Choo choo — Real Mode active.", "🕖 " + formatted_time(now, zone),
             render_location(snapshot.get("location"), now, zone)]
    for item in snapshot.get("telemetry", []):
        lines.append(render_telemetry(item, now, zone))
    if snapshot.get("prediction"):
        lines.append(render_prediction(snapshot["prediction"]))
    if snapshot.get("next_move"):
        lines.append("🎯 Move: " + str(snapshot["next_move"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        data = {"timezone": "Asia/Seoul", "location": {
            "name": "EXAMPLE Seoul City Hall", "latitude": 37.5665, "longitude": 126.9780,
            "source_kind": "user_pin"}, "telemetry": [
            {"domain": "weather", "evidence_kind": "unknown"},
            {"domain": "pollution", "name": "Demo PM2.5", "evidence_kind": "simulated",
             "value": 12, "unit": "µg/m³", "source": "synthetic example"}],
            "next_move": "Supply timestamped official observations."}
    else:
        data = json.loads(args.snapshot.read_text(encoding="utf-8"))
    print(render_snapshot(data))


if __name__ == "__main__":
    main()

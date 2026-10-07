#!/usr/bin/env python3
"""GP — web-fed, evidence-calibrated mobility engine.

Uses Open-Meteo's current conditions and hourly forecast through urllib.  The
prediction horizon is one complete hour because that is the forecast resolution
returned by this feed; it never pretends an hourly probability is a 30-minute
probability.  Traffic, pavement, and flooding remain caller-provided evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import hashlib, json, math, urllib.parse, urllib.request

LEDGER_DIR = Path("calibration_ledgers")
MIN_RECORDS, MIN_BIN = 20, 5

@dataclass(frozen=True)
class Anchor:
    city_id: str; name: str; latitude: float; longitude: float

@dataclass(frozen=True)
class WebObservation:
    city_id: str; observed_at: str; temperature_c: float; rh_pct: float
    rain_now: bool; forecast_p_next_hour: float; source: str
    traffic_flow_ratio: Optional[float] = None     # separately observed; 0 jam -> 1 free
    foot_surface_risk: Optional[float] = None      # direct visual observation only
    verified_ponding: Optional[bool] = None        # None = unknown

NODES = {
    "JP_TYO": Anchor("JP_TYO", "Tokyo", 35.6762, 139.6503),
    "UK_LON": Anchor("UK_LON", "London", 51.5072, -0.1276),
    "SG_SIN": Anchor("SG_SIN", "Singapore", 1.3521, 103.8198),
}

def clamp(x: float, low: float=0., high: float=1.) -> float: return max(low, min(high, x))
def ledger_path(city_id: str) -> Path:
    LEDGER_DIR.mkdir(exist_ok=True); return LEDGER_DIR / f"{city_id.lower()}_rain_hourly.json"
def load(path: Path) -> list[dict]:
    try:
        data=json.loads(path.read_text(encoding="utf-8")); return data if isinstance(data,list) else []
    except (OSError,json.JSONDecodeError): return []
def write(path: Path, rows: list[dict]) -> None: path.write_text(json.dumps(rows,indent=2),encoding="utf-8")

def fetch_open_meteo(anchor: Anchor) -> WebObservation:
    """Fetch live current weather plus the first forecast hour ending after now."""
    query = urllib.parse.urlencode({"latitude":anchor.latitude,"longitude":anchor.longitude,"timezone":"UTC","forecast_days":1,"current":"temperature_2m,relative_humidity_2m,precipitation","hourly":"precipitation_probability"})
    url = "https://api.open-meteo.com/v1/forecast?" + query
    with urllib.request.urlopen(url, timeout=15) as response:
        data = json.load(response)
    current, hourly = data["current"], data["hourly"]
    now = datetime.fromisoformat(current["time"].replace("Z","+00:00")).replace(tzinfo=timezone.utc)
    slots = [datetime.fromisoformat(x).replace(tzinfo=timezone.utc) for x in hourly["time"]]
    # Select the first *future* whole-hour forecast point. Its semantics are hourly, not 30 min.
    index = next((i for i,t in enumerate(slots) if t >= now + timedelta(hours=1)), None)
    if index is None: raise RuntimeError("Open-Meteo returned no future hourly precipitation probability")
    return WebObservation(anchor.city_id, now.isoformat(timespec="seconds"), float(current["temperature_2m"]), float(current["relative_humidity_2m"]), float(current.get("precipitation") or 0) > 0, clamp(float(hourly["precipitation_probability"][index])/100), "Open-Meteo forecast/current API")

def reliability(raw: float, rows: list[dict]) -> tuple[float,str,Optional[float],int]:
    resolved=[r for r in rows if r.get("outcome") in (0,1) and isinstance(r.get("raw_tp"),(int,float))]
    brier=round(sum((clamp(float(r["raw_tp"]))-r["outcome"])**2 for r in resolved)/len(resolved),4) if resolved else None
    if len(resolved)<MIN_RECORDS: return raw,"identity — insufficient resolved same-city hourly records",brier,len(resolved)
    bucket=min(4,int(raw*5)); peers=[r["outcome"] for r in resolved if min(4,int(clamp(float(r["raw_tp"]))*5))==bucket]
    if len(peers)<MIN_BIN: return raw,"identity — sparse reliability bin",brier,len(resolved)
    return round((sum(peers)+1)/(len(peers)+2),3),f"same-city Laplace reliability bin (n={len(peers)})",brier,len(resolved)

def record_id(anchor: Anchor, observed_at: str) -> str:
    return hashlib.sha256(f"{anchor.city_id}|hourly-rain|{observed_at}".encode()).hexdigest()[:16]
def log_once(path: Path, row: dict) -> bool:
    rows=load(path)
    if any(r.get("prediction_id")==row["prediction_id"] for r in rows): return False
    rows.append(row); write(path,rows); return True

def resolve_with_timestamped_observation(path: Path, prediction_id: str, outcome: bool, observed_at: str) -> bool:
    """Only resolve with an observation at/after due time—never with 'rain now'."""
    rows=load(path); when=datetime.fromisoformat(observed_at.replace("Z","+00:00"))
    for row in rows:
        if row.get("prediction_id")!=prediction_id or row.get("outcome") is not None: continue
        due=datetime.fromisoformat(row["due_at"].replace("Z","+00:00"))
        if when < due: return False
        row["outcome"]=int(outcome); row["resolved_at"]=observed_at; write(path,rows); return True
    return False

def evaluate(anchor: Anchor, obs: WebObservation) -> dict:
    path=ledger_path(anchor.city_id); raw=obs.forecast_p_next_hour; cal,method,brier,n=reliability(raw,load(path))
    observed=datetime.fromisoformat(obs.observed_at.replace("Z","+00:00")); due=(observed+timedelta(hours=1)).astimezone(timezone.utc).isoformat(timespec="seconds")
    row={"prediction_id":record_id(anchor,obs.observed_at),"city_id":anchor.city_id,"observed_at":obs.observed_at,"due_at":due,"event":"rain_occurs_in_next_complete_hour","horizon_minutes":60,"raw_tp":raw,"calibrated_tp":cal,"outcome":None,"source":obs.source}
    logged=log_once(path,row)
    vehicle="UNKNOWN — no independently observed traffic" if obs.traffic_flow_ratio is None else ("YELLOW — congestion" if 1-clamp(obs.traffic_flow_ratio)>=.60 else "GO — vehicle corridor fluid")
    foot="UNKNOWN — no direct foot-surface observation" if obs.foot_surface_risk is None else ("HOLD — rain or verified ponding" if obs.rain_now or obs.verified_ponding else "YELLOW — slippery surface" if clamp(obs.foot_surface_risk)>=.5 else "GO — observed footway clear")
    return {"anchor":asdict(anchor),"web_observation":asdict(obs),"rain_prediction":row,"logged":logged,"calibration":{"method":method,"same_city_resolved_n":n,"brier_raw":brier},"mobility":{"vehicle":vehicle,"foot":foot,"ponding":obs.verified_ponding},"limitations":["Hourly forecast probability is not a 30-minute probability.","Web weather does not supply reliable Google Maps traffic or line-of-sight pavement state.","A prediction resolves only with timestamped outcome evidence at/after due_at.","No cross-city pooling: climate and model error vary by city."]}

def main() -> None:
    # Fetches weather live. Add traffic/surface fields only from separate direct observations.
    reports=[evaluate(anchor,fetch_open_meteo(anchor)) for anchor in NODES.values()]
    Path("gp_web_operational_report.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    for report in reports:
        p=report["rain_prediction"]; a=report["anchor"]
        print(f"{a['city_id']} {a['name']}: {p['raw_tp']:.1%} raw -> {p['calibrated_tp']:.1%}; due {p['due_at']}; {report['calibration']['method']}")
if __name__ == "__main__": main()

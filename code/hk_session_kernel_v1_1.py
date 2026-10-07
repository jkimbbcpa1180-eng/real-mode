#!/usr/bin/env python3
"""
REAL MODE — HONG KONG SESSION KERNEL v1.1

v1.1 = v1.0 with minimal bug fixes. Every change is marked "# v1.1 FIX".
Feature list below is corrected to say what the code actually does:
  - HKMA monetary base ingestor (daily; day-over-day direction of
    mb_bf_disc_win_total)
  - HKMA interbank liquidity ingestor (daily; day-over-day direction of
    closing_balance)
  - HKO smart lamppost ingestor (one pi/di snapshot per call; the device
    updates ~every 10 min, but nothing here polls on a schedule)
  - HKMA/HKO observation → realmode ObservationEngine dict format
  - Anchor points + WGS84 local-projection helpers (helpers are defined but
    NOT used anywhere in this file)
  - Map link generator (Google, Google plus-code query, Naver, Kakao, Apple)
  - SHA-256 digests of the session record and of each anchor (the session
    digest does NOT cover the session's observations)
  - Session timeline (arrival → departure)
  - Calibration: only a resolutions dict + resolve_prediction(); nothing
    calls it, there is no CLI for it, and no calibration metric is computed.
    Confidence/strength values on observations are hard-coded constants,
    not calibrated.
  - CLI

The session kernel treats the Hong Kong visit as a bounded episode with a
beginning timestamp, a set of ingestors run during the visit, and an ending
timestamp. Data pulled during the session is tagged with the session ID so
that per-session calibration and coverage reports are possible.

Standard library only. Python 3.9+.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys  # v1.1 FIX: was only imported under __main__ -> NameError when imported
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERSION = "1.1"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_008.8
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

HKMA_BASE = "https://api.hkma.gov.hk/public"
HKO_LAMPPOST_BASE = ("https://data.weather.gov.hk/weatherAPI/"
                     "smart-lamppost/smart-lamppost.php")
HKO_LAMPPOST_LOCATIONS = ("https://www.hko.gov.hk/common/hko_data/"
                          "smart-lamppost/files/"
                          "smart_lamppost_met_device_location.json")

# v1.1 FIX: HKMA end_of_date and HKO TS are Hong Kong local time (UTC+8,
# no DST). v1.0 used time.mktime(), which interprets them in the *machine's*
# local zone (e.g. 1 h off on a KST box, 8 h off on a UTC server).
HKT = timezone(timedelta(hours=8))

# NOTE (v1.1): the constants below and clamp/sigmoid/wgs84_radii/
# ll_to_local_xy/local_xy_to_ll/haversine_m are unused in this file.
MIN_ACCURACY_M = 1.0
MAX_ACCURACY_M = 500.0
CHI2_GATE_999 = 13.82
MIN_DT_S = 1e-3
MAX_DT_S = 3600.0


# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def wgs84_radii(lat_deg):
    phi = math.radians(lat_deg)
    s2 = math.sin(phi) ** 2
    denom = 1.0 - WGS84_E2 * s2
    sqrt_denom = math.sqrt(denom)
    M = (WGS84_A * (1.0 - WGS84_E2)) / (denom * sqrt_denom)
    N = WGS84_A / sqrt_denom
    return M, N


def ll_to_local_xy(lat, lon, lat0, lon0):
    M, N = wgs84_radii(lat0)
    phi0 = math.radians(lat0)
    return (math.radians(lon - lon0) * N * math.cos(phi0),
            math.radians(lat - lat0) * M)


def local_xy_to_ll(east, north, lat0, lon0):
    M, N = wgs84_radii(lat0)
    phi0 = math.radians(lat0)
    return (lat0 + math.degrees(north / M),
            lon0 + math.degrees(east / (N * math.cos(phi0))))


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2)
    return EARTH_RADIUS_M * 2.0 * math.atan2(math.sqrt(a),
                                             math.sqrt(max(0.0, 1.0 - a)))


# ---------------------------------------------------------------------------
# SESSION MODEL
# ---------------------------------------------------------------------------

@dataclass
class HongKongSession:
    """
    A bounded Hong Kong episode.

    Fields:
      session_id:        unique identifier
      arrival_ts:        timestamp of arrival
      departure_ts:      timestamp of departure (None while active)
      arrival_label:     human-readable (e.g. "ICN → HKG")
      departure_label:   human-readable (e.g. "HKG → ICN")
      notes:             free-form
      ingest_runs:       list of {ingestor, timestamp, count} dicts
      anchor_lat/lon:    optional focal point during the visit
    """
    session_id: str
    arrival_ts: float
    departure_ts: Optional[float] = None
    arrival_label: str = ""
    departure_label: str = ""
    notes: str = ""
    ingest_runs: List[Dict[str, Any]] = field(default_factory=list)
    anchor_lat: Optional[float] = None
    anchor_lon: Optional[float] = None

    def is_active(self, at: Optional[float] = None) -> bool:
        t = at if at is not None else time.time()
        if self.departure_ts is None:
            return t >= self.arrival_ts
        return self.arrival_ts <= t <= self.departure_ts

    def duration_hours(self) -> Optional[float]:
        if self.departure_ts is None:
            return None
        return (self.departure_ts - self.arrival_ts) / 3600.0

    def record_ingest(self, ingestor: str, count: int,
                      timestamp: Optional[float] = None) -> None:
        self.ingest_runs.append({
            "ingestor": ingestor,
            "count": int(count),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
        })

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not self.session_id:
            raise ValueError("session_id required")
        if self.departure_ts is not None and self.departure_ts < self.arrival_ts:
            raise ValueError("departure_ts must be >= arrival_ts")


# ---------------------------------------------------------------------------
# ANCHOR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorPoint:
    name: str
    road_address: str
    jibun_address: str
    postal_code: str
    plus_code: str
    lat: float
    lon: float
    coordinate_system: str = "WGS84"

    def validate(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError("invalid latitude")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError("invalid longitude")


# ---------------------------------------------------------------------------
# PLUS CODE CHECK  (v1.1 addition: lets the self-test verify anchors offline)
# ---------------------------------------------------------------------------

_OLC_ALPHABET = "23456789CFGHJMPQRVWX"


def olc_encode(lat: float, lon: float, length: int = 10) -> str:
    """Open Location Code encoder (pair digits + grid refinement)."""
    lat = min(90.0, max(-90.0, lat))
    if lat == 90.0:
        lat -= 1e-9
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat_v = int(math.floor(round((lat + 90.0) * 25_000_000, 6)))
    lon_v = int(math.floor(round((lon + 180.0) * 8_192_000, 6)))
    tail = ""
    for _ in range(5):
        tail = _OLC_ALPHABET[(lat_v % 5) * 4 + (lon_v % 4)] + tail
        lat_v //= 5
        lon_v //= 4
    head = ""
    for _ in range(5):
        head = _OLC_ALPHABET[lat_v % 20] + _OLC_ALPHABET[lon_v % 20] + head
        lat_v //= 20
        lon_v //= 20
    code = head + tail
    code = code[:8] + "+" + code[8:]
    return code[:length + 1]


def plus_code_matches(anchor: "AnchorPoint") -> bool:
    """True iff anchor.lat/lon lies inside the cell named by anchor.plus_code."""
    local = anchor.plus_code.split()[0].upper()
    n_digits = len(local.replace("+", "")) + (8 - local.index("+"))
    full = olc_encode(anchor.lat, anchor.lon, max(10, n_digits))
    return full.endswith(local)


# Default anchors. The Korea anchor is an EXAMPLE location (Seoul City Hall),
# not anyone's home; replace it with your own if you use this kernel.
# The Hong Kong anchor below is Central Pier 7 as an example focal point.
KOREA_ANCHOR = AnchorPoint(
    name="Seoul City Hall (example anchor)",
    road_address="서울특별시 중구 세종대로 110",
    jibun_address="",
    postal_code="04524",
    plus_code="HX8H+J59 Seoul",
    lat=37.566535,
    lon=126.977969,
)

HK_ANCHOR = AnchorPoint(
    name="Central Piers Focal Point",
    road_address="Central, Hong Kong",
    jibun_address="",
    postal_code="",
    # v1.1 FIX: was "5Q2X+22 Hong Kong", which recovers to 7PJM5Q2X+22
    # (22.15006, 113.79756) — ~40 km WSW of these coordinates. The plus code
    # of 22.283770,114.161570 is 7PJP75M6+GJ.
    # (Not verified that these coordinates are actually Central Pier 7.)
    plus_code="75M6+GJ Hong Kong",
    lat=22.283770,
    lon=114.161570,
)


# ---------------------------------------------------------------------------
# MAP LINKS + DIGEST
# ---------------------------------------------------------------------------

def build_map_links(anchor: AnchorPoint) -> Dict[str, str]:
    q_name = urllib.parse.quote(anchor.name, safe="")
    q_addr = urllib.parse.quote(anchor.road_address, safe="")
    q_pc = urllib.parse.quote(anchor.plus_code, safe="")
    return {
        "google": (f"https://www.google.com/maps/search/?api=1"
                   f"&query={anchor.lat:.6f},{anchor.lon:.6f}"),
        "google_pluscode": (f"https://www.google.com/maps/search/?api=1"
                            f"&query={q_pc}"),
        "google_address": (f"https://www.google.com/maps/search/?api=1"
                           f"&query={q_addr}"),
        "naver_coord": (f"https://map.naver.com/v5/search/"
                        f"{anchor.lat:.6f},{anchor.lon:.6f}"),
        "naver_address": f"https://map.naver.com/p/search/{q_addr}",
        "kakao": (f"https://map.kakao.com/link/map/"
                  f"{q_name},{anchor.lat:.6f},{anchor.lon:.6f}"),
        "apple": (f"https://maps.apple.com/?ll={anchor.lat:.6f},"
                  f"{anchor.lon:.6f}&q={q_name}&z=19"),
    }


def anchor_digest(anchor: AnchorPoint) -> str:
    payload = json.dumps(asdict(anchor), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# HKMA INGESTORS
# ---------------------------------------------------------------------------

def _hk_date_ts(date_str: str) -> float:
    # v1.1 FIX: midnight Hong Kong time, independent of the machine's TZ.
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=HKT).timestamp()


def _chronological(records):
    # v1.1 FIX: the HKMA API returns records NEWEST FIRST (verified live
    # 2026-10-01: 2026-09-30, 2026-09-29, ...). v1.0 paired records in API
    # order, so "curr" was the OLDER day and every rise/fall was inverted
    # (and stamped with the older day's date). Sort oldest -> newest.
    return sorted((r for r in records if r.get("end_of_date")),
                  key=lambda r: r["end_of_date"])


class HKMAMonetaryIngestor:
    URL = (f"{HKMA_BASE}/market-data-and-statistics/"
           f"daily-monetary-statistics/daily-figures-monetary-base")

    def fetch(self, pagesize: int = 60, timeout: float = 15.0):
        params = urllib.parse.urlencode({"pagesize": pagesize})
        req = urllib.request.Request(
            f"{self.URL}?{params}",
            headers={"User-Agent": f"realmode-hk-session/{VERSION}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get(
                "result", {}).get("records", [])

    def to_observations(self, records):
        records = _chronological(records)  # v1.1 FIX
        obs = []
        for i in range(1, len(records)):
            prev, curr = records[i - 1], records[i]
            try:
                pv = float(prev.get("mb_bf_disc_win_total", 0))
                cv = float(curr.get("mb_bf_disc_win_total", 0))
            except (TypeError, ValueError):
                continue
            if pv <= 0:
                continue
            # v1.1 FIX: unchanged values were scored as a "fall" (outcome 0,
            # direction -1). For interbank closing_balance that was 14 of 29
            # live pairs. A no-change day is neither; skip it.
            if cv == pv:
                continue
            outcome = 1.0 if cv > pv else 0.0
            try:
                ts = _hk_date_ts(curr["end_of_date"])  # v1.1 FIX (was time.mktime)
            except (KeyError, ValueError):
                continue
            obs.append({
                "source": "HKMA_DAILY",
                "source_family": "HKMA_MONETARY",
                "confidence": 0.95,
                "timestamp": ts,
                "provenance_id": f"hkma_mb_{curr['end_of_date']}",
                "payload": {
                    "signals": [{
                        "claim_id": "monetary_base_rise",
                        "name": "Monetary base direction",
                        "direction": 1.0 if outcome == 1.0 else -1.0,
                        "strength": 0.9,
                        "confidence": 0.95,
                        "pattern_key": "hkma_monetary_trend",
                    }],
                    "resolved_outcome": outcome,
                    "value": cv,
                    "previous_value": pv,
                },
            })
        return obs


class HKMAInterbankIngestor:
    URL = (f"{HKMA_BASE}/market-data-and-statistics/"
           f"daily-monetary-statistics/daily-figures-interbank-liquidity")

    def fetch(self, pagesize: int = 60, timeout: float = 15.0):
        params = urllib.parse.urlencode({"pagesize": pagesize})
        req = urllib.request.Request(
            f"{self.URL}?{params}",
            headers={"User-Agent": f"realmode-hk-session/{VERSION}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get(
                "result", {}).get("records", [])

    def to_observations(self, records):
        records = _chronological(records)  # v1.1 FIX
        obs = []
        for i in range(1, len(records)):
            prev, curr = records[i - 1], records[i]
            try:
                pv = float(prev.get("closing_balance", 0))
                cv = float(curr.get("closing_balance", 0))
            except (TypeError, ValueError):
                continue
            if pv <= 0:
                continue
            # v1.1 FIX: unchanged values were scored as a "fall" (outcome 0,
            # direction -1). For interbank closing_balance that was 14 of 29
            # live pairs. A no-change day is neither; skip it.
            if cv == pv:
                continue
            outcome = 1.0 if cv > pv else 0.0
            try:
                ts = _hk_date_ts(curr["end_of_date"])  # v1.1 FIX (was time.mktime)
            except (KeyError, ValueError):
                continue
            obs.append({
                "source": "HKMA_INTERBANK",
                "source_family": "HKMA_LIQUIDITY",
                "confidence": 0.95,
                "timestamp": ts,
                "provenance_id": f"hkma_ib_{curr['end_of_date']}",
                "payload": {
                    "signals": [{
                        "claim_id": "interbank_liquidity_rise",
                        "name": "Interbank liquidity direction",
                        "direction": 1.0 if outcome == 1.0 else -1.0,
                        "strength": 0.85,
                        "confidence": 0.95,
                        "pattern_key": "hkma_liquidity_trend",
                    }],
                    "resolved_outcome": outcome,
                    "closing_balance": cv,
                    "previous_balance": pv,
                },
            })
        return obs


# ---------------------------------------------------------------------------
# HKO SMART LAMPPOST INGESTOR
# ---------------------------------------------------------------------------

class HKOSmartLamppostIngestor:
    """
    HKO smart lamppost meteorological ingestor.

    Requires BOTH pi (lamppost ID) and di (device ID).
    The static device-location database lists valid pi values.
    """

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._locations: Optional[List[Dict[str, Any]]] = None

    def load_locations(self):
        if self._locations is not None:
            return self._locations
        req = urllib.request.Request(
            HKO_LAMPPOST_LOCATIONS,
            headers={"User-Agent": f"realmode-hk-session/{VERSION}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self._locations = json.loads(r.read().decode("utf-8"))
        return self._locations

    def fetch_one(self, pi: str, di: str = "01"):
        params = urllib.parse.urlencode({"pi": pi, "di": di})
        req = urllib.request.Request(
            f"{HKO_LAMPPOST_BASE}?{params}",
            headers={"User-Agent": f"realmode-hk-session/{VERSION}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def to_observations(self, raw, pi: str, di: str = "01"):
        # Invalid pi/di returns {"message": "No record found"} (no BODY) -> [].
        body = raw.get("BODY", {}).get("HKO", {}) if isinstance(raw, dict) else {}
        # v1.1 FIX: the live field is "T0" (T-zero), not "TO" (letter O).
        # v1.0 therefore returned 0 observations even for live lampposts
        # such as GF3637 (verified live). Keep "TO" as a fallback.
        temp = body.get("T0", body.get("TO"))
        humidity = body.get("RH")
        wind = body.get("WS")
        if temp is None:
            return []
        try:
            temp_f = float(temp)
        except (TypeError, ValueError):
            return []
        # NOTE (v1.1, not changed): humidity > 70 % -> +1 is an arbitrary
        # threshold; HK RH is usually > 70 %, so this is almost always +1.
        direction = 1.0 if (humidity and float(humidity) > 70.0) else -1.0
        # v1.1 FIX: use the sensor's own reading time (TS, HKT
        # yyyymmddHHMMSS) instead of fetch time, so re-fetching the same
        # reading yields the same provenance_id (and is de-duplicated).
        try:
            ts = (datetime.strptime(body["TS"], "%Y%m%d%H%M%S")
                  .replace(tzinfo=HKT).timestamp())
        except (KeyError, TypeError, ValueError):
            ts = time.time()
        return [{
            "source": f"HKO_LAMPPOST_{pi}_{di}",
            "source_family": "HKO_WEATHER",
            "confidence": 0.85,
            "timestamp": ts,
            "provenance_id": f"hko_lamppost_{pi}_{di}_{int(ts)}",
            "payload": {
                "signals": [{
                    "claim_id": f"temp_humidity_{pi}",
                    "name": f"Lamppost {pi} temp/humidity",
                    "direction": direction,
                    "strength": 0.7,
                    "confidence": 0.85,
                    "pattern_key": "hk_weather_trend",
                }],
                "temperature": temp_f,
                "humidity": float(humidity) if humidity else None,
                "wind_speed": float(wind) if wind else None,
                "pi": pi,
                "di": di,
            },
        }]


# ---------------------------------------------------------------------------
# SESSION LEDGER
# ---------------------------------------------------------------------------

class HKSessionLedger:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.sessions: Dict[str, HongKongSession] = {}
        self.observations: List[Dict[str, Any]] = []
        self.resolutions: Dict[str, Dict[str, int]] = {}
        if self.path and self.path.exists():
            self.load()

    def add_session(self, session: HongKongSession):
        session.validate()
        if session.session_id in self.sessions:
            raise ValueError(f"duplicate session_id: {session.session_id}")
        self.sessions[session.session_id] = session
        self.save()

    def close_session(self, session_id: str,
                      departure_ts: Optional[float] = None) -> None:
        if session_id not in self.sessions:
            raise KeyError(f"unknown session_id: {session_id}")
        s = self.sessions[session_id]
        s.departure_ts = (time.time() if departure_ts is None
                          else float(departure_ts))
        s.validate()
        self.save()

    def record_observations(self, session_id: str,
                            observations: Sequence[Dict[str, Any]],
                            ingestor: str) -> int:
        if session_id not in self.sessions:
            raise KeyError(f"unknown session_id: {session_id}")
        # v1.1 FIX: re-running an ingestor appended the same 29 HKMA records
        # again (same provenance_id). Skip (session_id, provenance_id) pairs
        # already in the ledger; count only what was actually added.
        seen = {(o.get("session_id"), o.get("provenance_id"))
                for o in self.observations if o.get("provenance_id")}
        added = 0
        for obs in observations:
            obs = dict(obs)
            obs["session_id"] = session_id
            key = (session_id, obs.get("provenance_id"))
            if obs.get("provenance_id") and key in seen:
                continue
            seen.add(key)
            self.observations.append(obs)
            added += 1
        self.sessions[session_id].record_ingest(ingestor, added)
        self.save()
        return added

    def resolve_prediction(self, provenance_id: str, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        self.resolutions.setdefault(provenance_id, {})
        self.resolutions[provenance_id]["outcome"] = int(outcome)
        self.resolutions[provenance_id]["resolved_at"] = time.time()
        self.save()

    def session_observations(self, session_id: str):
        return [o for o in self.observations
                if o.get("session_id") == session_id]

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": VERSION,
            "sessions": [asdict(s) for s in self.sessions.values()],
            "observations": self.observations,
            "resolutions": self.resolutions,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2,
                                  ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(self.path)

    def load(self):
        if not self.path:
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw.get("sessions", []):
            s = HongKongSession(**item)
            self.sessions[s.session_id] = s
        self.observations = list(raw.get("observations", []))
        self.resolutions = dict(raw.get("resolutions", {}))


# ---------------------------------------------------------------------------
# SESSION SUMMARY
# ---------------------------------------------------------------------------

def session_summary(session: HongKongSession,
                    ledger: HKSessionLedger) -> Dict[str, Any]:
    obs = ledger.session_observations(session.session_id)
    by_family: Dict[str, int] = {}
    for o in obs:
        fam = o.get("source_family", "unknown")
        by_family[fam] = by_family.get(fam, 0) + 1
    duration = session.duration_hours()
    return {
        "session_id": session.session_id,
        "arrival_ts": session.arrival_ts,
        "departure_ts": session.departure_ts,
        "duration_hours": round(duration, 3) if duration is not None else None,
        "is_active": session.is_active(),
        "arrival_label": session.arrival_label,
        "departure_label": session.departure_label,
        "ingest_runs": session.ingest_runs,
        "total_observations": len(obs),
        "observations_by_family": by_family,
        "digest": session.digest(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _begin_session(ledger: HKSessionLedger,  # v1.1 FIX: was "HongKongSession and HKSessionLedger"
                   arrival_label: str, notes: str,
                   anchor_lat: Optional[float] = None,
                   anchor_lon: Optional[float] = None,
                   arrival_ts: Optional[float] = None) -> HongKongSession:
    session = HongKongSession(
        session_id=str(uuid.uuid4()),
        arrival_ts=(arrival_ts if arrival_ts is not None else time.time()),
        arrival_label=arrival_label,
        notes=notes,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon)
    ledger.add_session(session)
    return session


def _demo():
    print("=== REAL MODE — HONG KONG SESSION DEMO ===\n")
    ledger = HKSessionLedger("hk_session_demo.json")

    session = _begin_session(
        ledger,
        arrival_label="ICN → HKG",
        notes="Hong Kong visit session; HKMA + HKO ingestors enabled",
        anchor_lat=HK_ANCHOR.lat,
        anchor_lon=HK_ANCHOR.lon)
    print(f"Opened session: {session.session_id}")
    print(f"  arrival: {datetime.fromtimestamp(session.arrival_ts, tz=timezone.utc).isoformat()}")

    # --- HKMA monetary base ---
    try:
        mon = HKMAMonetaryIngestor()
        records = mon.fetch(pagesize=30)
        obs = mon.to_observations(records)
        ledger.record_observations(session.session_id, obs, "hkma_monetary")
        print(f"  HKMA monetary: fetched {len(records)} records, "
              f"{len(obs)} observations")
    except Exception as e:
        print(f"  HKMA monetary failed: {e}")

    # --- HKMA interbank ---
    try:
        ib = HKMAInterbankIngestor()
        records = ib.fetch(pagesize=30)
        obs = ib.to_observations(records)
        ledger.record_observations(session.session_id, obs, "hkma_interbank")
        print(f"  HKMA interbank: fetched {len(records)} records, "
              f"{len(obs)} observations")
    except Exception as e:
        print(f"  HKMA interbank failed: {e}")

    # --- HKO smart lamppost (single device) ---
    # The specific pi/di pair depends on the HKO location database.
    # This demo attempts one known-format pair; a production run would
    # enumerate valid pi values from load_locations().
    try:
        hko = HKOSmartLamppostIngestor()
        raw = hko.fetch_one("GF3637", "01")
        obs = hko.to_observations(raw, "GF3637", "01")
        ledger.record_observations(session.session_id, obs, "hko_lamppost")
        print(f"  HKO lamppost: {len(obs)} observations")
    except Exception as e:
        print(f"  HKO lamppost failed (expected if pi/di not live): {e}")

    # Close session.
    # NOTE (v1.1, unchanged): the demo closes the session seconds after
    # opening it, so duration_hours is ~0.002. Real use: --begin / --close.
    ledger.close_session(session.session_id)
    summary = session_summary(session, ledger)
    print("\n--- session summary ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # Anchor links.
    print("\n--- Hong Kong anchor links ---")
    for k, v in build_map_links(HK_ANCHOR).items():
        print(f"  {k:20s}: {v}")
    print(f"\n  anchor digest: {anchor_digest(HK_ANCHOR)}")

    # Korea anchor (departure destination).
    print("\n--- Korea anchor links ---")
    for k, v in build_map_links(KOREA_ANCHOR).items():
        print(f"  {k:20s}: {v}")
    print(f"\n  anchor digest: {anchor_digest(KOREA_ANCHOR)}")

    # Save state.
    ledger.save()
    print(f"\nLedger saved to {ledger.path}")


def main(argv=None) -> int:  # v1.1 FIX: return an exit code (errors exited 0)
    ap = argparse.ArgumentParser(
        description=f"Real Mode Hong Kong Session Kernel v{VERSION}")
    ap.add_argument("--ledger", default="hk_session.json")
    ap.add_argument("--begin", metavar="ARRIVAL_LABEL",
                    help="Begin a new HK session")
    ap.add_argument("--notes", default="")
    ap.add_argument("--anchor-lat", type=float)
    ap.add_argument("--anchor-lon", type=float)
    ap.add_argument("--close", metavar="SESSION_ID",
                    help="Close an existing session")
    ap.add_argument("--ingest", choices=["hkma_monetary", "hkma_interbank",
                                         "hko_lamppost"],
                    help="Run an ingestor against the active session")
    ap.add_argument("--pi", help="HKO lamppost pi value (with --ingest hko_lamppost)")
    ap.add_argument("--di", default="01", help="HKO lamppost di value")
    ap.add_argument("--summary", metavar="SESSION_ID")
    ap.add_argument("--anchor", choices=["hk", "korea"],
                    help="Print anchor links and exit")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="Run offline unit tests and exit")  # v1.1

    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.demo:
        _demo()
        return 0

    if args.anchor:
        anchor = HK_ANCHOR if args.anchor == "hk" else KOREA_ANCHOR
        print(json.dumps({
            "anchor": asdict(anchor),
            "links": build_map_links(anchor),
            "digest": anchor_digest(anchor),
            "plus_code_matches_latlon": plus_code_matches(anchor),  # v1.1
        }, indent=2, ensure_ascii=False))
        return 0

    ledger = HKSessionLedger(args.ledger)

    if args.begin:
        session = _begin_session(
            ledger, args.begin, args.notes,
            anchor_lat=args.anchor_lat, anchor_lon=args.anchor_lon)
        print(f"opened session {session.session_id}")
        return 0

    if args.close:
        # v1.1 FIX: unknown id printed a raw KeyError traceback.
        if args.close not in ledger.sessions:
            print(f"error: unknown session {args.close}", file=sys.stderr)
            return 1
        ledger.close_session(args.close)
        print(f"closed session {args.close}")
        return 0

    if args.ingest:
        active = [s for s in ledger.sessions.values() if s.is_active()]
        if not active:
            print("error: no active session", file=sys.stderr)
            return 1
        # v1.1 FIX: with >1 active session v1.0 silently used the oldest.
        if len(active) > 1:
            print(f"warning: {len(active)} active sessions; using most "
                  f"recent arrival", file=sys.stderr)
        sid = max(active, key=lambda s: s.arrival_ts).session_id
        if args.ingest == "hkma_monetary":
            records = HKMAMonetaryIngestor().fetch(pagesize=30)
            obs = HKMAMonetaryIngestor().to_observations(records)
        elif args.ingest == "hkma_interbank":
            records = HKMAInterbankIngestor().fetch(pagesize=30)
            obs = HKMAInterbankIngestor().to_observations(records)
        elif args.ingest == "hko_lamppost":
            if not args.pi:
                print("error: --pi required for hko_lamppost", file=sys.stderr)
                return 1
            hko = HKOSmartLamppostIngestor()
            raw = hko.fetch_one(args.pi, args.di)
            obs = hko.to_observations(raw, args.pi, args.di)
        else:
            obs = []
        added = ledger.record_observations(sid, obs, args.ingest)
        print(f"ingested {added} new observations ({len(obs)} parsed) "
              f"into session {sid}")
        return 0

    if args.summary:
        if args.summary not in ledger.sessions:
            print(f"error: unknown session {args.summary}", file=sys.stderr)
            return 1
        s = ledger.sessions[args.summary]
        print(json.dumps(session_summary(s, ledger),
                         indent=2, ensure_ascii=False, default=str))
        return 0

    ap.print_help()
    return 0


# ---------------------------------------------------------------------------
# OFFLINE SELF-TEST (v1.1)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import os
    import tempfile
    import typing
    import unittest

    # Fixture mirrors live HKMA shape/order (newest first), 2026-10-01 pull.
    MB_NEWEST_FIRST = [
        {"end_of_date": "2026-09-30", "mb_bf_disc_win_total": 2090204},
        {"end_of_date": "2026-09-29", "mb_bf_disc_win_total": 2087110},
        {"end_of_date": "2026-09-28", "mb_bf_disc_win_total": 2086204},
        {"end_of_date": "2026-09-25", "mb_bf_disc_win_total": 2087362},
    ]
    IB_NEWEST_FIRST = [
        {"end_of_date": "2026-09-30", "closing_balance": 54108},
        {"end_of_date": "2026-09-29", "closing_balance": 54108},
        {"end_of_date": "2026-09-28", "closing_balance": 54156},
    ]
    HKO_LIVE = {"BD": "00", "DI": "01", "PI": "GF3637", "BODY": {"HKO": {
        "RH": "74.7", "T0": "31.7", "TS": "20261001181011", "WS": "3",
        "WD": "126", "VN": "1.0"}}}

    class T(unittest.TestCase):
        def test_monetary_direction_newest_first(self):
            obs = HKMAMonetaryIngestor().to_observations(MB_NEWEST_FIRST)
            got = {o["provenance_id"]: o["payload"]["resolved_outcome"] for o in obs}
            # 09-25 -> 09-28 fell, 09-28 -> 09-29 rose, 09-29 -> 09-30 rose
            self.assertEqual(got, {"hkma_mb_2026-09-28": 0.0,
                                   "hkma_mb_2026-09-29": 1.0,
                                   "hkma_mb_2026-09-30": 1.0})
            o = [x for x in obs if x["provenance_id"] == "hkma_mb_2026-09-30"][0]
            self.assertEqual(o["payload"]["value"], 2090204.0)
            self.assertEqual(o["payload"]["previous_value"], 2087110.0)

        def test_order_invariance(self):
            a = HKMAMonetaryIngestor().to_observations(MB_NEWEST_FIRST)
            b = HKMAMonetaryIngestor().to_observations(MB_NEWEST_FIRST[::-1])
            self.assertEqual(a, b)

        def test_interbank_tie_skipped(self):
            obs = HKMAInterbankIngestor().to_observations(IB_NEWEST_FIRST)
            self.assertEqual([o["provenance_id"] for o in obs], ["hkma_ib_2026-09-29"])
            self.assertEqual(obs[0]["payload"]["resolved_outcome"], 0.0)

        def test_hk_timestamp_tz_independent(self):
            old = os.environ.get("TZ")
            try:
                vals = []
                for tz in ("UTC", "Asia/Seoul", "America/Los_Angeles"):
                    os.environ["TZ"] = tz
                    time.tzset()
                    vals.append(_hk_date_ts("2026-09-30"))
            finally:
                if old is None:
                    os.environ.pop("TZ", None)
                else:
                    os.environ["TZ"] = old
                time.tzset()
            self.assertEqual(len(set(vals)), 1)
            self.assertEqual(vals[0], datetime(2026, 9, 29, 16, tzinfo=timezone.utc).timestamp())

        def test_hko_t0_and_sensor_ts(self):
            obs = HKOSmartLamppostIngestor().to_observations(HKO_LIVE, "GF3637", "01")
            self.assertEqual(len(obs), 1)
            self.assertEqual(obs[0]["payload"]["temperature"], 31.7)
            self.assertEqual(obs[0]["timestamp"],
                             datetime(2026, 10, 1, 10, 10, 11, tzinfo=timezone.utc).timestamp())

        def test_hko_no_record(self):
            self.assertEqual(HKOSmartLamppostIngestor().to_observations(
                {"message": "No record found"}, "DF3637", "01"), [])

        def test_ledger_dedup_and_roundtrip(self):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "l.json")
                led = HKSessionLedger(path)
                s = _begin_session(led, "ICN → HKG", "t", arrival_ts=1000.0)
                obs = HKMAMonetaryIngestor().to_observations(MB_NEWEST_FIRST)
                self.assertEqual(led.record_observations(s.session_id, obs, "m"), 3)
                self.assertEqual(led.record_observations(s.session_id, obs, "m"), 0)
                self.assertEqual(len(led.session_observations(s.session_id)), 3)
                led.close_session(s.session_id, departure_ts=4600.0)
                led2 = HKSessionLedger(path)
                self.assertEqual(led2.sessions[s.session_id], led.sessions[s.session_id])
                self.assertEqual(led2.sessions[s.session_id].duration_hours(), 1.0)
                self.assertFalse(led2.sessions[s.session_id].is_active(at=5000.0))

        def test_close_validation(self):
            led = HKSessionLedger(None)
            s = _begin_session(led, "x", "", arrival_ts=1000.0)
            with self.assertRaises(ValueError):
                led.close_session(s.session_id, departure_ts=10.0)

        def test_main_importable_no_nameerror(self):
            with tempfile.TemporaryDirectory() as d:
                rc = main(["--ledger", os.path.join(d, "l.json"), "--summary", "nope"])
                self.assertEqual(rc, 1)
                rc = main(["--ledger", os.path.join(d, "l.json"), "--close", "nope"])
                self.assertEqual(rc, 1)

        def test_annotation(self):
            hints = typing.get_type_hints(_begin_session)
            self.assertIs(hints["ledger"], HKSessionLedger)

        def test_olc_encoder_known_values(self):
            # Reference values from Google's openlocationcode library.
            self.assertEqual(olc_encode(22.283770, 114.161570), "7PJP75M6+GJ")
            self.assertEqual(olc_encode(37.566535, 126.977969, 11), "8Q98HX8H+J59")

        def test_anchor_plus_codes(self):
            self.assertTrue(plus_code_matches(HK_ANCHOR))
            old_hk = AnchorPoint("x", "", "", "", "5Q2X+22", HK_ANCHOR.lat, HK_ANCHOR.lon)
            self.assertFalse(plus_code_matches(old_hk))
            # Korea example anchor matches at 11 and 10 digits.
            self.assertTrue(plus_code_matches(KOREA_ANCHOR))
            k10 = AnchorPoint("x", "", "", "", "HX8H+J5", KOREA_ANCHOR.lat, KOREA_ANCHOR.lon)
            self.assertTrue(plus_code_matches(k10))

    res = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(T))
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

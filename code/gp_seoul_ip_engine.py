#!/usr/bin/env python3
"""GP Seoul IP Fused Engine — live weather + Korean open-source research.

Two predictions are created on each --deploy run and deliberately kept apart:
  WEATHER GP: rain during one precisely defined future hourly interval.
  IP GP: a research lead gains corroboration from >=2 independently named
         Korean open-web sources by a stated due time.

Map links are navigation context, never corroboration or a probability input.
Monte Carlo samples a probability for display/decision support only. It never
creates outcomes. A record can be resolved only with timestamped evidence at
or after its due time. Standard library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Optional
import argparse
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SEOUL = {"code": "KR_SEL", "name": "Seoul", "lat": 37.5665, "lon": 126.9780}
LEDGER_DIR = Path("gp_seoul_ip_ledgers")
WEATHER_LEDGER = LEDGER_DIR / "weather_hourly.json"
IP_LEDGER = LEDGER_DIR / "ip_research.json"
REPORT = Path("gp_seoul_ip_report.json")
RUNS, MIN_RECORDS, MIN_BIN = 10_000, 30, 8
UA = "GP-Seoul-IP/2.0 (open-source research runner)"


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return dt.astimezone(timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load(path: Path) -> list[dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace-in-place avoids a half-written ledger if execution stops mid-save.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def brier(rows: list[dict]) -> Optional[float]:
    done = [r for r in rows if r.get("outcome") in (0, 1) and isinstance(r.get("raw_probability"), (int, float))]
    return None if not done else round(sum((clamp(float(r["raw_probability"])) - int(r["outcome"])) ** 2 for r in done) / len(done), 4)


def laplace_calibrate(raw: float, rows: list[dict], cohort: str) -> tuple[float, str, Optional[float], int]:
    """Only resolves records from the identical event/cohort definition."""
    done = [r for r in rows if r.get("cohort") == cohort and r.get("outcome") in (0, 1)]
    score = brier(done)
    if len(done) < MIN_RECORDS:
        return raw, "identity: insufficient matched resolved records", score, len(done)
    bucket = min(4, int(clamp(raw) * 5))
    peers = [int(r["outcome"]) for r in done if min(4, int(clamp(float(r["raw_probability"])) * 5)) == bucket]
    if len(peers) < MIN_BIN:
        return raw, "identity: sparse matched reliability bin", score, len(done)
    return round((sum(peers) + 1) / (len(peers) + 2), 3), f"Laplace reliability bin n={len(peers)}", score, len(done)


def mc(probability: float, seed: str) -> dict:
    rng = random.Random(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16))
    hits = sum(rng.random() < probability for _ in range(RUNS))
    return {"runs": RUNS, "hits": hits, "probability": round(hits / RUNS, 4)}


def add_record(path: Path, record: dict) -> bool:
    rows = load(path)
    if any(r.get("prediction_id") == record["prediction_id"] for r in rows):
        return False
    rows.append(record)
    save(path, rows)
    return True


def active_pending(path: Path, cohort: str, at: datetime) -> Optional[dict]:
    """Avoid correlated duplicate forecasts for one still-open event window."""
    for row in load(path):
        if row.get("cohort") != cohort or row.get("outcome") is not None:
            continue
        try:
            if parse_iso(row["due_at"]) > at:
                return row
        except (KeyError, TypeError, ValueError):
            continue
    return None


def resolve(kind: str, prediction_id: str, outcome: int, observed_at: str, source: str) -> bool:
    """Record a post-horizon resolution with a named evidence source.

    A weather source must cover the entire forecast interval; an IP source must
    identify the corroborating publisher(s) or primary source(s).
    """
    if kind not in {"weather", "ip"} or outcome not in (0, 1) or not source.strip():
        raise ValueError("kind, binary outcome, and non-empty evidence source are required")
    path = WEATHER_LEDGER if kind == "weather" else IP_LEDGER
    rows = load(path)
    when = parse_iso(observed_at)
    for row in rows:
        if row.get("prediction_id") != prediction_id or row.get("outcome") is not None:
            continue
        if when < parse_iso(row["due_at"]):
            return False
        row["outcome"] = int(bool(outcome))
        row["resolved_at"] = to_iso(when)
        row["resolution_source"] = source.strip()
        save(path, rows)
        return True
    return False


def next_complete_hour(at: datetime) -> datetime:
    """Return the next clock-hour boundary in UTC, never an already-active hour."""
    base = at.replace(minute=0, second=0, microsecond=0)
    return base + timedelta(hours=1)


def parse_utc_feed_time(value: str) -> datetime:
    """Open-Meteo returns zone-less times when queried with timezone=UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_weather() -> dict:
    args = urllib.parse.urlencode({
        "latitude": SEOUL["lat"], "longitude": SEOUL["lon"], "timezone": "UTC", "forecast_days": 2,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "hourly": "precipitation_probability",
    })
    request = urllib.request.Request("https://api.open-meteo.com/v1/forecast?" + args, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=15) as response:
        feed = json.load(response)
    current = feed["current"]
    observed_at = parse_utc_feed_time(current["time"])
    target_start = next_complete_hour(observed_at)
    hourly_times = [parse_utc_feed_time(x) for x in feed["hourly"]["time"]]
    # Tolerate provider formatting/rounding differences without selecting a
    # neighboring forecast hour.
    index = next((i for i, value in enumerate(hourly_times) if abs((value - target_start).total_seconds()) < 60), None)
    if index is None:
        raise RuntimeError("forecast does not contain the next complete hourly interval")
    return {
        "source": "Open-Meteo current + hourly precipitation_probability", "observed_at": to_iso(observed_at),
        "temperature_c": float(current["temperature_2m"]), "relative_humidity_pct": float(current["relative_humidity_2m"]),
        "rain_now": float(current.get("precipitation") or 0) > 0,
        "period_start": to_iso(target_start), "period_end": to_iso(target_start + timedelta(hours=1)),
        "raw_probability": clamp(float(feed["hourly"]["precipitation_probability"][index]) / 100),
    }


def weather_prediction(weather: dict) -> dict:
    raw = weather["raw_probability"]
    cohort = "KR_SEL|rain_during_next_complete_hour|open_meteo_hourly"
    gp, method, score, n = laplace_calibrate(raw, load(WEATHER_LEDGER), cohort)
    pid = digest(f"{cohort}|{weather['period_start']}")  # idempotent within a forecast interval
    record = {
        "prediction_id": pid, "kind": "weather", "cohort": cohort,
        "event": "measurable rain occurs in Seoul during [period_start, period_end)",
        "observed_at": weather["observed_at"], "period_start": weather["period_start"], "due_at": weather["period_end"],
        "raw_probability": raw, "gp_probability": gp, "outcome": None, "resolved_at": None,
        "forecast_source": weather["source"], "monte_carlo": mc(gp, pid),
    }
    return {"record": record, "logged": add_record(WEATHER_LEDGER, record), "calibration": {"method": method, "matched_n": n, "brier": score}}


@dataclass(frozen=True)
class Article:
    title: str
    source: str
    url: str
    published_at: Optional[str]
    query: str


@dataclass(frozen=True)
class MapContext:
    """A topic-keyword location reference, not a Google Maps observation."""
    label: str
    district: str
    latitude: float
    longitude: float
    maps_search_url: str
    match_type: str


def clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", unescape(value))).strip()


def korean_queries(topic: str) -> list[str]:
    return list(dict.fromkeys([topic, f"{topic} 최신", f"{topic} 발표", f"{topic} 분석", f"{topic} 자료"]))


def map_context(topic: str) -> MapContext:
    """Build a usable Maps search reference without claiming a Maps signal."""
    places = {
        "성수": ("Seongsu-dong", 37.5446, 127.0559),
        "판교": ("Pangyo Techno Valley", 37.4014, 127.1086),
        "강남": ("Gangnam", 37.4979, 127.0276),
        "여의도": ("Yeouido", 37.5215, 126.9242),
        "양재": ("Yangjae", 37.4782, 127.0396),
        "상암": ("Sangam DMC", 37.5794, 126.8898),
        "마곡": ("Magok", 37.5602, 126.8285),
        "가산": ("Gasan Digital Complex", 37.4804, 126.8829),
    }
    for keyword, (label, lat, lon) in places.items():
        if keyword in topic:
            query = urllib.parse.quote(f"{topic} {label}")
            return MapContext(label, keyword, lat, lon, f"https://www.google.com/maps/search/?api=1&query={query}", "topic keyword reference")
    query = urllib.parse.quote(topic + " Seoul")
    return MapContext("Seoul", "Seoul", SEOUL["lat"], SEOUL["lon"], f"https://www.google.com/maps/search/?api=1&query={query}", "city fallback reference")


def fetch_korean_news(query: str) -> list[Article]:
    args = urllib.parse.urlencode({"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"})
    request = urllib.request.Request("https://news.google.com/rss/search?" + args, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=15) as response:
        root = ET.fromstring(response.read())
    result: list[Article] = []
    for item in root.findall("./channel/item"):
        title, url = clean_html(item.findtext("title", "")), item.findtext("link", "").strip()
        source = clean_html(item.findtext("source", "unknown")) or "unknown"
        published = item.findtext("pubDate")
        try:
            published = to_iso(parsedate_to_datetime(published)) if published else None
        except (TypeError, ValueError):
            published = None
        if title and url:
            result.append(Article(title, source, url, published, query))
    return result


def research_prediction(topic: str, horizon_hours: int, claim: Optional[str] = None) -> dict:
    articles: list[Article] = []
    failures: list[str] = []
    for query in korean_queries(topic):
        try:
            articles.extend(fetch_korean_news(query))
        except (OSError, ValueError, ET.ParseError) as exc:
            failures.append(f"{query}: {exc.__class__.__name__}")
    seen: set[str] = set()
    articles = [a for a in articles if not (re.sub(r"\W+", "", a.title.casefold()) in seen or seen.add(re.sub(r"\W+", "", a.title.casefold())))]
    if not articles:
        # A failed/empty search is absence of evidence, not a low-probability forecast.
        raise RuntimeError("no Korean open-web articles fetched; no IP prediction logged")
    # RSS publisher labels are a diversity proxy, not domain-verified independence.
    sources = {a.source.casefold() for a in articles if a.source.casefold() not in {"unknown", "google news"}}
    coverage = len({a.query for a in articles})
    # Explicit provisional model: predicts future corroboration, not truth.
    raw = round(clamp(0.15 + 0.12 * min(len(sources), 5) + 0.05 * min(coverage, 4)), 3)
    claim_text = (claim or topic).strip()
    if not claim_text:
        raise ValueError("a non-empty topic or claim is required")
    claim_key = digest(claim_text.casefold())
    cohort = f"KR_SEL|independent_korean_source_corroboration|{claim_key}"
    gp, method, score, n = laplace_calibrate(raw, load(IP_LEDGER), cohort)
    observed = now_utc()
    due = observed + timedelta(hours=horizon_hours)
    existing = active_pending(IP_LEDGER, cohort, observed)
    if existing is not None:
        return {
            "record": existing, "logged": False,
            "calibration": {"method": method, "matched_n": n, "brier": score},
            "discovery": {
                "queries": korean_queries(topic), "articles": [asdict(a) for a in articles[:80]],
                "article_count": len(articles), "publisher_label_diversity": len(sources), "query_coverage": coverage,
                "map_context": asdict(map_context(topic)), "fetch_failures": failures,
                "limitations": "An overlapping pending forecast already exists; fresh articles are reported but no correlated ledger row is added.",
            },
        }
    pid = digest(f"{cohort}|{to_iso(observed)}|{horizon_hours}")
    record = {
        "prediction_id": pid, "kind": "ip", "cohort": cohort, "topic": topic, "claim": claim_text,
        "event": "at least two independently named Korean open-web sources corroborate claim by due_at",
        "observed_at": to_iso(observed), "due_at": to_iso(due), "raw_probability": raw, "gp_probability": gp,
        "outcome": None, "resolved_at": None, "monte_carlo": mc(gp, pid),
    }
    return {
        "record": record, "logged": add_record(IP_LEDGER, record),
        "calibration": {"method": method, "matched_n": n, "brier": score},
        "discovery": {
            "queries": korean_queries(topic), "articles": [asdict(a) for a in articles[:80]],
            "article_count": len(articles), "publisher_label_diversity": len(sources), "query_coverage": coverage,
            "map_context": asdict(map_context(topic)), "fetch_failures": failures,
            "limitations": "Publisher labels are not proof of independent ownership or primary-source status.",
        },
    }


def deploy(topic: str, horizon_hours: int, claim: Optional[str] = None) -> dict:
    try:
        weather = fetch_weather()
        weather_result: dict = {"telemetry": weather, "prediction": weather_prediction(weather)}
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        weather_result = {"status": "unavailable", "reason": exc.__class__.__name__}
    try:
        ip_result = research_prediction(topic, horizon_hours, claim)
    except (OSError, ValueError, RuntimeError) as exc:
        ip_result = {"status": "unavailable", "reason": exc.__class__.__name__}
    return {
        "metadata": {"title": "GP Seoul IP Fused Deployment", "generated_at": to_iso(now_utc()), "anchor": SEOUL},
        "weather_gp": weather_result, "research_ip_gp": ip_result,
        "rules": [
            "Weather GP and research IP GP have independent ledgers and calibration cohorts.",
            "The weather due time is the END of the exact forecast hourly interval, never observed_at + one hour.",
            "Monte Carlo samples a stated probability; it cannot resolve an outcome.",
            "Resolution requires timestamped evidence at or after due_at and records its source.",
            "Map context is navigation only; it is not Google Maps traffic, place verification, or a probability input.",
            "Korean-source discovery is a lead generator; open primary sources before treating a claim as true.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy fused Seoul weather GP and Korean-source IP GP.")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--topic")
    parser.add_argument("--claim", help="Specific claim to be corroborated; defaults to --topic")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--resolve", nargs=3, metavar=("KIND", "PREDICTION_ID", "OUTCOME"))
    parser.add_argument("--observed-at")
    parser.add_argument("--source", help="URL or concise description of post-horizon resolution evidence")
    args = parser.parse_args()
    if args.resolve:
        if args.resolve[0] not in {"weather", "ip"} or not args.observed_at or not args.source:
            parser.error("--resolve requires KIND weather|ip, --observed-at with timezone, and --source")
        print("resolved" if resolve(args.resolve[0], args.resolve[1], int(args.resolve[2]), args.observed_at, args.source) else "not resolved")
        return
    if not args.deploy or not args.topic:
        parser.error("use --deploy --topic 'research topic'")
    if not 1 <= args.horizon_hours <= 168:
        parser.error("--horizon-hours must be 1 through 168")
    report = deploy(args.topic, args.horizon_hours, args.claim)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FUSED SEOUL DEPLOYMENT | topic={args.topic} | report={REPORT.resolve()}")
    if "prediction" in report.get("weather_gp", {}):
        p = report["weather_gp"]["prediction"]["record"]
        print(f"WEATHER GP={p['gp_probability']:.1%} | {p['period_start']} to {p['due_at']} | id={p['prediction_id']}")
    if "record" in report.get("research_ip_gp", {}):
        p = report["research_ip_gp"]["record"]
        print(f"IP GP={p['gp_probability']:.1%} | due={p['due_at']} | id={p['prediction_id']}")


if __name__ == "__main__":
    main()

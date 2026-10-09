"""
REAL MODE: Telemetry / uplink layer for drone_core_v2 (the drone's own data only).

1. Live stream: one compact, versioned, timestamped JSON-lines message per tick
   (pose + sigma, heading + sigma, mag_ok, action, nearest obstacle, battery),
   rate-limited by an ASSUMED bandwidth budget (token bucket). A bounded queue
   drops the oldest NON-critical messages when the link is down; safety and
   action-change events go to a separate, larger queue and are sent first.
   Every message carries a sequence number and a CRC32 of its canonical body.
2. Incremental cloud map (primary): during flight the drone sends map DELTAS (grid
   cells whose quantised value changed + new points), each message versioned,
   sequence-numbered, CRC32-checked and zlib-compressed. The cloud-side
   CloudMapAssembler (in-memory fake) rebuilds the map; applying a message twice
   or out of order is harmless (highest version per cell wins, points are
   id-keyed). A bounded per-link backlog drops the oldest deltas during long
   outages and schedules a snapshot resync. The drone never has to return to
   report; a final full-map hash comparison is the reconciliation check.
3. Final map export (ObstacleMap + flight summary) as canonical JSON + SHA-256,
   chunked with per-chunk SHA-256 for resumable upload.
4. Links: WiFi / LTE / satellite (Starlink-like) profiles with ASSUMED bandwidth,
   latency and outage statistics (placeholders, NOT measurements or vendor specs)
   and a LinkManager that picks the preferred available link per message and fails
   over. Satellite internet is modelled ONLY as a data link: it provides no
   positioning and no GPS improvement here (no official positioning service exists;
   signal-of-opportunity positioning is experimental research and out of scope).
   A satellite terminal is an optional power load (drone_core_v2.SAT_TERMINAL_W,
   ASSUMED) and adds mass, so it needs a larger airframe.

Transport is an abstract interface; in-memory and file fakes are provided. No
network code is included. Encryption and authentication of a real link are OUT
OF SCOPE here and must be added before real use (CRC32 detects accidental
corruption only; it is not tamper protection).

The flight loop only calls UplinkManager.on_tick(), which enqueues in O(1); map
sync (MapSync.collect/pump) runs in the uplink worker, outside the flight tick.
Sending happens in flush(): either called by the caller outside the control
loop, or by a background thread (threaded=True). Transport exceptions are
caught and counted, never propagated to the flight loop.

Dependencies: Python standard library (+ numpy for array conversion).
License: CC0 1.0 Universal (public domain). Copy, modify, use freely.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

SCHEMA_VERSION = 1
MAP_SCHEMA_VERSION = 1


# ===========================================================================
# Message encoding: JSON line {"crc": crc32(body), "body": {...}}
# ===========================================================================
def _canonical(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")


def _r(x, nd=3):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def encode_message(body: Dict) -> bytes:
    """One JSON line; the CRC32 covers the canonical encoding of the body."""
    return _canonical({"crc": zlib.crc32(_canonical(body)) & 0xFFFFFFFF, "body": body}) + b"\n"


def decode_message(line: bytes) -> Dict:
    """Return the body, or raise ValueError on bad JSON, missing fields, CRC or schema mismatch."""
    try:
        obj = json.loads(line.decode("utf-8"))
        body, crc = obj["body"], int(obj["crc"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed message: {e}") from e
    if zlib.crc32(_canonical(body)) & 0xFFFFFFFF != crc:
        raise ValueError("CRC32 mismatch")
    if body.get("v") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {body.get('v')}")
    return body


def build_stream_body(tick_out: Dict, seq: int, t: float, kind: str = "state",
                      reason: Optional[str] = None) -> Dict:
    """Compact per-tick body from a drone_core_v2 process_flight_tick() output."""
    pt = tick_out["predictive_telemetry"]
    fg = tick_out["flight_guidance"]
    hd = tick_out.get("heading", {})
    pw = tick_out["power_management"]
    body = {
        "v": SCHEMA_VERSION, "seq": int(seq), "t": _r(t, 3), "kind": kind,
        "pos": _r(pt["extrapolated_pos"]), "pos_sig": _r(pt["pos_sigma_m"]),
        "vel": _r(pt["extrapolated_vel"]),
        "yaw": _r(hd.get("yaw_enu_deg"), 2), "yaw_sig": _r(hd.get("heading_sigma_deg"), 2),
        "mag_ok": bool(hd.get("mag_ok", False)),
        "act": fg["action"], "mis": fg["mission_action"],
        "near": _r(fg["closest_obstacle_m"]), "blind": bool(fg["sonar_blind"]),
        "hdg_unc": bool(fg.get("heading_uncertain", False)),
        "soc": _r(pw["soc"], 4),
    }
    if reason:
        body["why"] = reason
    return body


# ===========================================================================
# Transport interface + fakes
# ===========================================================================
class Transport:
    """Abstract link. send() must not block for long; return False if the link is down."""

    def is_up(self) -> bool:
        raise NotImplementedError

    def send(self, data: bytes) -> bool:
        raise NotImplementedError


class InMemoryTransport(Transport):
    def __init__(self):
        self.up = True
        self.sent: List[bytes] = []
        self.delay_s = 0.0          # to simulate a slow link in tests

    def is_up(self) -> bool:
        return self.up

    def send(self, data: bytes) -> bool:
        if not self.up:
            return False
        if self.delay_s:
            time.sleep(self.delay_s)
        self.sent.append(bytes(data))
        return True


class FileTransport(Transport):
    """Appends each message to a local file (a stand-in for a link)."""

    def __init__(self, path: str):
        self.path = path
        self.up = True

    def is_up(self) -> bool:
        return self.up

    def send(self, data: bytes) -> bool:
        if not self.up:
            return False
        with open(self.path, "ab") as f:
            f.write(data)
        return True


# ===========================================================================
# Live stream manager
# ===========================================================================
@dataclass
class UplinkConfig:
    bandwidth_bytes_per_s: float = 2000.0    # ASSUMED link budget for the live stream
    burst_bytes: float = 4000.0              # ASSUMED token-bucket depth
    state_rate_hz: float = 5.0               # ASSUMED periodic state message rate
    max_state_queue: int = 300               # ASSUMED (non-critical; oldest dropped)
    max_event_queue: int = 2000              # ASSUMED (critical; dropped only if this overflows)
    max_sends_per_flush: int = 50            # bounded work per flush call


SAFETY_ACTIONS = {"COLLISION_AVOIDANCE_BRAKE", "COLLISION_LIKELY_MAX_BRAKE", "HOLD_STANDOFF"}


class UplinkManager:
    """
    on_tick(): builds the message and enqueues it (O(1), never raises).
    Critical events: action or mission change, any safety action, sonar-blind or
    heading-uncertain transitions. Periodic state messages are rate limited.
    flush(): sends events first, then states, oldest first, within the token bucket.
    """

    def __init__(self, transport: Transport, config: Optional[UplinkConfig] = None, threaded: bool = False):
        self.tr = transport
        self.cfg = config or UplinkConfig()
        self.events: deque = deque()
        self.states: deque = deque()
        self.lock = threading.Lock()
        self.seq = 0
        self.tokens = self.cfg.burst_bytes
        self._last_refill: Optional[float] = None
        self._last_state_t: Optional[float] = None
        self._prev: Dict = {}
        self.stats = {"built": 0, "events": 0, "states": 0, "sent": 0, "sent_bytes": 0,
                      "dropped_state": 0, "dropped_event": 0, "skipped_rate": 0,
                      "transport_errors": 0, "max_on_tick_s": 0.0}
        self._stop = threading.Event()
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _classify(self, tick_out: Dict) -> Optional[str]:
        fg = tick_out["flight_guidance"]
        cur = {"act": fg["action"], "mis": fg["mission_action"], "blind": fg["sonar_blind"],
               "hdg_unc": fg.get("heading_uncertain", False)}
        reasons = [k for k in cur if k in self._prev and cur[k] != self._prev[k]]
        if not self._prev:
            reasons.append("start")
        if fg["action"] in SAFETY_ACTIONS:
            reasons.append("safety")
        self._prev = cur
        return ",".join(reasons) if reasons else None

    def on_tick(self, tick_out: Dict, t: float) -> None:
        t0 = time.perf_counter()
        try:
            reason = self._classify(tick_out)
            due = self._last_state_t is None or t - self._last_state_t >= 1.0 / self.cfg.state_rate_hz - 1e-9
            if reason is None and not due:
                self.stats["skipped_rate"] += 1
                return
            self.seq += 1
            kind = "event" if reason else "state"
            msg = encode_message(build_stream_body(tick_out, self.seq, t, kind, reason))
            self.stats["built"] += 1
            with self.lock:
                if reason:
                    self.events.append(msg)
                    self.stats["events"] += 1
                    if len(self.events) > self.cfg.max_event_queue:
                        self.events.popleft()
                        self.stats["dropped_event"] += 1
                else:
                    self.states.append(msg)
                    self.stats["states"] += 1
                    if len(self.states) > self.cfg.max_state_queue:
                        self.states.popleft()
                        self.stats["dropped_state"] += 1
            self._last_state_t = t
        except Exception:                     # the flight loop must never see an uplink error
            self.stats["transport_errors"] += 1
        finally:
            self.stats["max_on_tick_s"] = max(self.stats["max_on_tick_s"], time.perf_counter() - t0)

    def flush(self, now: float) -> int:
        """Send what the bandwidth budget allows. Returns the number of messages sent."""
        if self._last_refill is not None:
            self.tokens = min(self.cfg.burst_bytes,
                              self.tokens + (now - self._last_refill) * self.cfg.bandwidth_bytes_per_s)
        self._last_refill = now
        n = 0
        try:
            if not self.tr.is_up():
                return 0
            while n < self.cfg.max_sends_per_flush:
                with self.lock:
                    q = self.events if self.events else self.states
                    if not q or len(q[0]) > self.tokens:
                        break
                    msg = q[0]
                if not self.tr.send(msg):
                    break
                with self.lock:
                    if q and q[0] is msg:
                        q.popleft()
                self.tokens -= len(msg)
                self.stats["sent"] += 1
                self.stats["sent_bytes"] += len(msg)
                n += 1
        except Exception:
            self.stats["transport_errors"] += 1
        return n

    def _worker(self) -> None:
        while not self._stop.is_set():
            self.flush(time.monotonic())
            self._stop.wait(0.02)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def queued(self) -> Dict[str, int]:
        with self.lock:
            return {"events": len(self.events), "states": len(self.states)}


# ===========================================================================
# Final map export + chunked, resumable upload
# ===========================================================================
def export_map(core, flight_summary: Optional[Dict] = None, created_t: float = 0.0) -> Dict:
    """ObstacleMap -> JSON-able dict. Covariances already include the heading-sigma term."""
    pts = []
    for o in core.map.items:
        pts.append({"pos": _r(o.pos_enu, 4), "cov": _r(np.asarray(o.cov).tolist(), 6),
                    "last_seen": _r(o.last_seen_s, 3), "hits": int(o.hits),
                    "ir_consistent_hits": int(o.ir_consistent_hits)})
    return {
        "map_schema": MAP_SCHEMA_VERSION, "created_t": _r(created_t, 3), "frame": "local ENU (m)",
        "note": "cov is sensor + heading-sigma covariance; absolute positions also carry the pose sigma",
        "points": pts, "summary": flight_summary or {},
    }


def map_to_bytes(export: Dict) -> bytes:
    return _canonical(export)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_map_file(path: str, export: Dict) -> Dict:
    data = map_to_bytes(export)
    with open(path, "wb") as f:
        f.write(data)
    manifest = {"file": path, "size": len(data), "sha256": sha256_hex(data)}
    with open(path + ".sha256", "w") as f:
        f.write(f"{manifest['sha256']}  {path}\n")
    return manifest


def read_map_file(path: str, expected_sha256: str) -> Dict:
    with open(path, "rb") as f:
        data = f.read()
    if sha256_hex(data) != expected_sha256:
        raise ValueError("map file checksum mismatch")
    return json.loads(data.decode("utf-8"))


def make_manifest(data: bytes, chunk_size: int, upload_id: str) -> Dict:
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
    return {"upload_id": upload_id, "size": len(data), "sha256": sha256_hex(data),
            "chunk_size": chunk_size, "n_chunks": len(chunks),
            "chunk_sha256": [sha256_hex(c) for c in chunks]}


class ChunkSink:
    """Abstract receiver for resumable uploads."""

    def received(self, upload_id: str) -> Set[int]:
        raise NotImplementedError

    def put_chunk(self, upload_id: str, index: int, data: bytes, sha: str) -> bool:
        raise NotImplementedError


class InMemoryChunkSink(ChunkSink):
    def __init__(self):
        self.store: Dict[str, Dict[int, bytes]] = {}
        self.fail_after: Optional[int] = None   # simulate a dropped link after N chunks
        self.puts = 0

    def received(self, upload_id: str) -> Set[int]:
        return set(self.store.get(upload_id, {}))

    def put_chunk(self, upload_id: str, index: int, data: bytes, sha: str) -> bool:
        if self.fail_after is not None and self.puts >= self.fail_after:
            return False
        if sha256_hex(data) != sha:              # receiver verifies each chunk
            return False
        self.store.setdefault(upload_id, {})[index] = bytes(data)
        self.puts += 1
        return True

    def assemble(self, manifest: Dict) -> bytes:
        parts = self.store.get(manifest["upload_id"], {})
        if set(parts) != set(range(manifest["n_chunks"])):
            raise ValueError("incomplete upload")
        data = b"".join(parts[i] for i in range(manifest["n_chunks"]))
        if sha256_hex(data) != manifest["sha256"]:
            raise ValueError("assembled checksum mismatch")
        return data


def upload_chunked(data: bytes, manifest: Dict, sink: ChunkSink, max_chunks: Optional[int] = None) -> Dict:
    """Send only the chunks the sink does not already have. Safe to call again to resume."""
    have = sink.received(manifest["upload_id"])
    cs = manifest["chunk_size"]
    sent, failed = 0, False
    for i in range(manifest["n_chunks"]):
        if i in have:
            continue
        if max_chunks is not None and sent >= max_chunks:
            break
        if not sink.put_chunk(manifest["upload_id"], i, data[i * cs:(i + 1) * cs], manifest["chunk_sha256"][i]):
            failed = True
            break
        sent += 1
    done = len(sink.received(manifest["upload_id"])) == manifest["n_chunks"]
    return {"sent_this_call": sent, "complete": done, "interrupted": failed}


# ===========================================================================
# Links: WiFi / LTE / satellite (Starlink-like) profiles + failover manager.
# ALL profile numbers are ASSUMED placeholders for simulation, NOT measured and
# NOT vendor specifications. No real radio or network code is included.
# Satellite internet is used here ONLY as a data link: it is NOT a positioning
# source and gives no GPS improvement in this code.
# ===========================================================================
@dataclass
class LinkProfile:
    name: str
    bandwidth_Bps: float        # usable uplink payload bytes/s (ASSUMED)
    latency_s: float            # one-way delivery delay (ASSUMED)
    mean_up_s: float            # outage model: exponential up / down episodes (ASSUMED)
    mean_down_s: float
    range_m: Optional[float] = None   # max distance from the ground station (WiFi only)
    power_w: float = 0.0        # radio/terminal draw while enabled (ASSUMED)
    cost_rank: int = 0          # lower is preferred when several links are up


LINK_PROFILES: Dict[str, LinkProfile] = {
    "wifi": LinkProfile("wifi", 1_000_000.0, 0.005, 120.0, 1.0, range_m=100.0, power_w=1.0, cost_rank=0),
    "lte": LinkProfile("lte", 250_000.0, 0.060, 60.0, 4.0, power_w=3.0, cost_rank=1),
    # "tens of watts" order only; the value is an ASSUMED placeholder
    "satellite": LinkProfile("satellite", 100_000.0, 0.050, 90.0, 3.0, power_w=50.0, cost_rank=2),
}


class SimLink(Transport):
    """Simulated link with outage episodes, range limit, bandwidth and latency.
    Delivered messages are handed to `deliver(t_arrival, data)`."""

    def __init__(self, profile: LinkProfile, rng: np.random.Generator, deliver,
                 station_pos: Optional[np.ndarray] = None):
        self.p = profile
        self.rng = rng
        self.deliver = deliver
        self.station = None if station_pos is None else np.asarray(station_pos, float)
        self._up = True
        self._next_switch = float(rng.exponential(profile.mean_up_s))
        self._in_range = True
        self.t = 0.0
        self.budget = 0.0
        self.bytes_sent = 0
        self.up_time_s = 0.0
        self.total_time_s = 0.0

    def step(self, t: float, dt: float, drone_pos: Optional[np.ndarray] = None) -> None:
        self.t = t
        while t >= self._next_switch:
            self._up = not self._up
            self._next_switch += float(self.rng.exponential(self.p.mean_up_s if self._up else self.p.mean_down_s))
        if self.p.range_m is not None and self.station is not None and drone_pos is not None:
            self._in_range = float(np.linalg.norm(np.asarray(drone_pos, float) - self.station)) <= self.p.range_m
        self.budget = min(self.budget + self.p.bandwidth_Bps * dt, self.p.bandwidth_Bps * max(dt, 0.1))
        self.total_time_s += dt
        self.up_time_s += dt * int(self.is_up())

    def is_up(self) -> bool:
        return self._up and self._in_range

    def can_send(self, n: int) -> bool:
        return self.is_up() and self.budget >= n

    def send(self, data: bytes) -> bool:
        if not self.can_send(len(data)):
            return False
        self.budget -= len(data)
        self.bytes_sent += len(data)
        self.deliver(self.t + self.p.latency_s, data)
        return True


class LinkManager:
    """Picks the preferred available link (lowest cost_rank that is up) per message and
    fails over when it goes down. Never raises into the caller."""

    def __init__(self, links: List[SimLink]):
        self.links = sorted(links, key=lambda l: l.p.cost_rank)
        self.current: Optional[str] = None
        self.failovers = 0
        self.sends_by_link: Dict[str, int] = {l.p.name: 0 for l in self.links}

    def any_up(self) -> bool:
        return any(l.is_up() for l in self.links)

    def _by_name(self, name: str) -> "SimLink":
        return next(l for l in self.links if l.p.name == name)

    def send(self, data: bytes) -> bool:
        for l in self.links:
            if l.can_send(len(data)):
                # a failover is a switch because the previous link went DOWN (not budget spill)
                if self.current is not None and l.p.name != self.current and \
                        not self._by_name(self.current).is_up():
                    self.failovers += 1
                self.current = l.p.name
                self.sends_by_link[l.p.name] += 1
                return l.send(data)
        return False

    def energy_wh(self) -> Dict[str, float]:
        """Radio energy with each link enabled for the whole mission (ASSUMED power)."""
        return {l.p.name: round(l.p.power_w * l.total_time_s / 3600.0, 4) for l in self.links}


# ===========================================================================
# Incremental cloud map: drone-side MapSync -> CloudMapAssembler
# Cells and points are sent as ABSOLUTE values with a version (= delta sequence of
# their last change), so applying a message twice or out of order is harmless:
# the cloud keeps the highest version per cell. Points are immutable (id-keyed).
# A snapshot is the same message type carrying everything, used to resync after
# the drone had to drop deltas (bounded backlog) or the cloud reports a gap.
# ===========================================================================
def _map_canonical_hash(cells: Dict, points: Dict) -> str:
    c = sorted([list(k) + list(v[:2]) for k, v in cells.items()])
    p = sorted([[pid] + list(v) for pid, v in points.items()])
    return sha256_hex(_canonical({"cells": c, "points": p}))


def encode_map_message(body: Dict) -> bytes:
    """Map messages are compressed: b"Z" + zlib(JSON line with CRC32)."""
    return b"Z" + zlib.compress(encode_message(body), 6)


def decode_map_message(data: bytes) -> Dict:
    if data[:1] == b"Z":
        try:
            data = zlib.decompress(data[1:])
        except zlib.error as e:
            raise ValueError(f"bad compressed frame: {e}") from None
    return decode_message(data)


def _q_point(pt) -> Tuple[int, int, int, int]:
    return (int(round(pt.pos[0] * 1000)), int(round(pt.pos[1] * 1000)), int(round(pt.pos[2] * 1000)),
            int(round(pt.sigma * 1000)))


class SyncChannel:
    """One uplink path (a LinkManager) with its own bounded DELTA backlog. When it
    overflows, the oldest deltas are dropped and a snapshot resync is scheduled for the
    next time a link is up. The snapshot replaces the queued deltas, has its own queue
    (never dropped) and is sent first."""

    def __init__(self, link: "LinkManager", max_backlog_bytes: int = 2_000_000):
        self.link = link
        self.max_backlog = max_backlog_bytes
        self.queue: deque = deque()
        self.snap_queue: deque = deque()
        self.backlog = 0
        self.need_snapshot = False
        self.stats = {"dropped_msgs": 0, "resyncs": 0, "snapshot_msgs": 0, "snapshot_bytes": 0,
                      "sent_msgs": 0, "sent_bytes": 0}

    def enqueue(self, data: bytes) -> None:
        self.queue.append(data)
        self.backlog += len(data)
        while self.backlog > self.max_backlog and self.queue:
            d = self.queue.popleft()
            self.backlog -= len(d)
            self.stats["dropped_msgs"] += 1
            self.need_snapshot = True

    def pump(self, sync: "MapSync", t: float) -> int:
        if self.need_snapshot and self.link.any_up():
            snap = sync.snapshot(t)
            self.snap_queue = deque(snap)
            self.queue.clear()
            self.backlog = 0
            self.need_snapshot = False
            self.stats["resyncs"] += 1
            self.stats["snapshot_msgs"] += len(snap)
            self.stats["snapshot_bytes"] += sum(len(d) for d in snap)
        sent = 0
        for q, is_delta in ((self.snap_queue, False), (self.queue, True)):
            while q:
                d = q[0]
                if not self.link.send(d):
                    return sent
                q.popleft()
                if is_delta:
                    self.backlog -= len(d)
                sent += 1
                self.stats["sent_msgs"] += 1
                self.stats["sent_bytes"] += len(d)
        return sent


class MapSync:
    """Drone side producer. collect() turns the grid cells whose QUANTISED value changed
    (and new points) into versioned, CRC-checked, compressed delta messages and hands them
    to every channel. collect()/pump() run in the uplink worker, never in the flight tick;
    both are O(changed entries)."""

    def __init__(self, recon_map, max_entries_per_msg: int = 400):
        self.map = recon_map
        self.max_entries = max_entries_per_msg
        self.seq = 0
        self.sent_points = 0
        self.version: Dict = {}
        self.last_sent: Dict = {}
        self.channels: List[SyncChannel] = []
        self.stats = {"delta_msgs": 0, "delta_bytes": 0}

    def add_channel(self, link: "LinkManager", max_backlog_bytes: int = 2_000_000) -> SyncChannel:
        ch = SyncChannel(link, max_backlog_bytes)
        self.channels.append(ch)
        return ch

    def _entries(self, keys, pts) -> List[Dict]:
        cells = [[k[0], k[1], k[2], *self.last_sent[k], self.version[k]] for k in keys]
        pl = [[p.pid, *_q_point(p)] for p in pts]
        msgs = []
        i = j = 0
        while True:
            c = cells[i:i + self.max_entries]
            p = pl[j:j + max(self.max_entries - len(c), 0)]
            i += len(c)
            j += len(p)
            msgs.append({"c": c, "p": p, "part": len(msgs)})
            if i >= len(cells) and j >= len(pl):
                break
        for m in msgs:
            m["nparts"] = len(msgs)
        return msgs

    def collect(self, t: float) -> List[bytes]:
        g = self.map.grid
        keys = []
        for k in g.drain_dirty():
            q = g.quantized(k)
            if self.last_sent.get(k) != q:
                self.last_sent[k] = q
                keys.append(k)
        pts = self.map.points[self.sent_points:]
        self.sent_points = len(self.map.points)
        if not keys and not pts:
            return []
        self.seq += 1
        for k in keys:
            self.version[k] = self.seq
        out = []
        for m in self._entries(keys, pts):
            m.update({"v": MAP_SCHEMA_VERSION, "kind": "map_delta", "seq": self.seq, "t": _r(t)})
            data = encode_map_message(m)
            out.append(data)
            self.stats["delta_msgs"] += 1
            self.stats["delta_bytes"] += len(data)
            for ch in self.channels:
                ch.enqueue(data)
        return out

    def snapshot(self, t: float) -> List[bytes]:
        """Everything collected so far (values as last sent, with their versions)."""
        keys = list(self.last_sent)
        out = []
        for m in self._entries(keys, self.map.points[:self.sent_points]):
            m.update({"v": MAP_SCHEMA_VERSION, "kind": "map_snapshot", "seq": self.seq, "t": _r(t)})
            out.append(encode_map_message(m))
        return out

    def pump(self, t: float) -> int:
        return sum(ch.pump(self, t) for ch in self.channels)

    def quantized_state(self) -> Tuple[Dict, Dict]:
        """The drone's map as of the last collect() (what the cloud should converge to)."""
        pts = {p.pid: _q_point(p) for p in self.map.points[:self.sent_points]}
        return dict(self.last_sent), pts

    def map_hash(self) -> str:
        return _map_canonical_hash(*self.quantized_state())


class CloudMapAssembler:
    """Cloud side (in-memory fake). Idempotent and order-tolerant; CRC-checked."""

    def __init__(self):
        self.cells: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        self.points: Dict[int, Tuple[int, int, int, int]] = {}
        self.seen: Set[Tuple[str, int, int]] = set()
        self.delta_parts: Dict[int, int] = {}
        self.snapshot_seq = 0
        self.stats = {"received": 0, "duplicates": 0, "bad_crc": 0, "applied_cells": 0}
        self._pending: List[Tuple[float, bytes]] = []

    def deliver(self, t_arrival: float, data: bytes) -> None:   # used by SimLink
        self._pending.append((t_arrival, data))

    def poll(self, now: float) -> int:
        ready = [d for t, d in self._pending if t <= now]
        self._pending = [(t, d) for t, d in self._pending if t > now]
        for d in ready:
            self.receive(d)
        return len(ready)

    def receive(self, line: bytes) -> bool:
        try:
            m = decode_map_message(line)
        except ValueError:
            self.stats["bad_crc"] += 1
            return False
        key = (m["kind"], int(m["seq"]), int(m["part"]))
        if key in self.seen:
            self.stats["duplicates"] += 1
            return False
        self.seen.add(key)
        self.stats["received"] += 1
        if m["kind"] == "map_delta":
            self.delta_parts.setdefault(int(m["seq"]), int(m["nparts"]))
        else:
            self.snapshot_seq = max(self.snapshot_seq, int(m["seq"]))
        for c in m["c"]:
            k, val, ver = (c[0], c[1], c[2]), (c[3], c[4]), c[5]
            cur = self.cells.get(k)
            if cur is None or cur[2] < ver:
                self.cells[k] = (val[0], val[1], ver)
                self.stats["applied_cells"] += 1
        for p in m["p"]:
            self.points.setdefault(int(p[0]), tuple(p[1:5]))
        return True

    def missing_deltas(self) -> List[int]:
        """Delta sequences not fully received and not covered by a later snapshot."""
        if not self.delta_parts:
            return []
        hi = max(self.delta_parts)
        miss = []
        for s in range(self.snapshot_seq + 1, hi + 1):
            n = self.delta_parts.get(s)
            if n is None or any(("map_delta", s, i) not in self.seen for i in range(n)):
                miss.append(s)
        return miss

    def map_hash(self) -> str:
        return _map_canonical_hash({k: v[:2] for k, v in self.cells.items()}, self.points)

    def fraction_present(self, drone_cells: Dict, drone_points: Dict, occupied_only_keys=None) -> Dict:
        """Fraction of the drone's current map that the cloud holds with identical values."""
        same = sum(1 for k, v in drone_cells.items() if k in self.cells and tuple(self.cells[k][:2]) == tuple(v))
        pts = sum(1 for pid in drone_points if pid in self.points)
        out = {"cells": same / max(len(drone_cells), 1), "points": pts / max(len(drone_points), 1)}
        if occupied_only_keys is not None:
            occ = list(occupied_only_keys)
            out["occupied_cells"] = sum(1 for k in occ if k in self.cells and
                                        tuple(self.cells[k][:2]) == tuple(drone_cells[k])) / max(len(occ), 1)
        return out

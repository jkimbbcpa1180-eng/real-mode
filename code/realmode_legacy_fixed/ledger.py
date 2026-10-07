"""Real Mode legacy ledger, repaired release 1.0 (Python 3.9+).

TP is a user-entered score, not verified truth or calibrated probability.
mean = sum(p)/n; geo = exp(sum(log(p))/n).
Legacy odds pool = sigmoid(sum(logit(p))); assumes no evidence model.
No claim of Bayesian inference is made. Boundary odds use eps=1e-6.
Persistence is atomic, single-writer only; no concurrent-process locking.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ISO = "%Y-%m-%dT%H:%M:%S"


def now():
    return datetime.now(timezone.utc).isoformat()


def number(value, label="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def text(value, label="text", blank=True):
    if not isinstance(value, str) or (not blank and not value.strip()):
        raise ValueError(f"invalid {label}")
    return value


def strings(values):
    if not isinstance(values, list):
        raise ValueError("expected list of strings")
    return [text(v) for v in values]


def timestamp(value):
    text(value, "timestamp", False)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Legacy naive dates are explicitly interpreted as UTC on import.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class TP:
    value: float

    def __post_init__(self):
        self.value = number(self.value, "TP")
        if not 0 <= self.value <= 1:
            raise ValueError("TP must be within [0, 1]")

    def bump(self, delta):
        return TP(max(0, min(1, self.value + number(delta))))

    def fade(self, factor):
        factor = TP(factor).value
        return TP(self.value * factor)

    def __str__(self):
        return f"{self.value:.2f}"


@dataclass
class Mirror:
    text: str
    created_at: str = field(default_factory=now)

    def __post_init__(self):
        text(self.text)
        self.created_at = timestamp(self.created_at)


@dataclass
class Entry:
    entry_id: str
    title: str
    content: str
    tp: TP = field(default_factory=lambda: TP(0.5))
    mirrors: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)

    def __post_init__(self):
        text(self.entry_id, "entry ID", False)
        text(self.title)
        text(self.content)
        if not isinstance(self.tp, TP):
            raise ValueError("tp must be TP")
        self.tp = TP(self.tp.value)
        if not isinstance(self.mirrors, list) or any(not isinstance(m, Mirror) for m in self.mirrors):
            raise ValueError("invalid mirrors")
        self.mirrors = copy.deepcopy(self.mirrors)
        self.tags = list(dict.fromkeys(strings(self.tags)))
        self.created_at = timestamp(self.created_at)
        self.updated_at = timestamp(self.updated_at)

    def attach_mirror(self, value):
        self.mirrors.append(Mirror(value))
        self.updated_at = now()

    def tag(self, *words):
        self.tags = list(dict.fromkeys(self.tags + strings(list(words))))
        self.updated_at = now()


@dataclass
class Streak:
    name: str
    start_date: str
    last_date: str
    count: int
    notes: list = field(default_factory=list)

    def __post_init__(self):
        text(self.name, "streak name", False)
        start, last = date.fromisoformat(self.start_date), date.fromisoformat(self.last_date)
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("count must be positive integer")
        if (last - start).days != self.count - 1:
            raise ValueError("inconsistent consecutive streak")
        self.notes = strings(self.notes)


@dataclass
class HabitSnapshot:
    date: str
    metrics: dict

    def __post_init__(self):
        date.fromisoformat(self.date)
        if not isinstance(self.metrics, dict):
            raise ValueError("metrics must be a mapping")
        self.metrics = {text(k, "metric", False): number(v, k) for k, v in self.metrics.items()}


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


class RealModeFull:
    def __init__(self, store_path="real_mode_full.json", timezone_name="Asia/Seoul", clock=None):
        self.store_path = str(store_path)
        self.zone = ZoneInfo(timezone_name)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._entries, self._streaks, self._snapshots = {}, {}, []
        self._load()

    def _today(self):
        instant = self.clock()
        if instant.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetime")
        return instant.astimezone(self.zone).date()

    @staticmethod
    def _decode(raw):
        if not isinstance(raw, dict):
            raise ValueError("archive must be an object")
        collections = [raw.get(k, []) for k in ("entries", "streaks", "snapshots")]
        if any(not isinstance(c, list) for c in collections):
            raise ValueError("archive collections must be lists")
        entries, streaks = {}, {}
        for source in collections[0]:
            d = dict(source)
            value = d.pop("tp", 0.5)
            d["tp"] = TP(value["value"] if isinstance(value, dict) else value)
            d["mirrors"] = [Mirror(**m) for m in d.get("mirrors", [])]
            for k in ("created_at", "updated_at"):
                if d.get(k) is None:
                    d[k] = d.get("created_at") or now()
            e = Entry(**d)
            if e.entry_id in entries:
                raise ValueError("duplicate entry ID")
            entries[e.entry_id] = e
        for source in collections[1]:
            s = Streak(**source)
            if s.name in streaks:
                raise ValueError("duplicate streak name")
            streaks[s.name] = s
        return entries, streaks, [HabitSnapshot(**s) for s in collections[2]]

    def _load(self):
        if Path(self.store_path).exists():
            self._entries, self._streaks, self._snapshots = self._decode(read_json(self.store_path))

    def _save(self):
        data = {"version": 1, "entries": [asdict(e) for e in self._entries.values()],
                "streaks": [asdict(s) for s in self._streaks.values()],
                "snapshots": [asdict(s) for s in self._snapshots], "saved_at": now()}
        atomic_write(self.store_path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

    def _commit(self, entries=None, streaks=None, snapshots=None):
        old = self._entries, self._streaks, self._snapshots
        self._entries = entries if entries is not None else self._entries
        self._streaks = streaks if streaks is not None else self._streaks
        self._snapshots = snapshots if snapshots is not None else self._snapshots
        try:
            self._save()
        except (OSError, ValueError, TypeError):
            self._entries, self._streaks, self._snapshots = old
            raise

    def log(self, title, content, tp=0.5, tags=None, entry_id=None):
        eid = entry_id if entry_id is not None else "RMF_" + uuid.uuid4().hex
        if eid in self._entries:
            raise ValueError("entry ID exists; use update_entry")
        e = Entry(eid, title, content, TP(tp), tags=[] if tags is None else tags)
        self._commit(entries={**self._entries, eid: e})
        return copy.deepcopy(e)

    def update_entry(self, entry_id, **values):
        if set(values) - {"title", "content", "tp", "tags"}:
            raise ValueError("unknown update fields")
        e = self.get(entry_id)
        if e is None:
            return None
        for k, v in values.items():
            setattr(e, k, TP(v.value if isinstance(v, TP) else v) if k == "tp" else copy.deepcopy(v))
        e.updated_at = now()
        e.__post_init__()
        self._commit(entries={**self._entries, entry_id: e})
        return copy.deepcopy(e)

    def attach_mirror(self, entry_id, value):
        e = self.get(entry_id)
        if e is None:
            raise KeyError(entry_id)
        e.attach_mirror(value)
        self._commit(entries={**self._entries, entry_id: e})
        return copy.deepcopy(e)

    def tag_entry(self, entry_id, *words):
        e = self.get(entry_id)
        if e is None:
            raise KeyError(entry_id)
        return self.update_entry(entry_id, tags=e.tags + list(words))

    def delete_entry(self, entry_id):
        if entry_id not in self._entries:
            return False
        entries = dict(self._entries)
        del entries[entry_id]
        self._commit(entries=entries)
        return True

    def get(self, entry_id):
        return copy.deepcopy(self._entries.get(entry_id))

    def list(self, tag=None, text=None):
        items = [e for e in self._entries.values() if (tag is None or tag in e.tags)
                 and (text is None or text.casefold() in (e.title + "\n" + e.content).casefold())]
        return copy.deepcopy(sorted(items, key=lambda e: (e.created_at, e.entry_id)))

    def start_streak(self, name, note=""):
        today = self._today().isoformat()
        text(note)
        s = Streak(name, today, today, 1, [note] if note else [])
        self._commit(streaks={**self._streaks, name: s})
        return copy.deepcopy(s)

    def tick_streak(self, name, note=""):
        s = self.streak_status(name)
        if s is None:
            return None
        text(note)
        today, last = self._today(), date.fromisoformat(s.last_date)
        if today < last:
            raise ValueError("cannot tick backwards")
        if today == last:
            return s
        if today == last + timedelta(days=1):
            s.count += 1
        else:
            s.count, s.start_date = 1, today.isoformat()
        s.last_date = today.isoformat()
        if note:
            s.notes.append(note)
        s.__post_init__()
        self._commit(streaks={**self._streaks, name: s})
        return copy.deepcopy(s)

    def streak_status(self, name):
        return copy.deepcopy(self._streaks.get(name))

    def snapshot(self, metrics):
        s = HabitSnapshot(self._today().isoformat(), metrics)
        self._commit(snapshots=self._snapshots + [s])
        return copy.deepcopy(s)

    @staticmethod
    def combine_tp(*tps, mode="mean"):
        if mode not in ("mean", "geo", "odds_pool", "bayes"):
            raise ValueError("unknown combination mode")
        vals = [TP(t.value).value for t in tps]
        if mode == "bayes":
            warnings.warn("bayes is a legacy odds-pool alias, not Bayesian inference", UserWarning)
        if not vals:
            return TP(0.5)
        if mode == "mean":
            return TP(sum(vals) / len(vals))
        if mode == "geo":
            return TP(0 if 0 in vals else math.exp(sum(math.log(v) for v in vals) / len(vals)))
        z = sum(math.log(p / (1-p)) for p in [max(1e-6, min(1-1e-6, v)) for v in vals])
        if z >= 0:
            return TP(1 / (1 + math.exp(-z)))
        ez = math.exp(z)
        return TP(ez / (1 + ez))

    mirror_line = staticmethod(Mirror)

    def import_json(self, path):
        incoming, _, _ = self._decode(read_json(path))
        additions = {k: v for k, v in incoming.items() if k not in self._entries}
        self._commit(entries={**self._entries, **additions})
        return len(additions)

    def export_markdown(self, path):
        lines = ["# Real Mode Archive", "TP values are user-entered scores."]
        for e in self.list():
            lines.extend([f"## {e.title}", f"ID: {e.entry_id}", f"TP: {e.tp}",
                          f"Tags: {', '.join(e.tags)}", f"Created: {e.created_at}",
                          f"Updated: {e.updated_at}", "", e.content])
            lines.extend(f"- {m.text} ({m.created_at})" for m in e.mirrors)
            lines.append("\n---\n")
        atomic_write(path, "\n".join(lines))
        return str(path)

    @staticmethod
    def pretty_entry(e):
        lines = [f"[{e.entry_id}] {e.title} (user score={e.tp})",
                 f"  Created: {e.created_at} | Updated: {e.updated_at}",
                 f"  Tags: {', '.join(e.tags) if e.tags else '-'}", f"  Content: {e.content}"]
        if e.mirrors:
            lines.append("  Mirrors:")
            lines.extend(f"    - {m.text} ({m.created_at})" for m in e.mirrors)
        return "\n".join(lines)

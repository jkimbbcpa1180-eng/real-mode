# SPDX-License-Identifier: CC0-1.0
"""
drone v3 shared helpers: imports the v2 modules from the parent folder (they are NOT
modified), seeds, the ASSUMED power model and a wind model.

Seeds (recorded once, used only as stated):
  tuning   : SEED_TUNE_* (settings may be chosen by looking at these)
  held-out : SEED_HOLD_* (never used before v3; evaluated once with frozen settings)

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(HERE)                       # workspace layout: v3/ inside the v2 dir
if os.path.isdir(os.path.join(V2_DIR, "v2")):         # repo layout: drone/v2 next to drone/v3
    V2_DIR = os.path.join(V2_DIR, "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

import numpy as np  # noqa: E402

import drone_core_v2 as v2  # noqa: E402,F401
import drone_recon_v2 as R  # noqa: E402,F401
import drone_slam_v2 as S  # noqa: E402,F401
import drone_uplink_v2 as U  # noqa: E402,F401

# --- seeds -----------------------------------------------------------------
SEEDS = {
    "safe_return": {"tuning": 20263006, "heldout": 20264006},
    "repass": {"tuning": 20263004, "heldout": 20264004},
    "swarm": {"tuning": 20263001, "heldout": 20264001, "fresh": 20265001},   # fresh: after the v3.1 avoidance change
    "change": {"tuning": 20263005, "heldout": 20264005},
    "lidar": {"tuning": 20263003, "heldout": 20264003},
    "thermal": {"tuning": 20263002, "heldout": 20264002},
    "ground_ref": {"tuning": 20263007, "heldout": 20264007},
}

# --- ASSUMED power model (not measured) ------------------------------------
P_BASE_W = 60.5                 # hover 55 + avionics 2 + sonar 3.5 (v2 critical load)
P_SPEED_W_PER_MPS2 = 0.30       # ASSUMED extra power per (airspeed m/s)^2
BATTERY_WH = 45.0               # v2 default capacity


def power_w(airspeed_mps: float, extra_w: float = 0.0) -> float:
    return P_BASE_W + P_SPEED_W_PER_MPS2 * airspeed_mps ** 2 + extra_w


class Wind:
    """Horizontal wind: mean (level, direction) + slow random-walk drift of the mean
    + Gauss-Markov gusts. ASSUMED model."""

    def __init__(self, level_mps: float, rng: np.random.Generator, gust_frac: float = 0.25,
                 gust_tau_s: float = 4.0, drift_frac_per_min: float = 0.10):
        a = rng.uniform(0, 2 * math.pi)
        self.mean = level_mps * np.array([math.cos(a), math.sin(a)])
        self.level = float(level_mps)
        self.gust = np.zeros(2)
        self.sg = gust_frac * level_mps
        self.tau = gust_tau_s
        self.drift = drift_frac_per_min * level_mps / math.sqrt(60.0)
        self.rng = rng

    def step(self, dt: float) -> np.ndarray:
        phi = math.exp(-dt / self.tau)
        self.gust = phi * self.gust + math.sqrt(1 - phi * phi) * self.rng.normal(0, self.sg, 2) if self.sg > 0 else self.gust
        if self.drift > 0:
            self.mean = self.mean + self.rng.normal(0, self.drift * math.sqrt(dt), 2)
        return np.array([*(self.mean + self.gust), 0.0])


def fmt(x, n=3):
    return None if x is None else round(float(x), n)


def stats(x) -> dict:
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 3), "mean": round(float(np.mean(x)), 3),
            "p90": round(float(np.percentile(x, 90)), 3), "max": round(float(np.max(x)), 3)}


def pct(flags) -> float:
    flags = list(flags)
    return round(100.0 * sum(bool(f) for f in flags) / max(len(flags), 1), 1)

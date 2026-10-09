# SPDX-License-Identifier: CC0-1.0
"""
drone v3 sensors: vectorised ray casting, a simulated lidar (feature 3) and a simulated
low-resolution thermal camera (feature 2). All sensor numbers are ASSUMED (typical of
small commercial units), not taken from a specific datasheet.

Lidar: 16 (or 32) channels, +/-15 deg vertical, 360 deg spin, 100 m range, 2.5 cm range
noise; per-surface dropout (dark / wet surfaces return weakly) rising with range.
Mass 0.6 kg and 10 W, charged to the battery model (mission_v3.Flight.power_w).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import common_v3 as C
from common_v3 import np

GROUND = -1
NOHIT = -2


def raycast_many(o: np.ndarray, U_: np.ndarray, boxes: Sequence, max_r: float,
                 ground_box: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Rays from one origin o along unit rows of U_. Returns (range, hit id): id = box index,
    GROUND for the z = 0 plane inside ground_box (x/y), NOHIT if nothing within max_r."""
    n = len(U_)
    best = np.full(n, np.inf)
    hid = np.full(n, NOHIT, int)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / U_
        for b, (lo, hi) in enumerate(boxes):
            t1 = (np.asarray(lo) - o) * inv
            t2 = (np.asarray(hi) - o) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            ok = (tmax >= tmin) & (tmin > 1e-6) & (tmin < best)
            best[ok] = tmin[ok]; hid[ok] = b
        if ground_box is not None:
            tg = -o[2] / U_[:, 2]
            q = o[None, :2] + tg[:, None] * U_[:, :2]
            gl, gh = ground_box
            ok = (U_[:, 2] < -1e-9) & (tg > 0) & (tg < best) & np.all(q >= gl[:2], 1) & np.all(q <= gh[:2], 1)
            best[ok] = tg[ok]; hid[ok] = GROUND
    far = best > max_r
    best[far] = np.inf; hid[far] = NOHIT
    return best, hid


class LidarHead:
    name = "lidar"
    extra_power_w = 10.0          # ASSUMED
    extra_mass_kg = 0.6           # ASSUMED
    range_noise = 0.025
    bearing_noise = math.radians(0.1)
    max_range = 100.0

    def __init__(self, scene_boxes: Sequence, ground_box, dropout: Sequence[float], ground_dropout: float = 0.2,
                 channels: int = 16, az_per_tick: int = 24, max_pts_per_tick: int = 120):
        """dropout[i]: base dropout probability of box i (dark/wet surfaces high)."""
        self.boxes = list(scene_boxes)
        self.ground_box = ground_box
        self.dropout = np.asarray(dropout, float)
        self.ground_dropout = float(ground_dropout)
        self.el = np.radians(np.linspace(-15.0, 15.0, channels))
        self.az_per_tick = az_per_tick
        self.max_pts = max_pts_per_tick
        self.phase = 0.0

    def scan(self, p: np.ndarray, look: np.ndarray, scene, rng) -> List:
        # a 10 Hz spin sampled at az_per_tick azimuths (subsampled to keep the sim fast)
        az = self.phase + np.linspace(0, 2 * math.pi, self.az_per_tick, endpoint=False) + rng.uniform(0, 2 * math.pi / self.az_per_tick)
        self.phase += 0.37
        A, E = np.meshgrid(az, self.el)
        A, E = A.ravel(), E.ravel()
        U_ = np.column_stack([np.cos(A) * np.cos(E), np.sin(A) * np.cos(E), np.sin(E)])
        r, hid = raycast_many(p, U_, self.boxes, self.max_range, self.ground_box)
        hit = hid != NOHIT
        pd = np.where(hid == GROUND, self.ground_dropout, self.dropout[np.clip(hid, 0, None)] if len(self.dropout) else 0.0)
        rf = np.where(np.isfinite(r), r, 0.0)
        pd = 1.0 - (1.0 - pd) * np.exp(-np.maximum(rf - 60.0, 0.0) / 40.0)   # weaker far returns
        keep = hit & (rng.random(len(r)) >= pd)
        idx = np.nonzero(keep)[0]
        if len(idx) > self.max_pts:
            idx = rng.choice(idx, self.max_pts, replace=False)
        out = []
        for i in idx:
            um = U_[i] + rng.normal(0, self.bearing_noise, 3)
            out.append((um / np.linalg.norm(um), float(r[i] + rng.normal(0, self.range_noise)),
                        self.range_noise, self.bearing_noise))
        return out

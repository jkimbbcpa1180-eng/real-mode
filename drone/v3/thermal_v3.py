# SPDX-License-Identifier: CC0-1.0
"""
drone v3 feature 2: low-resolution thermal camera painted onto the 3D map, hot-spot detection.

Simulated camera (ASSUMED, typical of a 32x24 microbolometer / thermopile array):
55 x 35 deg FOV, 2 Hz, NETD 0.2 K, each pixel = mean of 3x3 sub-rays (so a hot spot smaller
than the pixel footprint is diluted - the main range limit). Surfaces: facade ~ambient with a
solar gradient, ground, sky (-20 C), vents (45-80 C, r 0.15-0.3 m), people outdoors
(apparent ~30 C, 0.5 x 0.4 x 1.7 m), glass (emissivity 0.1: shows the reflected sky, or with
probability 0.4 a reflected warm object -> the expected false alarm), people indoors (behind
walls: in the truth list but physically undetectable - LWIR does not pass walls or glass).

Hot pixels (> frame median + DT_K) are back-projected with the drone's *estimated* pose and
the surface depth (the map's depth along the ray; simulated as true depth + 0.1 m noise) into
a 0.5 m grid; cells seen hot in >= MIN_FRAMES frames form detections.

The trajectory is a kinematic orbit (no wind / flight loop) with a GPS-like Gauss-Markov
pose error; mission_v3 can attach the camera to the real flight loop via a hook
(ThermalCam.frame), but the MC below uses the kinematic orbit to stay cheap.

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import common_v3 as C
from common_v3 import np
import sensors_v3 as SE

NX, NY = 32, 24
HFOV, VFOV = math.radians(55.0), math.radians(35.0)
NETD_K = 0.2
SUB = 3
DT_K = 6.0             # hot threshold above the frame median (frozen before tuning run)
MIN_FRAMES = 2
CELL_M = 0.5
MATCH_R_M = 1.0
SKY_C = -20.0
extra_mass_kg = 0.05   # ASSUMED (small radiometric core)
extra_power_w = 1.0


def make_world(sc: Dict, rng) -> Dict:
    lo, hi = (np.asarray(x, float) for x in sc["building"])
    amb = float(rng.uniform(5, 25))
    faces = [(0, -1), (0, +1), (1, -1), (1, +1)]          # (axis, sign)
    def on_face(ax, sg, w, h):
        p = np.empty(3)
        p[ax] = hi[ax] if sg > 0 else lo[ax]
        o = 1 - ax
        p[o] = rng.uniform(lo[o] + w, hi[o] - w)
        p[2] = rng.uniform(1.0, max(hi[2] - 1.0, 1.5))
        return p
    vents, glass = [], []
    for _ in range(int(rng.integers(2, 5))):
        ax, sg = faces[rng.integers(4)]
        vents.append({"ax": ax, "sg": sg, "c": on_face(ax, sg, 0.5, 0.5), "r": float(rng.uniform(0.15, 0.3)),
                      "T": float(rng.uniform(45, 80))})
    for _ in range(3):
        ax, sg = faces[rng.integers(4)]
        glass.append({"ax": ax, "sg": sg, "c": on_face(ax, sg, 1.2, 1.0), "hw": 1.0, "hh": 0.75,
                      "Tr": 50.0 if rng.random() < 0.4 else SKY_C})
    people = []
    for _ in range(int(rng.integers(1, 4))):
        ax, sg = faces[rng.integers(4)]
        c = on_face(ax, sg, 1.0, 0.5); c[ax] += sg * rng.uniform(1.5, 6.0); c[2] = 0.0
        people.append((c + np.array([-0.25, -0.2, 0.0]), c + np.array([0.25, 0.2, 1.7])))
    indoor = [((lo + hi) / 2 * np.array([1, 1, 0]) + np.array([0, 0, 1.0])) for _ in range(int(rng.integers(0, 3)))]
    return {"lo": lo, "hi": hi, "amb": amb, "vents": vents, "glass": glass, "people": people, "indoor": indoor,
            "boxes": [(lo, hi)] + people}


def truth(world) -> List[Dict]:
    t = [{"kind": "vent", "p": v["c"], "visible": True} for v in world["vents"]]
    t += [{"kind": "person", "p": (a + b) / 2, "visible": True} for a, b in world["people"]]
    t += [{"kind": "person_indoor", "p": p, "visible": False} for p in world["indoor"]]
    return t


def _temps(world, P, hid):
    T = np.full(len(hid), SKY_C)
    T[hid == SE.GROUND] = world["amb"] + 3.0
    T[hid >= 1] = 30.0
    b = hid == 0
    if b.any():
        q = P[b]
        lo, hi = world["lo"], world["hi"]
        tb = world["amb"] + 4.0 * (q[:, 2] / max(hi[2], 1.0))           # solar / stack gradient
        for v in world["vents"]:
            on = np.abs(q[:, v["ax"]] - v["c"][v["ax"]]) < 0.05
            d = np.linalg.norm((q - v["c"])[:, [1 - v["ax"], 2]], axis=1)
            tb = np.where(on & (d < v["r"]), v["T"], tb)
        for g in world["glass"]:
            on = np.abs(q[:, g["ax"]] - g["c"][g["ax"]]) < 0.05
            o = 1 - g["ax"]
            ing = on & (np.abs(q[:, o] - g["c"][o]) < g["hw"]) & (np.abs(q[:, 2] - g["c"][2]) < g["hh"])
            tb = np.where(ing, 0.1 * world["amb"] + 0.9 * g["Tr"], tb)
        T[b] = tb
    return T


def frame(world, p, look, rng, ground_box):
    """One thermal frame from true position p looking along look (unit). Returns per-pixel
    apparent temperature, centre-ray unit vectors and centre-ray depth."""
    f = look / np.linalg.norm(look)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    ax = (np.arange(NX * SUB) + 0.5) / (NX * SUB) - 0.5
    ay = (np.arange(NY * SUB) + 0.5) / (NY * SUB) - 0.5
    X, Y = np.meshgrid(np.tan(HFOV / 2) * 2 * ax, np.tan(VFOV / 2) * 2 * ay)
    D = f[None, :] + X.ravel()[:, None] * r[None, :] + Y.ravel()[:, None] * u[None, :]
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    rng_, hid = SE.raycast_many(p, D, world["boxes"], 200.0, ground_box)
    P = p + np.where(np.isfinite(rng_), rng_, 0.0)[:, None] * D
    T = _temps(world, P, hid).reshape(NY, SUB, NX, SUB).mean(axis=(1, 3)) + rng.normal(0, NETD_K, (NY, NX))
    ci = (NY * SUB) * 0 + (np.arange(NY)[:, None] * SUB + SUB // 2) * NX * SUB + (np.arange(NX)[None, :] * SUB + SUB // 2)
    return T, D[ci.ravel()], rng_[ci.ravel()]


def orbit(world, standoff, z, speed=2.0, hz=2.0):
    lo, hi = world["lo"] - standoff, world["hi"] + standoff
    cs = [np.array([lo[0], lo[1]]), np.array([hi[0], lo[1]]), np.array([hi[0], hi[1]]), np.array([lo[0], hi[1]])]
    pts = []
    for i in range(4):
        a, b = cs[i], cs[(i + 1) % 4]
        n = max(int(np.linalg.norm(b - a) / (speed / hz)), 1)
        pts += [a + (b - a) * k / n for k in range(n)]
    return [np.array([q[0], q[1], z]) for q in pts]


def run(sc, world, standoff, gps_sigma, rng):
    lo, hi = world["lo"], world["hi"]
    centre = (lo + hi) / 2
    gb = (np.asarray(sc["area"][0]) - 40, np.asarray(sc["area"][1]) + 40)
    z = float(min(max(hi[2] / 2, 2.0), hi[2] + 2))
    e, phi = np.zeros(3), math.exp(-0.5 / 30.0)
    cells: Dict = {}
    for k, p in enumerate(orbit(world, standoff, z)):
        e = phi * e + math.sqrt(1 - phi * phi) * gps_sigma * rng.normal(size=3)
        look = centre - p; look[2] = 0.0
        T, D, dep = frame(world, p, look, rng, gb)
        surf = np.isfinite(dep)                       # pixels with a mapped surface (sky excluded)
        if not surf.any():
            continue
        hot = (T.ravel() > np.median(T.ravel()[surf]) + DT_K) & surf
        for i in np.nonzero(hot)[0]:
            q = (p + e) + (dep[i] + rng.normal(0, 0.1)) * D[i]
            key = tuple(np.floor(q / CELL_M).astype(int))
            cells.setdefault(key, set()).add(k)
    good = [np.array(k) * CELL_M + CELL_M / 2 for k, fr in cells.items() if len(fr) >= MIN_FRAMES]
    dets = _cluster(good)
    return dets


def _cluster(pts, r=1.5):
    """Single-linkage clusters (link distance r) -> centroids."""
    n = len(pts)
    lab = list(range(n))
    def root(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]; i = lab[i]
        return i
    P = np.array(pts) if n else np.zeros((0, 3))
    for i in range(n):
        for j in np.nonzero(np.linalg.norm(P[i + 1:] - P[i], axis=1) < r)[0] + i + 1:
            lab[root(j)] = root(i)
    groups: Dict = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(P[i])
    return [np.mean(g, axis=0) for g in groups.values()]


def score(dets, tr):
    used, tp = set(), 0
    hits = []
    for t in tr:
        best = None
        for j, d in enumerate(dets):
            dd = np.linalg.norm(d - t["p"]) if t["kind"] != "person" else np.linalg.norm((d - t["p"])[:2])
            lim = MATCH_R_M if t["kind"] != "person" else 1.2
            if j not in used and dd < lim and (best is None or dd < best[1]):
                best = (j, dd)
        if best:
            used.add(best[0]); tp += 1
        hits.append((t["kind"], best is not None))
    return {"tp": tp, "fp": len(dets) - len(used), "n_det": len(dets), "hits": hits}

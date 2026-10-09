# SPDX-License-Identifier: CC0-1.0
"""
drone v3, feature 5: change detection between two missions over the same site.

Mission A and mission B fly the same lawnmower template (each with its own GPS error).
Between them, small objects are added, removed or moved by a known distance.
  1. align: B's map is registered to A's map with the swarm cloud alignment
     (drone_slam_v2.register_translation on all points, translation only; changed
     objects are a small fraction and are down-weighted by the robust ICP);
  2. difference: a filtered point of one map is "unexplained" if the other map has no
     point within CHANGE_TOL_M but HAS points within OBSERVED_R_M (so it looked there);
     only points more than ABOVE_GROUND_M above the map's own ground level are candidates
     (ground coverage holes are not changes);
  3. unexplained points are clustered on a 2-D grid; clusters with >= MIN_CLUSTER_PTS
     points are reported as changes ("added" from B, "removed" from A).
v2 baseline: the same differencing without step 1 (maps in their own GPS frames).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import common_v3 as C
from common_v3 import np, R
import mc_recon_v2 as MR
import mission_v3 as MV
import swarm_v3 as SW

CHANGE_TOL_M = 0.45
OBSERVED_R_M = 2.0
MIN_CLUSTER_PTS = 12
ABOVE_GROUND_M = 0.15
CLUSTER_CELL_M = 0.5
MATCH_R_M = 1.5
SIZES = (0.3, 0.5, 1.0, 2.0)
MOVES_M = (1.0, 2.0, 4.0)


def make_changes(sc: Dict, rng: np.random.Generator, change: bool = True) -> Tuple[List, List, List[Dict]]:
    """Objects in the open area. Returns (boxes for mission A, boxes for mission B, truth events)."""
    a_lo, a_hi = sc["area"]
    busy = [sc["building"], sc["poi"]]

    def free_spot(s, taken):
        for _ in range(200):
            c = rng.uniform(a_lo[:2] + 2 + s, a_hi[:2] - 2 - s)
            lo, hi = np.array([c[0] - s / 2, c[1] - s / 2, 0.0]), np.array([c[0] + s / 2, c[1] + s / 2, s])
            if all(np.any(lo[:2] > b[1][:2] + 2.5) or np.any(hi[:2] < b[0][:2] - 2.5) for b in busy + taken):
                return lo, hi
        return None
    A, B, ev, taken = [], [], [], []
    kinds = ["static", "static"] + (["added", "added", "removed", "removed", "moved", "moved"] if change else [])
    for kind in kinds:
        s = float(rng.choice(SIZES))
        box = free_spot(s, taken)
        if box is None:
            continue
        taken.append(box)
        c = 0.5 * (box[0] + box[1])
        if kind == "static":
            A.append(box); B.append(box)
        elif kind == "added":
            B.append(box); ev.append(dict(kind="added", size=s, pos=c))
        elif kind == "removed":
            A.append(box); ev.append(dict(kind="removed", size=s, pos=c))
        else:
            d = float(rng.choice(MOVES_M))
            for _ in range(50):
                a = rng.uniform(0, 2 * math.pi)
                off = np.array([d * math.cos(a), d * math.sin(a), 0.0])
                nb = (box[0] + off, box[1] + off)
                if np.all(nb[0][:2] > a_lo[:2] + 1) and np.all(nb[1][:2] < a_hi[:2] - 1) and \
                        all(np.any(nb[0][:2] > b[1][:2] + 1.0) or np.any(nb[1][:2] < b[0][:2] - 1.0) for b in busy):
                    break
            A.append(box); B.append(nb); taken.append(nb)
            ev.append(dict(kind="moved", size=s, pos=c, new_pos=0.5 * (nb[0] + nb[1]), dist=d))
    return A, B, ev


def fly_map(sc: Dict, objects: List, gps: str, seed: int) -> Tuple[np.ndarray, MV.Flight]:
    wps, _, _ = MR.plan(sc, "lawnmower")
    scene = R.Scene([sc["building"], sc["poi"]] + list(objects), sc["area"][0], sc["area"][1], True)
    f = MV.Flight(sc, scene, MR.GPS_MODES[gps], seed, wps[0].pos)
    f.fly(wps)
    return f.rmap.filtered_points()[0], f


def _counts_within(Q: np.ndarray, P: np.ndarray, r: float) -> np.ndarray:
    import repass_v3 as RP
    idx = RP.PointIndex(P, r)
    return np.array([idx.count_within(q, r) for q in Q])


def detect(A: np.ndarray, B: np.ndarray) -> List[Dict]:
    out = []
    for kind, X, Y in (("added", B, A), ("removed", A, B)):
        if len(X) == 0 or len(Y) == 0:
            continue
        g = float(np.percentile(X[:, 2], 5))           # this map's ground level (own frame)
        X = X[X[:, 2] > g + ABOVE_GROUND_M]              # bare ground is not a change candidate
        if len(X) == 0:
            continue
        near = _counts_within(X, Y, CHANGE_TOL_M)
        seen = _counts_within(X, Y, OBSERVED_R_M)
        U_ = X[(near == 0) & (seen > 0)]
        if len(U_) == 0:
            continue
        keys = [tuple(k) for k in np.floor(U_[:, :2] / CLUSTER_CELL_M).astype(int)]
        groups: Dict = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        lab = set()
        for k0 in groups:
            if k0 in lab:
                continue
            comp, st = [], [k0]; lab.add(k0)
            while st:
                a = st.pop(); comp += groups[a]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        b = (a[0] + dx, a[1] + dy)
                        if b in groups and b not in lab:
                            lab.add(b); st.append(b)
            if len(comp) >= MIN_CLUSTER_PTS:
                P = U_[comp]
                out.append(dict(kind=kind, pos=P.mean(0), n=len(comp), extent=float(np.ptp(P[:, :2], 0).max())))
    return out


def score(dets: List[Dict], ev: List[Dict], b_offset: np.ndarray) -> Dict:
    """Truth matching in mission A's frame (b_offset: estimate of B-frame minus A-frame).
    A moved object is two truth changes (removed at old, added at new)."""
    truths = []
    for e in ev:
        if e["kind"] in ("added", "removed"):
            truths.append((e["kind"], e["pos"], e["size"]))
        else:
            truths.append(("removed", e["pos"], e["size"])); truths.append(("added", e["new_pos"], e["size"]))
    hit = [False] * len(truths)
    tp = 0
    for d in dets:
        best, bd = None, MATCH_R_M
        for i, (k, p, s) in enumerate(truths):
            dd = float(np.linalg.norm((d["pos"] - p)[:2]))
            if k == d["kind"] and dd < bd + 0.5 * s:
                best, bd = i, dd
        if best is not None:
            tp += 1
            hit[best] = True
    return dict(n_det=len(dets), tp=tp, fp=len(dets) - tp, n_truth=len(truths),
                truth_hits=[(k, s, h) for (k, p, s), h in zip(truths, hit)])


def run_pair(sc: Dict, gps: str, change: bool, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    Aobj, Bobj, ev = make_changes(sc, rng, change)
    A, fa = fly_map(sc, Aobj, gps, seed + 1)
    B, fb = fly_map(sc, Bobj, gps, seed + 2)
    true_rel = fb.gps_err_sum / max(fb.gps_err_n, 1) - fa.gps_err_sum / max(fa.gps_err_n, 1)
    al = SW.align_to(A, B, np.random.default_rng(seed + 3))
    c = al["c"] if al["ok"] else np.zeros(3)
    out = dict(id=sc["id"], gps=gps, change=change, events=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                             for k, v in e.items()} for e in ev],
               true_rel_offset=true_rel.tolist(), align_c=c.tolist(), align_ok=al["ok"], align_nobs=al["n_obs"],
               align_err_before_m=float(np.linalg.norm(true_rel)), align_err_after_m=float(np.linalg.norm(true_rel - c)))
    for mode, Bm in (("v2_noalign", B), ("v3_aligned", B - c)):
        dets = detect(A, Bm)
        out[mode] = score(dets, ev, c)
    return out

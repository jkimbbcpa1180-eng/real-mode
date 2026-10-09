# SPDX-License-Identifier: CC0-1.0
"""
drone v3, feature 4: "go look closer" re-passes.

After the first pass the mission computer looks at its OWN map (no truth):
  1. structures = 2-D connected clusters of mapped points above the ground; each gets a
     robust bounding box (1st..99th percentile) in the estimated frame;
  2. the surfaces the mission was asked for (facades for perimeter/POI passes, roofs +
     ground for a lawnmower) are sampled on a 0.5 m grid on those boxes;
  3. a sample is a GAP if fewer than GAP_MIN_PTS filtered points lie within GAP_TOL_M
     (never seen, or seen too thinly = low detail / weak evidence);
  4. gaps are clustered per face; each cluster gets a short pass at a closer standoff,
     looking at the face;
  5. passes are chosen greedily by gap samples per second of flight, subject to:
     - clearance: every waypoint and transit sample keeps SAFETY_M from mapped points,
       transits that would not are routed over the top at a safe altitude;
     - no-fly boxes and building interiors (interior only with explicit authorization,
       as in drone_recon_v2.interior_room_sweep) are never entered;
     - battery: extra energy + return-home energy stays above the v2 RETURN_HOME
       reserve (25 % of capacity) and the extra time stays under max_extra_s.
Runs on the mission computer between passes; it does not touch the flight loop.

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import common_v3 as C
from common_v3 import np, R

GAP_TOL_M = 0.3
GAP_MIN_PTS = 3                 # fewer points than this within GAP_TOL_M = gap ("low detail")
SAMPLE_M = 0.5
MIN_CLUSTER = 4                 # samples (= 1 m^2)
SAFETY_M = 1.5                  # clearance from mapped points for re-pass waypoints/transits
REPASS_STANDOFF_FRAC = 0.75     # re-pass standoff = 0.75 x template standoff (closer)
MIN_STANDOFF_M = 1.5
RESERVE_FRAC = 0.25             # v2 return_home_soc
NOFLY_MARGIN_M = 2.0            # keep this far outside no-fly boxes (navigation error + path start)
STRUCT_CELL_M = 1.0
STRUCT_MIN_Z = 0.4


@dataclass
class GapCluster:
    normal: np.ndarray           # outward unit normal of the face
    samples: np.ndarray          # (n,3) gap samples
    face: str

    @property
    def center(self) -> np.ndarray:
        return self.samples.mean(0)


@dataclass
class RepassPlan:
    waypoints: List = field(default_factory=list)
    chosen: List[GapCluster] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=lambda: {"nofly": 0, "interior": 0, "clearance": 0, "budget": 0})
    est_time_s: float = 0.0
    est_energy_wh: float = 0.0
    n_gap_samples: int = 0


# --- map analysis -------------------------------------------------------------
class PointIndex:
    """Grid hash for neighbour counts / nearest distance among map points."""

    def __init__(self, pts: np.ndarray, cell: float):
        self.pts, self.cell = np.asarray(pts, float).reshape(-1, 3), float(cell)
        self.b: Dict[Tuple[int, int, int], List[int]] = {}
        for i, k in enumerate(map(tuple, np.floor(self.pts / self.cell).astype(int))):
            self.b.setdefault(k, []).append(i)

    def _near(self, q: np.ndarray, r: float) -> np.ndarray:
        n = int(math.ceil(r / self.cell))
        k = np.floor(q / self.cell).astype(int)
        idx = []
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for dz in range(-n, n + 1):
                    idx += self.b.get((k[0] + dx, k[1] + dy, k[2] + dz), [])
        return np.array(idx, int)

    def count_within(self, q: np.ndarray, r: float) -> int:
        idx = self._near(q, r)
        if len(idx) == 0:
            return 0
        return int((np.linalg.norm(self.pts[idx] - q, axis=1) <= r).sum())

    def clear(self, q: np.ndarray, r: float) -> bool:
        return self.count_within(q, r) == 0


def structures(pts: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    P = pts[pts[:, 2] > STRUCT_MIN_Z]
    if len(P) == 0:
        return []
    keys = np.floor(P[:, :2] / STRUCT_CELL_M).astype(int)
    lab: Dict[Tuple[int, int], int] = {}
    cells = {tuple(k) for k in keys}
    nlab = 0
    for c in cells:
        if c in lab:
            continue
        stack = [c]; lab[c] = nlab
        while stack:
            a = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = (a[0] + dx, a[1] + dy)
                    if b in cells and b not in lab:
                        lab[b] = nlab; stack.append(b)
        nlab += 1
    li = np.array([lab[tuple(k)] for k in keys])
    out = []
    for j in range(nlab):
        Q = P[li == j]
        if len(Q) < 30:
            continue
        lo = np.percentile(Q, 1, axis=0); hi = np.percentile(Q, 99, axis=0)
        lo[2] = 0.0
        out.append((lo, hi))
    return out


def face_samples(lo: np.ndarray, hi: np.ndarray, faces: str, z_min: float = 0.5
                 ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """(face name, outward normal, samples) for the requested faces of a box."""
    out = []
    zs = np.arange(z_min, hi[2] - 0.1 + 1e-9, SAMPLE_M)
    if "sides" in faces and len(zs):
        for name, n, fixed, axis in (("-x", np.array([-1.0, 0, 0]), lo[0], 1), ("+x", np.array([1.0, 0, 0]), hi[0], 1),
                                     ("-y", np.array([0, -1.0, 0]), lo[1], 0), ("+y", np.array([0, 1.0, 0]), hi[1], 0)):
            ts = np.arange(lo[axis] + 0.25, hi[axis] - 0.25 + 1e-9, SAMPLE_M)
            T, Z = np.meshgrid(ts, zs)
            S = np.zeros((T.size, 3)); S[:, axis] = T.ravel(); S[:, 1 - axis] = fixed; S[:, 2] = Z.ravel()
            out.append((name, n, S))
    if "top" in faces:
        xs = np.arange(lo[0] + 0.25, hi[0] - 0.25 + 1e-9, SAMPLE_M)
        ys = np.arange(lo[1] + 0.25, hi[1] - 0.25 + 1e-9, SAMPLE_M)
        X, Y = np.meshgrid(xs, ys)
        out.append(("top", np.array([0, 0, 1.0]), np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, hi[2])])))
    return out


def ground_samples(area_lo: np.ndarray, area_hi: np.ndarray, boxes) -> np.ndarray:
    xs = np.arange(area_lo[0] + 0.25, area_hi[0] - 0.25 + 1e-9, SAMPLE_M)
    ys = np.arange(area_lo[1] + 0.25, area_hi[1] - 0.25 + 1e-9, SAMPLE_M)
    X, Y = np.meshgrid(xs, ys)
    G = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    keep = np.ones(len(G), bool)
    for lo, hi in boxes:
        keep &= ~((G[:, 0] >= lo[0] - 0.3) & (G[:, 0] <= hi[0] + 0.3) & (G[:, 1] >= lo[1] - 0.3) & (G[:, 1] <= hi[1] + 0.3))
    return G[keep]


def cluster_samples(S: np.ndarray, cell: float = 1.5) -> List[np.ndarray]:
    keys = [tuple(k) for k in np.floor(S / cell).astype(int)]
    groups: Dict[Tuple, List[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    lab, out = {}, []
    for k0 in groups:
        if k0 in lab:
            continue
        comp, stack = [], [k0]; lab[k0] = True
        while stack:
            a = stack.pop(); comp += groups[a]
            for d in np.ndindex(3, 3, 3):
                b = (a[0] + d[0] - 1, a[1] + d[1] - 1, a[2] + d[2] - 1)
                if b in groups and b not in lab:
                    lab[b] = True; stack.append(b)
        out.append(S[comp])
    return out


def find_gaps(rmap: R.ReconMap, mission: str, area: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              focus: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[List[GapCluster], Dict]:
    """mission: 'facades' (perimeter / POI passes) or 'roofs+ground' (lawnmower).
    focus: optional (lo, hi) box in the estimated frame; only structures inside it."""
    pts, _ = rmap.filtered_points()
    info = {"n_points": int(len(pts)), "structures": 0, "samples": 0, "gap_samples": 0}
    if len(pts) == 0:
        return [], info
    idx = PointIndex(pts, GAP_TOL_M)
    boxes = structures(pts)
    if focus is not None:
        flo, fhi = focus
        boxes = [(lo, hi) for lo, hi in boxes
                 if np.all(0.5 * (lo + hi)[:2] >= flo[:2]) and np.all(0.5 * (lo + hi)[:2] <= fhi[:2])]
    info["structures"] = len(boxes)
    faces = []
    for lo, hi in boxes:
        faces += face_samples(lo, hi, "sides" if mission == "facades" else "top")
    if mission != "facades" and area is not None:
        faces.append(("ground", np.array([0, 0, 1.0]), ground_samples(area[0], area[1], boxes)))
    out = []
    for name, n, S in faces:
        info["samples"] += len(S)
        gap = np.array([idx.count_within(q, GAP_TOL_M) < GAP_MIN_PTS for q in S], bool)
        info["gap_samples"] += int(gap.sum())
        for G in cluster_samples(S[gap]) if gap.any() else []:
            if len(G) >= MIN_CLUSTER:
                out.append(GapCluster(n, G, name))
    return out, info


# --- planning -------------------------------------------------------------------
def _in_box(q: np.ndarray, box, margin: float = 0.0) -> bool:
    lo, hi = box
    return bool(np.all(q >= np.asarray(lo) - margin) and np.all(q <= np.asarray(hi) + margin))


def pass_waypoints(g: GapCluster, standoff: float, beam_half: float, overlap: float = 0.3,
                   step: float = 0.5) -> List:
    """Short boustrophedon in front of the gap cluster, looking at the face."""
    n = g.normal / np.linalg.norm(g.normal)
    sw = max(R.swath_width(standoff, beam_half) * (1.0 - overlap), 0.3)
    S = g.samples
    if abs(n[2]) > 0.9:                     # roof / ground: horizontal lanes, look down
        a, b = S[:, :2].min(0) - 0.5, S[:, :2].max(0) + 0.5
        z = float(S[:, 2].max()) + standoff
        ys = np.arange(a[1], b[1] + sw * 0.5, sw) if b[1] - a[1] > sw else [0.5 * (a[1] + b[1])]
        pts = []
        for i, y in enumerate(ys):
            xa, xb = (a[0], b[0]) if i % 2 == 0 else (b[0], a[0])
            pts += [np.array([xa, y, z]), np.array([xb, y, z])]
        look = np.array([0, 0, -1.0])
    else:
        t = np.array([-n[1], n[0], 0.0])
        s_t = S @ t
        z0, z1 = max(float(S[:, 2].min()), 0.5), float(S[:, 2].max())
        face_off = float((S @ n).mean())
        base = n * (face_off + standoff)
        zs = np.arange(z0, z1 + sw * 0.5, sw) if z1 - z0 > sw else [0.5 * (z0 + z1)]
        pts = []
        for i, z in enumerate(zs):
            ta, tb = (s_t.min() - 0.5, s_t.max() + 0.5) if i % 2 == 0 else (s_t.max() + 0.5, s_t.min() - 0.5)
            pts += [base + t * ta + np.array([0, 0, z]) - n * 0.0, base + t * tb + np.array([0, 0, z])]
        pts = [np.array([p[0], p[1], p[2]]) for p in pts]
        look = -n
    return [R.Waypoint(p, look.copy()) for p in R._densify(pts, step)]


def transit(a: np.ndarray, b: np.ndarray, idx: PointIndex, safe_z: float, look_hint: np.ndarray,
            step: float = 0.5, boxes: Sequence = ()) -> List:
    """Straight if every sample keeps SAFETY_M from mapped points and from the mapped
    structure boxes, else up (at the start point) - over at safe_z - down."""
    def clear_seg(p, q):
        L = float(np.linalg.norm(q - p))
        for s in np.linspace(0, 1, max(int(L / 0.5), 1) + 1):
            x = p + (q - p) * s
            if not idx.clear(x, SAFETY_M) or any(_in_box(x, bx, SAFETY_M) for bx in boxes):
                return False
        return True
    if clear_seg(a, b):
        pts = [a, b]
    else:
        up_a, up_b = np.array([a[0], a[1], safe_z]), np.array([b[0], b[1], safe_z])
        pts = [a, up_a, up_b, b]
    d = (b - a); d[2] = 0
    look = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else look_hint
    return [R.Waypoint(p, look.copy()) for p in R._densify(pts, step)]


def path_len(wps) -> float:
    return float(sum(np.linalg.norm(b.pos - a.pos) for a, b in zip(wps[:-1], wps[1:])))


def plan_repasses(gaps: List[GapCluster], rmap: R.ReconMap, start: np.ndarray, home: np.ndarray,
                  standoff: float, beam_half: float, energy_wh: float, capacity_wh: float,
                  power_w: float, speed: float, max_extra_s: float = 300.0,
                  nofly: Sequence = (), interior_authorized: bool = False) -> RepassPlan:
    plan = RepassPlan()
    pts, _ = rmap.filtered_points()
    idx = PointIndex(pts, SAFETY_M)
    boxes = structures(pts)
    safe_z = max((float(pts[:, 2].max()) if len(pts) else 5.0) + 3.0, float(np.asarray(start)[2]))
    so = max(MIN_STANDOFF_M, REPASS_STANDOFF_FRAC * standoff)
    cands = []
    for g in gaps:
        w = pass_waypoints(g, so, beam_half)
        why = None
        for wp in w:
            if any(_in_box(wp.pos, nb, NOFLY_MARGIN_M) for nb in nofly):
                why = "nofly"; break
            if not interior_authorized and any(_in_box(wp.pos, b, min(SAFETY_M, so * 0.8)) for b in boxes):
                why = "interior"; break
            if wp.pos[2] < 0.5 or not idx.clear(wp.pos, min(SAFETY_M, so * 0.8)):
                why = "clearance"; break
        if why:
            plan.skipped[why] += 1
            continue
        cands.append((g, w))
    cur = np.asarray(start, float)
    t_used = 0.0
    budget_wh = energy_wh - RESERVE_FRAC * capacity_wh
    while cands:
        best, best_v, best_tr, best_dt = None, -1.0, None, 0.0
        for i, (g, w) in enumerate(cands):
            tr = transit(cur, w[0].pos, idx, safe_z, w[0].look_dir, boxes=boxes)
            if any(_in_box(q.pos, nb, NOFLY_MARGIN_M) for q in tr for nb in nofly):
                continue           # not reachable from here without entering a no-fly zone
            dt = (path_len(tr) + path_len(w)) / speed
            v = len(g.samples) / max(dt, 1.0)
            if v > best_v:
                best, best_v, best_tr, best_dt = i, v, tr, dt
        if best is None:
            plan.skipped["nofly"] += len(cands)
            break
        g, w = cands.pop(best)
        home_s = float(np.linalg.norm(w[-1].pos - home)) / speed
        e_need = (t_used + best_dt + home_s) * power_w / 3600.0
        if t_used + best_dt > max_extra_s or e_need > budget_wh:
            plan.skipped["budget"] += 1
            continue
        plan.waypoints += best_tr + w
        plan.chosen.append(g)
        t_used += best_dt
        cur = w[-1].pos
    plan.est_time_s = t_used
    plan.est_energy_wh = t_used * power_w / 3600.0
    plan.n_gap_samples = int(sum(len(g.samples) for g in plan.chosen))
    # safety assertion on the final list (also checked in tests)
    for wp in plan.waypoints:
        assert not any(_in_box(wp.pos, nb, NOFLY_MARGIN_M) for nb in nofly)
    return plan

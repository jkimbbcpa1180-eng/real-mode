# SPDX-License-Identifier: CC0-1.0
"""
drone v3, feature 1: swarm of N drones (1..4) on one recon area.

* Split: a lawnmower area is cut into N strips along x (2 m overlap); a perimeter orbit
  is cut into N angular sectors (10 deg overlap), every drone flying all levels of its
  sector. Overlaps are what lets the cloud align the drones' maps.
* Deconfliction: altitude layers (drone k flies its transits and lawnmower lanes
  LAYER_M * k higher) + a separation monitor on the positions the drones broadcast
  (their own estimates, not truth). v3.1: reciprocal avoidance (reciprocal_avoidance):
  for every pair predicted to come closer than SEP_H_M (3-D, on estimates; the margin over
  the 2 m near-miss distance is the buffer for GPS error) within AVOID_HORIZON_S, the required change of relative velocity is split half/half
  between BOTH drones (ORCA-like), with a "both keep right" sideways term for head-on
  geometry, and a vertical-layer fallback (upper drone climbs, lower descends) once they
  are inside 0.75 * SEP_H_M. Pairs that are not closing and not inside 0.75 * SEP_H_M are
  left alone (parallel lawnmower neighbours would otherwise crawl). (v3.0 had one-sided priority hold + climb, which caused
  contacts on the held-out seed.)
  Truth separations are logged: near miss < NEAR_MISS_M, collision < COLLISION_M.
* Uplink: each drone streams map deltas with drone_uplink_v2.MapSync into its own
  CloudMapAssembler (one per drone in the cloud; the existing assembler is reused
  unchanged). A drone without a direct link forwards through other drones (multi-hop,
  at most MAX_HOPS, mesh range MESH_RANGE_M) using bounded relay queues
  (RELAY_QUEUE_MSGS per relay, back-pressure when full: nothing is silently dropped).
* Merge + alignment: the cloud aligns drone k's map to the union of the already aligned
  maps with drone_slam_v2.register_translation on the overlap (translation only,
  observable directions only) and merges them.
Each drone has its own GPS error (independent Gauss-Markov), so without alignment the
maps disagree by the difference of their GPS errors.

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import common_v3 as C
from common_v3 import np, R, S, U
import mc_recon_v2 as MR
import mission_v3 as MV

LAYER_M = 1.0                   # ASSUMED altitude layer spacing
SEP_H_M = 4.0
SEP_V_M = 1.5
LOOKAHEAD_S = 1.5
AVOID_HORIZON_S = 3.0
AVOID_VMAX_EACH = 2.0           # cap on each drone's avoidance velocity
LAYER_FALLBACK_MPS = 0.6        # vertical split speed per drone in the layer fallback


def reciprocal_avoidance(P: Sequence[np.ndarray], V: Sequence[np.ndarray], active: Sequence[bool],
                         r_safe: float = SEP_H_M, horizon: float = AVOID_HORIZON_S) -> List[np.ndarray]:
    """Avoidance velocities for N drones from broadcast positions P and velocities V.
    Each conflicting pair shares the needed relative-velocity change equally (both yield)."""
    n = len(P)
    A = [np.zeros(3) for _ in range(n)]
    w = np.ones(3)                                        # isotropic: the safety distance is 3-D
    for a in range(n):
        for b in range(a + 1, n):
            if not (active[a] and active[b]):
                continue
            d, v = P[a] - P[b], V[a] - V[b]
            dw, vw = d * w, v * w
            vv = float(vw @ vw)
            ts = float(np.clip(-(dw @ vw) / vv, 0.0, horizon)) if vv > 1e-9 else 0.0
            c = dw + vw * ts
            dmin = float(np.linalg.norm(c))
            nd = float(np.linalg.norm(dw))
            inner = nd < 0.75 * r_safe
            if dmin >= r_safe or (ts <= 0.0 and not inner):   # not closing and not inside: no action
                continue                                       # (parallel neighbours keep flying)
            nrm = c / dmin if dmin > 1e-6 else (dw / nd if nd > 1e-6 else np.array([1.0, 0.0, 0.0]))
            vh = v[:2]
            if np.linalg.norm(vh) > 0.1:
                right = np.array([vh[1], -vh[0], 0.0]) / np.linalg.norm(vh)
                cr = float(c @ right)
                if abs(cr) < 0.25 * r_safe:                   # near head-on: add a sideways term on the
                    nrm = nrm + (0.5 if cr >= 0 else -0.5) * right   # escape side (keep right if exactly head-on)
            nrm = nrm / w
            nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)
            mag = min((r_safe - dmin) / max(ts, 0.5), 2.0 * AVOID_VMAX_EACH)
            A[a] += 0.5 * mag * nrm
            A[b] -= 0.5 * mag * nrm
            if inner:                                        # layer fallback: split vertically
                up = 1.0 if (d[2] > 0 or (d[2] == 0 and a < b)) else -1.0
                A[a][2] += up * LAYER_FALLBACK_MPS
                A[b][2] -= up * LAYER_FALLBACK_MPS
    for k in range(n):
        s = float(np.linalg.norm(A[k]))
        if s > AVOID_VMAX_EACH:
            A[k] *= AVOID_VMAX_EACH / s
    return A
NEAR_MISS_M = 2.0
COLLISION_M = 0.5
PAD_SPACING_M = 4.0
MESH_RANGE_M = 50.0             # ASSUMED drone-to-drone radio range
MAX_HOPS = 3
RELAY_QUEUE_MSGS = 40
MESH_MSGS_PER_TICK = 20         # ASSUMED drone-to-drone throughput per 0.1 s tick
STRIP_OVERLAP_M = 2.0
SECTOR_OVERLAP_DEG = 10.0
ALIGN_MAX_PTS = 2500
ALIGN_NEAR_M = 1.0
COLLECT_EVERY_S = 1.0
STATION_DIST_M = (80.0, 115.0)  # ASSUMED: station near the edge of the 100 m WiFi range
COVERAGE_EVERY_S = 30.0


# --- task split -------------------------------------------------------------------
def split_lawnmower(sc: Dict, n: int) -> List[List]:
    a_lo, a_hi = sc["area"]
    b_hi, p_hi = sc["building"][1], sc["poi"][1]
    top = max(b_hi[2], p_hi[2])
    xs = np.linspace(a_lo[0], a_hi[0], n + 1)
    out = []
    for k in range(n):
        lo = np.array([xs[k] - (STRIP_OVERLAP_M if k > 0 else 0.0), a_lo[1], a_lo[2]])
        hi = np.array([xs[k + 1] + (STRIP_OVERLAP_M if k < n - 1 else 0.0), a_hi[1], a_hi[2]])
        # strips are cut along x, lanes run along y so each strip keeps the full lane spacing
        w = R.lawnmower(np.array([lo[1], lo[0], 0]), np.array([hi[1], hi[0], 0]), top, MR.STANDOFF["lawnmower"],
                        MR.BEAM_HALF, MR.OVERLAP, max_range_m=MR.SONAR_MAX)
        wps = [R.Waypoint(np.array([q.pos[1], q.pos[0], q.pos[2] + LAYER_M * k]), q.look_dir) for q in w]
        out.append(wps)
    return out


def split_perimeter(sc: Dict, n: int) -> List[List]:
    b_lo, b_hi = sc["building"]
    full = R.perimeter_orbit(b_lo, b_hi, MR.STANDOFF["perimeter_orbit"], MR.BEAM_HALF, MR.OVERLAP)
    c = 0.5 * (b_lo + b_hi)
    ang = np.array([math.atan2(w.pos[1] - c[1], w.pos[0] - c[0]) % (2 * math.pi) for w in full])
    zs = sorted({round(float(w.pos[2]), 3) for w in full})
    ov = math.radians(SECTOR_OVERLAP_DEG)
    out = []
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n - (ov if n > 1 else 0), 2 * math.pi * (k + 1) / n + (ov if n > 1 else 0)
        wps = []
        for li, z in enumerate(zs):
            sel = []
            for i, w in enumerate(full):
                if abs(w.pos[2] - z) > 1e-3:
                    continue
                a = ang[i]
                inside = (a0 <= a <= a1) or (a0 <= a + 2 * math.pi <= a1) or (a0 <= a - 2 * math.pi <= a1)
                if n == 1 or inside:
                    sel.append((((a - a0) % (2 * math.pi)), w))
            sel.sort(key=lambda x: x[0])
            lv = [w for _, w in sel]
            if li % 2 == 1:
                lv = lv[::-1]
            wps += lv
        out.append(wps)
    return out


def pads(sc: Dict, n: int) -> List[np.ndarray]:
    """Take-off pads in a row on the area edge facing the ground station."""
    a_lo, a_hi = sc["area"]
    st = sc["station"]
    c = 0.5 * (a_lo + a_hi)
    d = st[:2] - c[:2]
    d /= max(np.linalg.norm(d), 1e-9)
    base = c[:2] + d * (0.5 * float(np.linalg.norm(a_hi[:2] - a_lo[:2])) + 3.0)
    t = np.array([-d[1], d[0]])
    return [np.array([*(base + t * PAD_SPACING_M * (k - (n - 1) / 2)), 0.5]) for k in range(n)]


def with_launch(pad: np.ndarray, wps: List, layer_z: float) -> List:
    """Climb at the pad to the drone's transit layer (above every structure of the site
    plan), transit at that layer, descend vertically onto the first task waypoint; after
    the task climb back to the layer, fly back over the pad and land."""
    first = wps[0].pos
    z = max(layer_z, first[2])
    up = np.array([pad[0], pad[1], z])
    over = np.array([first[0], first[1], z])
    look = wps[0].look_dir
    pre = [R.Waypoint(q, look) for q in R._densify([pad, up, over, first], 0.5)]
    last = wps[-1].pos
    post = [R.Waypoint(q, wps[-1].look_dir) for q in
            R._densify([last, np.array([last[0], last[1], z]), up, pad], 0.5)]
    return pre + list(wps) + post


def station_for(sc: Dict) -> np.ndarray:
    """Ground station on the line of sc["station"], placed at the edge of WiFi range
    (ASSUMED site layout: STATION_DIST_M from the area centre) so that parts of the area
    are in direct range and parts are not."""
    rng = np.random.default_rng(sc["seed"] + 77)
    c = 0.5 * (sc["area"][0] + sc["area"][1])
    d = sc["station"][:2] - c[:2]
    d = d / max(np.linalg.norm(d), 1e-9)
    r = rng.uniform(*STATION_DIST_M)
    return np.array([c[0] + d[0] * r, c[1] + d[1] * r, 2.0])


# --- uplink with relay ----------------------------------------------------------------
class SwarmComms:
    """Direct WiFi per drone (+ optional LTE), relay through other drones when enabled."""

    def __init__(self, n: int, station: np.ndarray, seed: int, relay: bool, lte: bool = False):
        self.n, self.relay = n, relay
        self.clouds = [U.CloudMapAssembler() for _ in range(n)]
        self.links: List[List[U.SimLink]] = []
        for i in range(n):
            rng = np.random.default_rng(seed + 31 * i)
            ls = [U.SimLink(U.LINK_PROFILES["wifi"], np.random.default_rng(rng.integers(0, 2**31)),
                            self._deliver_fn(i), station_pos=station)]
            if lte:
                ls.append(U.SimLink(U.LINK_PROFILES["lte"], np.random.default_rng(rng.integers(0, 2**31)), self._deliver_fn(i)))
            self.links.append(ls)
        self.relay_q: List[deque] = [deque() for _ in range(n)]
        self.pos = [np.zeros(3) for _ in range(n)]
        self.stats = {"relayed_msgs": 0, "relay_queue_max": 0, "relay_full_refusals": 0, "hops_hist": {}}
        self.mesh_budget = [0] * n
        self._routes: Dict[int, Optional[int]] = {}

    def _deliver_fn(self, origin_link_owner: int):
        def f(t, data):
            o, payload = data.split(b"|", 1)
            self.clouds[int(o)].deliver(t, payload)
        return f

    def step(self, t: float, positions: Sequence[np.ndarray]) -> None:
        self.pos = [np.asarray(p, float) for p in positions]
        for i, ls in enumerate(self.links):
            for l in ls:
                l.step(t, MV.DT, self.pos[i])
        self.mesh_budget = [MESH_MSGS_PER_TICK] * self.n
        self._routes = {i: self._route(i) for i in range(self.n)}

    def direct_up(self, i: int) -> bool:
        return any(l.is_up() for l in self.links[i])

    def _route(self, i: int) -> Optional[int]:
        """Next hop from i towards any drone with a direct link (BFS, <= MAX_HOPS)."""
        if self.direct_up(i) or not self.relay:
            return None
        prev = {i: None}
        frontier = [i]
        for _ in range(MAX_HOPS):
            nxt = []
            for a in frontier:
                for b in range(self.n):
                    if b in prev or np.linalg.norm(self.pos[a] - self.pos[b]) > MESH_RANGE_M:
                        continue
                    prev[b] = a
                    if self.direct_up(b):
                        hop = b
                        while prev[hop] != i:
                            hop = prev[hop]
                        return hop
                    nxt.append(b)
            frontier = nxt
        return None

    def _send_direct(self, i: int, data: bytes) -> bool:
        for l in sorted(self.links[i], key=lambda l: l.p.cost_rank):
            if l.can_send(len(data)):
                return l.send(data)
        return False

    def send_from(self, i: int, tagged: bytes) -> bool:
        if self.direct_up(i):
            return self._send_direct(i, tagged)
        j = self._routes.get(i)
        if j is None or self.mesh_budget[i] <= 0:
            return False
        if len(self.relay_q[j]) >= RELAY_QUEUE_MSGS:
            self.stats["relay_full_refusals"] += 1
            return False
        self.mesh_budget[i] -= 1
        self.relay_q[j].append(tagged)
        self.stats["relayed_msgs"] += 1
        self.stats["relay_queue_max"] = max(self.stats["relay_queue_max"], len(self.relay_q[j]))
        return True

    def pump_relays(self) -> None:
        for j in range(self.n):
            q = self.relay_q[j]
            while q:
                if not self.send_from(j, q[0]):
                    break
                q.popleft()

    def link_for(self, i: int):
        comms = self

        class _L:
            def any_up(self_inner):
                return comms.direct_up(i) or comms._routes.get(i) is not None

            def send(self_inner, data: bytes) -> bool:
                return comms.send_from(i, str(i).encode() + b"|" + data)
        return _L()

    def poll(self, t: float) -> None:
        for c in self.clouds:
            c.poll(t)


def cloud_points(cloud: U.CloudMapAssembler, filtered: bool = True) -> np.ndarray:
    """Points held by the cloud; filtered = only points whose evidence-grid cell the cloud
    holds as occupied (the same rule as ReconMap.filtered_points on the drone)."""
    if not cloud.points:
        return np.zeros((0, 3))
    P = np.array([v[:3] for v in cloud.points.values()], float) / 1000.0
    if not filtered:
        return P
    thr = R.L_OCC * R.QUANT_LO
    occ = {k for k, v in cloud.cells.items() if v[0] > thr}
    K = np.floor(P / R.GRID_RES_M).astype(int)
    keep = np.array([tuple(k) in occ for k in K], bool)
    return P[keep]


# --- alignment -------------------------------------------------------------------
def _near_mask(A: np.ndarray, B: np.ndarray, r: float) -> np.ndarray:
    """Points of A with a point of B within r (grid hash)."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(len(A), bool)
    cell = r
    keys = {}
    for k in map(tuple, np.floor(B / cell).astype(int)):
        keys[k] = True
    KA = np.floor(A / cell).astype(int)
    out = np.zeros(len(A), bool)
    for i, k in enumerate(KA):
        for d in np.ndindex(3, 3, 3):
            if (k[0] + d[0] - 1, k[1] + d[1] - 1, k[2] + d[2] - 1) in keys:
                out[i] = True
                break
    return out


def align_to(ref: np.ndarray, pts: np.ndarray, rng: np.random.Generator, sigma: float = 0.1) -> Dict:
    """Translation c (pts - c ~ ref) on the overlap, observable directions only."""
    out = {"c": np.zeros(3), "n_obs": 0, "n_match": 0, "ok": False, "overlap_pts": 0}
    m = _near_mask(pts, ref, ALIGN_NEAR_M + 1.0)
    Sp = pts[m]
    out["overlap_pts"] = int(len(Sp))
    if len(Sp) < 50:
        return out
    if len(Sp) > ALIGN_MAX_PTS:
        Sp = Sp[rng.choice(len(Sp), ALIGN_MAX_PTS, replace=False)]
    mr = _near_mask(ref, Sp, ALIGN_NEAR_M + 1.0)
    M = ref[mr]
    if len(M) > 2 * ALIGN_MAX_PTS:
        M = M[rng.choice(len(M), 2 * ALIGN_MAX_PTS, replace=False)]
    Mn, Mpl = S.local_normals(M, M)
    M, Mn = M[Mpl], Mn[Mpl]
    Sn, Spl = S.local_normals(Sp, Sp)
    Sn = np.where(Spl[:, None], Sn, 0.0)
    cov = np.repeat((np.eye(3) * sigma ** 2)[None], len(Sp), 0)
    mcov = np.repeat((np.eye(3) * sigma ** 2)[None], len(M), 0)
    r = S.register_translation(Sp, cov, M, Mn, mcov, np.zeros(3), S_n=Sn)
    if r["ok"]:
        out.update(c=r["c"], n_obs=int(r["n_obs"]), n_match=int(r["n_match"]), ok=True, V_obs=r["V_obs"])
    return out


def consistency(ref: np.ndarray, pts: np.ndarray) -> Optional[float]:
    """Median distance from pts (on the overlap) to the nearest ref point."""
    m = _near_mask(pts, ref, ALIGN_NEAR_M + 1.0)
    P = pts[m]
    if len(P) < 20:
        return None
    P = P[:2000]
    d = []
    for i in range(0, len(P), 250):
        D2 = ((P[i:i + 250, None, :] - ref[None, ::max(1, len(ref) // 6000), :]) ** 2).sum(-1)
        d.append(np.sqrt(D2.min(1)))
    return float(np.median(np.concatenate(d)))


# --- mission -----------------------------------------------------------------------
def run_swarm(sc: Dict, template: str, n: int, gps_mode: str = "gps_standard", deconflict: bool = True,
              relay: bool = True, align: bool = True, lte: bool = False, coverage_curve: bool = True,
              max_s: float = 900.0, record: bool = False) -> Dict:
    _, scene, targets = MR.plan(sc, template)
    tasks = split_lawnmower(sc, n) if template == "lawnmower" else split_perimeter(sc, n)
    pp = pads(sc, n)
    flights, plans = [], []
    for k in range(n):
        top = max(sc["building"][1][2], sc["poi"][1][2])
        transit_z = max(top + 3.0, tasks[k][0].pos[2]) + LAYER_M * k
        wps = with_launch(pp[k], tasks[k], transit_z)
        f = MV.Flight(sc, scene, MR.GPS_MODES[gps_mode], sc["seed"] + 1000 * (k + 1) + 17, wps[0].pos)
        flights.append(f)
        plans.append(wps)
    comms = SwarmComms(n, station_for(sc), sc["seed"] + 5, relay, lte)
    syncs = []
    for k, f in enumerate(flights):
        sy = U.MapSync(f.rmap)
        ch = sy.add_channel(comms.link_for(k))
        syncs.append((sy, ch))
    iters = [f.fly_iter(w, max_s=max_s) for f, w in zip(flights, plans)]
    done = [False] * n
    t = 0.0
    min_sep, near, coll_ticks = float("inf"), 0, 0
    min_sep_at: Dict = {}
    near_pairs_prev = set()
    curve = []
    out_of_range_ticks = [0] * n
    holds = [0] * n
    while not all(done) and t < max_s:
        t += MV.DT
        # reciprocal avoidance on broadcast ESTIMATES (both drones of a pair yield)
        for f in flights:
            f.avoid_v = None
        if deconflict:
            Pb = [f.p_hat for f in flights]
            Vb = [(f.path[-1] - f.path[-2]) / MV.DT if len(f.path) > 1 else np.zeros(3) for f in flights]
            Av = reciprocal_avoidance(Pb, Vb, [not x for x in done])
            for k, f in enumerate(flights):
                if not done[k] and float(np.linalg.norm(Av[k])) > 1e-9:
                    f.avoid_v = Av[k]
        for k in range(n):
            if done[k]:
                continue
            try:
                r = next(iters[k])
                if flights[k].avoid_v is not None:
                    holds[k] += 1
                if r:
                    done[k] = True
            except StopIteration:
                done[k] = True
        # truth separation
        pairs = set()
        for a in range(n):
            for b in range(a + 1, n):
                d = float(np.linalg.norm(flights[a].p - flights[b].p))
                if d < min_sep:
                    min_sep = d
                    min_sep_at = dict(t=round(t, 1), pair=(a, b), z=(round(float(flights[a].p[2]), 2), round(float(flights[b].p[2]), 2)),
                                      est_d=round(float(np.linalg.norm(flights[a].p_hat - flights[b].p_hat)), 2))
                if d < NEAR_MISS_M:
                    pairs.add((a, b))
                if d < COLLISION_M:
                    coll_ticks += 1
        near += len(pairs - near_pairs_prev)            # count episodes, not ticks
        near_pairs_prev = pairs
        # uplink worker
        comms.step(t, [f.p for f in flights])
        for k in range(n):
            if not comms.direct_up(k):
                out_of_range_ticks[k] += 1
        if abs(t / COLLECT_EVERY_S - round(t / COLLECT_EVERY_S)) < 1e-6:
            for sy, _ in syncs:
                sy.collect(t)
        for sy, ch in syncs:
            ch.pump(sy, t)
        comms.pump_relays()
        comms.poll(t)
        if coverage_curve and abs(t / COVERAGE_EVERY_S - round(t / COVERAGE_EVERY_S)) < 1e-6:
            P = np.concatenate([cloud_points(c) for c in comms.clouds])
            curve.append((round(t, 1), R.coverage(targets, P, R.COVERAGE_TOL_M) if len(P) else 0.0))
    t_end = t
    for sy, _ in syncs:
        sy.collect(t_end)
    for _ in range(50):                    # a few seconds more of in-flight delivery (hover at the end)
        t += MV.DT
        comms.step(t, [f.p for f in flights])
        for sy, ch in syncs:
            ch.pump(sy, t)
        comms.pump_relays()
        comms.poll(t)
    deliv = []
    for k, (sy, ch) in enumerate(syncs):
        cells, pts_q = sy.quantized_state()
        fr = comms.clouds[k].fraction_present(cells, pts_q, flights[k].rmap.grid.occupied())
        deliv.append(fr["points"])
    # --- merge + alignment (cloud side, on the delivered maps) ---
    rng = np.random.default_rng(sc["seed"] + 99)
    per = [cloud_points(c) for c in comms.clouds]
    true_off = [f.gps_err_sum / max(f.gps_err_n, 1) for f in flights]
    align_rows = []
    merged = per[0].copy()
    merged_raw = per[0].copy()
    for k in range(1, n):
        P = per[k]
        before = consistency(merged, P)
        rel_true = true_off[k] - true_off[0]
        row = {"drone": k, "consistency_before_m": before, "true_rel_offset_m": float(np.linalg.norm(rel_true))}
        if align and len(P):
            a = align_to(merged, P, rng)
            c = a["c"] if a["ok"] else np.zeros(3)
            row.update(ok=a["ok"], n_obs=a["n_obs"], overlap_pts=a["overlap_pts"])
            Pa = P - c
            row["consistency_after_m"] = consistency(merged, Pa)
            row["offset_err_after_m"] = float(np.linalg.norm(rel_true - c))
            if a["ok"] and a["n_obs"]:
                V = a["V_obs"]
                row["offset_err_obs_before_m"] = float(np.linalg.norm(V.T @ rel_true))
                row["offset_err_obs_after_m"] = float(np.linalg.norm(V.T @ (rel_true - c)))
        else:
            Pa = P
        align_rows.append(row)
        merged = np.concatenate([merged, Pa]) if len(Pa) else merged
        merged_raw = np.concatenate([merged_raw, P]) if len(P) else merged_raw
    cov_merged = R.coverage(targets, merged, R.COVERAGE_TOL_M) if len(merged) else 0.0
    cov_raw = R.coverage(targets, merged_raw, R.COVERAGE_TOL_M) if len(merged_raw) else 0.0
    # onboard union (no uplink losses) for reference
    onboard = np.concatenate([f.rmap.filtered_points()[0] for f in flights])
    rec = dict(_merged=merged, _merged_raw=merged_raw, _per=per, _paths=[np.array(f.path) for f in flights],
               _scene=scene) if record else {}
    return dict(**rec, id=sc["id"], template=template, n=n, gps=gps_mode, deconflict=deconflict, relay=relay, align=align,
                mission_s=round(t_end, 1), completed=all(done), min_sep_m=min_sep, near_miss_episodes=near,
                collision_ticks=coll_ticks, min_sep_at=min_sep_at, hold_ticks=holds, avoid_ticks=holds, out_of_range_frac=[o / max(t_end / MV.DT, 1) for o in out_of_range_ticks],
                delivered_points_frac=deliv, relay_stats=dict(comms.stats), coverage_curve=curve,
                coverage_merged_aligned=cov_merged, coverage_merged_raw=cov_raw,
                coverage_onboard_union=R.coverage(targets, onboard, R.COVERAGE_TOL_M) if len(onboard) else 0.0,
                alignment=align_rows, energy_wh=[f.energy_used_wh for f in flights],
                min_clear_m=min(f.min_clear for f in flights))

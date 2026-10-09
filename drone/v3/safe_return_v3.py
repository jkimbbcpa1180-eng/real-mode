# SPDX-License-Identifier: CC0-1.0
"""
drone v3, feature 6: offline safe-return brain (no GPS, no link).

When GPS and the link are lost (e.g. jamming) the drone must get back to its take-off
pad with only the core filter (drone_core_v2.DragConsistentKF), the sonar map it built
before the loss, the compass, an air-relative velocity estimate and the pad marker.

Modes compared by the Monte Carlo (mc_safe_return_v3.py):
  "v2"  baseline: the v2 pieces only. Straight line to the recorded home at 8 m/s, the core
        KF dead-reckons on the air-relative velocity (v2 has no wind state, so wind is
        unobserved after the loss), v2 battery rule (LAND in place at SoC <= 15 %), range
        updates to the pad beacon via kf.update_landmark_range when within range.
  "v3_noslam"  ablation: v3 without the map-based position fixes.
  "v3"  this module: wind estimated before the loss and carried as a velocity bias;
        sonar scans registered against the pre-loss map (drone_slam_v2.register_translation)
        give position fixes in observable directions only; energy-weighted visibility path
        around the mapped structures with per-leg airspeed for least energy per ground
        metre; land-in-place fallback when energy x margin + reserve exceeds what is left;
        pad found with a downward marker detector, expanding-square search if not seen.
Both modes share the same simulated physics and the same sonar "slide" safety layer.

Everything here is a simulation with ASSUMED sensor and power numbers (labelled).
Runs on the mission computer, not in the flight-control loop (see test file).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import common_v3 as C
from common_v3 import np, v2, R, S

# --- ASSUMED parameters (frozen before the held-out run) ---------------------
DT = 0.1
ALT_M = 6.0
MISSION_SPEED = 4.0             # m/s ground, with GPS
V2_RETURN_SPEED = 8.0           # m/s ground command (ASSUMED typical return-to-home speed)
V_AIR_MIN, V_AIR_MAX = 3.0, 14.0
VEL_TAU_S = 1.0                 # truth: ground velocity follows command with this lag
GPS_SIGMA, GPS_TAU, GPS_WHITE = 0.5, 30.0, 0.10
GPS_VEL_SIGMA = 0.10
AIR_VEL_NOISE = 0.20            # m/s, air-relative velocity estimate (tilt/drag), ASSUMED
AIR_VEL_SCALE_SIGMA = 0.05      # 5 % scale error, ASSUMED
COMPASS_BIAS_SIGMA_DEG = 1.0    # ASSUMED residual compass bias
BARO_SIGMA = 0.3
SONAR_MAX = 15.0
SONAR_RANGE_SIGMA = 0.05
SONAR_BEARING_SIGMA = 0.02
PING_DT = 0.2
PAD_BEACON_RANGE_M = 10.0       # v2: ranging beacon on the pad (known landmark), ASSUMED
PAD_BEACON_SIGMA = 0.10
PAD_DETECT_R_M = 6.0            # v3: downward camera sees the pad marker within this radius
SEARCH_ALT_M = 15.0             # v3: climb for the pad search (wider footprint)
PAD_DETECT_R_SEARCH_M = 12.0    # v3: footprint radius at SEARCH_ALT_M (ASSUMED +-40 deg FOV)
SEARCH_SPACING_M = 15.0
HOLD_SPEED = 3.0                # max correction speed while descending
PAD_DETECT_P = 0.9              # per 0.5 s frame
PAD_REL_SIGMA = 0.10
SOC_LAND_V2 = 0.15              # v2 min_safe_soc
RESERVE_FRAC = 0.05             # v3: keep 5 % of capacity at touchdown
ENERGY_MARGIN = 1.3             # v3: multiply predicted energy
SEARCH_BUDGET_S = 90.0          # v3: expanding-square search time budget
WIND_AVG_TAU_S = 60.0           # v3: wind = running mean of (KF velocity - air velocity)
WIND_FIX_GAIN = 0.5             # v3: wind correction from position-fix innovations
GUST_HP_TAU_S = 60.0            # v3: wind changes = high-passed (bias-corrected INS velocity - air velocity)
IMU_ACC_NOISE = 0.10            # m/s^2, ASSUMED
IMU_ACC_BIAS_SIGMA = 0.03       # m/s^2, ASSUMED
IMU_BIAS_RW = 0.001             # m/s^2 per sqrt(s) bias random walk, ASSUMED
INFLATE_M = 4.0                 # path planning clearance around mapped structures
SAFETY_R_M = 4.0                # slide layer looks this far
COLLIDE_M = 0.25                # truth distance = collision (drone radius)
NEAR_MISS_M = 1.0
SUCCESS_R_M = 3.0               # landed within this of the true pad (ASSUMED success rule)
REG_EVERY_S = 1.0
REG_GATE = 16.0
T_MAX_S = 1500.0


# --- scenario -----------------------------------------------------------------
def make_scenarios(n: int, rng: np.random.Generator) -> List[Dict]:
    out = []
    for i in range(n):
        W, D, H = rng.uniform(10, 24), rng.uniform(10, 24), rng.uniform(8, 15)
        b = (np.array([-W / 2, -D / 2, 0.0]), np.array([W / 2, D / 2, H]))
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(80, 250)
        home = np.array([dist * math.cos(ang), dist * math.sin(ang), ALT_M])
        # a mast/tree near the line home->building, offset sideways (outbound transit misses it)
        f = rng.uniform(0.3, 0.7)
        side = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 8.0)
        nrm = np.array([-math.sin(ang), math.cos(ang), 0.0])
        mc = f * home + side * nrm
        mast = (np.array([mc[0] - 1, mc[1] - 1, 0.0]), np.array([mc[0] + 1, mc[1] + 1, 20.0]))
        out.append(dict(id=i, building=b, mast=mast, home=home, seed=int(rng.integers(0, 2**31 - 1)),
                        soc0=float(rng.uniform(0.08, 0.45))))
    return out


def mission_waypoints(sc: Dict) -> List[np.ndarray]:
    lo, hi = sc["building"]
    s = 4.0
    corners = [np.array([lo[0] - s, lo[1] - s, ALT_M]), np.array([hi[0] + s, lo[1] - s, ALT_M]),
               np.array([hi[0] + s, hi[1] + s, ALT_M]), np.array([lo[0] - s, hi[1] + s, ALT_M])]
    h = sc["home"]
    k = int(np.argmin([np.linalg.norm(c - h) for c in corners]))
    ring = corners[k:] + corners[:k]
    return [h.copy()] + ring + [ring[0]] + [h.copy()]


def path_length(wps: List[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(wps[:-1], wps[1:])))


# --- geometry helpers ---------------------------------------------------------
def box_dist_2d(p: np.ndarray, box) -> Tuple[float, np.ndarray]:
    lo, hi = box
    q = np.clip(p[:2], lo[:2], hi[:2])
    d = p[:2] - q
    n = float(np.hypot(*d))
    if n < 1e-9:     # inside: push out through the nearest face
        dd = np.array([p[0] - lo[0], hi[0] - p[0], p[1] - lo[1], hi[1] - p[1]])
        k = int(np.argmin(dd))
        out = [np.array([-1, 0.0]), np.array([1, 0.0]), np.array([0, -1.0]), np.array([0, 1.0])][k]
        return -float(dd[k]), np.array([*out, 0.0])
    return n, np.array([*(d / n), 0.0])


def seg_hits_rect(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """2-D segment vs open rectangle (slab test)."""
    t0, t1 = 0.0, 1.0
    d = b[:2] - a[:2]
    for k in range(2):
        if abs(d[k]) < 1e-12:
            if a[k] <= lo[k] or a[k] >= hi[k]:
                return False
            continue
        ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 >= t1:
            return False
    return True


def slide_safety(p_true: np.ndarray, v_cmd: np.ndarray, boxes) -> np.ndarray:
    """Shared sonar safety layer (both modes): remove the approaching component near a
    surface, add a sideways slide when blocked, push out when closer than 2 m."""
    v = v_cmd.copy()
    for box in boxes:
        d, n = box_dist_2d(p_true, box)
        if d > SAFETY_R_M:
            continue
        vn = float(v @ n)
        if vn < 0:
            v = v - vn * n
            t = np.array([-n[1], n[0], 0.0])
            vt = float(v @ t)
            if abs(vt) < 1.0:
                v = v + (1.0 if vt >= 0 else -1.0) * t * (1.0 - abs(vt))
        if d < 2.0:
            v = v + n * 1.5 * (2.0 - d)
    return v


def range_rate_guard(p_true: np.ndarray, v_true: np.ndarray, v_air_cmd: np.ndarray, boxes) -> np.ndarray:
    """Shared sonar range-rate guard (both modes): if the measured closing speed to a
    surface exceeds 0.6*(d - 0.8) m/s, push the air command outwards."""
    out = v_air_cmd.copy()
    for box in boxes:
        d, n = box_dist_2d(p_true, box)
        if d > SAFETY_R_M:
            continue
        closing = -float(v_true @ n)
        allow = max(0.0, 0.6 * (d - 0.8))
        if closing > allow:
            out = out + n * 2.0 * (closing - allow)
    return out


def ground_speed_for(va: float, d: np.ndarray, w: np.ndarray) -> float:
    """Ground speed along unit d when flying airspeed va in wind w (<=0: impossible)."""
    wa = float(w[:2] @ d[:2])
    wc2 = float(w[:2] @ w[:2]) - wa * wa
    if va * va <= wc2:
        return -1.0
    return wa + math.sqrt(va * va - wc2)


def best_leg_speed(d: np.ndarray, w: np.ndarray, extra_w: float = 0.0) -> Tuple[float, float]:
    """Airspeed minimising energy per ground metre along d. Returns (va, J per metre)."""
    best = (V_AIR_MAX, float("inf"))
    for va in np.arange(V_AIR_MIN, V_AIR_MAX + 1e-9, 0.25):
        vg = ground_speed_for(va, d, w)
        if vg <= 0.3:
            continue
        e = C.power_w(va, extra_w) / vg
        if e < best[1]:
            best = (float(va), e)
    return best


def mapped_obstacles(points: np.ndarray, cell: float = 2.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Connected 2-D grid components of the mapped points -> bounding rectangles."""
    if len(points) == 0:
        return []
    keys = {tuple(k) for k in np.floor(points[:, :2] / cell).astype(int)}
    seen, rects = set(), []
    for k in keys:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        a = np.array(comp)
        if len(a) < 2:
            continue
        rects.append((a.min(0) * cell, (a.max(0) + 1) * cell))
    return rects


def plan_path(start: np.ndarray, goal: np.ndarray, rects, wind: np.ndarray, inflate: float = INFLATE_M
              ) -> Tuple[List[np.ndarray], List[float], float]:
    """Energy-weighted visibility graph (start, goal, inflated rectangle corners).
    Returns (waypoints, per-leg airspeed, predicted energy J). Empty if no path."""
    inf = [(lo - inflate, hi + inflate) for lo, hi in rects]
    nodes = [start[:2], goal[:2]]
    for lo, hi in inf:
        e = 0.05
        nodes += [np.array([lo[0] - e, lo[1] - e]), np.array([hi[0] + e, lo[1] - e]),
                  np.array([hi[0] + e, hi[1] + e]), np.array([lo[0] - e, hi[1] + e])]
    nodes = [n for i, n in enumerate(nodes)
             if i < 2 or not any(np.all(n > lo) and np.all(n < hi) for lo, hi in inf)]
    N = len(nodes)

    def free(a, b):
        for lo, hi in inf:
            if np.all(a > lo) and np.all(a < hi):      # start inside the inflated zone: allow leaving
                continue
            if seg_hits_rect(a, b, lo, hi):
                return False
        return True

    def cost(a, b):
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            return 0.0, V_AIR_MIN
        va, jpm = best_leg_speed(np.array([*(b - a) / L, 0.0]), wind)
        return jpm * L, va

    dist = [float("inf")] * N
    prev = [-1] * N
    dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        dcur, i = heapq.heappop(pq)
        if dcur > dist[i]:
            continue
        if i == 1:
            break
        for j in range(N):
            if j == i or not free(nodes[i], nodes[j]):
                continue
            c, _ = cost(nodes[i], nodes[j])
            if dcur + c < dist[j]:
                dist[j] = dcur + c
                prev[j] = i
                heapq.heappush(pq, (dist[j], j))
    if not math.isfinite(dist[1]):
        return [], [], float("inf")
    idx = [1]
    while idx[-1] != 0:
        idx.append(prev[idx[-1]])
    idx = idx[::-1]
    wps = [np.array([*nodes[i], ALT_M]) for i in idx]
    speeds = [cost(nodes[a], nodes[b])[1] for a, b in zip(idx[:-1], idx[1:])]
    return wps, speeds, dist[1]


# --- simulator ------------------------------------------------------------------
def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class ReturnResult:
    mode: str
    onset_frac: float
    wind_mps: float
    soc_at_loss: float
    outcome: str = "timeout"          # landed / depleted / timeout / done_before_loss
    landed_in_place: bool = False
    landing_err_m: Optional[float] = None
    success: bool = False
    collision: bool = False
    near_miss: bool = False
    min_clear_m: float = float("inf")
    return_time_s: Optional[float] = None
    energy_used_wh: float = 0.0
    slam_fixes: int = 0
    pad_seen: bool = False
    est_err_at_land_m: Optional[float] = None
    traj: Optional[Dict] = None


class SafeReturnSim:
    """One mission: GPS mission (perimeter orbit) -> GPS + link loss -> offline return."""

    def __init__(self, sc: Dict, mode: str, onset_frac: float, wind_mps: float, seed: int,
                 record: bool = False):
        assert mode in ("v2", "v3", "v3_noslam")
        self.v3 = mode.startswith("v3")
        self.sc, self.mode, self.record = sc, mode, record
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.boxes = [sc["building"], sc["mast"]]
        lo = np.minimum(sc["building"][0], sc["home"]) - 30
        hi = np.maximum(sc["building"][1], sc["home"]) + 30
        self.scene = R.Scene(self.boxes, lo, hi, False)
        self.wind = C.Wind(wind_mps, rng)
        self.home_true = sc["home"].copy()
        self.gps_bias = rng.normal(0, GPS_SIGMA, 3) * np.array([1, 1, 0])
        self.hdg_bias = math.radians(rng.normal(0, COMPASS_BIAS_SIGMA_DEG))
        self.air_scale = 1.0 + rng.normal(0, AIR_VEL_SCALE_SIGMA)
        self.acc_bias = rng.normal(0, IMU_ACC_BIAS_SIGMA, 3) * np.array([1, 1, 0])
        self.gust_var = 0.0
        self.cal_A = np.zeros((4, 4)); self.cal_g = np.zeros(4)
        self.cal = np.zeros(4)          # v3: [wind x, wind y, air-data scale, heading] calibration
        self.v_ins = np.zeros(3)        # IMU-integrated velocity (drifts with bias)
        self.u_prev = None
        self.gust_hp = np.zeros(3)
        self.v_ins_raw = np.zeros(3)
        self.b_est = np.zeros(3)        # v3: IMU bias, LS slope of (INS - GPS/KF velocity) while GPS is up
        self.bsum = np.zeros((5, 3))    # n, sum t, sum t^2, sum e, sum t e
        self.t_last_fix = None
        self.p = self.home_true.copy()
        self.v = np.zeros(3)
        self.kf = v2.DragConsistentKF(drag_k=0.18, mass_kg=1.45, jerk_psd=25.0,
                                      sigma_pos_meas=GPS_SIGMA, sigma_vel_meas=GPS_VEL_SIGMA)
        z0 = self._gps()
        self.kf.initialize(v2.AirframeState(0.0, z0, np.zeros(3), np.zeros(3), np.zeros(3),
                                            pos_sigma_m=np.full(3, GPS_SIGMA)))
        self.home_rec = z0.copy()                     # where the drone believes home is
        self.home_rec[2] = ALT_M
        self.wps = mission_waypoints(sc)
        self.t_loss = onset_frac * path_length(self.wps) / MISSION_SPEED
        self.res = ReturnResult(mode, onset_frac, wind_mps, sc["soc0"])
        self.w_ctrl = np.zeros(3)
        self.climbed = False
        self.acc_true = np.zeros(3)
        self.w_hat = np.zeros(3)
        self.map_pts: List[np.ndarray] = []
        self.map_cov: List[np.ndarray] = []
        self.scan: List[Tuple[np.ndarray, np.ndarray]] = []
        self.energy_wh = sc["soc0"] * C.BATTERY_WH

    # sensors ------------------------------------------------------------------
    def _gps(self) -> np.ndarray:
        return self.p + self.gps_bias + self.rng.normal(0, GPS_WHITE, 3)

    def _step_gps_bias(self) -> None:
        phi = math.exp(-DT / GPS_TAU)
        self.gps_bias = phi * self.gps_bias + math.sqrt(1 - phi * phi) * self.rng.normal(0, GPS_SIGMA, 3) * np.array([1, 1, 0])

    def _air_vel_meas(self, wind: np.ndarray) -> np.ndarray:
        va = self.v - wind
        return rot_z(self.hdg_bias) @ (self.air_scale * va) + self.rng.normal(0, AIR_VEL_NOISE, 3) * np.array([1, 1, 0.3])

    def _clearance(self) -> Tuple[float, int]:
        ds = [box_dist_2d(self.p, b)[0] for b in self.boxes]
        k = int(np.argmin(ds))
        return float(ds[k]), k

    def _ping(self) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """Sonar fan steered at the nearest surface; returns (u_measured, r, point cov)."""
        d, k = self._clearance()
        if d > SONAR_MAX:
            return []
        lo, hi = self.boxes[k]
        q = np.clip(self.p, lo, hi)
        q[2] = ALT_M
        c = q - self.p
        az0 = math.atan2(c[1], c[0])
        out = []
        for daz in np.radians(np.arange(-30, 31, 7.5)):
            for el in np.radians((-8.0, 0.0, 8.0)):
                u = np.array([math.cos(az0 + daz) * math.cos(el), math.sin(az0 + daz) * math.cos(el), math.sin(el)])
                r = self.scene.raycast(self.p, u, SONAR_MAX)
                if r is None:
                    continue
                rm = r + self.rng.normal(0, SONAR_RANGE_SIGMA)
                um = rot_z(self.hdg_bias + self.rng.normal(0, SONAR_BEARING_SIGMA)) @ u
                cov = R.point_covariance(rm, um, SONAR_RANGE_SIGMA, SONAR_BEARING_SIGMA,
                                         sig_heading=math.radians(COMPASS_BIAS_SIGMA_DEG))
                out.append((um, rm, cov))
        return out

    # map / localisation --------------------------------------------------------
    def _freeze_map(self) -> None:
        if not self.map_pts:
            self.M = np.zeros((0, 3)); self.Mn = np.zeros((0, 3)); self.Mc = np.zeros((0, 3, 3))
            self.rects = []
            return
        P = np.array(self.map_pts); Cv = np.array(self.map_cov)
        _, keep = np.unique(np.floor(P / 0.25).astype(int), axis=0, return_index=True)
        P, Cv = P[keep], Cv[keep]
        if len(P) > 4000:
            sel = self.rng.choice(len(P), 4000, replace=False)
            P, Cv = P[sel], Cv[sel]
        nrm = np.zeros_like(P); pl = np.zeros(len(P), bool)
        for i in range(0, len(P), 400):
            nrm[i:i + 400], pl[i:i + 400] = S.local_normals(P[i:i + 400], P)
        self.M, self.Mn, self.Mc = P[pl], nrm[pl], Cv[pl]
        self.rects = mapped_obstacles(P)

    def _slam_fix(self) -> None:
        if len(self.scan) < 20 or len(self.M) < 20:
            self.scan = []
            return
        Sp = np.array([s[0] for s in self.scan]); Sc = np.array([s[1] for s in self.scan])
        self.scan = []
        Sn, Spl = S.local_normals(Sp, Sp)
        Sn = np.where(Spl[:, None], Sn, 0.0)
        out = S.register_translation(Sp, Sc, self.M, self.Mn, self.Mc, np.zeros(3), S_n=Sn)
        if not out["ok"] or out["n_obs"] == 0:
            return
        V, ev = S.observable_basis(out["A"], 1.0)
        if V.shape[1] == 0:
            return
        c = out["c"]
        Rf_obs = V @ np.diag(4.0 / ev) @ V.T                  # x4: scan smear over 1 s
        Uc = np.eye(3) - V @ V.T
        Rf = Rf_obs + 1e4 * Uc
        Pp = self.kf.P[0:3, 0:3]
        Sinn = V.T @ (Pp + Rf_obs) @ V
        nis = float((V.T @ c) @ np.linalg.solve(Sinn, V.T @ c))
        if nis > REG_GATE:
            return
        z = self.kf.x[0:3] - V @ (V.T @ c)
        self._wind_from_fix(-(V @ (V.T @ c)))
        self.kf.update_pos(z, 0.5 * (Rf + Rf.T))
        self.res.slam_fixes += 1

    @staticmethod
    def _cal_H(va: np.ndarray) -> np.ndarray:
        return np.array([[1.0, 0.0, va[0], -va[1]], [0.0, 1.0, va[1], va[0]]])

    def _cal_update(self, t: float, a: float, r: np.ndarray, va_raw: np.ndarray) -> None:
        """Air-data calibration while GPS is up: (GPS/KF velocity - measured air velocity)
        = wind + scale * va + heading * J va. Exponentially weighted least squares, ridge
        prior on scale (5 %) and heading (1 deg)."""
        H = self._cal_H(va_raw)
        y = r[:2]
        res = y - H @ self.cal
        self.gust_var = (1 - a) * self.gust_var + a * float(res @ res) / 2
        self.cal_A = (1 - a) * self.cal_A + a * H.T @ H
        self.cal_g = (1 - a) * self.cal_g + a * H.T @ y
        if int(round(t / DT)) % 10 == 0:
            n_eff = max(1.0, min(t, WIND_AVG_TAU_S) / 4.0)
            ridge = (self.gust_var + 0.04) / n_eff * np.diag([1e-6, 1e-6, 1 / AIR_VEL_SCALE_SIGMA ** 2,
                                                               1 / math.radians(COMPASS_BIAS_SIGMA_DEG) ** 2])
            self.cal = np.linalg.solve(self.cal_A + ridge + 1e-9 * np.eye(4), self.cal_g)
            self.w_hat = np.array([self.cal[0], self.cal[1], 0.0])

    def _air_cal(self, va_raw: np.ndarray) -> np.ndarray:
        out = va_raw.copy()
        out[:2] = va_raw[:2] + self.cal[2] * va_raw[:2] + self.cal[3] * np.array([-va_raw[1], va_raw[0]])
        return out

    def _wind_from_fix(self, innov: np.ndarray) -> None:
        """Position-fix innovation / time since the previous fix = mean velocity (wind)
        error over that interval; fold part of it into the wind estimate and the KF velocity."""
        t = self._t
        if self.t_last_fix is not None:
            dt = min(max(t - self.t_last_fix, 1.0), 60.0)
            dv = WIND_FIX_GAIN * np.asarray(innov, float) * np.array([1, 1, 0]) / dt
            self.w_hat = self.w_hat + dv
            self.kf.x[3:6] = self.kf.x[3:6] + dv
        self.t_last_fix = t

    # control -----------------------------------------------------------------
    def _p_est(self) -> np.ndarray:
        return self.kf.x[0:3].copy()

    def _toward(self, target: np.ndarray, speed: float) -> np.ndarray:
        d = target - self._p_est()
        d[2] = 0.0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return np.zeros(3)
        return d / L * min(speed, 0.8 * L + 0.2)

    def _plan_v3(self) -> None:
        wps, speeds, e_j = plan_path(self._p_est(), self.home_rec, self.rects, self.w_hat)
        self.ret_wps, self.ret_speeds, self.ret_e_wh = wps, speeds, e_j / 3600.0

    def _v3_energy_ok(self) -> bool:
        if not self.ret_wps:
            return False
        p_hover = C.power_w(0.0)
        need = self.ret_e_wh * ENERGY_MARGIN + 6.0 * p_hover / 3600.0
        spare = self.energy_wh - need - RESERVE_FRAC * C.BATTERY_WH
        self.search_budget_s = max(0.0, min(SEARCH_BUDGET_S, spare * 3600.0 / C.power_w(6.0)))
        return spare >= 0.0

    def _search_wps(self) -> List[np.ndarray]:
        """Expanding square around the believed home, flown after climbing to SEARCH_ALT_M."""
        if not self.climbed:
            self.climbed = True
            e = 1.45 * 9.81 * (SEARCH_ALT_M - ALT_M) / 0.5 / 3600.0     # climb, 50 % efficiency (ASSUMED)
            self.energy_wh -= e
            self.res.energy_used_wh += e
        out, p, s = [self.home_rec.copy()], self.home_rec.copy(), SEARCH_SPACING_M
        dirs = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, -1.0, 0])]
        for k in range(10):
            p = p + dirs[k % 4] * s * (k // 2 + 1)
            out.append(p.copy())
        return out

    # main loop -----------------------------------------------------------------
    def run(self) -> ReturnResult:
        res, rng = self.res, self.rng
        t, wp_i, lost = 0.0, 1, False
        phase = "mission"           # mission / return / search / servo / descend / done
        t_descend, t_last_ping, t_last_reg, t_last_plan, t_last_pad = None, -1.0, 0.0, -99.0, -99.0
        t_pad_seen, search, search_i, t_search0 = -99.0, [], 0, 0.0
        tr = dict(t=[], p=[], est=[], phase=[]) if self.record else None
        while t < T_MAX_S:
            self._t = t
            wind = self.wind.step(DT)
            self._step_gps_bias()
            self.kf.predict(DT)
            v_air_raw = self._air_vel_meas(wind)
            v_air = self._air_cal(v_air_raw) if self.v3 else v_air_raw
            if t > 0:
                self.acc_bias = self.acc_bias + rng.normal(0, IMU_BIAS_RW * math.sqrt(DT), 3) * np.array([1, 1, 0])
                acc_m = self.acc_true + self.acc_bias + rng.normal(0, IMU_ACC_NOISE, 3) * np.array([1, 1, 0])
                if self.v3:
                    acc_m = acc_m - self.b_est
                # the v2 KF state "a" is the drag-free (thrust) acceleration: dv/dt = a - lam*v.
                # An IMU measures the kinematic acceleration, so add lam*v back (both modes).
                self.kf.update_acc(acc_m + self.kf.lam * self.kf.x[3:6] * np.array([1, 1, 0]),
                                   np.eye(3) * (IMU_ACC_NOISE ** 2 + IMU_ACC_BIAS_SIGMA ** 2))
                self.v_ins = self.v_ins + acc_m * DT
                self.v_ins_raw = self.v_ins_raw + (acc_m + (self.b_est if self.v3 else 0.0)) * DT
            u = (self.v_ins - v_air) * np.array([1, 1, 0])
            if self.u_prev is not None:
                al = GUST_HP_TAU_S / (GUST_HP_TAU_S + DT)
                self.gust_hp = al * (self.gust_hp + u - self.u_prev)
            self.u_prev = u
            p_est = self._p_est()
            if not lost and t >= self.t_loss:
                lost = True
                t_lost = t
                self.t_last_fix = t                  # last GPS fix
                if self.v3:
                    self._freeze_map()
                    self._plan_v3()
                    phase = "return" if self._v3_energy_ok() else "descend"
                    res.landed_in_place = phase == "descend"
                else:
                    phase = "return" if self.energy_wh / C.BATTERY_WH > SOC_LAND_V2 else "descend"
                    res.landed_in_place = phase == "descend"
                if phase == "descend":
                    t_descend = t
                    self.hold = p_est
            # --- estimation ---
            if not lost:
                if int(round(t / DT)) % 2 == 0:
                    self.kf.update_pos(self._gps(), np.eye(3) * GPS_SIGMA ** 2)
                    self.kf.update_vel(self.v + rng.normal(0, GPS_VEL_SIGMA, 3), np.eye(3) * GPS_VEL_SIGMA ** 2)
                a = max(DT / WIND_AVG_TAU_S, DT / (t + DT))
                self._cal_update(t, a, self.kf.x[3:6] - v_air_raw, v_air_raw)
                e = self.v_ins_raw - self.kf.x[3:6]
                self.bsum += np.array([np.ones(3), np.full(3, t), np.full(3, t * t), e, t * e])
                n, st, stt, se, ste = self.bsum
                den = n * stt - st * st
                if self.v3 and t > 10.0 and int(round(t / DT)) % 10 == 0:
                    b_new = np.where(den > 0, (n * ste - st * se) / np.maximum(den, 1e-12), 0.0) * np.array([1, 1, 0])
                    self.v_ins = self.v_ins - (b_new - self.b_est) * t   # re-base the corrected INS
                    self.b_est = b_new
            else:
                w_used = self.w_hat + self.gust_hp if self.v3 else np.zeros(3)
                sv = AIR_VEL_NOISE ** 2 + (AIR_VEL_SCALE_SIGMA * np.linalg.norm(v_air)) ** 2
                if self.v3:
                    sv += 0.3 ** 2
                self.kf.update_vel(v_air + w_used, np.eye(3) * sv)
            a = DT / 3.0
            self.w_ctrl = (1 - a) * self.w_ctrl + a * (self.kf.x[3:6] - v_air) * np.array([1, 1, 0])
            # sonar
            if t - t_last_ping >= PING_DT - 1e-9:
                t_last_ping = t
                pe = self._p_est()
                for um, rm, cov in self._ping():
                    q = pe + rm * um
                    if not lost:
                        self.map_pts.append(q); self.map_cov.append(cov)
                    elif self.v3:
                        self.scan.append((q, cov))
            if lost and self.mode == "v3" and t - t_last_reg >= REG_EVERY_S:
                t_last_reg = t
                self._slam_fix()
            # pad sensing (every 0.5 s)
            if lost and t - t_last_pad >= 0.5:
                t_last_pad = t
                dh = float(np.hypot(*(self.p - self.home_true)[:2]))
                if self.mode == "v2" and dh <= PAD_BEACON_RANGE_M:
                    rng_m = float(np.linalg.norm(self.p - (self.home_true - np.array([0, 0, ALT_M]))))
                    lm = self.home_rec - np.array([0, 0, ALT_M])
                    self.kf.update_landmark_range(lm, rng_m + rng.normal(0, PAD_BEACON_SIGMA), PAD_BEACON_SIGMA)
                if self.v3 and dh <= (PAD_DETECT_R_SEARCH_M if self.climbed else PAD_DETECT_R_M) and rng.random() < PAD_DETECT_P:
                    rel = rot_z(self.hdg_bias) @ (self.home_true - self.p) + rng.normal(0, PAD_REL_SIGMA, 3)
                    z = self.home_rec - rel
                    z[2] = self.kf.x[2]
                    self._wind_from_fix(z - self.kf.x[0:3])
                    self.kf.update_pos(z, np.diag([PAD_REL_SIGMA ** 2, PAD_REL_SIGMA ** 2, 1e4]))
                    t_pad_seen = t
                    res.pad_seen = True
                    if phase in ("return", "search"):
                        phase = "servo"
            p_est = self._p_est()
            # --- guidance ---
            if phase == "mission":
                tgt = self.wps[wp_i]
                if np.linalg.norm((tgt - p_est)[:2]) < 1.5:
                    wp_i += 1
                    if wp_i >= len(self.wps):
                        res.outcome = "done_before_loss"
                        break
                    tgt = self.wps[wp_i]
                v_cmd = self._toward(tgt, MISSION_SPEED)
            elif phase == "return" and self.mode == "v2":
                if self.energy_wh / C.BATTERY_WH <= SOC_LAND_V2:
                    phase, t_descend, self.hold = "descend", t, p_est
                    res.landed_in_place = True
                    v_cmd = np.zeros(3)
                elif np.linalg.norm((self.home_rec - p_est)[:2]) < 1.0:
                    phase, t_descend, self.hold = "descend", t, self.home_rec.copy()
                    v_cmd = np.zeros(3)
                else:
                    v_cmd = self._toward(self.home_rec, V2_RETURN_SPEED)
            elif phase == "return":
                if t - t_last_plan >= 5.0:
                    t_last_plan = t
                    self._plan_v3()
                    if not self._v3_energy_ok():
                        phase, t_descend, self.hold = "descend", t, p_est
                        res.landed_in_place = True
                if phase == "return":
                    if len(self.ret_wps) >= 2 and np.linalg.norm((self.ret_wps[1] - p_est)[:2]) < 1.5 \
                            and len(self.ret_wps) > 2:
                        self.ret_wps.pop(0); self.ret_speeds.pop(0)
                    if np.linalg.norm((self.home_rec - p_est)[:2]) < 1.5:
                        phase, search, search_i, t_search0 = "search", self._search_wps(), 0, t
                        v_cmd = np.zeros(3)
                    else:
                        tgt = self.ret_wps[1] if len(self.ret_wps) >= 2 else self.home_rec
                        d = tgt - p_est; d[2] = 0
                        L = max(float(np.linalg.norm(d)), 1e-6)
                        vg = ground_speed_for(self.ret_speeds[0] if self.ret_speeds else 6.0, d / L, self.w_hat)
                        v_cmd = self._toward(tgt, max(vg, 1.0))
                else:
                    v_cmd = np.zeros(3)
            elif phase == "search":
                if t - t_search0 > self.search_budget_s or search_i >= len(search):
                    phase, t_descend, self.hold = "descend", t, self.home_rec.copy()
                    v_cmd = np.zeros(3)
                else:
                    if np.linalg.norm((search[search_i] - p_est)[:2]) < 1.0:
                        search_i += 1
                    v_cmd = self._toward(search[min(search_i, len(search) - 1)], 5.0)
            elif phase == "servo":
                if t - t_pad_seen > 3.0:
                    phase, search, search_i, t_search0 = "search", self._search_wps(), 0, t
                if np.linalg.norm((self.home_rec - p_est)[:2]) < 0.3:
                    phase, t_descend, self.hold = "descend", t, self.home_rec.copy()
                v_cmd = self._toward(self.home_rec, 2.0)
            elif phase == "descend":
                v_cmd = self._toward(self.hold, HOLD_SPEED)
                if t - t_descend >= (SEARCH_ALT_M if self.climbed else ALT_M) / 1.0:
                    res.outcome = "landed"
                    break
            # --- safety layer + truth dynamics ---
            v_cmd = slide_safety(self.p, v_cmd, self.boxes)
            v_air_cmd = range_rate_guard(self.p, self.v, v_cmd - self.w_ctrl, self.boxes)
            sa = float(np.linalg.norm(v_air_cmd[:2]))
            if sa > V_AIR_MAX:
                v_air_cmd = v_air_cmd * (V_AIR_MAX / sa)
            v_old = self.v.copy()
            self.v = self.v + (v_air_cmd + wind - self.v) * (DT / VEL_TAU_S)
            self.v[2] = 0.0
            self.acc_true = (self.v - v_old) / DT
            self.p = self.p + self.v * DT
            dmin, _ = self._clearance()
            if lost:
                res.min_clear_m = min(res.min_clear_m, dmin)
                pw = C.power_w(float(np.linalg.norm((self.v - wind)[:2])))
                self.energy_wh -= pw * DT / 3600.0
                res.energy_used_wh += pw * DT / 3600.0
                if self.energy_wh <= 0.0:
                    res.outcome = "depleted"
                    break
            if tr is not None and int(round(t / DT)) % 5 == 0:
                tr["t"].append(t); tr["p"].append(self.p.copy()); tr["est"].append(self._p_est()); tr["phase"].append(phase)
            t += DT
        res.collision = res.min_clear_m < COLLIDE_M
        res.near_miss = (not res.collision) and res.min_clear_m < NEAR_MISS_M
        if lost:
            res.return_time_s = t - t_lost
        if res.outcome == "landed":
            res.landing_err_m = float(np.hypot(*(self.p - self.home_true)[:2]))
            res.est_err_at_land_m = float(np.hypot(*(self.p - self._p_est())[:2]))
            res.success = (res.landing_err_m <= SUCCESS_R_M) and not res.collision
        if tr is not None:
            tr.update(home=self.home_true, home_rec=self.home_rec, boxes=self.boxes,
                      t_loss=self.t_loss, map=np.array(self.map_pts) if self.map_pts else np.zeros((0, 3)))
            res.traj = tr
        return res


def run_return(sc: Dict, mode: str, onset_frac: float, wind_mps: float, seed: int, record: bool = False
               ) -> ReturnResult:
    return SafeReturnSim(sc, mode, onset_frac, wind_mps, seed, record).run()

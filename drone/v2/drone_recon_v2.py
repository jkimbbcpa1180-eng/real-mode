"""
REAL MODE: Reconnaissance / mapping layer for drone_core_v2 (downstream of the
flight loop; it never feeds back into guidance).

1. Range-dependent resolution. Each sonar hit becomes a point with covariance
     range noise along the ray + (range * bearing sigma)^2 across it
     + (range * heading sigma)^2 about the vertical + the pose covariance,
   the same model drone_core_v2 uses for obstacle points. Closer = smaller sigma.
2. 3-D evidence grid (sparse, log-odds). Free space is ray-cast along each beam up
   to just before the hit; the hit evidence is spread over a footprint of radius
   ~2 * lateral sigma, so far hits are smeared over several cells and close hits
   land in one. Each cell keeps the smallest lateral sigma that hit it, so the
   effective detail of the map improves where the drone flew closer.
3. Export: ASCII PLY point cloud (x y z sigma + colour) and an optional PNG
   render (matplotlib, if installed; otherwise the render is skipped with a note).
4. Mission templates: lawnmower area sweep, perimeter orbit, point-of-interest
   close pass. Waypoint spacing comes from standoff * tan(beam half-angle) with an
   overlap fraction, so neighbouring swaths overlap.
5. Authorized interior mapping: the drone flies INSIDE a room (same sonar grid);
   planning refuses unless mission_config["interior_authorized"] is True. Airborne
   ultrasound reflects almost entirely at walls, so this layer maps SURFACES the
   drone can see; it does not and cannot image through walls.
6. Magnetic disturbance log: field magnitude/dip departures along the flight path
   are logged as "possible wiring/steel nearby" only (no claim of seeing inside).
7. Coverage metric for KNOWN sim scenes: fraction of true surface samples that
   have a reconstructed point within a tolerance (ASSUMED 0.3 m), and the
   accuracy of reconstructed points against the true surfaces.

All numeric settings are ASSUMED (labelled). Dependencies: numpy (matplotlib optional).
License: CC0 1.0 Universal (public domain). Copy, modify, use freely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GRID_RES_M = 0.25          # ASSUMED evidence grid cell size
L_HIT = 0.85               # ASSUMED log-odds added by one hit (spread over the footprint)
L_MISS = -0.40             # ASSUMED log-odds added by a beam passing through a cell
L_MIN, L_MAX = -2.0, 3.5   # clamps
L_OCC = 0.80               # occupied if log-odds above this
FOOTPRINT_K = 2.0          # footprint radius = K * lateral sigma
MAX_FOOTPRINT_CELLS = 4    # cap of the footprint radius in cells
COVERAGE_TOL_M = 0.30      # ASSUMED accuracy tolerance for "observed"
QUANT_LO = 10              # log-odds quantisation for sync / hashing (0.1)
QUANT_SIG = 100            # cell best-sigma quantisation (cm)


# ===========================================================================
# 1. Range-dependent point covariance
# ===========================================================================
def ray_basis(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, float) / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    e_az = np.cross(z, u)
    if np.linalg.norm(e_az) < 1e-6:          # looking straight up/down
        e_az = np.array([0.0, 1.0, 0.0])
    e_az /= np.linalg.norm(e_az)
    e_el = np.cross(u, e_az)
    return u, e_az, e_el


def point_covariance(r: float, u: np.ndarray, sig_range: float, sig_bearing: float,
                     sig_heading: Optional[float] = None,
                     pose_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """Covariance of p + r*u. Lateral terms grow linearly with range."""
    u, e_az, e_el = ray_basis(u)
    cov = (sig_range ** 2) * np.outer(u, u) \
        + (r * sig_bearing) ** 2 * (np.outer(e_az, e_az) + np.outer(e_el, e_el))
    if sig_heading:
        j = np.cross(np.array([0.0, 0.0, 1.0]), r * u)   # d(point)/d(yaw)
        cov = cov + np.outer(j, j) * sig_heading ** 2
    if pose_cov is not None:
        cov = cov + np.asarray(pose_cov, float)
    return 0.5 * (cov + cov.T)


def lateral_sigma(cov: np.ndarray, u: np.ndarray) -> float:
    """Largest sigma across the ray (what limits detail)."""
    _, e_az, e_el = ray_basis(u)
    B = np.stack([e_az, e_el], axis=1)
    return float(math.sqrt(max(np.linalg.eigvalsh(B.T @ cov @ B).max(), 0.0)))


# ===========================================================================
# 2. Sparse 3-D evidence grid
# ===========================================================================
Key = Tuple[int, int, int]


class EvidenceGrid:
    def __init__(self, res: float = GRID_RES_M):
        self.res = float(res)
        self.lo: Dict[Key, float] = {}
        self.best_sigma: Dict[Key, float] = {}
        self.dirty: set = set()        # cells changed since the map-sync layer last drained it
        self.counter = 0
        self.n_beams = 0

    def key(self, p: np.ndarray) -> Key:
        return (int(math.floor(p[0] / self.res)), int(math.floor(p[1] / self.res)),
                int(math.floor(p[2] / self.res)))

    def center(self, k: Key) -> np.ndarray:
        return (np.array(k, float) + 0.5) * self.res

    def _add(self, k: Key, dl: float) -> None:
        old = self.lo.get(k)
        v = min(max((0.0 if old is None else old) + dl, L_MIN), L_MAX)
        if v != old:
            self.lo[k] = v
            self._touch(k)

    def _touch(self, k: Key) -> None:
        self.counter += 1
        self.dirty.add(k)

    def integrate_beam(self, origin: np.ndarray, hit: Optional[np.ndarray], sig_range: float,
                       sig_lat: float) -> None:
        """One beam. A beam without a hit adds nothing (a soft absorber cannot be ruled out)."""
        if hit is not None:
            self.integrate_ping(origin, [(np.asarray(hit, float), sig_range, sig_lat)])

    def integrate_ping(self, origin: np.ndarray, hits: Sequence[Tuple[np.ndarray, float, float]]) -> None:
        """
        All beams of one ping from the same origin. Free space: every cell crossed by a beam
        from the origin up to (hit - margin), margin = max(2 sigma_range, 2 sigma_lat, res),
        gets L_MISS (once per beam). Hit: L_HIT spread over a DISC perpendicular to the ray,
        radius FOOTPRINT_K * sigma_lat (>= half a cell, <= MAX_FOOTPRINT_CELLS cells), with
        Gaussian weights normalised to sum 1, so a far (wide) hit puts less evidence in any
        one cell than a close (narrow) hit. Cells near the footprint centre keep the
        smallest lateral sigma that hit them (effective detail).
        """
        origin = np.asarray(origin, float)
        miss_keys: Dict[Key, int] = {}
        hit_list = []
        for hit, sig_range, sig_lat in hits:
            hit = np.asarray(hit, float)
            d = hit - origin
            rng = float(np.linalg.norm(d))
            if rng < 1e-6:
                continue
            self.n_beams += 1
            u = d / rng
            margin = max(2.0 * sig_range, 2.0 * sig_lat, self.res)
            free_len = rng - margin
            if free_len > 0:
                n = int(free_len / (0.5 * self.res)) + 1
                S = origin + np.linspace(0.0, free_len, n)[:, None] * u
                K = np.floor(S / self.res).astype(np.int64)
                keep = np.ones(len(K), bool)
                keep[1:] = np.any(K[1:] != K[:-1], axis=1)
                for k in set(map(tuple, K[keep].tolist())):
                    miss_keys[k] = miss_keys.get(k, 0) + 1
            hit_list.append((hit, u, sig_lat))
        for k, c in miss_keys.items():
            self._add(k, L_MISS * c)
        for hit, u, sig_lat in hit_list:
            rad = min(max(FOOTPRINT_K * sig_lat, 0.5 * self.res), MAX_FOOTPRINT_CELLS * self.res)
            m = int(math.ceil(rad / self.res))
            kc = np.floor(hit / self.res).astype(np.int64)
            ar = np.arange(-m, m + 1)
            O = np.stack(np.meshgrid(ar, ar, ar, indexing="ij"), -1).reshape(-1, 3)
            Kc = kc + O
            C = (Kc + 0.5) * self.res - hit
            along = C @ u
            perp = np.linalg.norm(C - along[:, None] * u, axis=1)
            mask = (np.abs(along) <= 0.5 * self.res * math.sqrt(3.0)) & (perp <= rad + 0.5 * self.res)
            if not np.any(mask):
                mask[np.argmin(np.linalg.norm(C, axis=1))] = True
            sl = max(sig_lat, 0.5 * self.res)
            w = np.exp(-0.5 * (perp[mask] / sl) ** 2 - 0.5 * (along[mask] / (0.5 * self.res)) ** 2)
            w = w / w.sum()
            wmax = float(w.max())
            for k, w_ in zip(map(tuple, Kc[mask].tolist()), w):
                self._add(k, L_HIT * float(w_))
                if w_ >= 0.5 * wmax and sig_lat < self.best_sigma.get(k, float("inf")):
                    self.best_sigma[k] = float(sig_lat)
                    self._touch(k)

    def occupied(self) -> List[Key]:
        return [k for k, v in self.lo.items() if v > L_OCC]

    def occupied_points(self) -> Tuple[np.ndarray, np.ndarray]:
        ks = self.occupied()
        if not ks:
            return np.zeros((0, 3)), np.zeros(0)
        P = np.array([self.center(k) for k in ks])
        S = np.array([self.best_sigma.get(k, float("nan")) for k in ks])
        return P, S

    def drain_dirty(self) -> set:
        d, self.dirty = self.dirty, set()
        return d

    def quantized(self, k: Key) -> Tuple[int, int]:
        bs = self.best_sigma.get(k)
        return (int(round(self.lo.get(k, 0.0) * QUANT_LO)),
                -1 if bs is None else int(round(bs * QUANT_SIG)))


@dataclass
class ReconPoint:
    pid: int
    pos: np.ndarray
    sigma: float
    t: float


class ReconMap:
    """Point cloud (every accepted hit, with sigma) + evidence grid."""

    def __init__(self, res: float = GRID_RES_M):
        self.grid = EvidenceGrid(res)
        self.points: List[ReconPoint] = []

    def add_hit(self, origin: np.ndarray, u: np.ndarray, r: float, cov: np.ndarray,
                sig_range: float, t: float) -> ReconPoint:
        return self.add_ping(origin, [(u, r, cov)], sig_range, t)[0]

    def add_ping(self, origin: np.ndarray, beams: Sequence[Tuple[np.ndarray, float, np.ndarray]],
                 sig_range: float, t: float) -> List[ReconPoint]:
        """beams: (unit ray, range, point covariance) for each hit of one ping."""
        origin = np.asarray(origin, float)
        out, hits = [], []
        for u, r, cov in beams:
            u = np.asarray(u, float)
            hit = origin + r * u
            hits.append((hit, sig_range, lateral_sigma(cov, u)))
            sig = float(math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
            p = ReconPoint(len(self.points), hit, sig, float(t))
            self.points.append(p)
            out.append(p)
        self.grid.integrate_ping(origin, hits)
        return out

    def filtered_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Points whose grid cell ended up occupied (multipath / spikes are mostly not)."""
        if not self.points:
            return np.zeros((0, 3)), np.zeros(0)
        occ = set(self.grid.occupied())
        keep = [p for p in self.points if self.grid.key(p.pos) in occ]
        if not keep:
            return np.zeros((0, 3)), np.zeros(0)
        return np.array([p.pos for p in keep]), np.array([p.sigma for p in keep])

    def effective_detail_m(self) -> Optional[float]:
        _, S = self.grid.occupied_points()
        S = S[np.isfinite(S)]
        return float(np.median(S)) if len(S) else None


# ===========================================================================
# 3. Export: PLY + PNG
# ===========================================================================
def _sigma_colour(s: np.ndarray, s_max: float = 0.5) -> np.ndarray:
    f = np.clip(np.nan_to_num(s, nan=s_max) / s_max, 0.0, 1.0)
    return np.stack([255 * f, 255 * (1 - f), 60 * np.ones_like(f)], axis=1).astype(int)


def write_ply(path: str, xyz: np.ndarray, sigma: np.ndarray, comment: str = "") -> int:
    """ASCII PLY: x y z sigma red green blue (green = small sigma, red = large)."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    sigma = np.asarray(sigma, float).reshape(-1)
    col = _sigma_colour(sigma)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment drone_recon_v2 export; sigma = 1-sigma position uncertainty in metres\n")
        if comment:
            f.write(f"comment {comment}\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nproperty float sigma\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, s, c in zip(xyz, sigma, col):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {s:.4f} {c[0]} {c[1]} {c[2]}\n")
    return len(xyz)


def read_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    start = lines.index("end_header") + 1
    rows = np.array([[float(x) for x in l.split()[:4]] for l in lines[start:start + n]]).reshape(-1, 4)
    return rows[:, :3], rows[:, 3]


def render_png(path: str, xyz: np.ndarray, sigma: np.ndarray, boxes: Sequence = (),
               path_xyz: Optional[np.ndarray] = None, title: str = "") -> Dict:
    """3-D scatter coloured by sigma. Returns {"written": bool, "note": str}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib is optional
        return {"written": False, "note": f"matplotlib not available ({type(e).__name__}); PNG skipped"}
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(xyz):
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.nan_to_num(sigma, nan=0.5),
                        cmap="RdYlGn_r", s=1.5, vmin=0.0, vmax=0.5)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="point sigma (m)")
    for b in boxes:
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        for a in range(3):
            for c1 in (0, 1):
                for c2 in (0, 1):
                    p0, p1 = lo.copy(), lo.copy()
                    o = [i for i in range(3) if i != a]
                    p0[o[0]] = p1[o[0]] = (lo, hi)[c1][o[0]]
                    p0[o[1]] = p1[o[1]] = (lo, hi)[c2][o[1]]
                    p0[a], p1[a] = lo[a], hi[a]
                    ax.plot(*zip(p0, p1), color="k", lw=0.6)
    if path_xyz is not None and len(path_xyz):
        P = np.asarray(path_xyz, float)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="b", lw=0.5, alpha=0.6)
    ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
    ax.set_title(title or "drone_recon_v2: reconstructed points (truth boxes in black)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"written": True, "note": "matplotlib render"}


# ===========================================================================
# 4. Mission templates
# ===========================================================================
@dataclass
class Waypoint:
    pos: np.ndarray
    look_dir: np.ndarray          # unit vector the sonar head is pointed along


def swath_width(standoff_m: float, beam_half_rad: float) -> float:
    return 2.0 * standoff_m * math.tan(beam_half_rad)


def _densify(pts: List[np.ndarray], step: float) -> List[np.ndarray]:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return out


LAWNMOWER_RANGE_MARGIN = 0.9   # ASSUMED: plan the ground to be within 90% of the sonar range
LAWNMOWER_MIN_ROOF_CLEARANCE_M = 2.0   # ASSUMED minimum standoff above the highest roof


def lawnmower(area_min: Sequence[float], area_max: Sequence[float], top_z: float, standoff_m: float,
              beam_half_rad: float, overlap: float = 0.3, step_m: float = 1.0,
              max_range_m: Optional[float] = None) -> List[Waypoint]:
    """Nadir-looking sweep at top_z + standoff over the area box; lane spacing =
    swath(standoff) * (1 - overlap). Sees roofs and open ground, NOT facades.
    If max_range_m is given, the standoff is reduced so the ground at the beam edge stays
    within LAWNMOWER_RANGE_MARGIN of the sonar range (never below the minimum roof
    clearance; raises ValueError if that is impossible)."""
    if max_range_m is not None:
        reach = LAWNMOWER_RANGE_MARGIN * max_range_m * math.cos(beam_half_rad)
        standoff_m = min(standoff_m, reach - top_z)
        if standoff_m < LAWNMOWER_MIN_ROOF_CLEARANCE_M:
            raise ValueError("ground out of sonar range above these roofs; split the area or map ground separately")
    lane = swath_width(standoff_m, beam_half_rad) * (1.0 - overlap)
    z = top_z + standoff_m
    x0, y0 = area_min[0], area_min[1]
    x1, y1 = area_max[0], area_max[1]
    n = max(int(math.ceil((y1 - y0) / lane)) + 1, 2)
    ys = np.linspace(y0, y1, n)
    pts = []
    for i, y in enumerate(ys):
        a, b = (x0, x1) if i % 2 == 0 else (x1, x0)
        pts += [np.array([a, y, z]), np.array([b, y, z])]
    down = np.array([0.0, 0.0, -1.0])
    return [Waypoint(p, down) for p in _densify(pts, step_m)]


def _nearest_on_box_2d(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([np.clip(p[0], lo[0], hi[0]), np.clip(p[1], lo[1], hi[1])])


def perimeter_orbit(box_min: Sequence[float], box_max: Sequence[float], standoff_m: float,
                    beam_half_rad: float, overlap: float = 0.3, z_min: float = 0.5,
                    step_m: float = 0.5) -> List[Waypoint]:
    """Loops at standoff around the box footprint (rounded corners), sonar looking at the
    nearest facade point; altitude levels spaced by swath * (1 - overlap) from z_min to the top."""
    lo, hi = np.asarray(box_min, float), np.asarray(box_max, float)
    sw = swath_width(standoff_m, beam_half_rad) * (1.0 - overlap)
    n_lv = max(int(math.ceil((hi[2] - z_min) / sw)), 1)
    levels = np.linspace(z_min + 0.5 * sw, max(hi[2] - 0.5 * sw, z_min + 0.5 * sw), n_lv)
    ring = []
    corners = [(hi[0], lo[1], -math.pi / 2), (hi[0], hi[1], 0.0), (lo[0], hi[1], math.pi / 2),
               (lo[0], lo[1], math.pi)]
    for cx, cy, a0 in corners:
        for a in np.linspace(a0, a0 + math.pi / 2, 7):
            ring.append(np.array([cx + standoff_m * math.cos(a), cy + standoff_m * math.sin(a)]))
    ring.append(ring[0])
    wps = []
    for z in levels:
        for p2 in _densify([np.array([*q, z]) for q in ring], step_m):
            n2 = _nearest_on_box_2d(p2, lo, hi)
            d = np.array([n2[0] - p2[0], n2[1] - p2[1], 0.0])
            wps.append(Waypoint(p2, d / max(np.linalg.norm(d), 1e-9)))
    return wps


def poi_close_pass(poi_center: Sequence[float], poi_half_xy: float, poi_top_z: float,
                   standoff_m: float, beam_half_rad: float, overlap: float = 0.3,
                   z_min: float = 0.5, step_m: float = 0.3) -> List[Waypoint]:
    """Full circles around a point of interest at a SHORT standoff from its surface,
    sonar looking at the POI axis; levels spaced by swath * (1 - overlap)."""
    c = np.asarray(poi_center, float)
    R = poi_half_xy * math.sqrt(2.0) + standoff_m
    sw = swath_width(standoff_m, beam_half_rad) * (1.0 - overlap)
    n_lv = max(int(math.ceil((poi_top_z - z_min) / sw)), 1)
    levels = np.linspace(z_min + 0.5 * sw, max(poi_top_z - 0.5 * sw, z_min + 0.5 * sw), n_lv)
    n_ang = max(int(math.ceil(2 * math.pi * R / step_m)), 12)
    wps = []
    for z in levels:
        for a in np.linspace(0.0, 2 * math.pi, n_ang + 1):
            p = np.array([c[0] + R * math.cos(a), c[1] + R * math.sin(a), z])
            d = np.array([c[0] - p[0], c[1] - p[1], 0.0])
            wps.append(Waypoint(p, d / np.linalg.norm(d)))
    return wps


class AuthorizationError(PermissionError):
    pass


def interior_room_sweep(room_min: Sequence[float], room_max: Sequence[float], standoff_m: float,
                        beam_half_rad: float, mission_config: Dict, overlap: float = 0.3,
                        step_m: float = 0.5) -> List[Waypoint]:
    """AUTHORIZED interior mapping: the drone flies INSIDE the room and maps its walls with
    the same sonar grid. Refuses to plan unless mission_config["interior_authorized"] is True
    (the operator confirms the owner's permission). This is not through-wall sensing:
    nothing behind a wall is observed."""
    if mission_config.get("interior_authorized") is not True:
        raise AuthorizationError("interior mapping requires mission_config['interior_authorized'] = True "
                                 "(owner's permission); refusing to plan")
    lo, hi = np.asarray(room_min, float), np.asarray(room_max, float)
    inner_lo = lo + np.array([standoff_m, standoff_m, 0.0])
    inner_hi = hi - np.array([standoff_m, standoff_m, 0.0])
    if np.any(inner_hi[:2] <= inner_lo[:2]):
        raise ValueError("room too small for this standoff")
    z = 0.5 * (lo[2] + hi[2])
    ring = [np.array([inner_lo[0], inner_lo[1], z]), np.array([inner_hi[0], inner_lo[1], z]),
            np.array([inner_hi[0], inner_hi[1], z]), np.array([inner_lo[0], inner_hi[1], z]),
            np.array([inner_lo[0], inner_lo[1], z])]
    wps = []
    sw = swath_width(standoff_m, beam_half_rad) * (1.0 - overlap)
    for p in _densify(ring, step_m):
        # look at the nearest wall, then sweep the head up and down (ASSUMED gimbal) so the
        # floor/ceiling band is covered; the sweep is encoded as extra waypoints
        dists = [p[0] - lo[0], hi[0] - p[0], p[1] - lo[1], hi[1] - p[1]]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = dirs[int(np.argmin(dists))]
        dmin = min(dists)
        n_el = max(int(math.ceil((hi[2] - lo[2]) / max(sw, 1e-3))), 1)
        for zt in np.linspace(lo[2] + 0.5 * sw, hi[2] - 0.5 * sw, n_el):
            v = np.array([dx * dmin, dy * dmin, zt - z])
            wps.append(Waypoint(p.copy(), v / np.linalg.norm(v)))
    # corners: the perpendicular looks above leave ~standoff-wide strips next to each corner
    # unobserved, so at each ring corner the head pans across the corner
    pan_step = 2.0 * beam_half_rad * (1.0 - overlap)
    n_el = max(int(math.ceil((hi[2] - lo[2]) / max(sw, 1e-3))), 1)
    for cx, cy in ((inner_lo[0], inner_lo[1]), (inner_hi[0], inner_lo[1]),
                   (inner_hi[0], inner_hi[1]), (inner_lo[0], inner_hi[1])):
        wx = lo[0] if cx == inner_lo[0] else hi[0]
        wy = lo[1] if cy == inner_lo[1] else hi[1]
        a0 = math.atan2(0.0, wx - cx)            # toward the x wall
        a1 = math.atan2(wy - cy, 0.0)            # toward the y wall
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        n_pan = max(int(math.ceil(abs(da) / pan_step)), 1)
        for a in np.linspace(a0, a0 + da, n_pan + 1):
            for zt in np.linspace(lo[2] + 0.5 * sw, hi[2] - 0.5 * sw, n_el):
                horiz = np.array([math.cos(a), math.sin(a)])
                # distance to the wall hit along this heading
                tx = (wx - cx) / horiz[0] if abs(horiz[0]) > 1e-9 else np.inf
                ty = (wy - cy) / horiz[1] if abs(horiz[1]) > 1e-9 else np.inf
                dist = min(t for t in (tx, ty) if t > 0)
                v = np.array([horiz[0] * dist, horiz[1] * dist, zt - z])
                wps.append(Waypoint(np.array([cx, cy, z]), v / np.linalg.norm(v)))
    # floor and ceiling: lanes at mid height, looking straight down and straight up
    h = 0.5 * (hi[2] - lo[2])
    lane = swath_width(h, beam_half_rad) * (1.0 - overlap)
    n = max(int(math.ceil((hi[1] - lo[1]) / lane)), 1)
    for look in (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])):
        pts = []
        for i, y in enumerate(np.linspace(lo[1] + 0.5 * lane, hi[1] - 0.5 * lane, n)):
            a, b = (inner_lo[0], inner_hi[0]) if i % 2 == 0 else (inner_hi[0], inner_lo[0])
            y = float(np.clip(y, inner_lo[1], inner_hi[1]))
            pts += [np.array([a, y, z]), np.array([b, y, z])]
        wps += [Waypoint(q, look) for q in _densify(pts, step_m)]
    return wps


TEMPLATES = ("lawnmower", "perimeter_orbit", "poi_close_pass", "interior_room_authorized")


# ===========================================================================
# Magnetic disturbance log (flight-path only; NOT interior sensing)
# ===========================================================================
class MagDisturbanceLog:
    """
    Logs where the measured field magnitude or dip departs from the site model by more
    than the thresholds, AT THE DRONE'S POSITION. Label: "possible wiring/steel nearby".
    It does not locate anything behind a wall; external magnetic sensing only picks up
    currents/steel very close to the sensor.
    """
    LABEL = "possible wiring/steel nearby (field disturbance at drone position)"

    def __init__(self, field_ut: float, dip_rad: float, field_thr_frac: float = 0.10,
                 dip_thr_rad: float = math.radians(5.0)):
        self.field_ut, self.dip_rad = field_ut, dip_rad
        self.field_thr, self.dip_thr = field_thr_frac * field_ut, dip_thr_rad
        self.events: List[Dict] = []

    def observe(self, t: float, pos: np.ndarray, field_ut: Optional[float], dip_rad: Optional[float]) -> bool:
        if field_ut is None or dip_rad is None:
            return False
        df, dd = field_ut - self.field_ut, dip_rad - self.dip_rad
        if abs(df) > self.field_thr or abs(dd) > self.dip_thr:
            self.events.append({"t": round(float(t), 3), "pos": [round(float(x), 2) for x in pos],
                                "field_dev_ut": round(float(df), 2), "dip_dev_deg": round(math.degrees(dd), 2),
                                "label": self.LABEL})
            return True
        return False


# ===========================================================================
# 5. Known scenes: ray casting, surface samples, coverage
# ===========================================================================
@dataclass
class Scene:
    boxes: List[Tuple[np.ndarray, np.ndarray]]       # axis-aligned (min, max)
    area_min: np.ndarray
    area_max: np.ndarray
    ground: bool = True
    names: List[str] = field(default_factory=list)
    rooms: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)   # hollow, seen from inside

    def raycast(self, o: np.ndarray, u: np.ndarray, max_r: float) -> Optional[float]:
        for lo, hi in self.rooms:     # inside a room: the ray ends at the inner wall
            if np.all(o > lo) and np.all(o < hi):
                with np.errstate(divide="ignore"):
                    t = np.where(u > 1e-12, (hi - o) / np.where(u > 1e-12, u, 1.0),
                                 np.where(u < -1e-12, (lo - o) / np.where(u < -1e-12, u, 1.0), np.inf))
                tt = float(np.min(t))
                return tt if tt <= max_r else None
        best = None
        for lo, hi in self.boxes:
            t0, t1 = 0.0, max_r
            ok = True
            for a in range(3):
                if abs(u[a]) < 1e-12:
                    if o[a] < lo[a] or o[a] > hi[a]:
                        ok = False
                        break
                    continue
                ta, tb = (lo[a] - o[a]) / u[a], (hi[a] - o[a]) / u[a]
                if ta > tb:
                    ta, tb = tb, ta
                t0, t1 = max(t0, ta), min(t1, tb)
                if t0 > t1:
                    ok = False
                    break
            if ok and t0 > 1e-6 and (best is None or t0 < best):
                best = t0
        if self.ground and u[2] < -1e-9:
            tg = -o[2] / u[2]
            if 0 < tg <= max_r and (best is None or tg < best):
                q = o + tg * u
                if self.area_min[0] - 20 <= q[0] <= self.area_max[0] + 20 and \
                   self.area_min[1] - 20 <= q[1] <= self.area_max[1] + 20:
                    best = tg
        return best

    def surface_distance(self, Q: np.ndarray) -> np.ndarray:
        """Distance of each point to the nearest true surface (box faces or ground)."""
        Q = np.asarray(Q, float).reshape(-1, 3)
        d = np.abs(Q[:, 2]) if self.ground else np.full(len(Q), np.inf)
        for lo, hi in self.rooms:
            d = np.minimum(d, np.min(np.abs(np.concatenate([Q - lo, hi - Q], axis=1)), axis=1))
        for lo, hi in self.boxes:
            outside = np.maximum(np.maximum(lo - Q, Q - hi), 0.0)
            d_out = np.linalg.norm(outside, axis=1)
            inside = np.all((Q >= lo) & (Q <= hi), axis=1)
            d_in = np.min(np.minimum(Q - lo, hi - Q), axis=1)
            d = np.minimum(d, np.where(inside, d_in, d_out))
        return d


def box_surface_samples(lo: np.ndarray, hi: np.ndarray, spacing: float, faces: str = "sides+top"
                        ) -> np.ndarray:
    """Samples on the vertical faces (and the top if requested) of a box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = []
    zs = np.arange(lo[2] + spacing / 2, hi[2], spacing)
    if "sides" in faces:
        for x in np.arange(lo[0] + spacing / 2, hi[0], spacing):
            for z in zs:
                out += [(x, lo[1], z), (x, hi[1], z)]
        for y in np.arange(lo[1] + spacing / 2, hi[1], spacing):
            for z in zs:
                out += [(lo[0], y, z), (hi[0], y, z)]
    if "top" in faces:
        for x in np.arange(lo[0] + spacing / 2, hi[0], spacing):
            for y in np.arange(lo[1] + spacing / 2, hi[1], spacing):
                out.append((x, y, hi[2]))
    return np.array(out, float).reshape(-1, 3)


def room_samples(lo: np.ndarray, hi: np.ndarray, spacing: float) -> np.ndarray:
    """Samples on the inner walls, floor and ceiling of a room."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    S = [box_surface_samples(lo, hi, spacing, "sides+top")]
    xs = np.arange(lo[0] + spacing / 2, hi[0], spacing)
    ys = np.arange(lo[1] + spacing / 2, hi[1], spacing)
    S.append(np.array([(x, y, lo[2]) for x in xs for y in ys]).reshape(-1, 3))
    return np.concatenate(S)


def ground_samples(area_min, area_max, boxes, spacing: float) -> np.ndarray:
    xs = np.arange(area_min[0] + spacing / 2, area_max[0], spacing)
    ys = np.arange(area_min[1] + spacing / 2, area_max[1], spacing)
    G = np.array([(x, y, 0.0) for x in xs for y in ys]).reshape(-1, 3)
    keep = np.ones(len(G), bool)
    for lo, hi in boxes:   # ground under a box is not observable
        keep &= ~((G[:, 0] >= lo[0]) & (G[:, 0] <= hi[0]) & (G[:, 1] >= lo[1]) & (G[:, 1] <= hi[1]))
    return G[keep]


def coverage(samples: np.ndarray, points: np.ndarray, tol: float = COVERAGE_TOL_M) -> float:
    """Fraction of true surface samples with at least one reconstructed point within tol."""
    samples = np.asarray(samples, float).reshape(-1, 3)
    points = np.asarray(points, float).reshape(-1, 3)
    if len(samples) == 0:
        return float("nan")
    if len(points) == 0:
        return 0.0
    cell = tol
    buckets: Dict[Key, List[int]] = {}
    keys = np.floor(points / cell).astype(int)
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    hit = 0
    for s in samples:
        k0 = np.floor(s / cell).astype(int)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    idx = buckets.get((k0[0] + dx, k0[1] + dy, k0[2] + dz))
                    if idx and np.min(np.linalg.norm(points[idx] - s, axis=1)) <= tol:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        hit += int(found)
    return hit / len(samples)

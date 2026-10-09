"""
REAL MODE: Sonar SLAM layer (translation-only) for drone_core_v2 + drone_recon_v2.

What it estimates: the slowly varying position error of the core's estimate
(dominated by correlated GPS error), c(t) = p_core(t) - p_true(t), sampled at
keyframes (one per SLAM_KF_DT_S of sonar pings). Heading comes from the core's
magnetometer/gyro filter and is NOT re-estimated here (translation-only SLAM).

1. Keyframes: the pings of each keyframe interval are stored relative to the core's
   pose (core frame); within one keyframe c is treated as constant.
2. Local surface normals (PCA of nearby points of the last few keyframes); only
   clearly planar points (thin and two-dimensional neighbourhoods) become
   registration targets. Corners, edges, multipath spikes are not used.
3. Scan-to-map registration: point-to-plane ICP of a keyframe's points against the
   planar points of all EARLIER keyframes (corrected with their current estimates).
   Revisits of earlier passes (perimeter rings, lawnmower lanes, POI circles) give
   loop-closure constraints through the same step.
4. Degeneracy: the 3x3 information matrix of the registration is eigen-decomposed;
   only directions with enough information (relative AND absolute thresholds) are
   used. A single flat wall constrains only its normal: the along-track and
   vertical directions are reported unobservable and get no correction.
5. Each registration becomes linear constraints n.(c_k - c_j) = n.(s - m) between
   keyframe k and the earlier keyframes j that own the matched map points, with an
   information matrix scaled down to at most SLAM_N_EFF effective independent
   matches (point errors within one scan are correlated: heading, plane fit).
6. Absolute anchor: c is modelled as the ASSUMED GPS error process (Gauss-Markov,
   zero mean, SLAM_GPS_SIGMA_M, SLAM_GPS_TAU_S). The map registration fixes the map's
   SHAPE; its absolute position can only come from averaging GPS over the mission,
   so the absolute map offset stays of the order of the mission-mean GPS error.
7. Back-end: linear least squares over all keyframes (dense solve; size 3 x number
   of keyframes), giving the online (filtered) estimate after each keyframe and a
   smoothed estimate at mission end, both with covariance. The map is re-integrated
   with the smoothed poses at the end (rebuild_map).

This layer runs on the mapping worker. It never runs inside
AutonomousDroneCore.process_flight_tick and never changes the flight estimate; the
corrected position (corrected_position) is published for the map with its covariance.

All numeric settings are ASSUMED (labelled). Dependencies: numpy only.
License: CC0 1.0 Universal (public domain). Copy, modify, use freely.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SLAM_KF_DT_S = 1.0            # ASSUMED keyframe length (s)
SLAM_GPS_SIGMA_M = 0.5        # ASSUMED sigma of the core's GPS-driven position error (GPS spec)
SLAM_GPS_TAU_S = 30.0         # ASSUMED correlation time of that error (GPS spec)
NORMAL_K = 12                 # neighbours for the local plane fit
NORMAL_MAX_R_M = 1.5          # neighbours farther than this are ignored
NORMAL_MIN_PTS = 6
PLANAR_MAX_THICK_M = 0.08     # ASSUMED: sqrt(smallest PCA eigenvalue) must be below this
PLANAR_MAX_RATIO = 0.10       # smallest / middle PCA eigenvalue
PLANAR_MIN_LINE_RATIO = 0.10  # middle / largest PCA eigenvalue: not a line of points
NORMAL_HISTORY_KF = 3         # keyframes used for normals (this one + previous)
ICP_ITERS = 10
ICP_GATE_M = (1.5, 0.5)       # nearest-neighbour gate, first -> last iteration
ICP_PLANE_GATE_M = 0.35       # point-to-plane residual gate
ICP_MIN_MATCHES = 15            # tuned on the tuning scenes (25 -> 15)
NORMAL_AGREE_COS = math.cos(math.radians(25.0))   # scan and map normals must agree (corners/edges)
SIGMA_PLANE_M = 0.03          # ASSUMED plane roughness / plane-fit error
SLAM_N_EFF = 15.0             # ASSUMED cap on effective independent matches per registration (tuned: 30 -> 15)
OBS_REL = 0.05                # direction observable if eigenvalue >= OBS_REL * largest ...
OBS_MAX_SIGMA_M = 0.10        # ... and the implied (scaled) sigma along it is below this
CONSIST_CHI2 = 16.27          # 3-dof 99.9% gate: registration vs prediction
MAX_SCAN_PTS = 250
MAX_REF_PTS = 6000
BBOX_MARGIN_M = 2.0


@dataclass
class Keyframe:
    idx: int
    t0: float
    t1: float
    t_mid: float = 0.0
    pings: List[Tuple[float, np.ndarray, list]] = field(default_factory=list)   # (t, origin_core, beams)
    pts: Optional[np.ndarray] = None       # (N,3) core frame
    covs: Optional[np.ndarray] = None      # (N,3,3) point covariance without pose
    normals: Optional[np.ndarray] = None
    planar: Optional[np.ndarray] = None
    lo: Optional[np.ndarray] = None        # bbox (core frame)
    hi: Optional[np.ndarray] = None


def local_normals(Q: np.ndarray, Ref: np.ndarray, k: Optional[int] = None, max_r: Optional[float] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Normals of the points Q from a PCA of their nearest neighbours in Ref.
    Returns (normals (N,3), planar mask (N,))."""
    k = NORMAL_K if k is None else k
    max_r = NORMAL_MAX_R_M if max_r is None else max_r
    n = len(Q)
    if n == 0 or len(Ref) < NORMAL_MIN_PTS:
        return np.zeros((n, 3)), np.zeros(n, bool)
    D2 = ((Q[:, None, :] - Ref[None, :, :]) ** 2).sum(-1)
    kk = min(k, Ref.shape[0])
    idx = np.argpartition(D2, kk - 1, axis=1)[:, :kk]
    d2 = np.take_along_axis(D2, idx, 1)
    w = (d2 <= max_r * max_r).astype(float)
    cnt = w.sum(1)
    Nb = Ref[idx]                                               # (n,k,3)
    mu = (Nb * w[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]
    X = (Nb - mu[:, None, :]) * w[..., None]
    C = np.einsum("nki,nkj->nij", X, X) / np.maximum(cnt, 1)[:, None, None]
    ev, evec = np.linalg.eigh(C)
    normals = evec[:, :, 0]
    l0, l1, l2 = np.maximum(ev[:, 0], 0.0), np.maximum(ev[:, 1], 1e-12), np.maximum(ev[:, 2], 1e-12)
    planar = ((cnt >= NORMAL_MIN_PTS) & (np.sqrt(l0) <= PLANAR_MAX_THICK_M)
              & (l0 <= PLANAR_MAX_RATIO * l1) & (l1 >= PLANAR_MIN_LINE_RATIO * l2))
    return normals, planar


def observable_basis(A: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-directions of the information matrix A that are observable.
    Returns (V (3,m) observable directions, eigenvalues (m,)) after scaling A by scale."""
    ev, V = np.linalg.eigh(0.5 * (A + A.T) * scale)
    if ev[-1] <= 0:
        return np.zeros((3, 0)), np.zeros(0)
    keep = (ev >= OBS_REL * ev[-1]) & (ev >= 1.0 / OBS_MAX_SIGMA_M ** 2)
    return V[:, keep], ev[keep]


def register_translation(S: np.ndarray, S_cov: np.ndarray, M: np.ndarray, M_n: np.ndarray,
                         M_cov: np.ndarray, c0: np.ndarray, iters: int = ICP_ITERS,
                         S_n: Optional[np.ndarray] = None) -> Dict:
    """
    Point-to-plane ICP for a pure translation. Scan points S (core frame, corrected
    position = S - c) against map points M (already corrected) with normals M_n.
    If scan normals S_n are given, a match also needs |n_scan . n_map| >= NORMAL_AGREE_COS
    (stops points on one face matching the plane of the neighbouring face at an edge).
    Returns dict(c, A (unscaled 3x3 information), V_obs, n_obs, idx_s, idx_m, ok).
    Unobservable directions keep the initial value c0.
    """
    c = np.asarray(c0, float).copy()
    out = dict(c=c, A=np.zeros((3, 3)), V_obs=np.zeros((3, 0)), n_obs=0, idx_s=np.zeros(0, int),
               idx_m=np.zeros(0, int), w=np.zeros(0), ok=False, n_match=0)
    if len(S) < ICP_MIN_MATCHES or len(M) < ICP_MIN_MATCHES:
        return out
    gates = np.linspace(ICP_GATE_M[0], ICP_GATE_M[1], iters)
    MM = (M ** 2).sum(1)
    for it in range(iters):
        Sc = S - c
        D2 = (Sc ** 2).sum(1)[:, None] + MM[None, :] - 2.0 * Sc @ M.T
        j = np.argmin(D2, axis=1)
        d = np.sqrt(np.maximum(D2[np.arange(len(S)), j], 0.0))
        nrm = M_n[j]
        r = np.einsum("ij,ij->i", nrm, Sc - M[j])
        sel = (d <= gates[it]) & (np.abs(r) <= ICP_PLANE_GATE_M)
        if S_n is not None:
            sel &= np.abs(np.einsum("ij,ij->i", S_n, nrm)) >= NORMAL_AGREE_COS
        if sel.sum() < ICP_MIN_MATCHES:
            return out
        ns, rs = nrm[sel], r[sel]
        var = (np.einsum("ni,nij,nj->n", ns, S_cov[sel], ns) + np.einsum("ni,nij,nj->n", ns, M_cov[j[sel]], ns)
               + SIGMA_PLANE_M ** 2)
        # robust (Cauchy-like) down-weighting of large residuals
        w = 1.0 / var / (1.0 + (rs ** 2) / (4.0 * var))
        A = np.einsum("n,ni,nj->ij", w, ns, ns)
        g = np.einsum("n,ni->i", w * rs, ns)
        V, ev = observable_basis(A, 1.0)
        if V.shape[1] == 0:
            return out
        delta = V @ ((V.T @ g) / ev)
        c = c + delta
        if np.linalg.norm(delta) < 1e-3 and it >= 2:
            break
    out.update(c=c, A=A, V_obs=V, n_obs=V.shape[1], idx_s=np.nonzero(sel)[0], idx_m=j[sel], w=w,
               ok=True, n_match=int(sel.sum()))
    return out


class SonarSLAM:
    """Translation-only sonar SLAM over keyframes; see module docstring."""

    def __init__(self, gps_sigma_m: float = SLAM_GPS_SIGMA_M, gps_tau_s: float = SLAM_GPS_TAU_S,
                 kf_dt_s: float = SLAM_KF_DT_S, n_eff: float = SLAM_N_EFF, loop_closure: bool = True,
                 min_kf_gap_for_loop: int = 0):
        self.sig = float(gps_sigma_m)
        self.tau = float(gps_tau_s)
        self.dt = float(kf_dt_s)
        self.n_eff = float(n_eff)
        self.loop_closure = loop_closure      # False: register only against the previous keyframes (odometry)
        self.kfs: List[Keyframe] = []
        self.cur: Optional[Keyframe] = None
        self.factors: List[Tuple[int, int, np.ndarray, np.ndarray]] = []   # (k, j, A, b)
        self.est = np.zeros((0, 3))
        self.cov_online: List[np.ndarray] = []
        self.cov_rel_online: List[np.ndarray] = []
        self._A_reg = np.zeros((3, 3))
        self.est_online: List[np.ndarray] = []
        self.stats = dict(registrations=0, rejected_few=0, rejected_degenerate=0, rejected_inconsistent=0,
                          n_obs_hist=[0, 0, 0, 0], step_ms=[], loop_factors=0)
        self._H = np.zeros((0, 0))
        self._g = np.zeros(0)
        self._P_last = np.eye(3) * self.sig ** 2

    # ------------------------------------------------------------------ input
    def add_ping(self, t: float, origin_core: np.ndarray, beams: Sequence[Tuple[np.ndarray, float, np.ndarray]]
                 ) -> Optional[int]:
        """beams: (unit ray in the world/ENU frame from the core heading, range, point covariance
        WITHOUT the position covariance). Returns the index of a keyframe closed by this ping."""
        closed = None
        if self.cur is None:
            self.cur = Keyframe(0, t, t + self.dt)
        while t >= self.cur.t1:
            closed = self._close_keyframe()
            self.cur = Keyframe(len(self.kfs), self.cur.t1, self.cur.t1 + self.dt)
        self.cur.pings.append((float(t), np.asarray(origin_core, float).copy(), list(beams)))
        return closed

    def finish(self) -> Optional[int]:
        if self.cur is not None and self.cur.pings:
            k = self._close_keyframe()
            self.cur = None
            return k
        return None

    # --------------------------------------------------------------- keyframes
    def _close_keyframe(self) -> Optional[int]:
        t0 = time.perf_counter()
        kf = self.cur
        kf.idx = len(self.kfs)
        pts, covs = [], []
        for t, o, beams in kf.pings:
            for u, r, cov in beams:
                pts.append(o + r * np.asarray(u, float))
                covs.append(cov)
        kf.t_mid = 0.5 * (kf.t0 + kf.t1)
        kf.pts = np.array(pts).reshape(-1, 3)
        kf.covs = np.array(covs).reshape(-1, 3, 3)
        if len(kf.pts):
            kf.lo, kf.hi = kf.pts.min(0), kf.pts.max(0)
        else:
            kf.lo = kf.hi = np.zeros(3)
        self.kfs.append(kf)
        k = kf.idx
        self._grow(k + 1)
        # prior / process factor
        phi = math.exp(-self.dt / self.tau)
        if k == 0:
            self._add_block(0, 0, np.eye(3) / self.sig ** 2)
        else:
            W = np.eye(3) / (self.sig ** 2 * (1.0 - phi * phi))
            self._add_block(k - 1, k - 1, phi * phi * W)
            self._add_block(k, k, W)
            self._add_block(k - 1, k, -phi * W)
            self._add_block(k, k - 1, -phi * W)
        c_pred = phi * self.est[k - 1] if k > 0 else np.zeros(3)
        P_pred = (phi * phi * self._P_last + (1 - phi * phi) * self.sig ** 2 * np.eye(3)) if k > 0 \
            else self.sig ** 2 * np.eye(3)
        self._A_reg = np.zeros((3, 3))
        # local surface normals of this keyframe (with the predicted correction and the previous
        # keyframes' current estimates); planar points are registration inputs and later targets
        if len(kf.pts):
            refs = [kf.pts - c_pred] + [self.kfs[j].pts - self.est[j]
                                        for j in range(max(0, k - NORMAL_HISTORY_KF + 1), k)]
            kf.normals, kf.planar = local_normals(kf.pts - c_pred, np.concatenate(refs))
        else:
            kf.normals, kf.planar = np.zeros((0, 3)), np.zeros(0, bool)
        if k > 0 and len(kf.pts):
            self._register(kf, c_pred, P_pred)
        self._solve(k)
        # local relative covariance (keyframe vs. the map it was registered to, or vs. the
        # previous keyframe in unregistered directions); used for the online map footprint
        W = np.eye(3) / (self.sig ** 2 * (1.0 - phi * phi)) if k > 0 else np.eye(3) / self.sig ** 2
        self.cov_rel_online.append(np.linalg.inv(self._A_reg + W))
        self.stats["step_ms"].append((time.perf_counter() - t0) * 1000.0)
        return k

    def _register(self, kf: Keyframe, c_pred: np.ndarray, P_pred: np.ndarray) -> None:
        k = kf.idx
        S, S_cov, S_n = kf.pts[kf.planar], kf.covs[kf.planar], kf.normals[kf.planar]
        if len(S) < ICP_MIN_MATCHES:
            self.stats["rejected_few"] += 1
            return
        if len(S) > MAX_SCAN_PTS:
            sel = np.linspace(0, len(S) - 1, MAX_SCAN_PTS).astype(int)
            S, S_cov, S_n = S[sel], S_cov[sel], S_n[sel]
        lo, hi = S.min(0) - c_pred - BBOX_MARGIN_M, S.max(0) - c_pred + BBOX_MARGIN_M
        Ms, Mn, Mc, Mk = [], [], [], []
        js = range(k) if self.loop_closure else range(max(0, k - NORMAL_HISTORY_KF), k)
        for j in js:
            o = self.kfs[j]
            if o.planar is None or not o.planar.any():
                continue
            clo, chi = o.lo - self.est[j], o.hi - self.est[j]
            if np.any(chi < lo) or np.any(clo > hi):
                continue
            m = o.planar
            Ms.append(o.pts[m] - self.est[j]); Mn.append(o.normals[m]); Mc.append(o.covs[m])
            Mk.append(np.full(int(m.sum()), j))
        if not Ms:
            self.stats["rejected_few"] += 1
            return
        M, Mn, Mc, Mk = np.concatenate(Ms), np.concatenate(Mn), np.concatenate(Mc), np.concatenate(Mk)
        if len(M) > MAX_REF_PTS:
            sel = np.random.default_rng(k).choice(len(M), MAX_REF_PTS, replace=False)
            M, Mn, Mc, Mk = M[sel], Mn[sel], Mc[sel], Mk[sel]
        res = register_translation(S, S_cov, M, Mn, Mc, c_pred, S_n=S_n)
        if not res["ok"]:
            self.stats["rejected_few"] += 1
            return
        n_match = res["n_match"]
        scale = min(1.0, self.n_eff / max(n_match, 1))
        V, ev = observable_basis(res["A"], scale)
        self.stats["n_obs_hist"][V.shape[1]] += 1
        if V.shape[1] == 0:
            self.stats["rejected_degenerate"] += 1
            return
        # consistency with the prediction, in the observable subspace
        d = V.T @ (res["c"] - c_pred)
        Sg = V.T @ P_pred @ V + np.diag(1.0 / ev)
        if float(d @ np.linalg.solve(Sg, d)) > CONSIST_CHI2:
            self.stats["rejected_inconsistent"] += 1
            return
        self.stats["registrations"] += 1
        Pobs = V @ V.T                                     # projector onto observable directions
        iS, iM, w = res["idx_s"], res["idx_m"], res["w"]
        ns = Mn[iM]
        # residual model: n.(c_k - c_j) = n.(s_core - m_core); m_core = m_corrected + c_j(est)
        m_core = M[iM] + self.est[Mk[iM]]
        rhs = np.einsum("ij,ij->i", ns, S[iS] - m_core)
        for j in np.unique(Mk[iM]):
            sel = Mk[iM] == j
            A = np.einsum("n,ni,nj->ij", w[sel], ns[sel], ns[sel]) * scale
            b = np.einsum("n,ni->i", w[sel] * rhs[sel], ns[sel]) * scale
            A = Pobs @ A @ Pobs
            b = Pobs @ b
            self.factors.append((k, int(j), A, b))
            if k - j > NORMAL_HISTORY_KF:
                self.stats["loop_factors"] += 1
            self._add_block(k, k, A); self._add_block(int(j), int(j), A)
            self._add_block(k, int(j), -A); self._add_block(int(j), k, -A)
            self._g[3 * k:3 * k + 3] += b
            self._A_reg += A
            self._g[3 * j:3 * j + 3] -= b

    # ----------------------------------------------------------------- back-end
    def _grow(self, n: int) -> None:
        cap = self._H.shape[0] // 3
        if n > cap:
            new = max(n, 2 * cap, 64)
            H = np.zeros((3 * new, 3 * new)); g = np.zeros(3 * new)
            H[:3 * cap, :3 * cap] = self._H; g[:3 * cap] = self._g
            self._H, self._g = H, g
        if self.est.shape[0] < n:
            self.est = np.vstack([self.est, np.zeros((n - self.est.shape[0], 3))])

    def _add_block(self, i: int, j: int, B: np.ndarray) -> None:
        self._H[3 * i:3 * i + 3, 3 * j:3 * j + 3] += B

    def _solve(self, k: int) -> None:
        n = 3 * (k + 1)
        H = self._H[:n, :n]
        E = np.zeros((n, 3)); E[n - 3:, :] = np.eye(3)
        X = np.linalg.solve(H, np.column_stack([self._g[:n], E]))
        self.est[:k + 1] = X[:, 0].reshape(-1, 3)
        P = X[n - 3:, 1:]
        self._P_last = 0.5 * (P + P.T)
        self.est_online.append(self.est[k].copy())
        self.cov_online.append(self._P_last.copy())

    def smoothed(self) -> Tuple[np.ndarray, np.ndarray]:
        """(estimates (K,3), covariances (K,3,3)) using all constraints."""
        K = len(self.kfs)
        if K == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3))
        n = 3 * K
        H = self._H[:n, :n]
        Hinv = np.linalg.inv(H)
        c = Hinv @ self._g[:n]
        covs = np.array([Hinv[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(K)])
        return c.reshape(-1, 3), covs

    def smoothed_full(self) -> Dict[str, np.ndarray]:
        """Smoothed estimates with absolute covariances, covariances RELATIVE to the map datum
        (cov of c_k - mean(c), i.e. how well each keyframe sits in the map's shape) and the
        covariance of the datum itself (cov of mean(c): the common absolute offset)."""
        K = len(self.kfs)
        if K == 0:
            z = np.zeros((0, 3, 3))
            return dict(c=np.zeros((0, 3)), P=z, P_rel=z, P_datum=np.eye(3) * self.sig ** 2)
        n = 3 * K
        Hinv = np.linalg.inv(self._H[:n, :n])
        c = (Hinv @ self._g[:n]).reshape(-1, 3)
        B = Hinv.reshape(K, 3, K, 3)
        P = np.array([B[i, :, i, :] for i in range(K)])
        row = B.sum(axis=2) / K                                   # cov(c_k, mean c)  (K,3,3)
        P_datum = B.sum(axis=(0, 2)) / (K * K)
        P_rel = P - row - np.transpose(row, (0, 2, 1)) + P_datum[None]
        P_rel = 0.5 * (P_rel + np.transpose(P_rel, (0, 2, 1)))
        return dict(c=c, P=P, P_rel=P_rel, P_datum=0.5 * (P_datum + P_datum.T))

    def correction_at(self, t: float, smoothed: Optional[Tuple[np.ndarray, np.ndarray]] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Correction c(t) (core position minus true position) and its covariance:
        linear interpolation between keyframe mid-times (online estimate if smoothed is None)."""
        if not self.kfs:
            return np.zeros(3), np.eye(3) * self.sig ** 2
        if smoothed is None:
            C = np.array(self.est_online); P = np.array(self.cov_online)
        else:
            C, P = smoothed
        tm = np.array([k.t_mid for k in self.kfs[:len(C)]])
        if t <= tm[0]:
            return C[0], P[0]
        if t >= tm[-1]:
            return C[-1], P[-1]
        i = int(np.searchsorted(tm, t)) - 1
        a = (t - tm[i]) / max(tm[i + 1] - tm[i], 1e-9)
        return (1 - a) * C[i] + a * C[i + 1], (1 - a) * P[i] + a * P[i + 1]

    def corrected_position(self, t: float, p_core: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c, P = self.correction_at(t)
        return np.asarray(p_core, float) - c, P

    def rebuild_map(self, recon_map, sig_range: float, full: Optional[Dict[str, np.ndarray]] = None):
        """Re-integrate every stored ping into recon_map (a drone_recon_v2.ReconMap) with the
        smoothed corrections. The point covariance gets the position covariance RELATIVE to the
        map datum (the common absolute offset does not blur the map; it is reported once as
        recon_map.datum_cov)."""
        full = self.smoothed_full() if full is None else full
        sm_rel = (full["c"], full["P_rel"])
        for kf in self.kfs:
            for t, o, beams in kf.pings:
                c, P = self.correction_at(t, sm_rel)
                recon_map.add_ping(o - c, [(u, r, cov + P) for u, r, cov in beams], sig_range, t)
        recon_map.datum_cov = full["P_datum"]
        return recon_map


def _self_test() -> None:
    rng = np.random.default_rng(1)
    # two walls + floor: offset recovered
    pts = []
    for _ in range(400):
        f = rng.integers(3)
        a, b = rng.uniform(0, 4, 2)
        pts.append([0.0, a, b] if f == 0 else ([a, 0.0, b] if f == 1 else [a, b, 0.0]))
    M = np.array(pts)
    N, pl = local_normals(M, M)
    assert pl.mean() > 0.7, pl.mean()
    off = np.array([0.3, -0.2, 0.15])
    S = M[::2] + off + rng.normal(0, 0.01, (200, 3))
    cov = np.tile(np.eye(3) * 1e-4, (len(M), 1, 1))
    r = register_translation(S, cov[:200], M[pl], N[pl], cov[pl], np.zeros(3))
    assert r["ok"] and r["n_obs"] == 3 and np.linalg.norm(r["c"] - off) < 0.03, r["c"]
    print("drone_slam_v2 self-test passed")


if __name__ == "__main__":
    _self_test()

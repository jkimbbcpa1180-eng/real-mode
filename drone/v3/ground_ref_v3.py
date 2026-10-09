# SPDX-License-Identifier: CC0-1.0
"""
drone v3: ground-reference mode. Absolute anchors for the v2 sonar SLAM, which on its
own cannot observe the common GPS datum offset (drone_slam_v2 module docstring, item 6).

The SLAM state is c_k = core position - true position at keyframe k (mostly correlated
GPS error, modelled as Gauss-Markov sigma/tau). Anchors are linear factors on c_k:

(a) Surveyed take-off pad. The drone dwells DWELL_S on the pad before take-off and after
    landing. Pad survey error is PAD_SURVEY_SIG_M (RTK / total station), and placement on
    the pad (antenna offset, landing spot) is PAD_PLACE_SIG_M. The dwell-mean of c is
    linked to the first and last keyframes through the GM model, with the exact
    conditional mean a*c_k and variance (averaging window + climb/return gap). So the
    anchor weakens with the time between pad and keyframe: between anchors the GPS error
    still drifts, and the covariance says so.
(b) Ground control markers: N surveyed markers (coded targets, ASSUMED to be detected by a
    downward camera; the sonar cannot identify a marker), within MARKER_RANGE_M slant
    range and MARKER_NADIR_DEG of nadir. Each tick has P_DET of detection and
    P_FALSE_MATCH of a false match (wrong spot 0.5-3 m away). Per keyframe and marker, the
    detections are cleaned around their median and combined into one factor. Its
    covariance is the random noise / n, plus heading error x range (not averaged: it is
    correlated within the keyframe), plus survey error x SURVEY_SHARE (the survey error is
    common to every sighting of that marker, so it must not be averaged away).
    Outliers: online, a per-factor chi-square gate against the SLAM prediction (keeps the
    online map clean). At mission end every factor (gated ones included, since the online
    prediction may itself have been pulled by a bad marker) goes into a leave-one-marker-out
    chi-square test that removes the worst inconsistent marker, repeated; then single
    factors that disagree with the final solution are dropped. A wrong marker is only
    detectable if another anchor constrains the same time (co-visible markers, or SLAM
    links to nearby times); a marker seen alone is not checkable.
(c) Relative mode: the pad's coordinates are taken from the GPS during the dwell (no
    survey). The map is then tied to the take-off point. It is reported as a separate
    pad-relative metric and does not count toward absolute accuracy.

All numbers are ASSUMED. License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import common_v3 as C
from common_v3 import np, R, S
import mc_recon_v2 as MR
import mission_v3 as MV

DWELL_S = 20.0
CLIMB_MPS = 1.0
PAD_SURVEY_SIG_M = 0.025
PAD_PLACE_SIG_M = 0.03
CORE_VS_GPS_SIG_M = 0.05        # core filter error vs raw GPS while sitting on the pad
MARKER_SURVEY_SIG_M = 0.025
MARKER_RANGE_M = 30.0          # 0.5 m coded target, downward camera
MARKER_NADIR_DEG = 50.0
P_DET = 0.3                     # per tick (10 Hz) when in view
P_FALSE_MATCH = 0.03
MEAS_SIG_M = 0.02               # + MEAS_SIG_PER_M * range, per axis
MEAS_SIG_PER_M = 0.01
SURVEY_SHARE = 10.0             # survey variance multiplier per factor (shared error, see above)
FACTOR_FLOOR_SIG_M = 0.05       # per-factor floor (slowly varying heading / timing errors)
CLEAN_R_M = 0.3                 # per-keyframe detections farther than this from their median are dropped
GATE_CHI2 = 11.34               # 3 dof, 99 %
MARKER_CHI2 = 11.34


def gm_window(sig: float, tau: float, lags: np.ndarray) -> Tuple[float, float]:
    """For a stationary GM process: mean over samples at time lags l_i from c_k is
    a*c_k + w, w ~ N(0, v I). Returns (a, v)."""
    l = np.asarray(lags, float)
    a = float(np.mean(np.exp(-l / tau)))
    K = np.exp(-np.abs(l[:, None] - l[None, :]) / tau)
    v = sig * sig * (float(K.mean()) - a * a)
    return a, max(v, 0.0)


class AnchoredSLAM(S.SonarSLAM):
    """SonarSLAM + absolute anchor factors (pad, markers) with outlier rejection."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.pending: Dict[int, List] = {}
        self.anchors: List[Dict] = []
        self.gate_rejects = 0
        self.rejected_markers: List[int] = []
        self.marker_counts: Dict[int, List[int]] = {}     # marker -> [accepted, gated online]
        self.gated: List[Dict] = []
        self.final_factor_rejects = 0

    # -- anchors ---------------------------------------------------------------------
    def _factor(self, k: int, z: np.ndarray, Rm: np.ndarray, a: float, src) -> Dict:
        Ri = np.linalg.inv(Rm)
        f = dict(k=k, z=np.asarray(z, float), R=Rm, a=a, src=src, H=a * a * Ri, g=a * Ri @ z)
        self._add_block(k, k, f["H"]); self._g[3 * k:3 * k + 3] += f["g"]
        self.anchors.append(f)
        return f

    def add_now(self, k: int, z, Rm, a: float = 1.0, src="pad") -> None:
        self._factor(k, z, Rm, a, src)

    def queue_detection(self, m: int, z: np.ndarray, var_rand: float, var_fixed: float) -> None:
        k = len(self.kfs)
        self.pending.setdefault(k, []).append((m, np.asarray(z, float), var_rand, var_fixed))

    def _solve(self, k: int) -> None:
        dets = self.pending.pop(k, [])
        if dets:
            phi = math.exp(-self.dt / self.tau)
            c_pred = phi * self.est[k - 1] if k > 0 else np.zeros(3)
            P_pred = (phi * phi * self._P_last + (1 - phi * phi) * self.sig ** 2 * np.eye(3)) if k > 0 \
                else self.sig ** 2 * np.eye(3)
            for m in sorted({d[0] for d in dets}):
                Z = np.array([d[1] for d in dets if d[0] == m])
                med = np.median(Z, axis=0)
                keep = np.linalg.norm(Z - med, axis=1) <= CLEAN_R_M
                if keep.sum() == 0:
                    continue
                vr = np.mean([d[2] for d in dets if d[0] == m])
                vf = np.mean([d[3] for d in dets if d[0] == m])
                z = Z[keep].mean(0)
                Rm = np.eye(3) * (vr / keep.sum() + vf + FACTOR_FLOOR_SIG_M ** 2)
                r = z - c_pred
                cnt = self.marker_counts.setdefault(m, [0, 0])
                if float(r @ np.linalg.solve(P_pred + Rm, r)) > GATE_CHI2:
                    self.gate_rejects += 1
                    cnt[1] += 1
                    self.gated.append(dict(k=k, z=z, R=Rm, src=("marker", m)))   # re-tested at the end
                    continue
                cnt[0] += 1
                self._factor(k, z, Rm, 1.0, ("marker", m))
        super()._solve(k)

    def _remove(self, f: Dict) -> None:
        k = f["k"]
        self._add_block(k, k, -f["H"]); self._g[3 * k:3 * k + 3] -= f["g"]

    def reject_bad_markers(self) -> List[int]:
        """Leave-one-marker-out test at mission end; removes the worst inconsistent marker
        and repeats. Returns the rejected marker ids."""
        K = len(self.kfs)
        n = 3 * K
        # the online gate may have been fooled by a bad marker seen early: give every marker
        # all its factors back and decide per marker with the leave-one-out test
        for f in self.gated:
            self._factor(f["k"], f["z"], f["R"], 1.0, f["src"])
        self.gated = []
        while True:
            ids = sorted({f["src"][1] for f in self.anchors if isinstance(f["src"], tuple)} - set(self.rejected_markers))
            worst, wchi = None, MARKER_CHI2
            for m in ids:
                fs = [f for f in self.anchors if f["src"] == ("marker", m)]
                H = self._H[:n, :n].copy(); g = self._g[:n].copy()
                for f in fs:
                    k = f["k"]; H[3 * k:3 * k + 3, 3 * k:3 * k + 3] -= f["H"]; g[3 * k:3 * k + 3] -= f["g"]
                Hi = np.linalg.inv(H)
                c = Hi @ g
                r = np.mean([f["z"] - c[3 * f["k"]:3 * f["k"] + 3] for f in fs], axis=0)
                Pm = np.mean([Hi[3 * f["k"]:3 * f["k"] + 3, 3 * f["k"]:3 * f["k"] + 3] + f["R"] for f in fs], axis=0)
                chi = float(r @ np.linalg.solve(Pm, r))
                if chi > wchi:
                    worst, wchi = m, chi
            if worst is None:
                break
            for f in [f for f in self.anchors if f["src"] == ("marker", worst)]:
                self._remove(f)
            self.anchors = [f for f in self.anchors if f["src"] != ("marker", worst)]
            self.rejected_markers.append(worst)
        # remaining markers: drop single factors that disagree with the solution (false matches)
        Hi = np.linalg.inv(self._H[:n, :n])
        c = Hi @ self._g[:n]
        drop = set()
        for i, f in enumerate(self.anchors):
            if not isinstance(f["src"], tuple):
                continue
            k = f["k"]
            r = f["z"] - c[3 * k:3 * k + 3]
            if float(r @ np.linalg.solve(Hi[3 * k:3 * k + 3, 3 * k:3 * k + 3] + f["R"], r)) > GATE_CHI2:
                self._remove(f)
                drop.add(i)
        self.anchors = [f for i, f in enumerate(self.anchors) if i not in drop]
        self.final_factor_rejects += len(drop)
        return list(self.rejected_markers)


# --- scenario helpers ---------------------------------------------------------------
def place_markers(wps: Sequence, n: int, rng) -> np.ndarray:
    """n markers on the ground under the planned path, evenly spaced along it (+-1 m)."""
    P = np.array([w.pos for w in wps])
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for i in range(n):
        si = (i + 0.5) / n * s[-1]
        j = min(int(np.searchsorted(s, si, side="right")) - 1, len(seg) - 1)
        a = (si - s[j]) / max(seg[j], 1e-9)
        q = P[j] + a * (P[j + 1] - P[j])
        out.append(np.array([q[0] + rng.uniform(-1, 1), q[1] + rng.uniform(-1, 1), 0.0]))
    return np.array(out).reshape(-1, 3)


def gm_chain(c0: np.ndarray, sig: float, tau: float, n: int, rng) -> np.ndarray:
    """n further samples (DT apart) of the true GM error after c0 (time-reversible)."""
    phi = math.exp(-MV.DT / tau)
    out, c = [], np.asarray(c0, float).copy()
    for _ in range(n):
        c = phi * c + math.sqrt(1 - phi * phi) * sig * rng.normal(size=3)
        out.append(c.copy())
    return np.array(out)


CONFIGS = ("slam", "pad", "pad_m3", "pad_m5", "relative")


def run(sc: Dict, template: str, gps: Tuple[float, float], seed: int, moved_marker: bool = False,
        configs: Sequence[str] = CONFIGS, slam_sig: float = 0.5, slam_tau: float = 30.0,
        record: bool = False) -> Dict:
    """One mission; every config gets its own SLAM fed with the SAME pings (paired)."""
    rng = np.random.default_rng(seed + 77)
    wps, scene, targets = MR.plan(sc, template)
    f = MV.Flight(sc, scene, gps, seed, wps[0].pos)
    sig_t, tau_t = gps
    pad = np.array([wps[0].pos[0], wps[0].pos[1], 0.0])
    markers = place_markers(wps, 5, rng)
    actual = markers.copy()
    if moved_marker:                                  # marker #1 moved / wrongly surveyed
        ang = rng.uniform(0, 2 * math.pi)
        actual[1] += rng.uniform(0.5, 1.5) * np.array([math.cos(ang), math.sin(ang), 0.0])
    survey = markers + rng.normal(0, MARKER_SURVEY_SIG_M, markers.shape)
    nm = {"slam": 0, "pad": 0, "pad_m3": 3, "pad_m5": 5, "relative": 0}
    sl = {c: AnchoredSLAM(gps_sigma_m=slam_sig, gps_tau_s=slam_tau) for c in configs}
    # --- take-off dwell (before t = 0): backward GM chain from the flight's initial error
    gap_up = wps[0].pos[2] / CLIMB_MPS
    n_dw, n_gap = int(DWELL_S / MV.DT), int(gap_up / MV.DT)
    back = gm_chain(f.gps_err, sig_t, tau_t, n_gap + n_dw, rng)[n_gap:]
    m_up = back.mean(0) + rng.normal(0, CORE_VS_GPS_SIG_M, 3)
    lags_up = (n_gap + 1 + np.arange(n_dw)) * MV.DT
    a_up, v_up = gm_window(slam_sig, slam_tau, lags_up)
    pad_survey = pad + rng.normal(0, PAD_SURVEY_SIG_M, 3)
    place = rng.normal(0, PAD_PLACE_SIG_M, 3)
    z_up_abs = m_up + (pad - pad_survey) + place      # measured core pos - surveyed pad pos
    R_pad = np.eye(3) * (PAD_SURVEY_SIG_M ** 2 + PAD_PLACE_SIG_M ** 2 + CORE_VS_GPS_SIG_M ** 2)
    det_n = np.zeros(5, int); false_n = 0
    tick_t, tick_e = [], []

    def hook(fl, all_hits, p_hat, P, sig_h, dyaw):
        nonlocal false_n
        t = fl.t
        tick_t.append(t); tick_e.append(p_hat - fl.p)
        beams = []
        for sen, bms in all_hits:
            for u, r_m, _ in bms:
                beams.append((u, r_m, R.point_covariance(r_m, u, sen.range_noise, sen.bearing_noise, sig_h, None)))
        if beams:
            for s_ in sl.values():
                s_.add_ping(t, p_hat, beams)
        Rz = MR._rz(dyaw)
        for m in range(5):
            d = actual[m] - fl.p
            r = float(np.linalg.norm(d))
            if r > MARKER_RANGE_M or math.degrees(math.acos(max(-1.0, min(1.0, -d[2] / max(r, 1e-9))))) > MARKER_NADIR_DEG:
                continue
            if fl.srng.random() >= P_DET:
                continue
            sg = MEAS_SIG_M + MEAS_SIG_PER_M * r
            dm = d + fl.srng.normal(0, sg, 3)
            if fl.srng.random() < P_FALSE_MATCH:
                a_ = fl.srng.uniform(0, 2 * math.pi)
                dm = dm + fl.srng.uniform(0.5, 3.0) * np.array([math.cos(a_), math.sin(a_), 0.0])
                false_n += 1
            det_n[m] += 1
            z = p_hat + Rz @ dm - survey[m]
            for c, s_ in sl.items():
                if m < nm[c]:
                    s_.queue_detection(m, z, sg * sg, (r * sig_h) ** 2 + SURVEY_SHARE * MARKER_SURVEY_SIG_M ** 2)

    f.hooks.append(hook)
    # pad factor on keyframe 0 must be in before keyframe 0 is solved: queue via a tiny wrapper
    for c, s_ in sl.items():
        if c in ("pad", "pad_m3", "pad_m5", "relative"):
            z0 = z_up_abs if c != "relative" else (m_up - m_up)
            Rp = R_pad if c != "relative" else np.eye(3) * CORE_VS_GPS_SIG_M ** 2
            s_._pad0 = (z0, Rp / max(a_up, 1e-6) ** 0 + np.eye(3) * v_up, a_up)
            orig = s_._solve

            def _solve(k, s_=s_, orig=orig):
                if k == 0 and getattr(s_, "_pad0", None) is not None:
                    z0_, R0_, a0_ = s_._pad0
                    s_._factor(0, z0_, R0_, a0_, "pad_up"); s_._pad0 = None
                orig(k)
            s_._solve = _solve
    f.fly(wps)
    for s_ in sl.values():
        s_.finish()
    # --- return + landing dwell: forward chain from the last error
    p_end = f.p
    gap_dn = float(np.linalg.norm(p_end[:2] - pad[:2])) / MV.SPEED + p_end[2] / CLIMB_MPS
    n_gap = int(gap_dn / MV.DT)
    fwd = gm_chain(f.gps_err, sig_t, tau_t, n_gap + n_dw, rng)[n_gap:]
    m_dn = fwd.mean(0) + rng.normal(0, CORE_VS_GPS_SIG_M, 3)
    a_dn, v_dn = gm_window(slam_sig, slam_tau, (n_gap + 1 + np.arange(n_dw)) * MV.DT)
    z_dn_abs = m_dn + (pad - pad_survey) + rng.normal(0, PAD_PLACE_SIG_M, 3)
    out = dict(id=sc["id"], template=template, gps=list(gps), moved_marker=moved_marker, mission_s=round(f.t, 1),
               marker_dets=det_n.tolist(), false_matches=int(false_n), pad_gap_up_s=round(gap_up, 1),
               pad_gap_down_s=round(gap_dn, 1))
    tt, te = np.array(tick_t), np.array(tick_e)
    for c, s_ in sl.items():
        K = len(s_.kfs)
        if K == 0:
            continue
        if c in ("pad", "pad_m3", "pad_m5"):
            s_.add_now(K - 1, z_dn_abs, R_pad + np.eye(3) * v_dn, a_dn, "pad_down")
        elif c == "relative":
            s_.add_now(K - 1, m_dn - m_up, np.eye(3) * (2 * CORE_VS_GPS_SIG_M ** 2 + v_dn), a_dn, "pad_down")
        rej = s_.reject_bad_markers()
        full = s_.smoothed_full()
        rm = s_.rebuild_map(R.ReconMap(), MR.RANGE_NOISE, full)
        c_true = np.zeros((K, 3)); ok = np.zeros(K, bool)
        for i, kf in enumerate(s_.kfs):
            msk = (tt >= kf.t0 - 1e-9) & (tt < kf.t1 - 1e-9)
            if msk.any():
                c_true[i] = te[msk].mean(0); ok[i] = True
        e = c_true[ok] - full["c"][ok]
        nees = np.array([float(x @ np.linalg.solve(p_, x)) for x, p_ in zip(e, full["P"][ok])])
        shift = None
        if c == "relative":                           # evaluate in the pad frame
            shift = -back.mean(0)
        mm = MR._map_metrics(rm, scene, targets, shift=shift)
        out[c] = dict(coverage=mm["coverage"], acc_median=mm["acc_median"], acc_p95=mm["acc_p95"],
                      nees_mean=float(nees.mean()), nees_le_7_81=float(np.mean(nees <= 7.81)),
                      datum_err_m=float(np.linalg.norm(e.mean(0))), nav_err_median=float(np.median(np.linalg.norm(e, axis=1))),
                      gate_rejects=s_.gate_rejects, rejected_markers=rej,
                      marker_factors=sum(1 for a in s_.anchors if isinstance(a["src"], tuple)))
        if record:
            out[c]["_e"] = e; out[c]["_c"] = full["c"][ok]; out[c]["_ct"] = c_true[ok]
    return out

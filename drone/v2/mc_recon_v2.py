"""
Monte Carlo for the recon layer: mission templates over KNOWN sim scenes.

For each scene and template the truth drone follows the template's waypoints at an
ASSUMED speed. drone_core_v2 runs every tick on noisy telemetry (GPS-like position
with correlated error, gyro with bias, magnetometer) and supplies the pose and
heading estimates; its flight guidance is not changed by any of this. An ASSUMED
gimballed multibeam sonar head (5 x 5 beams inside +/-15 deg) pings at 10 Hz;
hits are converted to world points with the ESTIMATED pose/heading and fused into
drone_recon_v2's evidence grid. At mission end: coverage of the template's target
surfaces within the tolerance, point accuracy vs truth, effective detail, and the
incremental cloud-map uplink over simulated WiFi / LTE / satellite links.

Held-out scenes come from a different seed and are never used for tuning.
What this sim does NOT model: sonar beam patterns/specular loss on oblique faces
beyond a fixed per-beam miss probability, wind, vehicle dynamics (kinematic path),
occlusion of the radio links, real radio behaviour, SLAM (the pose is NOT corrected
by the map).

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # one BLAS thread per worker process (8 processes)
import numpy as np  # noqa: E402

import drone_core_v2 as v2
import drone_recon_v2 as R
import drone_uplink_v2 as U
import drone_slam_v2 as S

SEED = 20261009
DT = 0.1                    # tick and ping period (10 Hz)
SPEED = 2.0                 # ASSUMED mission speed (m/s)
BEAM_HALF = math.radians(15.0)
N_BEAM = 5                  # ASSUMED 5 x 5 multibeam head (gimballed)
SONAR_MAX = 15.0
RANGE_NOISE = 0.015
BEARING_NOISE = math.radians(1.0)
P_PING_DROP = 0.10
P_BEAM_MISS = 0.10          # ASSUMED per-beam miss (weak/specular)
P_MULTIPATH = 0.05          # ASSUMED per-beam late echo
STANDOFF = {"lawnmower": 6.0, "perimeter_orbit": 4.0, "poi_close_pass": 2.0, "interior_room_authorized": 1.2}
OVERLAP = 0.3
GPS_MODES = {"gps_standard": (0.5, 30.0), "gps_rtk": (0.03, 30.0)}   # ASSUMED (sigma m, GM tau s)
INDOOR_POS = (0.10, 10.0)   # ASSUMED indoor positioning source (e.g. UWB): sigma m, tau s
VEL_NOISE = 0.10
ACC_NOISE = 0.30
GYRO_BIAS_DPS = 1.0
MAG_NOISE_UT = 0.5
HARD_IRON_RESID_UT = 0.3
YAW_INIT_ERR_DEG = 5.0
MAX_MISSION_S = 600.0
LINK_CONFIGS = {"wifi_only": ("wifi",), "lte_only": ("lte",), "satellite_only": ("satellite",),
                "wifi+lte+satellite": ("wifi", "lte", "satellite")}
WIFI_STATION_DIST_M = (60.0, 150.0)   # ASSUMED ground-station distance from the scene centre
COLLECT_EVERY_S = 1.0


def make_scenes(n: int, rng: np.random.Generator) -> List[Dict]:
    out = []
    for i in range(n):
        W, D, H = rng.uniform(8, 16), rng.uniform(8, 16), rng.uniform(4, 9)
        b_lo, b_hi = np.array([-W / 2, -D / 2, 0.0]), np.array([W / 2, D / 2, H])
        ang = rng.uniform(0, 2 * math.pi)
        gap = rng.uniform(6, 10)
        half = rng.uniform(0.4, 0.8)
        ph = rng.uniform(3, 6)
        rr = max(W, D) / 2 * 1.0 + gap
        pc = np.array([rr * math.cos(ang), rr * math.sin(ang)])
        p_lo, p_hi = np.array([pc[0] - half, pc[1] - half, 0.0]), np.array([pc[0] + half, pc[1] + half, ph])
        lo = np.minimum(b_lo, p_lo) - np.array([4, 4, 0]); hi = np.maximum(b_hi, p_hi) + np.array([4, 4, 0])
        room = np.array([rng.uniform(5, 10), rng.uniform(4, 8), rng.uniform(2.6, 3.5)])
        st_ang = rng.uniform(0, 2 * math.pi)
        st_d = rng.uniform(*WIFI_STATION_DIST_M)
        out.append(dict(id=i, building=(b_lo, b_hi), poi=(p_lo, p_hi), poi_half=half, area=(lo, hi),
                        room=(np.zeros(3), room), station=np.array([st_d * math.cos(st_ang), st_d * math.sin(st_ang), 2.0]),
                        seed=int(rng.integers(0, 2**31 - 1))))
    return out


def plan(sc: Dict, template: str):
    b_lo, b_hi = sc["building"]; p_lo, p_hi = sc["poi"]; a_lo, a_hi = sc["area"]
    if template == "lawnmower":
        wps = R.lawnmower(a_lo, a_hi, max(b_hi[2], p_hi[2]), STANDOFF[template], BEAM_HALF, OVERLAP,
                          max_range_m=SONAR_MAX)
        scene = R.Scene([sc["building"], sc["poi"]], a_lo, a_hi, True)
        targets = np.concatenate([R.box_surface_samples(b_lo, b_hi, 0.25, "top"),
                                  R.box_surface_samples(p_lo, p_hi, 0.25, "top"),
                                  R.ground_samples(a_lo, a_hi, [sc["building"], sc["poi"]], 0.25)])
    elif template == "perimeter_orbit":
        wps = R.perimeter_orbit(b_lo, b_hi, STANDOFF[template], BEAM_HALF, OVERLAP)
        scene = R.Scene([sc["building"], sc["poi"]], a_lo, a_hi, True)
        targets = R.box_surface_samples(b_lo, b_hi, 0.25, "sides")
    elif template == "poi_close_pass":
        c = 0.5 * (p_lo + p_hi)
        wps = R.poi_close_pass(c, sc["poi_half"], p_hi[2], STANDOFF[template], BEAM_HALF, OVERLAP)
        scene = R.Scene([sc["building"], sc["poi"]], a_lo, a_hi, True)
        targets = R.box_surface_samples(p_lo, p_hi, 0.25, "sides")
    else:
        r_lo, r_hi = sc["room"]
        wps = R.interior_room_sweep(r_lo, r_hi, STANDOFF[template], BEAM_HALF, {"interior_authorized": True}, OVERLAP)
        scene = R.Scene([], r_lo - 1, r_hi + 1, False, rooms=[sc["room"]])
        targets = R.room_samples(r_lo, r_hi, 0.25)
    return wps, scene, targets


def _beam_dirs(look: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    """N_BEAM x N_BEAM directions, one jittered inside each cell of the +/-15 deg cone
    (a random sample of a dense fan, so no fixed gaps between beams)."""
    look = look / np.linalg.norm(look)
    up = np.array([0.0, 0.0, 1.0]) if abs(look[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, look); e1 /= np.linalg.norm(e1)
    e2 = np.cross(look, e1)
    out = []
    edges = np.linspace(-BEAM_HALF, BEAM_HALF, N_BEAM + 1)
    for ia in range(N_BEAM):
        for ib in range(N_BEAM):
            a = rng.uniform(edges[ia], edges[ia + 1]); b = rng.uniform(edges[ib], edges[ib + 1])
            d = look + math.tan(a) * e1 + math.tan(b) * e2
            out.append(d / np.linalg.norm(d))
    return out


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def run_mission(sc: Dict, template: str, gps_mode: str, links=tuple(LINK_CONFIGS), record=False,
                slam: bool = False, slam_kwargs: Dict = None, gps_override=None) -> Dict:
    rng = np.random.default_rng(sc["seed"] + 101 * (1 + list(R.TEMPLATES).index(template)))
    wps, scene, targets = plan(sc, template)
    indoor = template == "interior_room_authorized"
    gsig, gtau = INDOOR_POS if indoor else (gps_override or GPS_MODES[gps_mode])
    core = v2.AutonomousDroneCore(heading_init_sigma_rad=math.radians(YAW_INIT_ERR_DEG))
    cal = core.heading.cal
    Dd, Ii, F = cal.declination_rad, cal.dip_rad, cal.field_ut
    b_w = F * np.array([math.cos(Ii) * math.sin(Dd), math.cos(Ii) * math.cos(Dd), -math.sin(Ii)])
    gyro_bias = math.radians(rng.uniform(-GYRO_BIAS_DPS, GYRO_BIAS_DPS))
    yaw_err0 = math.radians(rng.normal(0, YAW_INIT_ERR_DEG))
    hard_resid = rng.normal(0, HARD_IRON_RESID_UT, 3)
    rmap = R.ReconMap()
    # uplink: one independent sync + cloud per link configuration (same map)
    sync = U.MapSync(rmap)
    ul = {}
    for name in links:
        cloud = U.CloudMapAssembler()
        lrng = np.random.default_rng(sc["seed"] + 7 + len(name))
        lks = [U.SimLink(U.LINK_PROFILES[l], np.random.default_rng(lrng.integers(0, 2**31)), cloud.deliver,
                         station_pos=sc["station"] if l == "wifi" else None) for l in LINK_CONFIGS[name]]
        lm = U.LinkManager(lks)
        ul[name] = (sync.add_channel(lm), lm, cloud, lks)
    p = wps[0].pos.copy()
    i_wp = 0
    phi = math.exp(-DT / gtau)
    gps_err = rng.normal(0, gsig, 3)
    yaw_prev = None
    t = 0.0
    path = []
    hdg_err, pos_err = [], []
    slam_obj = S.SonarSLAM(**(slam_kwargs or {})) if slam else None
    rmap_online = R.ReconMap() if slam else None
    tick_t, tick_err, tick_core_nees = [], [], []
    sync_ms = []
    min_clear = float("inf")
    while i_wp < len(wps) and t < MAX_MISSION_S:
        t += DT
        # kinematic motion toward the next waypoint (at most one waypoint per tick = dwell)
        tgt = wps[i_wp].pos
        d = tgt - p
        dist = float(np.linalg.norm(d))
        step = SPEED * DT
        p_old = p
        if dist <= step:
            p = tgt.copy(); i_wp += 1
        else:
            p = p + d / dist * step
        # truth velocity over this tick (the telemetry velocity is generated from it; before
        # this fix it was the commanded SPEED even on ticks that stopped short at a waypoint)
        vel = (p - p_old) / DT
        look = wps[min(i_wp, len(wps) - 1)].look_dir
        hz = np.array([look[0], look[1]])
        if np.linalg.norm(hz) > 0.1:
            yaw = math.atan2(hz[1], hz[0])
        elif np.linalg.norm(vel[:2]) > 1e-6:
            yaw = math.atan2(vel[1], vel[0])
        else:
            yaw = yaw_prev if yaw_prev is not None else 0.0
        rate = 0.0 if yaw_prev is None else ((yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / DT
        yaw_prev = yaw
        path.append(p.copy())
        min_clear = min(min_clear, float(scene.surface_distance(p[None])[0]))
        gps_err = phi * gps_err + math.sqrt(1 - phi * phi) * rng.normal(0, gsig, 3)
        tm = v2.AirframeState(timestamp_s=t, pos_m=p + gps_err, vel_mps=vel + rng.normal(0, VEL_NOISE, 3),
                              acc_mps2=rng.normal(0, ACC_NOISE, 3), jerk_mps3=np.zeros(3),
                              attitude=v2.Attitude.from_euler(0.0, 0.0, yaw + yaw_err0),
                              pos_sigma_m=np.full(3, gsig), vel_sigma_mps=np.full(3, VEL_NOISE))
        tm.acc_sigma_mps2 = np.full(3, ACC_NOISE)
        tm.gyro_rps = np.array([0.0, 0.0, rate + gyro_bias]) + rng.normal(0, v2.GYRO_ARW_RAD_S_SQRT_HZ / math.sqrt(DT), 3)
        tm.mag_body_ut = _rz(yaw).T @ b_w + hard_resid + rng.normal(0, MAG_NOISE_UT, 3)
        out = core.process_flight_tick(tm, 0.0, [], horizon_s=DT, desired_velocity_mps=vel)
        p_hat = core.kf.x[0:3].copy()
        P = core.kf.P[0:3, 0:3]
        he = out["heading"]
        yaw_hat = math.radians(he["yaw_enu_deg"])
        sig_h = math.radians(he["heading_sigma_deg"])
        dyaw = (yaw_hat - yaw + math.pi) % (2 * math.pi) - math.pi
        hdg_err.append(abs(math.degrees(dyaw)))
        pos_err.append(float(np.linalg.norm(p_hat - p)))
        if slam:
            tick_t.append(t); tick_err.append(p_hat - p)
            e_ = p_hat - p
            tick_core_nees.append(float(e_ @ np.linalg.solve(P, e_)))
        if rng.random() >= P_PING_DROP:
            Rz = _rz(dyaw)
            beams = []
            beams_np = []
            for u in _beam_dirs(look, rng):
                if rng.random() < P_BEAM_MISS:
                    continue
                r = scene.raycast(p, u, SONAR_MAX)
                if r is None:
                    continue
                if rng.random() < P_MULTIPATH:
                    r *= rng.uniform(1.3, 2.5)
                    if r > SONAR_MAX:
                        continue
                r_m = r + rng.normal(0, RANGE_NOISE)
                um = u + rng.normal(0, BEARING_NOISE, 3); um /= np.linalg.norm(um)
                u_est = Rz @ um
                cov = R.point_covariance(r_m, u_est, RANGE_NOISE, BEARING_NOISE, sig_h, P)
                beams.append((u_est, r_m, cov))
                if slam:
                    beams_np.append((u_est, r_m, R.point_covariance(r_m, u_est, RANGE_NOISE, BEARING_NOISE, sig_h, None)))
            if beams:
                rmap.add_ping(p_hat, beams, RANGE_NOISE, t)
            if slam and beams_np:
                k_closed = slam_obj.add_ping(t, p_hat, beams_np)
                if k_closed is not None:
                    _slam_online_integrate(slam_obj, k_closed, rmap_online)
        # uplink (worker side; not part of the flight tick)
        t0 = time.perf_counter()
        if abs(t / COLLECT_EVERY_S - round(t / COLLECT_EVERY_S)) < 1e-6:
            sync.collect(t)
        for name, (ch, lm, cloud, lks) in ul.items():
            for l in lks:
                l.step(t, DT, p)
            ch.pump(sync, t)
            cloud.poll(t)
        sync_ms.append((time.perf_counter() - t0) * 1000.0)
    # --- metrics ---
    pts, sig = rmap.filtered_points()
    raw = np.array([q.pos for q in rmap.points]).reshape(-1, 3)
    acc = scene.surface_distance(pts) if len(pts) else np.array([np.nan])
    acc_raw = scene.surface_distance(raw) if len(raw) else np.array([np.nan])
    cov_f = R.coverage(targets, pts, R.COVERAGE_TOL_M)
    res = dict(id=sc["id"], template=template, gps_mode=("indoor_pos" if indoor else gps_mode),
               mission_s=round(t, 1), completed=i_wp >= len(wps), n_points=len(raw), n_filtered=len(pts),
               coverage=cov_f, acc_median=float(np.nanmedian(acc)), acc_p95=float(np.nanpercentile(acc, 95)),
               frac_points_within_tol=float(np.nanmean(acc <= R.COVERAGE_TOL_M)),
               raw_acc_p95=float(np.nanpercentile(acc_raw, 95)),
               effective_detail_m=rmap.effective_detail_m(),
               pos_err_median=float(np.median(pos_err)), hdg_err_median=float(np.median(hdg_err)),
               min_surface_dist=min_clear, uplink={})
    # final state + reconciliation
    sync.collect(t)
    for name, (ch, lm, cloud, lks) in ul.items():
        cloud.poll(t + 1.0)          # messages already in flight
        cells, pts_q = sync.quantized_state()
        occ = rmap.grid.occupied()
        frac = cloud.fraction_present(cells, pts_q, occ)
        hash_before = cloud.map_hash() == sync.map_hash()
        # final full-map reconciliation (landed, assumed reliable link): snapshot until hashes match
        missing = {k: v for k, v in cells.items() if tuple(cloud.cells.get(k, (None, None))[:2]) != tuple(v)}
        for d in sync.snapshot(t):      # (a real system would send only what the cloud lacks)
            cloud.receive(d)
        hash_after = cloud.map_hash() == sync.map_hash()
        res["uplink"][name] = dict(
            frac_cells=frac["cells"], frac_points=frac["points"], frac_occupied=frac["occupied_cells"],
            delta_Bps=sync.stats["delta_bytes"] / max(t, 1e-9), sent_Bps=ch.stats["sent_bytes"] / max(t, 1e-9),
            resyncs=ch.stats["resyncs"], dropped_msgs=ch.stats["dropped_msgs"], failovers=lm.failovers,
            cells_missing_at_end=len(missing),
            link_up_frac={l.p.name: round(l.up_time_s / max(l.total_time_s, 1e-9), 3) for l in lks},
            sends_by_link=dict(lm.sends_by_link), energy_wh=lm.energy_wh(),
            hash_match_before_reconcile=hash_before, hash_match_after_reconcile=hash_after)
    if slam:
        k_closed = slam_obj.finish()
        if k_closed is not None:
            _slam_online_integrate(slam_obj, k_closed, rmap_online)
        res["slam"] = _slam_metrics(slam_obj, rmap_online, scene, targets, np.array(tick_t), np.array(tick_err),
                                    pts, tick_core_nees, record)
    res["sync_ms_max"] = float(max(sync_ms)) if sync_ms else 0.0
    res["sync_ms_median"] = float(np.median(sync_ms)) if sync_ms else 0.0
    if record:
        res["_map"] = rmap; res["_path"] = np.array(path); res["_scene"] = scene
    return res


def _slam_online_integrate(sl, k: int, rmap_online) -> None:
    """Online map: integrate a just-closed keyframe's pings with the online (filtered) estimate."""
    t0 = time.perf_counter()
    c, P = sl.est_online[k], sl.cov_rel_online[k]
    for tp, o, bms in sl.kfs[k].pings:
        rmap_online.add_ping(o - c, [(u, r, cv + P) for u, r, cv in bms], RANGE_NOISE, tp)
    sl.stats.setdefault("online_map_ms", []).append((time.perf_counter() - t0) * 1000.0)


def _map_metrics(rm, scene, targets, shift=None) -> Dict:
    pts, _ = rm.filtered_points()
    if shift is not None and len(pts):
        pts = pts + shift
    acc = scene.surface_distance(pts) if len(pts) else np.array([np.nan])
    return dict(coverage=R.coverage(targets, pts, R.COVERAGE_TOL_M), acc_median=float(np.nanmedian(acc)),
                acc_p95=float(np.nanpercentile(acc, 95)), frac_points_within_tol=float(np.nanmean(acc <= R.COVERAGE_TOL_M)),
                effective_detail_m=rm.effective_detail_m(), n_filtered=len(pts))


def _slam_metrics(sl, rmap_online, scene, targets, tt, terr, base_pts, core_nees, record) -> Dict:
    t_rb = time.perf_counter()
    full = sl.smoothed_full()
    sm = (full["c"], full["P"])
    rmap_s = sl.rebuild_map(R.ReconMap(), RANGE_NOISE, full)
    rebuild_s = time.perf_counter() - t_rb
    # truth correction per keyframe = mean core error over the keyframe's ticks
    K = len(sl.kfs)
    c_true = np.zeros((K, 3)); ok = np.zeros(K, bool)
    for i, kf in enumerate(sl.kfs):
        m = (tt >= kf.t0 - 1e-9) & (tt < kf.t1 - 1e-9)
        if m.any():
            c_true[i] = terr[m].mean(0); ok[i] = True
    C_on, P_on = np.array(sl.est_online), np.array(sl.cov_online)
    C_sm, P_sm = sm

    def nees(C, P):
        e = c_true[ok] - C[ok]
        v = np.array([float(x @ np.linalg.solve(p_, x)) for x, p_ in zip(e, P[ok])])
        return v
    n_on, n_sm = nees(C_on, P_on), nees(C_sm, P_sm)
    # navigation error of the SLAM-corrected position, every tick
    nav_on = np.array([np.linalg.norm(e_ - sl.correction_at(t_)[0]) for t_, e_ in zip(tt, terr)])
    nav_sm = np.array([np.linalg.norm(e_ - sl.correction_at(t_, sm)[0]) for t_, e_ in zip(tt, terr)])
    datum = (c_true[ok] - C_sm[ok]).mean(0)     # common (absolute) offset left in the smoothed map
    base_off = c_true[ok].mean(0)
    out = dict(smoothed=_map_metrics(rmap_s, scene, targets), online=_map_metrics(rmap_online, scene, targets),
               smoothed_datum_removed=_map_metrics(rmap_s, scene, targets, shift=-datum),
               nav_err_online_median=float(np.median(nav_on)), nav_err_smoothed_median=float(np.median(nav_sm)),
               nav_err_smoothed_p95=float(np.percentile(nav_sm, 95)),
               nees_online_mean=float(n_on.mean()), nees_online_le_7_81=float(np.mean(n_on <= 7.81)),
               nees_smoothed_mean=float(n_sm.mean()), nees_smoothed_le_7_81=float(np.mean(n_sm <= 7.81)),
               core_nees_mean=float(np.mean(core_nees)), core_nees_le_7_81=float(np.mean(np.array(core_nees) <= 7.81)),
               datum_err_m=float(np.linalg.norm(datum)),
               datum_sigma_m=[round(float(math.sqrt(full["P_datum"][i, i])), 3) for i in range(3)],
               datum_nees=float(datum @ np.linalg.solve(full["P_datum"], datum)), datum_err_xyz=[round(float(x), 3) for x in datum],
               core_mean_err_m=float(np.linalg.norm(base_off)),
               keyframes=K, registrations=sl.stats["registrations"], loop_factors=sl.stats["loop_factors"],
               rejected_few=sl.stats["rejected_few"], rejected_degenerate=sl.stats["rejected_degenerate"],
               rejected_inconsistent=sl.stats["rejected_inconsistent"], n_obs_hist=list(sl.stats["n_obs_hist"]),
               step_ms_median=float(np.median(sl.stats["step_ms"])), step_ms_p95=float(np.percentile(sl.stats["step_ms"], 95)),
               step_ms_max=float(np.max(sl.stats["step_ms"])), end_smooth_rebuild_s=float(rebuild_s),
               online_map_ms_max=float(max(sl.stats.get("online_map_ms", [0.0]))))
    if record:
        out["_map"] = rmap_s
    return out


def _job(args):
    sc, template, gps = args
    return run_mission(sc, template, gps)


def summarize(rs: List[Dict]) -> Dict:
    def mp(key):
        x = np.array([r[key] for r in rs if r[key] is not None], float)
        return {"median": round(float(np.median(x)), 3), "p5": round(float(np.percentile(x, 5)), 3),
                "min": round(float(np.min(x)), 3)} if len(x) else None
    out = {"missions": len(rs),
           "coverage": mp("coverage"),
           "frac_missions_coverage_ge_90": round(float(np.mean([r["coverage"] >= 0.90 for r in rs])), 3),
           "point_acc_m": {"median_of_medians": round(float(np.median([r["acc_median"] for r in rs])), 3),
                           "median_of_p95": round(float(np.median([r["acc_p95"] for r in rs])), 3)},
           "frac_points_within_tol": round(float(np.median([r["frac_points_within_tol"] for r in rs])), 3),
           "raw_point_acc_p95_median": round(float(np.median([r["raw_acc_p95"] for r in rs])), 3),
           "effective_detail_m_median": round(float(np.median([r["effective_detail_m"] for r in rs
                                                               if r["effective_detail_m"] is not None])), 3),
           "pos_err_median_m": round(float(np.median([r["pos_err_median"] for r in rs])), 3),
           "hdg_err_median_deg": round(float(np.median([r["hdg_err_median"] for r in rs])), 3),
           "mission_s_median": round(float(np.median([r["mission_s"] for r in rs])), 1),
           "completed_frac": round(float(np.mean([r["completed"] for r in rs])), 3),
           "min_surface_dist_m": round(float(min(r["min_surface_dist"] for r in rs)), 3),
           "uplink_worker_ms_max": round(float(max(r["sync_ms_max"] for r in rs)), 2)}
    up = {}
    for name in LINK_CONFIGS:
        u = [r["uplink"][name] for r in rs if name in r["uplink"]]
        if not u:
            continue
        up[name] = {"frac_occupied_cells_in_cloud": {"median": round(float(np.median([x["frac_occupied"] for x in u])), 3),
                                                     "min": round(float(np.min([x["frac_occupied"] for x in u])), 3)},
                    "frac_all_cells_in_cloud_median": round(float(np.median([x["frac_cells"] for x in u])), 3),
                    "frac_points_in_cloud_median": round(float(np.median([x["frac_points"] for x in u])), 3),
                    "delta_kBps_median": round(float(np.median([x["delta_Bps"] for x in u])) / 1000, 2),
                    "sent_kBps_median": round(float(np.median([x["sent_Bps"] for x in u])) / 1000, 2),
                    "resyncs_total": int(sum(x["resyncs"] for x in u)),
                    "dropped_msgs_total": int(sum(x["dropped_msgs"] for x in u)),
                    "radio_energy_wh_median": {k: round(float(np.median([x["energy_wh"][k] for x in u])), 3)
                                               for k in u[0]["energy_wh"]},
                    "failovers_total": int(sum(x["failovers"] for x in u)),
                    "hash_match_before_reconcile": round(float(np.mean([x["hash_match_before_reconcile"] for x in u])), 3),
                    "hash_match_after_reconcile": round(float(np.mean([x["hash_match_after_reconcile"] for x in u])), 3)}
    out["uplink"] = up
    return out


def main(n_main: int = 12, n_hold: int = 12) -> Dict:
    scen = make_scenes(n_main, np.random.default_rng(SEED))
    scen_h = make_scenes(n_hold, np.random.default_rng(SEED + 1))
    jobs = []
    for tag, scs in (("main", scen), ("holdout", scen_h)):
        for tpl in R.TEMPLATES:
            modes = ("indoor_pos",) if tpl == "interior_room_authorized" else tuple(GPS_MODES)
            for g in modes:
                for sc in scs:
                    jobs.append((tag, tpl, g, sc))
    t0 = time.perf_counter()
    with Pool(8) as pool:
        res = pool.map(_job, [(sc, tpl, (g if g != "indoor_pos" else "gps_standard")) for _, tpl, g, sc in jobs])
    rep = {"seed_main": SEED, "seed_holdout": SEED + 1, "n_main": n_main, "n_holdout": n_hold,
           "settings": {"speed_mps": SPEED, "sonar": {"beams": f"{N_BEAM}x{N_BEAM} inside +/-15 deg (ASSUMED gimballed multibeam)",
                                                      "rate_hz": 1 / DT, "max_range_m": SONAR_MAX, "range_noise_m": RANGE_NOISE,
                                                      "bearing_noise_deg": 1.0, "p_ping_drop": P_PING_DROP,
                                                      "p_beam_miss": P_BEAM_MISS, "p_multipath": P_MULTIPATH},
                        "standoff_m": STANDOFF, "overlap": OVERLAP, "grid_res_m": R.GRID_RES_M,
                        "coverage_tol_m": R.COVERAGE_TOL_M, "gps_modes_sigma_tau": GPS_MODES,
                        "indoor_position_sigma_tau": INDOOR_POS, "gyro_bias_dps": GYRO_BIAS_DPS,
                        "links": {k: vars(v) for k, v in U.LINK_PROFILES.items()},
                        "wifi_station_dist_m": WIFI_STATION_DIST_M,
                        "targets": {"lawnmower": "roof tops + open ground in the area box",
                                    "perimeter_orbit": "building facades", "poi_close_pass": "POI column sides",
                                    "interior_room_authorized": "room walls, floor, ceiling (inside; authorized)"}},
           "results": {}}
    groups: Dict = {}
    for (tag, tpl, g, _), r in zip(jobs, res):
        groups.setdefault(f"{tag}|{tpl}|{g}", []).append(r)
    for k, rs in groups.items():
        rep["results"][k] = summarize(rs)
    rep["runtime_s"] = round(time.perf_counter() - t0, 1)
    return rep


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rep = main(n, n)
    with open("mc_recon_results_v2.json", "w") as f:
        json.dump(rep, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for k, v in rep["results"].items():
        print(k, "cov", v["coverage"], ">=90%:", v["frac_missions_coverage_ge_90"], "acc", v["point_acc_m"],
              "detail", v["effective_detail_m_median"])
    print("runtime_s", rep["runtime_s"])

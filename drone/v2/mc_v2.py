"""
Closed-loop Monte Carlo for drone_core_v1 vs drone_core_v2.

A point-mass drone with true linear drag and a first-order actuator flies
toward a wall or a pole at a random speed and heading. Each version's own
estimator and guidance output close the loop through the same simple flight
controller. Measured: collision rate, minimum clearance, position error of the
"now" estimate vs truth, and filter consistency (3D position NEES).

What this sim does NOT model: aerodynamics beyond linear drag, rotor wash,
attitude dynamics (yaw = heading, level flight), real sonar beam patterns and
specular misses on oblique surfaces, correlated GPS error (GPS noise here is
white), wind, moving obstacles, multiple obstacles, IR physics.

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Dict, List

import numpy as np

import drone_core_v1 as v1
import drone_core_v2 as v2

SEED = 20261008
DT = 0.05                 # control + telemetry tick (20 Hz)
PING_EVERY = 2            # sonar at 10 Hz (round trip at 15 m is ~88 ms)
H_SIM = 0.005             # truth integration step
TAU_LAT = 0.05            # ASSUMED telemetry latency (one tick)
ECHO_DELAY = 0.01         # echo reception this long before the tick
LAM = 0.18 / 1.45         # true drag / mass (same as the filter model)
A_TRUE_MAX = 5.0          # ASSUMED vehicle acceleration limit
ACT_TAU = 0.10            # ASSUMED actuator lag
DRONE_R = 0.25            # ASSUMED body radius (collision if clearance <= 0)
POLE_R = 0.30
BEAM_HALF = math.radians(15.0)   # ASSUMED sonar half-beam (horizontal)
SONAR_MAX = 15.0
RANGE_NOISE = 0.015       # true sonar range noise (1 sigma)
AZ_NOISE = math.radians(1.0)
P_MULTIPATH = 0.10        # per echo
P_SPIKE = 0.05            # per ping: spurious early low-SNR echo
P_DROPOUT = 0.10          # per ping: no data at all
VEL_NOISE = 0.10
ACC_NOISE = 0.30
IR_BIAS_TRUE = 0.03
K_V = 1.5                 # cruise velocity loop gain (1/s)

# Heading-drift scenario (all ASSUMED)
G = 9.81
GYRO_BIAS_DPS = 1.0       # true gyro-z bias drawn from U(-1, +1) deg/s
YAW_INIT_ERR_DEG = 5.0    # initial yaw error ~ N(0, 5 deg); the core is told sigma 5 deg
TILT_NOISE_DEG = 0.5      # roll/pitch error of the attitude source
MAG_NOISE_UT = 0.5
HARD_IRON_UT = 15.0       # true hard-iron offset ~ N(0, 15) uT per axis
HARD_IRON_CAL_ERR_UT = 0.3
SOFT_IRON_SIGMA = 0.03    # true soft-iron S = I + N(0, 0.03)
SOFT_IRON_CAL_ERR = 0.005
MOTOR_K_UT_PER_A = 0.3    # true motor coupling ~ N(0, 0.3) uT/A per axis
MOTOR_K_CAL_ERR = 0.10    # compensation coefficients off by 10%
BURST_START_P = 0.01      # per tick, when not in a burst (mean gap 5 s)
BURST_LEN_S = (0.5, 3.0)
BURST_UT = (10.0, 40.0)   # disturbance magnitude, random body-frame direction
ANOMALY_UT = 20.0         # spatial (indoor-like) anomaly amplitude, world frame
ANOMALY_WAVELENGTH_M = 10.0
LM_LATERAL_M = (-6.0, 0.0, 6.0)   # known landmarks on the obstacle plane, lateral offsets
LM_FOV_DEG = 60.0         # ASSUMED landmark camera half field of view
LM_BEARING_NOISE_DEG = 0.5
LM_EVERY = 4              # landmark bearings at 5 Hz
HEADING_MODES = ("mag_off", "mag_on", "mag_bursts", "mag_anomaly", "mag_anomaly_landmarks")

BRAKE_V2 = {"COLLISION_AVOIDANCE_BRAKE", "COLLISION_LIKELY_MAX_BRAKE"}
BRAKE_V1 = {"COLLISION_AVOIDANCE_FLARE", "COLLISION_AVOIDANCE_DEGENERATE"}


def make_scenarios(n: int, rng: np.random.Generator) -> List[Dict]:
    out = []
    for i in range(n):
        kind = "wall" if rng.random() < 0.7 else "pole"
        out.append(dict(
            id=i, kind=kind,
            v0=float(rng.uniform(2.0, 12.0)),
            heading=float(rng.uniform(-math.radians(30), math.radians(30))),
            dist=float(rng.uniform(18.0, 30.0)),
            pole_offset=float(rng.uniform(-0.3, 0.3)),
            gps_sigma=float(rng.choice([0.5, 1.0, 2.0])),
            temp_c=float(rng.uniform(0.0, 35.0)),
            rh=float(rng.uniform(10.0, 90.0)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return out


class World:
    def __init__(self, sc: Dict):
        self.sc = sc
        h = sc["heading"]
        self.dir = np.array([math.cos(h), math.sin(h), 0.0])
        if sc["kind"] == "pole":
            # pole on the flight path, offset laterally by pole_offset
            perp = np.array([-self.dir[1], self.dir[0], 0.0])
            self.c = self.dir * sc["dist"] + perp * sc["pole_offset"]
        self.c_true = v2.sound_speed_cramer(sc["temp_c"], sc["rh"])

    def clearance(self, p: np.ndarray) -> float:
        if self.sc["kind"] == "wall":
            return self.sc["dist"] - p[0] - DRONE_R
        return float(np.hypot(*(p[:2] - self.c[:2]))) - POLE_R - DRONE_R

    def surface_dist(self, q: np.ndarray) -> float:
        """Distance from a point to the true obstacle surface (errors along the surface not counted)."""
        if self.sc["kind"] == "wall":
            return abs(self.sc["dist"] - q[0])
        return abs(float(np.hypot(*(q[:2] - self.c[:2]))) - POLE_R)

    def sonar(self, p: np.ndarray, yaw: float):
        """(range to nearest surface point inside the beam, body azimuth, unit ray ENU) or None."""
        if self.sc["kind"] == "wall":
            d = self.sc["dist"] - p[0]
            if d <= 0:
                return None
            # wall normal direction (toward wall) is +x, i.e. angle 0
            off = (0.0 - yaw + math.pi) % (2 * math.pi) - math.pi   # body az of the normal
            az = float(np.clip(off, -BEAM_HALF, BEAM_HALF))
            ang = yaw + az
            if math.cos(ang) <= 1e-3:
                return None
            r = d / math.cos(ang)
        else:
            rel = self.c[:2] - p[:2]
            dc = float(np.hypot(*rel))
            ang = math.atan2(rel[1], rel[0])
            az = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(az) > BEAM_HALF + math.asin(min(POLE_R / max(dc, POLE_R), 1.0)):
                return None
            r = dc - POLE_R
        if r > SONAR_MAX or r < 0.05:
            return None
        u = np.array([math.cos(ang), math.sin(ang), 0.0])
        return r, az, u


def run_flight(sc: Dict, version: str, use_ir: bool = True, record: bool = False,
               v1_gate_off: bool = False, gps_tau_s: float = 0.0,
               heading_mode: str = "") -> Dict:
    """heading_mode: "" (yaw known, original sim) or one of HEADING_MODES (v2 only)."""
    rng = np.random.default_rng(sc["seed"])     # same noise stream for every version
    w = World(sc)
    t_end = min(sc["dist"] / sc["v0"] + 6.0, 25.0)
    n_ticks = int(round(t_end / DT))
    sub = int(round(DT / H_SIM))
    yaw = sc["heading"]
    att_kwargs = dict(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))
    if version == "v1":
        core = v1.AutonomousDroneCore(temperature_c=sc["temp_c"], humidity_pct=sc["rh"])
        if v1_gate_off:   # diagnostic: v1 with its distance gate disabled (everything else as-is)
            core._predictive_voxel_gate = lambda *a, **k: 1.0
        att = v1.Attitude(**att_kwargs)
        M = v1
    else:
        core = v2.AutonomousDroneCore(temperature_c=sc["temp_c"], humidity_pct=sc["rh"])
        att = v2.Attitude(**att_kwargs)
        M = v2
    hd = bool(heading_mode)
    if hd:
        # separate stream so the original noise draws are unchanged
        rh = np.random.default_rng(sc["seed"] + 7919)
        cal_site = v2.MagCalibration()
        D, I, F = cal_site.declination_rad, cal_site.dip_rad, cal_site.field_ut
        b_w = F * np.array([math.cos(I) * math.sin(D), math.cos(I) * math.cos(D), -math.sin(I)])
        gyro_bias = math.radians(rh.uniform(-GYRO_BIAS_DPS, GYRO_BIAS_DPS))
        yaw_init_err = math.radians(rh.normal(0, YAW_INIT_ERR_DEG))
        S_true = np.eye(3) + rh.normal(0, SOFT_IRON_SIGMA, (3, 3))
        h_true = rh.normal(0, HARD_IRON_UT, 3)
        k_true = rh.normal(0, MOTOR_K_UT_PER_A, 3)
        cal = v2.MagCalibration(
            hard_iron_ut=h_true + rh.normal(0, HARD_IRON_CAL_ERR_UT, 3),
            soft_iron=np.linalg.inv(S_true) @ (np.eye(3) + rh.normal(0, SOFT_IRON_CAL_ERR, (3, 3))),
            motor_ut_per_a=k_true * (1.0 + MOTOR_K_CAL_ERR))
        core = v2.AutonomousDroneCore(temperature_c=sc["temp_c"], humidity_pct=sc["rh"],
                                      mag_calibration=cal,
                                      heading_init_sigma_rad=math.radians(YAW_INIT_ERR_DEG))
        prev_rp = None
        burst_left, burst_vec = 0.0, np.zeros(3)
        h_err, h_nees, n_burst_ticks, n_hunc = [], [], 0, 0
        an_ph = rh.uniform(0, 2 * math.pi, 3)
        perp = np.array([-w.dir[1], w.dir[0], 0.0])
        lms = {f"L{i}": w.dir * sc["dist"] + perp * off for i, off in enumerate(LM_LATERAL_M)}
        if heading_mode == "mag_anomaly_landmarks":
            core.landmarks = {k: v.copy() for k, v in lms.items()}
    map_world_err, map_rel_err = [], []
    p = np.zeros(3); v = w.dir * sc["v0"]; a = np.zeros(3)
    a_cmd = np.zeros(3)
    hist_p, hist_v, hist_a = [p.copy()], [v.copy()], [a.copy()]   # at H_SIM steps
    errs, nees, min_clear, collided, n_brake = [], [], w.clearance(p), False, 0
    trace = []
    gps = sc["gps_sigma"]
    phi = math.exp(-DT / gps_tau_s) if gps_tau_s > 0 else 0.0   # Gauss-Markov GPS error option
    gps_err = rng.normal(0, gps, 3) if gps_tau_s > 0 else np.zeros(3)
    for k in range(1, n_ticks + 1):
        # integrate truth over one tick with the command held
        for _ in range(sub):
            a = a + (a_cmd - a) * (H_SIM / ACT_TAU)
            v = v + (a - LAM * v) * H_SIM
            p = p + v * H_SIM
            hist_p.append(p.copy()); hist_v.append(v.copy()); hist_a.append(a.copy())
            c = w.clearance(p)
            min_clear = min(min_clear, c)
            if c <= 0.0:
                collided = True
        if collided:
            break
        t = k * DT
        idx_tel = int(round((t - TAU_LAT) / H_SIM))
        gps_err = phi * gps_err + math.sqrt(1.0 - phi * phi) * rng.normal(0, gps, 3)
        tel_p, tel_v, tel_a = hist_p[idx_tel], hist_v[idx_tel], hist_a[idx_tel]
        telem = M.AirframeState(
            timestamp_s=t - TAU_LAT,
            pos_m=tel_p + gps_err,
            vel_mps=tel_v + rng.normal(0, VEL_NOISE, 3),
            acc_mps2=tel_a + rng.normal(0, ACC_NOISE, 3),
            jerk_mps3=np.zeros(3), attitude=att,
            pos_sigma_m=np.full(3, gps), vel_sigma_mps=np.full(3, VEL_NOISE),
        )
        if version != "v1":
            telem.acc_sigma_mps2 = np.full(3, ACC_NOISE)
        if hd:
            # true tilt from the horizontal thrust acceleration (a + drag), yaw fixed = heading
            a_th = tel_a + LAM * tel_v
            a_f = math.cos(yaw) * a_th[0] + math.sin(yaw) * a_th[1]
            a_l = -math.sin(yaw) * a_th[0] + math.cos(yaw) * a_th[1]
            roll_t, pitch_t = -math.atan2(a_l, G), math.atan2(a_f, G)
            if prev_rp is None:
                prev_rp = (roll_t, pitch_t)
            rdot, pdot = (roll_t - prev_rp[0]) / DT, (pitch_t - prev_rp[1]) / DT
            prev_rp = (roll_t, pitch_t)
            body_rates = np.array([rdot, pdot * math.cos(roll_t), -pdot * math.sin(roll_t)])  # yaw rate 0
            R_true = v2.Attitude.from_euler(roll_t, pitch_t, yaw).to_matrix()
            telem.attitude = v2.Attitude.from_euler(
                roll_t + math.radians(rh.normal(0, TILT_NOISE_DEG)),
                pitch_t + math.radians(rh.normal(0, TILT_NOISE_DEG)), yaw + yaw_init_err)
            telem.gyro_rps = (body_rates + np.array([0.0, 0.0, gyro_bias])
                              + rh.normal(0, v2.GYRO_ARW_RAD_S_SQRT_HZ / math.sqrt(DT), 3))
            if heading_mode == "mag_bursts":
                if burst_left <= 0.0 and rh.random() < BURST_START_P:
                    burst_left = rh.uniform(*BURST_LEN_S)
                    dvec = rh.normal(size=3)
                    burst_vec = dvec / np.linalg.norm(dvec) * rh.uniform(*BURST_UT)
                in_burst = burst_left > 0.0
                burst_left -= DT
                n_burst_ticks += int(in_burst)
            else:
                in_burst = False
            if heading_mode != "mag_off":
                current = 10.0 + 2.0 * float(np.linalg.norm(a_th))       # ASSUMED motor current model
                b_world_here = b_w
                if heading_mode.startswith("mag_anomaly"):
                    kx = 2 * math.pi / ANOMALY_WAVELENGTH_M
                    b_world_here = b_w + ANOMALY_UT * np.array([
                        math.sin(kx * tel_p[0] + an_ph[0]), math.sin(kx * tel_p[1] + an_ph[1]),
                        0.5 * math.sin(kx * (tel_p[0] + tel_p[1]) + an_ph[2])])
                b_body = R_true.T @ b_world_here + (burst_vec if in_burst else 0.0)
                telem.mag_body_ut = (S_true @ b_body + h_true + k_true * current
                                     + rh.normal(0, MAG_NOISE_UT, 3))
                telem.motor_current_a = current
        groups, irs = [], []
        draws = rng.random(8)
        if k % PING_EVERY == 0 and draws[0] >= P_DROPOUT:
            t_r = t - ECHO_DELAY
            i_r = int(round(t_r / H_SIM))
            pr, vr = hist_p[i_r], hist_v[i_r]
            hit = w.sonar(pr, yaw)
            echoes = []
            if hit is not None:
                r_recv, az, u = hit
                el0 = 0.0
                if hd:    # body-frame bearing of the hit through the true (tilted) attitude
                    bvec = R_true.T @ u
                    az, el0 = math.atan2(bvec[1], bvec[0]), math.asin(float(np.clip(bvec[2], -1, 1)))
                L = pr + r_recv * u
                tof = 2 * r_recv / w.c_true
                for _ in range(2):   # out-path from the emission position
                    i_e = max(0, int(round((t_r - tof) / H_SIM)))
                    tof = (float(np.linalg.norm(L - hist_p[i_e])) + r_recv) / w.c_true
                for j in range(3):
                    if rng.random() < P_MULTIPATH:
                        echoes.append(M.AcousticSensorReturn(
                            az + rng.normal(0, math.radians(5)), el0 + rng.normal(0, math.radians(2)),
                            tof * rng.uniform(1.3, 2.5), 0.4, t_r, float(rng.uniform(10, 18))))
                    else:
                        echoes.append(M.AcousticSensorReturn(
                            az + rng.normal(0, AZ_NOISE), el0 + rng.normal(0, AZ_NOISE),
                            tof + 2 * rng.normal(0, RANGE_NOISE) / w.c_true,
                            float(np.clip(0.85 + rng.normal(0, 0.05), 0, 1)), t_r,
                            float(rng.uniform(15, 30))))
                if draws[1] < P_SPIKE:
                    echoes.append(M.AcousticSensorReturn(
                        float(rng.uniform(-BEAM_HALF, BEAM_HALF)), 0.0,
                        tof * rng.uniform(0.1, 0.9), 0.2, t_r, float(rng.uniform(10, 14))))
                ir_kw = dict(depth_m=r_recv + IR_BIAS_TRUE + rng.normal(0, 0.02), timestamp_s=t_r,
                             confidence=0.85)
                if version != "v1":
                    ir_kw.update(azimuth_rad=az, elevation_rad=el0)
                irs.append(M.InfraredReturn(**ir_kw))
            groups.append(M.AcousticEchoGroup(echoes=echoes))
            if not irs:
                irs.append(M.InfraredReturn(depth_m=0.0, timestamp_s=t_r, confidence=0.0))
        bearings = None
        if hd and heading_mode == "mag_anomaly_landmarks" and k % LM_EVERY == 0:
            bearings = []
            for lid, lpos in lms.items():
                bvec = R_true.T @ (lpos - tel_p)
                bvec = bvec / max(np.linalg.norm(bvec), 1e-9)
                az_l, el_l = math.atan2(bvec[1], bvec[0]), math.asin(float(np.clip(bvec[2], -1, 1)))
                if abs(math.degrees(az_l)) <= LM_FOV_DEG and np.linalg.norm(lpos - tel_p) > 1.0:
                    bearings.append((lid, az_l + math.radians(rh.normal(0, LM_BEARING_NOISE_DEG)),
                                     el_l + math.radians(rh.normal(0, LM_BEARING_NOISE_DEG)),
                                     math.radians(LM_BEARING_NOISE_DEG)))
        # --- core tick ---
        if version == "v1":
            ir_list = irs if groups else []
            out = core.process_flight_tick(telem, TAU_LAT, groups, [i.depth_m for i in ir_list],
                                           infrared_returns=ir_list, horizon_s=DT)
            fg = out["flight_guidance"]
            braking = fg["action"] in BRAKE_V1
            safe_v = w.dir * sc["v0"]
        else:
            out = core.process_flight_tick(telem, TAU_LAT, groups,
                                           infrared_returns=(irs if (use_ir and groups) else []),
                                           horizon_s=DT, desired_velocity_mps=w.dir * sc["v0"],
                                           landmark_bearings=bearings)
            fg = out["flight_guidance"]
            braking = fg["action"] in BRAKE_V2
            safe_v = np.asarray(fg["safe_velocity_mps"], float)
        if hd:
            he = out["heading"]
            e_h = math.radians(he["yaw_enu_deg"]) - yaw
            e_h = (e_h + math.pi) % (2 * math.pi) - math.pi
            h_err.append(abs(math.degrees(e_h)))
            h_nees.append((e_h / math.radians(he["heading_sigma_deg"])) ** 2)
            n_hunc += int(out["flight_guidance"]["heading_uncertain"])
        dt_now = t - core._last_update_t
        p_hat, v_hat, _ = core.kf.state_at(dt_now)
        P = core.kf.covariance_at(dt_now)[0:3, 0:3]
        i_now = int(round(t / H_SIM))
        if version != "v1" and k % 2 == 0:
            for o in core.map.items:
                map_world_err.append(w.surface_dist(o.pos_enu))
                map_rel_err.append(w.surface_dist(hist_p[i_now] + (o.pos_enu - p_hat)))
        e = p_hat - hist_p[i_now]
        errs.append(float(np.linalg.norm(e)))
        try:
            nees.append(float(e @ np.linalg.solve(P, e)))
        except np.linalg.LinAlgError:
            nees.append(float("inf"))
        # --- shared flight controller ---
        v_des = safe_v
        a_ctl = K_V * (v_des - v_hat) + LAM * v_des
        a_ctl[2] = -1.0 * p_hat[2] - 1.5 * v_hat[2]
        if braking:
            n_brake += 1
            a_ctl = np.asarray(fg["command_accel_mps2"], float)
        nrm = float(np.linalg.norm(a_ctl))
        a_cmd = a_ctl * (A_TRUE_MAX / nrm) if nrm > A_TRUE_MAX else a_ctl
        if record:
            trace.append(dict(t=round(t, 3), clear=round(w.clearance(hist_p[i_now]), 3),
                              speed=round(float(np.linalg.norm(hist_v[i_now])), 3),
                              err=round(errs[-1], 3), action=fg["action"]))
    res = dict(id=sc["id"], kind=sc["kind"], v0=sc["v0"], gps=gps, collided=collided,
               min_clearance=float(min_clear), err=errs, nees=nees, brake_ticks=n_brake,
               map_world_err=map_world_err, map_rel_err=map_rel_err)
    if hd:
        res.update(h_err=h_err, h_nees=h_nees, gyro_bias_dps=math.degrees(gyro_bias),
                   mag_accepted=core.heading.n_accepted, mag_rejected=core.heading.n_rejected,
                   lm_accepted=core.heading.n_lm_accepted, lm_rejected=core.heading.n_lm_rejected,
                   burst_ticks=n_burst_ticks, ticks=len(h_err), hunc_ticks=n_hunc)
    if record:
        res["trace"] = trace
    return res


def summarize_heading(results: List[Dict], label: str) -> Dict:
    he = np.concatenate([r["h_err"] for r in results])
    hn = np.concatenate([r["h_nees"] for r in results])
    final = np.array([r["h_err"][-1] for r in results if r["h_err"]])
    acc = sum(r["mag_accepted"] for r in results)
    rej = sum(r["mag_rejected"] for r in results)
    base = summarize(results, label)["all"]
    mw = np.array([x for r in results for x in r["map_world_err"]] or [np.nan])
    mr = np.array([x for r in results for x in r["map_rel_err"]] or [np.nan])
    return {
        "label": label, "flights": len(results),
        "heading_err_deg": {"median": round(float(np.median(he)), 3),
                            "p95": round(float(np.percentile(he, 95)), 3),
                            "max": round(float(np.max(he)), 3)},
        "final_heading_err_deg": {"median": round(float(np.median(final)), 3),
                                  "p95": round(float(np.percentile(final, 95)), 3)},
        "heading_nees_mean_expect_1": round(float(np.mean(hn)), 3),
        "frac_heading_nees_le_3.84_expect_0.95": round(float(np.mean(hn <= 3.841)), 3),
        "mag_accepted": int(acc), "mag_rejected": int(rej),
        "landmark_bearings_accepted": int(sum(r["lm_accepted"] for r in results)),
        "landmark_bearings_rejected": int(sum(r["lm_rejected"] for r in results)),
        "frac_ticks_heading_uncertain": round(sum(r["hunc_ticks"] for r in results)
                                              / max(sum(r["ticks"] for r in results), 1), 3),
        "frac_flights_reached_2m": round(float(np.mean([r["min_clearance"] < 2.0 for r in results])), 3),
        "map_point_err_world_m": {"median": round(float(np.median(mw)), 3), "p95": round(float(np.percentile(mw, 95)), 3)},
        "map_point_err_relative_m": {"median": round(float(np.median(mr)), 3), "p95": round(float(np.percentile(mr, 95)), 3)},
        "frac_ticks_in_burst": round(sum(r["burst_ticks"] for r in results)
                                     / max(sum(r["ticks"] for r in results), 1), 3),
        "collision_rate": base["collision_rate"],
        "min_clearance_m": base["min_clearance_m"],
        "pos_err_m": base["pos_err_m"],
    }


def _medp95(x):
    if not x:
        return None
    return {"median": round(float(np.median(x)), 3), "p95": round(float(np.percentile(x, 95)), 3)}


def summarize(results: List[Dict], label: str) -> Dict:
    def agg(rs):
        if not rs:
            return None
        errs = np.concatenate([r["err"] for r in rs]) if any(r["err"] for r in rs) else np.array([np.nan])
        nees = np.concatenate([r["nees"] for r in rs]) if any(r["nees"] for r in rs) else np.array([np.nan])
        mc = np.array([r["min_clearance"] for r in rs])
        return {
            "flights": len(rs),
            "collision_rate": round(float(np.mean([r["collided"] for r in rs])), 4),
            "min_clearance_m": {"median": round(float(np.median(mc)), 3),
                                "p5": round(float(np.percentile(mc, 5)), 3),
                                "min": round(float(np.min(mc)), 3)},
            "pos_err_m": {"median": round(float(np.median(errs)), 3),
                          "p95": round(float(np.percentile(errs, 95)), 3)},
            "nees_mean_expect_3": round(float(np.mean(np.minimum(nees, 1e6))), 2),
            "map_point_err_world_m": _medp95([x for r in rs for x in r["map_world_err"]]),
            "map_point_err_relative_m": _medp95([x for r in rs for x in r["map_rel_err"]]),
            "frac_nees_le_7.81_expect_0.95": round(float(np.mean(nees <= 7.815)), 3),
        }
    out = {"label": label, "all": agg(results)}
    out["by_gps_sigma"] = {str(g): agg([r for r in results if r["gps"] == g]) for g in (0.5, 1.0, 2.0)}
    out["v0_le_9"] = agg([r for r in results if r["v0"] <= 9.0])
    out["v0_gt_9"] = agg([r for r in results if r["v0"] > 9.0])
    out["by_kind"] = {k: agg([r for r in results if r["kind"] == k]) for k in ("wall", "pole")}
    return out


def main(n: int = 300) -> Dict:
    rng = np.random.default_rng(SEED)
    scen = make_scenarios(n, rng)
    report = {"seed": SEED, "n_flights": n, "settings": {
        "dt_s": DT, "sonar_rate_hz": 1 / (DT * PING_EVERY), "telemetry_latency_s": TAU_LAT,
        "gps_sigma_m_choices": [0.5, 1.0, 2.0],
        "gps_noise": "white per tick (all runs except v2_gps_correlated_tau5s: Gauss-Markov, tau 5 s, same sigma)",
        "variants": {"v1": "drone_core_v1 unchanged", "v1_gate_off": "diagnostic: v1 with its drone-distance gate "
                     "disabled (voxels still fused into the pose)", "v2": "drone_core_v2",
                     "v2_no_ir": "v2 with no IR at all", "v2_gps_correlated_tau5s": "v2, filter still assumes white GPS"},
        "speed_mps": [2, 12], "heading_deg": [-30, 30], "obstacle_dist_m": [18, 30],
        "obstacles": "70% wall, 30% pole (r=0.3 m) on path", "drone_radius_m": DRONE_R,
        "vehicle_accel_limit_mps2": A_TRUE_MAX, "actuator_lag_s": ACT_TAU,
        "sonar": {"max_range_m": SONAR_MAX, "half_beam_deg": 15, "range_noise_m": RANGE_NOISE,
                  "p_multipath_per_echo": P_MULTIPATH, "p_spike_per_ping": P_SPIKE,
                  "p_dropout_per_ping": P_DROPOUT},
        "ir": "provided whenever sonar has a hit (generous to v1, which drops sonar without IR); bias 0.03 m",
        "controller": "same for both: P-loop on velocity toward v0*heading (v2: its safe_velocity_mps); "
                      "the version's command_accel is applied while its action is a brake/flare",
    }, "runs": {}}
    variants = [("v1", "v1", True, False, 0.0), ("v1_gate_off", "v1", True, True, 0.0),
                ("v2", "v2", True, False, 0.0), ("v2_no_ir", "v2", False, False, 0.0),
                ("v2_gps_correlated_tau5s", "v2", True, False, 5.0)]
    for label, ver, use_ir, gate_off, gtau in variants:
        t0 = time.perf_counter()
        res = [run_flight(sc, ver, use_ir, v1_gate_off=gate_off, gps_tau_s=gtau) for sc in scen]
        el = time.perf_counter() - t0
        s = summarize(res, label)
        s["runtime_s"] = round(el, 1)
        s["collided_ids"] = [r["id"] for r in res if r["collided"]][:50]
        report["runs"][label] = s
        print(f"{label}: {el:.1f}s  collisions={s['all']['collision_rate']}", file=sys.stderr)
    # held-out check: different scenario seed, v1 and v2
    scen_h = make_scenarios(n // 2, np.random.default_rng(SEED + 1))
    for label, ver in (("holdout_seed_v1", "v1"), ("holdout_seed_v2", "v2")):
        t0 = time.perf_counter()
        res = [run_flight(sc, ver) for sc in scen_h]
        s = summarize(res, label)
        s["runtime_s"] = round(time.perf_counter() - t0, 1)
        s["scenario_seed"] = SEED + 1
        report["runs"][label] = s
        print(f"{label}: collisions={s['all']['collision_rate']}", file=sys.stderr)
    # heading-drift scenario (v2 only): main seed and held-out seed kept separate
    report["heading_settings"] = {
        "gyro_z_bias_dps": [-GYRO_BIAS_DPS, GYRO_BIAS_DPS], "yaw_init_err_deg_sigma": YAW_INIT_ERR_DEG,
        "gyro_arw_rad_s_sqrt_hz": v2.GYRO_ARW_RAD_S_SQRT_HZ, "tilt_noise_deg": TILT_NOISE_DEG,
        "tilt_model": "roll/pitch from horizontal thrust acceleration, yaw = heading (constant)",
        "mag_noise_ut": MAG_NOISE_UT, "hard_iron_ut_sigma": HARD_IRON_UT,
        "hard_iron_cal_err_ut": HARD_IRON_CAL_ERR_UT, "soft_iron_sigma": SOFT_IRON_SIGMA,
        "soft_iron_cal_err": SOFT_IRON_CAL_ERR, "motor_k_ut_per_a_sigma": MOTOR_K_UT_PER_A,
        "motor_k_cal_err_frac": MOTOR_K_CAL_ERR, "motor_current_a": "10 + 2*|thrust accel|",
        "bursts": {"start_p_per_tick": BURST_START_P, "len_s": list(BURST_LEN_S), "ut": list(BURST_UT)},
        "site_field": "same assumed declination/field/dip in truth and core (site model assumed correct)",
        "sonar": "body-frame bearing computed through the true tilted attitude",
        "anomaly": {"ut": ANOMALY_UT, "wavelength_m": ANOMALY_WAVELENGTH_M,
                    "model": "world-frame sinusoidal field added to the Earth field, random phases"},
        "landmarks": {"lateral_m": list(LM_LATERAL_M), "placement": "on the obstacle plane ahead",
                      "fov_half_deg": LM_FOV_DEG, "bearing_noise_deg": LM_BEARING_NOISE_DEG,
                      "rate_hz": 1 / (DT * LM_EVERY), "occlusion": "not modeled"},
        "map_error_metric": "distance from each active map point to the true surface (along-surface error not counted); "
                            "world = as stored, relative = re-anchored at the true drone position (what guidance uses)",
    }
    report["heading"] = {}
    for tag, scs in (("main_seed", scen), ("holdout_seed", scen_h)):
        for mode in HEADING_MODES:
            t0 = time.perf_counter()
            res = [run_flight(sc, "v2", heading_mode=mode) for sc in scs]
            hs = summarize_heading(res, f"{tag}_{mode}")
            hs["runtime_s"] = round(time.perf_counter() - t0, 1)
            report["heading"][f"{tag}_{mode}"] = hs
            print(f"heading {tag} {mode}: err med {hs['heading_err_deg']['median']} "
                  f"nees {hs['heading_nees_mean_expect_1']} coll {hs['collision_rate']} "
                  f"maprel {hs['map_point_err_relative_m']}", file=sys.stderr)
    return report


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.perf_counter()
    rep = main(n)
    rep["total_runtime_s"] = round(time.perf_counter() - t0, 1)
    with open("mc_results_v2.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("all", "runtime_s")}
                      for k, v in rep["runs"].items()}, indent=2))

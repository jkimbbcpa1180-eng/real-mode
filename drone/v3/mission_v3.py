# SPDX-License-Identifier: CC0-1.0
"""
drone v3 mission runner.

Flight: the same kinematic truth model, noisy telemetry and drone_core_v2 flight tick as
mc_recon_v2.run_mission (copied here so v2 stays unmodified), split into a reusable
Flight object so a mission can continue (re-passes, a second sensor, a second drone).
Per tick the order is fixed: (1) flight tick on the core (flight-control loop), then
(2) mission-computer work (mapping, thermal, planning) that only READS the core state.
Nothing in (2) feeds back into (1); test_mission_v3 checks the flight outputs are
bit-identical with the heavy work on or off.

run_mission_v3(sc, template, ...) switches the v3 features on/off:
  repass      : "go look closer" re-passes after the first pass (repass_v3)
  swarm_n > 1 : the task split over N drones with relay + map alignment (swarm_v3)
The offline safe-return brain (safe_return_v3) uses its own 2-D wind/IMU simulation and
is run with safe_return_v3.run_return.
  lidar       : a LidarHead (sensors_v3) flown alongside the sonar; its mass and power
                are charged to the battery model (feature 3)
Change detection (change_v3.run_pair: two missions + alignment + differencing) and the
thermal camera (thermal_v3.run: kinematic orbit, not this flight loop) are run from their
own entry points.

License: CC0 1.0 Universal (public domain).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import common_v3 as C
from common_v3 import np, v2, R
import mc_recon_v2 as MR

DT = MR.DT
SPEED = MR.SPEED
AVOID_VMAX = 3.0          # m/s, cap on nominal + avoidance velocity (ASSUMED)
AVOID_MIN_Z_M = 1.0
AVOID_MIN_CLEAR_M = 1.0


class SonarHead:
    """The v2 MC sonar: gimballed 5x5 beams in +/-15 deg, 15 m, drops/misses/multipath."""
    name = "sonar"
    extra_power_w = 0.0              # already in the v2 critical load (acoustic 3.5 W)
    extra_mass_kg = 0.0
    range_noise = MR.RANGE_NOISE
    bearing_noise = MR.BEARING_NOISE
    max_range = MR.SONAR_MAX

    def scan(self, p: np.ndarray, look: np.ndarray, scene, rng) -> List:
        out = []
        if rng.random() < MR.P_PING_DROP:
            return out
        for u in MR._beam_dirs(look, rng):
            if rng.random() < MR.P_BEAM_MISS:
                continue
            r = scene.raycast(p, u, self.max_range)
            if r is None:
                continue
            if rng.random() < MR.P_MULTIPATH:
                r *= rng.uniform(1.3, 2.5)
                if r > self.max_range:
                    continue
            r_m = r + rng.normal(0, self.range_noise)
            um = u + rng.normal(0, self.bearing_noise, 3)
            out.append((um / np.linalg.norm(um), r_m, self.range_noise, self.bearing_noise))
        return out


class Flight:
    """One drone: truth kinematics + v2 core + map. fly() can be called repeatedly."""

    def __init__(self, sc: Dict, scene, gps: Sequence[float], seed: int, start: np.ndarray,
                 sensors: Sequence = (SonarHead(),), rmap: Optional[R.ReconMap] = None,
                 battery_wh: float = C.BATTERY_WH, gps_bias0: Optional[np.ndarray] = None):
        self.sc, self.scene = sc, scene
        self.rng = np.random.default_rng(seed)             # truth / telemetry noise
        self.srng = np.random.default_rng(seed + 7919)     # sensor noise (mission computer)
        rng = self.rng
        self.gsig, self.gtau = float(gps[0]), float(gps[1])
        self.core = v2.AutonomousDroneCore(heading_init_sigma_rad=math.radians(MR.YAW_INIT_ERR_DEG))
        cal = self.core.heading.cal
        Dd, Ii, F = cal.declination_rad, cal.dip_rad, cal.field_ut
        self.b_w = F * np.array([math.cos(Ii) * math.sin(Dd), math.cos(Ii) * math.cos(Dd), -math.sin(Ii)])
        self.gyro_bias = math.radians(rng.uniform(-MR.GYRO_BIAS_DPS, MR.GYRO_BIAS_DPS))
        self.yaw_err0 = math.radians(rng.normal(0, MR.YAW_INIT_ERR_DEG))
        self.hard_resid = rng.normal(0, MR.HARD_IRON_RESID_UT, 3)
        self.gps_err = rng.normal(0, self.gsig, 3) if gps_bias0 is None else np.asarray(gps_bias0, float).copy()
        self.sensors = list(sensors)
        self.rmap = rmap if rmap is not None else R.ReconMap()
        self.p = np.asarray(start, float).copy()
        self.yaw_prev = None
        self.t = 0.0
        self.path: List[np.ndarray] = []
        self.pos_err: List[float] = []
        self.min_clear = float("inf")
        self.energy_wh = float(battery_wh)
        self.capacity_wh = float(battery_wh)
        self.energy_used_wh = 0.0
        self.flight_log: List = []       # flight-tick outputs (for the isolation test)
        self.hooks: List[Callable] = []  # mission-computer work after each tick: f(flight, hits, p_hat, P, sig_h, dyaw)
        self.p_hat = self.p.copy()
        self.mission_ms: List[float] = []
        self.hold = False
        self.escape_up = False
        self.hold_ticks = 0
        self.avoid_v: Optional[np.ndarray] = None   # swarm reciprocal avoidance velocity (set per tick)
        self.avoid_ticks = 0
        self.gps_err_sum = np.zeros(3)   # truth: mean GPS error over the flight (for alignment checks)
        self.gps_err_n = 0

    def power_w(self, speed: float) -> float:
        # payload mass raises hover power as (m / m0)^1.5 (momentum theory, ASSUMED m0 = 1.45 kg, 55 W)
        dm = sum(s.extra_mass_kg for s in self.sensors)
        hover_extra = 55.0 * (((1.45 + dm) / 1.45) ** 1.5 - 1.0)
        return C.power_w(speed, sum(s.extra_power_w for s in self.sensors) + hover_extra)

    def fly(self, wps: Sequence, max_s: float = MR.MAX_MISSION_S, speed: float = SPEED,
            log_flight: bool = False, map_on: bool = True) -> bool:
        """Follow waypoints (R.Waypoint). Returns True if all were reached."""
        done = False
        for done in self.fly_iter(wps, max_s, speed, log_flight, map_on):
            pass
        return done

    def fly_iter(self, wps: Sequence, max_s: float = MR.MAX_MISSION_S, speed: float = SPEED,
                 log_flight: bool = False, map_on: bool = True):
        """Generator: one flight tick per step; yields True once all waypoints are reached.
        If self.hold is set before a step, the drone holds position for that tick."""
        rng = self.rng
        phi = math.exp(-DT / self.gtau)
        i_wp, t_end = 0, self.t + max_s
        while i_wp < len(wps) and self.t < t_end:
            self.t += DT
            t = self.t
            tgt = wps[i_wp].pos
            d = tgt - self.p
            dist = float(np.linalg.norm(d))
            step = 0.0 if self.hold else speed * DT
            p_old = self.p
            if self.hold:
                self.hold_ticks += 1
                if self.escape_up:           # swarm deconfliction: climb out of the conflict
                    self.p = self.p + np.array([0.0, 0.0, 1.0 * DT])
            elif self.avoid_v is not None:      # swarm: nominal velocity + reciprocal avoidance
                self.avoid_ticks += 1
                nom = d / dist * speed if dist > 1e-9 else np.zeros(3)
                ua = self.avoid_v / max(float(np.linalg.norm(self.avoid_v)), 1e-9)
                na = float(nom @ ua)
                if na < 0.0:                    # ORCA-like half-plane: drop the part of the nominal
                    nom = nom - na * ua         # velocity that drives into the conflict
                v = nom + self.avoid_v
                s = float(np.linalg.norm(v))
                if s > AVOID_VMAX:
                    v = v * (AVOID_VMAX / s)
                q = self.p + v * DT
                q[2] = max(q[2], AVOID_MIN_Z_M)
                # local obstacle sensing (ASSUMED; truth clearance used as its proxy): never let the
                # avoidance move the drone closer than AVOID_MIN_CLEAR_M to a surface -> vertical only
                cq = float(self.scene.surface_distance(q[None])[0])
                if cq < AVOID_MIN_CLEAR_M and cq < float(self.scene.surface_distance(self.p[None])[0]):
                    q = self.p + np.array([0.0, 0.0, v[2] * DT])
                    q[2] = max(q[2], AVOID_MIN_Z_M)
                self.p = q
                if float(np.linalg.norm(tgt - self.p)) <= speed * DT:
                    i_wp += 1
            elif dist <= step:
                self.p = tgt.copy(); i_wp += 1
            else:
                self.p = self.p + d / dist * step
            p = self.p
            vel = (p - p_old) / DT
            look = wps[min(i_wp, len(wps) - 1)].look_dir
            hz = np.array([look[0], look[1]])
            if np.linalg.norm(hz) > 0.1:
                yaw = math.atan2(hz[1], hz[0])
            elif np.linalg.norm(vel[:2]) > 1e-6:
                yaw = math.atan2(vel[1], vel[0])
            else:
                yaw = self.yaw_prev if self.yaw_prev is not None else 0.0
            rate = 0.0 if self.yaw_prev is None else ((yaw - self.yaw_prev + math.pi) % (2 * math.pi) - math.pi) / DT
            self.yaw_prev = yaw
            self.path.append(p.copy())
            self.min_clear = min(self.min_clear, float(self.scene.surface_distance(p[None])[0]))
            self.gps_err = phi * self.gps_err + math.sqrt(1 - phi * phi) * rng.normal(0, self.gsig, 3)
            self.gps_err_sum += self.gps_err
            self.gps_err_n += 1
            tm = v2.AirframeState(timestamp_s=t, pos_m=p + self.gps_err, vel_mps=vel + rng.normal(0, MR.VEL_NOISE, 3),
                                  acc_mps2=rng.normal(0, MR.ACC_NOISE, 3), jerk_mps3=np.zeros(3),
                                  attitude=v2.Attitude.from_euler(0.0, 0.0, yaw + self.yaw_err0),
                                  pos_sigma_m=np.full(3, self.gsig), vel_sigma_mps=np.full(3, MR.VEL_NOISE))
            tm.acc_sigma_mps2 = np.full(3, MR.ACC_NOISE)
            tm.gyro_rps = np.array([0.0, 0.0, rate + self.gyro_bias]) + rng.normal(0, v2.GYRO_ARW_RAD_S_SQRT_HZ / math.sqrt(DT), 3)
            tm.mag_body_ut = MR._rz(yaw).T @ self.b_w + self.hard_resid + rng.normal(0, MR.MAG_NOISE_UT, 3)
            # (1) flight-control loop
            out = self.core.process_flight_tick(tm, 0.0, [], horizon_s=DT, desired_velocity_mps=vel)
            if log_flight:
                fg = out["flight_guidance"]
                self.flight_log.append((np.asarray(fg["safe_velocity_mps"], float).copy(),
                                        np.asarray(fg["command_accel_mps2"], float).copy(),
                                        self.core.kf.x.copy()))
            e = self.power_w(float(np.linalg.norm(vel))) * DT / 3600.0
            self.energy_wh -= e
            self.energy_used_wh += e
            # (2) mission computer: reads the core state only
            p_hat = self.core.kf.x[0:3].copy()
            self.p_hat = p_hat
            P = self.core.kf.P[0:3, 0:3]
            he = out["heading"]
            yaw_hat = math.radians(he["yaw_enu_deg"])
            sig_h = math.radians(he["heading_sigma_deg"])
            dyaw = (yaw_hat - yaw + math.pi) % (2 * math.pi) - math.pi
            self.pos_err.append(float(np.linalg.norm(p_hat - p)))
            if not map_on:
                yield i_wp >= len(wps)
                continue
            t0 = time.perf_counter()
            Rz = MR._rz(dyaw)
            all_hits = []
            for sen in self.sensors:
                hits = sen.scan(p, look, self.scene, self.srng)
                beams = []
                for um, r_m, sr, sb in hits:
                    u_est = Rz @ um
                    beams.append((u_est, r_m, R.point_covariance(r_m, u_est, sr, sb, sig_h, P)))
                if beams:
                    self.rmap.add_ping(p_hat, beams, sen.range_noise, t)
                all_hits.append((sen, beams))
            for h in self.hooks:
                h(self, all_hits, p_hat, P, sig_h, dyaw)
            self.mission_ms.append((time.perf_counter() - t0) * 1000.0)
            yield i_wp >= len(wps)


def map_metrics(rmap: R.ReconMap, scene, targets) -> Dict:
    pts, _ = rmap.filtered_points()
    acc = scene.surface_distance(pts) if len(pts) else np.array([np.nan])
    return dict(coverage=R.coverage(targets, pts, R.COVERAGE_TOL_M) if len(targets) else float("nan"),
                acc_median=float(np.nanmedian(acc)), acc_p95=float(np.nanpercentile(acc, 95)),
                effective_detail_m=rmap.effective_detail_m(), n_filtered=int(len(pts)))


def run_mission_v3(sc: Dict, template: str, gps_mode: str = "gps_standard", repass: bool = False,
                   swarm_n: int = 1, seed_offset: int = 0, max_extra_s: float = 300.0,
                   lidar: bool = False, lidar_dropout: float = 0.3) -> Dict:
    """One recon mission with v3 features toggled. All off = the v2 behaviour."""
    if swarm_n > 1:
        import swarm_v3 as SW
        return SW.run_swarm(sc, template, swarm_n, gps_mode)
    import repass_v3 as RP
    wps, scene, targets = MR.plan(sc, template)
    sensors = [SonarHead()]
    if lidar:
        import sensors_v3 as SE
        a_lo, a_hi = sc["area"]
        sensors.append(SE.LidarHead([sc["building"], sc["poi"]], (np.asarray(a_lo) - 20, np.asarray(a_hi) + 20),
                                    [lidar_dropout, lidar_dropout]))
    f = Flight(sc, scene, MR.GPS_MODES[gps_mode], sc["seed"] + 13 + seed_offset, wps[0].pos, sensors=sensors)
    f.fly(wps)
    out = {"first_pass": map_metrics(f.rmap, scene, targets), "first_pass_s": round(f.t, 1),
           "first_pass_wh": f.energy_used_wh}
    if repass:
        t0, e0 = f.t, f.energy_used_wh
        mission = "facades" if template != "lawnmower" else "roofs+ground"
        focus = {"perimeter_orbit": sc["building"], "poi_close_pass": sc["poi"]}.get(template)
        focus = None if focus is None else (focus[0] - 2.0, focus[1] + 2.0)
        gaps, _ = RP.find_gaps(f.rmap, mission, area=sc["area"], focus=focus)
        plan = RP.plan_repasses(gaps, f.rmap, f.p_hat, wps[0].pos, MR.STANDOFF[template], MR.BEAM_HALF,
                                f.energy_wh, f.capacity_wh, f.power_w(SPEED), SPEED, max_extra_s)
        if plan.waypoints:
            f.fly(plan.waypoints)
        out.update(after_repass=map_metrics(f.rmap, scene, targets), extra_s=round(f.t - t0, 1),
                   extra_wh=f.energy_used_wh - e0, repasses=len(plan.chosen), skipped=plan.skipped)
    out["_flight"] = f
    return out

"""
REAL MODE: Integrated Autonomous Drone Navigation & Predictive Core (v2)
Unifies:
  1. Kalman filter (9-state [p, v, a], linear drag) fed by GPS position,
     velocity and IMU acceleration. Process noise Q is derived from the
     continuous model (white jerk through the drag dynamics, Van Loan method).
  2. Acoustic ranging with multi-echo median, consistency/SNR gating,
     sensor noise floors, round-trip motion compensation, Cramer (1993)
     sound speed, and optional IR cross-checks paired by bearing and time.
  3. An obstacle map: echoes from unknown obstacles update the MAP, never the
     drone's pose. Pose corrections come only from KNOWN landmarks.
  4. Power management: flight-critical loads are never scaled; optional
     sensors are shed; low charge gives RETURN_HOME or LAND.
  5. Guidance from closest point of approach, time to contact and stopping
     distance v^2 / (2 a_max), using the latency-extrapolated current state.
  6. Heading: a 2-state [yaw, gyro-z bias] filter driven by the gyro and
     corrected by a calibrated, tilt-compensated, gated magnetometer. Roll and
     pitch are taken from the telemetry attitude (not estimated here).

v2 fixes the issues listed in review_v1.md (see README.md for the mapping).

Assumptions & limitations (values marked ASSUMED below are settings, not
measurements):
  - Local ENU frame, point-mass airframe with linear drag. Body frame x forward,
    y left, z up; yaw is measured counter-clockwise from East (ENU convention).
  - Single-bounce acoustic model; multipath rejected by median + gating.
  - Sonar beam pattern, rotor wash and aerodynamics are not modeled.
  - Obstacles are treated as static points.

Dependencies: numpy only.
License: CC0 1.0 Universal (public domain). Copy, modify, use freely.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Settings (ASSUMED unless stated otherwise)
# ---------------------------------------------------------------------------
RANGE_SIGMA_FLOOR_M = 0.02        # ASSUMED sonar range noise floor (sensor spec stand-in)
ANGLE_SIGMA_FLOOR_RAD = math.radians(2.0)  # ASSUMED bearing noise floor (beam width stand-in)
MAX_RANGE_SPREAD_M = 0.15         # ASSUMED: echoes in a group must agree within this
MIN_GROUP_ECHOES = 1              # single echoes allowed, with the noise floor
IR_PAIR_MAX_ANGLE_RAD = math.radians(10.0)  # ASSUMED IR/acoustic pairing window
IR_PAIR_MAX_DT_S = 0.05           # ASSUMED
A_PLAN_MPS2 = 4.0                 # ASSUMED braking deceleration used for planning
A_EMERGENCY_MPS2 = 5.0            # ASSUMED vehicle braking limit
T_REACT_S = 0.30                  # ASSUMED ping interval 0.1 + latency 0.05 + tick 0.05 + actuator 0.1
DRONE_RADIUS_M = 0.25             # ASSUMED
STANDOFF_M = 1.0                  # ASSUMED clearance to keep from obstacles
BLIND_SPEED_LIMIT_MPS = 2.0       # ASSUMED speed limit when the sonar is blind
BLIND_AFTER_S = 0.35              # ASSUMED (3 missed pings at 10 Hz)
MAP_TIMEOUT_S = 0.5               # ASSUMED: forget a point this long after it is in view but not re-seen
MAP_MEMORY_S = 3.0                # ASSUMED: keep points that left the sonar field of view this long
SONAR_HALF_FOV_RAD = math.radians(15.0)  # ASSUMED sonar half beam width (used for map expiry only)
ULTRASONIC_ABSORPTION_DB_PER_M = 1.3  # ASSUMED ~40 kHz value, NOT verified (ISO 9613-1)
KIN_TRUST_REF_SIGMA_M = 1.0       # ASSUMED: kinematic trust = 1 / (1 + sigma_p / ref)
# Heading / magnetometer (all ASSUMED; set them for the actual site and sensors)
GYRO_ARW_RAD_S_SQRT_HZ = 3.0e-4   # ASSUMED gyro angle random walk (rad/s/sqrt(Hz))
GYRO_BIAS_RW_RAD_S_SQRT_S = 2.0e-5  # ASSUMED gyro-z bias random walk (rad/s/sqrt(s))
GYRO_BIAS_INIT_SIGMA_RAD_S = 0.02 # ASSUMED initial gyro-z bias sigma (~1.1 deg/s)
HEADING_INIT_SIGMA_RAD = math.radians(30.0)  # ASSUMED sigma of the telemetry yaw used at start
MAG_GATE_CHI2 = 6.63              # 99% chi-square, 1 dof (innovation gate)
MAG_REACQUIRE_AFTER = 20          # ASSUMED: consecutive field-OK innovation rejections before
                                  # the heading variance is inflated to re-acquire
LM_REACQUIRE_AFTER = 6            # ASSUMED: consecutive known-landmark bearing rejections before
                                  # the heading variance is inflated to the landmark innovation
HEADING_SLOW_SIGMA_RAD = math.radians(10.0)  # ASSUMED: above this heading sigma, slow down
HEADING_SLOW_SPEED_MPS = 2.0      # ASSUMED speed limit while heading is uncertain
HEADING_TRUST_REF_RAD = math.radians(10.0)   # ASSUMED: heading trust = 1 / (1 + sigma / ref)


# ===========================================================================
# Attitude: quaternion, body -> ENU (verified against to_matrix in tests)
# ===========================================================================
@dataclass
class Attitude:
    """Unit quaternion (w, x, y, z) mapping body -> local ENU: v_enu = q v_body q*."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def from_yaw(yaw_rad: float) -> "Attitude":
        return Attitude(math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> "Attitude":
        """ZYX convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return Attitude(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)

    def to_euler(self) -> Tuple[float, float, float]:
        """(roll, pitch, yaw) for R = Rz(yaw) Ry(pitch) Rx(roll)."""
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def normalize(self) -> "Attitude":
        n = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if n < 1e-12:
            return Attitude()
        return Attitude(self.w / n, self.x / n, self.y / n, self.z / n)

    def rotate(self, v_body: np.ndarray) -> np.ndarray:
        q = self.normalize()
        v = np.asarray(v_body, float)
        qv = np.array([q.x, q.y, q.z])
        t = 2.0 * np.cross(qv, v)
        return v + q.w * t + np.cross(qv, t)

    def inverse_rotate(self, v_enu: np.ndarray) -> np.ndarray:
        q = self.normalize()
        return Attitude(q.w, -q.x, -q.y, -q.z).rotate(v_enu)

    def to_matrix(self) -> np.ndarray:
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    w = math.fmod(a + math.pi, 2.0 * math.pi)
    if w <= 0.0:
        w += 2.0 * math.pi
    return w - math.pi


def body_ray(az: float, el: float) -> np.ndarray:
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


# ===========================================================================
# Data containers
# ===========================================================================
@dataclass
class AirframeState:
    timestamp_s: float              # time the telemetry describes (not arrival time)
    pos_m: np.ndarray
    vel_mps: np.ndarray
    acc_mps2: np.ndarray            # IMU acceleration (used as a measurement in v2)
    jerk_mps3: np.ndarray           # kept for compatibility; not used by the filter
    attitude: Attitude = field(default_factory=Attitude)
    pos_sigma_m: np.ndarray = field(default_factory=lambda: np.full(3, 0.05))
    vel_sigma_mps: np.ndarray = field(default_factory=lambda: np.full(3, 0.10))
    acc_sigma_mps2: np.ndarray = field(default_factory=lambda: np.full(3, 0.30))  # ASSUMED IMU
    gyro_rps: Optional[np.ndarray] = None       # mean body rates over the interval ending at timestamp
    mag_body_ut: Optional[np.ndarray] = None    # raw magnetometer, body frame, microtesla
    motor_current_a: float = 0.0                # total motor current (for mag compensation)


@dataclass
class AcousticSensorReturn:
    azimuth_rad: float          # body-frame sensor azimuth
    elevation_rad: float        # body-frame sensor elevation
    tof_seconds: float
    echo_amplitude: float
    timestamp_s: float = 0.0    # reception time
    snr_db: float = 20.0


@dataclass
class InfraredReturn:
    depth_m: float
    timestamp_s: float = 0.0
    confidence: float = 0.9
    surface_angle_rad: Optional[float] = None  # angle between ray and surface normal
    azimuth_rad: float = 0.0    # ASSUMED boresight if the sensor does not report bearing
    elevation_rad: float = 0.0


SAT_TERMINAL_W = 50.0   # ASSUMED average draw of a satellite terminal ("tens of watts"); not a spec


@dataclass
class BatteryState:
    capacity_wh: float
    remaining_wh: float
    voltage_v: float = 11.1
    draw_w: Dict[str, float] = field(default_factory=lambda: {
        "avionics": 2.0,
        "acoustic": 3.5,
        "infrared": 1.2,
        "motors_hover": 55.0,
        # OPTIONAL, ASSUMED (not measured): a satellite-internet terminal for the cloud
        # uplink. Only drawn when requested in duty_requests; shed like any optional load.
        "satellite_terminal": SAT_TERMINAL_W,
    })

    @property
    def soc(self) -> float:
        return max(0.0, min(1.0, self.remaining_wh / max(self.capacity_wh, 1e-9)))

    def consume(self, energy_wh: float) -> None:
        """Record energy actually drawn (from a battery monitor or a simulator). No reserve floor."""
        self.remaining_wh = max(0.0, self.remaining_wh - max(0.0, energy_wh))


# ===========================================================================
# Speed of sound: Cramer (1993)
# ===========================================================================
_CRAMER_A = (331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782,
             -1.82e-7, 3.73e-8, -2.93e-10, -85.20931, -0.228525, 5.91e-5,
             -2.835149, -2.15e-13, 29.179762, 0.000486)


def sound_speed_cramer(temperature_c: float, humidity_pct: float = 0.0,
                       pressure_pa: float = 101325.0, co2_frac: float = 0.0004) -> float:
    """Cramer (1993) speed of sound in humid air (valid roughly 0-30 C; extrapolated outside)."""
    a, t, p = _CRAMER_A, float(temperature_c), float(pressure_pa)
    T = t + 273.15
    psv = math.exp(1.2378847e-5 * T * T - 1.9121316e-2 * T + 33.93711047 - 6.3431645e3 / T)
    f = 1.00062 + 3.14e-8 * p + 5.6e-7 * t * t
    xw = max(0.0, min(humidity_pct, 100.0)) / 100.0 * f * psv / p
    xc = co2_frac
    return (a[0] + a[1] * t + a[2] * t * t
            + (a[3] + a[4] * t + a[5] * t * t) * xw
            + (a[6] + a[7] * t + a[8] * t * t) * p
            + (a[9] + a[10] * t + a[11] * t * t) * xc
            + a[12] * xw * xw + a[13] * p * p + a[14] * xc * xc + a[15] * xw * p * xc)


# ===========================================================================
# Multi-echo robust ranger with consistency/SNR gating
# ===========================================================================
@dataclass
class AcousticEchoGroup:
    echoes: List[AcousticSensorReturn]


class AcousticRanger:
    """
    Turns one group of echoes from one sensor ping into a range + bearing.

    Gating uses ONLY the ranger's own evidence (SNR, amplitude, range window,
    bearing jitter, and agreement between echoes). It never compares the echo
    with the drone's position: obstacles are not expected to be near the drone.
    Sigmas never drop below the sensor noise floors (ASSUMED values above).
    """

    def __init__(
        self,
        c_sound_mps: float,
        min_snr_db: float = 10.0,
        min_amplitude: float = 0.15,
        min_range_m: float = 0.1,
        max_range_m: float = 15.0,
        max_angle_jitter_rad: float = math.radians(8.0),
        max_range_spread_m: float = MAX_RANGE_SPREAD_M,
        range_sigma_floor_m: float = RANGE_SIGMA_FLOOR_M,
        angle_sigma_floor_rad: float = ANGLE_SIGMA_FLOOR_RAD,
    ):
        self.c_sound = float(c_sound_mps)
        self.min_snr_db = float(min_snr_db)
        self.min_amplitude = float(min_amplitude)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.max_angle_jitter_rad = float(max_angle_jitter_rad)
        self.max_range_spread_m = float(max_range_spread_m)
        self.range_sigma_floor_m = float(range_sigma_floor_m)
        self.angle_sigma_floor_rad = float(angle_sigma_floor_rad)

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        return float(np.median(np.abs(x - np.median(x))) * 1.4826) if len(x) > 1 else 0.0

    def estimate(self, group: AcousticEchoGroup) -> Optional[Dict]:
        """Return a range estimate dict, or None (with no echo / all echoes rejected)."""
        if not group.echoes:
            return None
        kept = [e for e in group.echoes
                if e.snr_db >= self.min_snr_db and e.echo_amplitude >= self.min_amplitude]
        if not kept:
            return None
        rng = np.array([0.5 * e.tof_seconds * self.c_sound for e in kept])
        ok = (rng >= self.min_range_m) & (rng <= self.max_range_m)
        kept = [e for e, k in zip(kept, ok) if k]
        if not kept:
            return None
        az = np.array([e.azimuth_rad for e in kept])
        el = np.array([e.elevation_rad for e in kept])
        el_med = float(np.median(el))
        d_ang = np.hypot((az - np.median(az)) * math.cos(el_med), el - el_med)  # azimuth scaled by cos(el)
        kept = [e for e, k in zip(kept, d_ang <= self.max_angle_jitter_rad) if k]
        if not kept:
            return None
        # Range consistency: keep echoes within max_range_spread of the
        # median range (multipath arrives late and is rejected here).
        rng = np.array([0.5 * e.tof_seconds * self.c_sound for e in kept])
        r_med = float(np.median(rng))
        close = np.abs(rng - r_med) <= self.max_range_spread_m
        if np.sum(close) < max(MIN_GROUP_ECHOES, 1):
            return None
        # If the median itself sits on a multipath cluster (e.g. 2 late, 1 early),
        # prefer the earliest consistent cluster: first arrivals are direct paths.
        order = np.argsort(rng)
        first = rng[order[0]]
        first_cluster = np.abs(rng - first) <= self.max_range_spread_m
        if np.sum(first_cluster) >= np.sum(close) or np.sum(first_cluster) >= 2:
            close = first_cluster
        kept = [e for e, k in zip(kept, close) if k]

        tofs = np.array([e.tof_seconds for e in kept])
        azs = np.array([e.azimuth_rad for e in kept])
        els = np.array([e.elevation_rad for e in kept])
        snrs = np.array([e.snr_db for e in kept])
        amps = np.array([e.echo_amplitude for e in kept])
        tof_med = float(np.median(tofs))
        range_med = 0.5 * tof_med * self.c_sound
        n_kept, n_total = len(kept), len(group.echoes)
        # Sigma: spread-based, never below the sensor floor. With one echo the
        # spread is unknown, so the floor is used (stated assumption).
        range_sigma = max(0.5 * self._mad(tofs) * self.c_sound, self.range_sigma_floor_m)
        az_sigma = max(self._mad(azs), self.angle_sigma_floor_rad)
        el_sigma = max(self._mad(els), self.angle_sigma_floor_rad)
        rep = AcousticSensorReturn(
            azimuth_rad=float(np.median(azs)), elevation_rad=float(np.median(els)),
            tof_seconds=tof_med, echo_amplitude=float(np.clip(np.mean(amps), 0.0, 1.0)),
            timestamp_s=float(np.median([e.timestamp_s for e in kept])),
            snr_db=float(np.mean(snrs)),
        )
        return {
            "representative": rep,
            "range_m": range_med,
            "range_sigma_m": range_sigma,
            "az_sigma_rad": az_sigma,
            "el_sigma_rad": el_sigma,
            "n_kept": n_kept,
            "n_total": n_total,
            "outlier_ratio": 1.0 - n_kept / n_total,
            "single_echo": n_kept == 1,
        }


# ===========================================================================
# Kalman filter: 9-state drag-consistent
# ===========================================================================
def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential (scaling and squaring + Taylor); fine for small matrices."""
    norm = float(np.max(np.sum(np.abs(M), axis=1)))
    s = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0.5 else 0
    A = M / (2.0 ** s)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, 16):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


class DragConsistentKF:
    """
    9-state [p, v, a] Kalman filter, per axis:
        dp/dt = v,  dv/dt = a - lam * v,  da/dt = w   (w: white jerk, PSD q)
    F = expm(A dt). Q is the exact discretization of the white-jerk noise through
    the same drag dynamics (Van Loan); for lam -> 0 it equals the standard
    white-jerk Q = q [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2],
    [dt^3/6, dt^2/2, dt]]. There is no separate white-acceleration term.

    Measurement updates:
      - update_pos(z_pos, R): GPS position
      - update_vel(z_vel, R): GPS velocity
      - update_acc(z_acc, R): IMU acceleration
      - update_landmark_range(...): range to a KNOWN landmark (pose correction)
    Obstacle echoes from unknown obstacles do NOT update this filter.
    """

    def __init__(
        self,
        drag_k: float,
        mass_kg: float,
        jerk_psd: float = 25.0,       # ASSUMED q (m^2/s^5); see README
        sigma_pos_meas: float = 0.05,
        sigma_vel_meas: float = 0.10,
    ):
        self.lam = drag_k / max(mass_kg, 1e-9)
        self.q = float(jerk_psd)
        self.R_pos = np.eye(3) * sigma_pos_meas**2
        self.R_vel = np.eye(3) * sigma_vel_meas**2
        self.x = np.zeros(9)
        self.P = np.eye(9) * 1.0
        self.initialized = False
        A = np.array([[0.0, 1.0, 0.0], [0.0, -self.lam, 1.0], [0.0, 0.0, 0.0]])
        self._A1 = A
        self._cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    def _FQ(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        key = round(float(dt), 9)
        if key in self._cache:
            return self._cache[key]
        A = self._A1
        G = np.array([[0.0], [0.0], [1.0]])
        M = np.zeros((6, 6))
        M[0:3, 0:3] = -A
        M[0:3, 3:6] = G @ G.T * self.q
        M[3:6, 3:6] = A.T
        E = _expm(M * dt)
        F1 = E[3:6, 3:6].T
        Q1 = F1 @ E[0:3, 3:6]
        Q1 = 0.5 * (Q1 + Q1.T)
        if dt < 0:
            Q1 = np.zeros((3, 3))   # back-projection carries no extra process noise term
        out = (np.kron(F1, np.eye(3)), np.kron(Q1, np.eye(3)))
        if len(self._cache) < 256:
            self._cache[key] = out
        return out

    def _F(self, dt: float) -> np.ndarray:
        return self._FQ(dt)[0]

    def _Q(self, dt: float) -> np.ndarray:
        return self._FQ(dt)[1]

    def initialize(self, state: AirframeState) -> None:
        self.x[0:3] = np.asarray(state.pos_m, float)
        self.x[3:6] = np.asarray(state.vel_mps, float)
        self.x[6:9] = np.asarray(state.acc_mps2, float)
        ps = np.asarray(state.pos_sigma_m, float) ** 2
        vs = np.asarray(state.vel_sigma_mps, float) ** 2
        acs = np.asarray(state.acc_sigma_mps2, float) ** 2
        self.P = np.diag(np.concatenate([ps, vs, acs]))
        self.initialized = True

    def predict(self, dt: float) -> None:
        if not self.initialized or dt <= 0:
            return
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _kalman_update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> float:
        """Joseph-form update. Returns the normalized innovation squared (NIS)."""
        y = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ y
        IKH = np.eye(9) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return float(y @ np.linalg.solve(S, y))

    def update_pos(self, z_pos, R=None) -> float:
        if not self.initialized:
            return float("nan")
        H = np.zeros((3, 9)); H[0:3, 0:3] = np.eye(3)
        return self._kalman_update(H, z_pos, self.R_pos if R is None else R)

    def update_vel(self, z_vel, R=None) -> float:
        if not self.initialized:
            return float("nan")
        H = np.zeros((3, 9)); H[0:3, 3:6] = np.eye(3)
        return self._kalman_update(H, z_vel, self.R_vel if R is None else R)

    def update_acc(self, z_acc, R) -> float:
        if not self.initialized:
            return float("nan")
        H = np.zeros((3, 9)); H[0:3, 6:9] = np.eye(3)
        return self._kalman_update(H, z_acc, R)

    def update_landmark_range(self, landmark_enu: np.ndarray, range_m: float,
                              range_sigma_m: float, gate_nis: float = 9.0) -> Optional[float]:
        """
        Pose correction from a range to a KNOWN, surveyed landmark:
        z = |l - p| + noise, H = -(l - p)^T / |l - p| on the position block.
        Innovations beyond gate_nis (chi-square, 1 dof) are rejected (returns None).
        """
        if not self.initialized:
            return None
        diff = np.asarray(landmark_enu, float) - self.x[0:3]
        d = float(np.linalg.norm(diff))
        if d < 1e-6:
            return None
        H = np.zeros((1, 9)); H[0, 0:3] = -diff / d
        S = float((H @ self.P @ H.T)[0, 0]) + range_sigma_m**2
        y = float(range_m) - d
        if y * y / S > gate_nis:
            return None
        K = (self.P @ H.T) / S
        self.x = self.x + (K * y).ravel()
        IKH = np.eye(9) - K @ H
        self.P = IKH @ self.P @ IKH.T + (K @ K.T) * range_sigma_m**2
        self.P = 0.5 * (self.P + self.P.T)
        return y * y / S

    def state_at(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """State extrapolated dt seconds from the filter time (dt may be negative)."""
        if not self.initialized:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        x = self._F(dt) @ self.x
        return x[0:3].copy(), x[3:6].copy(), x[6:9].copy()

    def covariance_at(self, dt: float) -> np.ndarray:
        if not self.initialized:
            return np.eye(9)
        F, Q = self._FQ(dt)
        return F @ self.P @ F.T + Q

    def sigmas(self, dt: float = 0.0) -> Dict[str, float]:
        """RMS per-axis 1-sigma of position / velocity at filter time + dt."""
        P = self.covariance_at(dt)
        return {"pos_sigma_m": math.sqrt(max(np.trace(P[0:3, 0:3]), 0.0) / 3.0),
                "vel_sigma_mps": math.sqrt(max(np.trace(P[3:6, 3:6]), 0.0) / 3.0)}

    def trust(self, dt: float = 0.0) -> float:
        """
        Monotone kinematic trust: 1 / (1 + sigma_p / KIN_TRUST_REF_SIGMA_M), no floor.
        sigma_p 0.1 m -> 0.91, 1 m -> 0.50, 3 m -> 0.25. It is a readable summary of
        the reported sigma, NOT a calibrated probability; use the sigmas for decisions.
        """
        if not self.initialized:
            return 0.0
        return 1.0 / (1.0 + self.sigmas(dt)["pos_sigma_m"] / KIN_TRUST_REF_SIGMA_M)


# ===========================================================================
# IR bias estimator (corrects IR only)
# ===========================================================================
class IRBiasEstimator:
    """
    Recursive estimate of the systematic IR depth bias (IR minus acoustic).
    Updates only when the acoustic range is tight (< 5 cm sigma), the IR
    confidence is high (> 0.7) and the residual is within max_bias_m + 3 sigma.
    The estimate corrects IR depths only; acoustic ranges are never shifted.
    """

    def __init__(self, alpha0: float = 0.05, min_alpha: float = 0.01, max_samples: int = 500,
                 ir_sigma_floor_m: float = 0.02, max_bias_m: float = 0.15):
        self.bias_m = 0.0
        self.alpha0 = alpha0
        self.min_alpha = min_alpha
        self.n = 0
        self.max_samples = max_samples
        self.ir_sigma_floor_m = ir_sigma_floor_m  # ASSUMED IR noise floor
        self.max_bias_m = max_bias_m              # ASSUMED largest plausible IR bias

    def update(self, acoustic_range_m: float, acoustic_sigma_m: float,
               ir_depth_m: float, ir_confidence: float) -> float:
        if ir_confidence < 0.7 or acoustic_sigma_m > 0.05:
            return self.bias_m
        resid = ir_depth_m - acoustic_range_m
        # Gate: residual must be explainable as bias + noise. A bias larger than
        # max_bias_m (ASSUMED) is treated as a wrong pairing / different surface.
        combined = math.sqrt(acoustic_sigma_m**2 + self.ir_sigma_floor_m**2)
        if abs(resid) > self.max_bias_m + 3.0 * combined:
            return self.bias_m
        self.n = min(self.n + 1, self.max_samples)
        alpha = max(self.min_alpha, self.alpha0 * (1.0 - self.n / self.max_samples))
        self.bias_m = (1.0 - alpha) * self.bias_m + alpha * resid
        return self.bias_m

    def correct(self, ir_depth_m: float) -> float:
        return ir_depth_m - self.bias_m


# ===========================================================================
# Heading: gyro-driven yaw + gyro-z bias, corrected by a magnetometer
# ===========================================================================
@dataclass
class MagCalibration:
    """
    Magnetometer model: raw = S (B_body + disturbance) + hard_iron + motor_k * I + noise.
    Correction: B_cal = soft_iron @ (raw - hard_iron - motor_ut_per_a * I), where soft_iron
    estimates S^-1. Site values (declination, field, dip) are ASSUMED placeholders: set
    them from a geomagnetic model (e.g. WMM) for the flying site.
    """
    hard_iron_ut: np.ndarray = field(default_factory=lambda: np.zeros(3))
    soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3))
    motor_ut_per_a: np.ndarray = field(default_factory=lambda: np.zeros(3))  # ASSUMED 0 until bench-measured
    declination_rad: float = math.radians(-8.0)   # ASSUMED (east positive)
    field_ut: float = 50.0                        # ASSUMED local total field
    field_tol_frac: float = 0.15                  # ASSUMED |B| tolerance
    dip_rad: float = math.radians(53.0)           # ASSUMED inclination (down positive)
    dip_tol_rad: float = math.radians(10.0)       # ASSUMED dip tolerance
    noise_ut: float = 0.5                         # ASSUMED per-axis noise after calibration
    tilt_sigma_rad: float = math.radians(0.5)     # ASSUMED roll/pitch accuracy of the attitude source
    heading_floor_rad: float = math.radians(1.0)  # ASSUMED residual calibration error floor


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class HeadingEstimator:
    """
    2-state Kalman filter x = [yaw, b_gz] (ENU yaw, gyro-z bias).
      predict: yaw_dot = (q sin(roll) + (r - b_gz) cos(roll)) / cos(pitch)   (ZYX Euler rate)
               b_gz is a random walk.
      update : magnetometer heading after hard/soft-iron and motor-current correction and
               tilt compensation with roll/pitch from the attitude source:
                 m = Ry(pitch) Rx(roll) B_cal  (level frame),  yaw = (pi/2 - D) - atan2(m_y, m_x)
    Gates, in order: |B| within field_tol of the expected field, dip within dip_tol, then a
    chi-square innovation gate with the innovation wrapped to (-pi, pi]. Rejected readings
    are counted and NOT fused, so the heading variance keeps growing with gyro-only drift.
    Before the first accepted reading (no magnetic lock yet), MAG_REACQUIRE_AFTER
    consecutive readings that pass the field/dip gates but fail the innovation gate
    increase the yaw variance by the innovation^2: a wrong initial yaw is then the
    likelier culprit. After lock this is NOT done, because a steady local distortion
    with a plausible |B| looks the same and would drag the heading. Inflation lowers
    confidence; it never raises it.
    LIMIT: a smoothly varying distortion with plausible |B| and dip that is present at
    the first lock (or drifts slowly within the gate) is NOT detectable from the
    magnetometer alone; the reported sigma is then overconfident. Known landmarks fix this.
    Known-landmark bearings (update_yaw) are geometric and independent of the local
    field, so they are applied BEFORE the magnetometer in each tick, and
    LM_REACQUIRE_AFTER consecutive landmark rejections inflate the yaw variance (a
    magnetic lock on a distorted field is then the likelier culprit).
    Noise: white per-reading terms (sensor noise / horizontal field, tilt error * tan(dip))
    go into R and average down. The calibration floor (heading_floor_rad) does not
    average down, so it is added to the REPORTED sigma (yaw_at) and to the gate, not to R.
    Only gyro-z bias is estimated; x/y gyro biases are ignored.
    """

    def __init__(self, cal: Optional[MagCalibration] = None,
                 gyro_arw: float = GYRO_ARW_RAD_S_SQRT_HZ,
                 bias_rw: float = GYRO_BIAS_RW_RAD_S_SQRT_S,
                 bias_init_sigma: float = GYRO_BIAS_INIT_SIGMA_RAD_S,
                 gate_chi2: float = MAG_GATE_CHI2,
                 reacquire_after: int = MAG_REACQUIRE_AFTER):
        self.cal = cal or MagCalibration()
        self.gyro_arw = gyro_arw
        self.bias_rw = bias_rw
        self.bias_init_sigma = bias_init_sigma
        self.gate_chi2 = gate_chi2
        self.reacquire_after = reacquire_after
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.t: Optional[float] = None
        self.initialized = False
        self.last_rate = 0.0
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_rejected_field = 0
        self.n_rejected_dip = 0
        self.n_rejected_innovation = 0
        self.n_reacquire = 0
        self._consec_innov_reject = 0
        self.n_lm_accepted = 0
        self.n_lm_rejected = 0
        self.n_lm_reacquire = 0
        self._consec_lm_reject = 0
        self.lm_reacquire_after = LM_REACQUIRE_AFTER

    def initialize(self, yaw: float, yaw_sigma: float, t: float) -> None:
        self.x = np.array([wrap_pi(yaw), 0.0])
        self.P = np.diag([yaw_sigma**2, self.bias_init_sigma**2])
        self.t = float(t)
        self.initialized = True

    @property
    def yaw(self) -> float:
        return float(self.x[0])

    @property
    def yaw_sigma(self) -> float:
        return math.sqrt(max(self.P[0, 0], 0.0))

    def predict(self, dt: float, gyro_rps: np.ndarray, roll: float, pitch: float) -> None:
        if dt <= 0:
            return
        cp = math.cos(pitch)
        if abs(cp) < 0.1:                 # near gimbal lock: Euler yaw rate undefined
            cp = math.copysign(0.1, cp if cp != 0 else 1.0)
        c = math.cos(roll) / cp
        q, r = float(gyro_rps[1]), float(gyro_rps[2])
        rate = (q * math.sin(roll) + (r - self.x[1]) * math.cos(roll)) / cp
        self.last_rate = rate
        self.x[0] = wrap_pi(self.x[0] + rate * dt)
        F = np.array([[1.0, -c * dt], [0.0, 1.0]])
        qb = self.bias_rw**2
        Q = np.array([[self.gyro_arw**2 * dt / cp**2 + qb * c * c * dt**3 / 3.0, -qb * c * dt**2 / 2.0],
                      [-qb * c * dt**2 / 2.0, qb * dt]])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        if self.t is not None:
            self.t += dt

    def calibrate(self, raw_ut: np.ndarray, motor_current_a: float = 0.0) -> np.ndarray:
        c = self.cal
        return c.soft_iron @ (np.asarray(raw_ut, float) - c.hard_iron_ut - c.motor_ut_per_a * motor_current_a)

    def mag_heading(self, b_cal: np.ndarray, roll: float, pitch: float) -> Dict:
        """Tilt-compensated yaw from a calibrated body-frame field, with |B|, dip and R."""
        m = _rot_y(pitch) @ _rot_x(roll) @ np.asarray(b_cal, float)
        bmag = float(np.linalg.norm(m))
        h = float(math.hypot(m[0], m[1]))
        dip = math.atan2(-m[2], h)
        yaw = wrap_pi(math.pi / 2 - self.cal.declination_rad - math.atan2(m[1], m[0]))
        c = self.cal
        r_var = ((c.noise_ut / max(h, 1e-6)) ** 2
                 + (c.tilt_sigma_rad * math.tan(min(abs(dip), math.radians(85)))) ** 2)
        return {"yaw": yaw, "field_ut": bmag, "dip_rad": dip, "R": r_var}

    def update_mag(self, raw_ut: np.ndarray, roll: float, pitch: float,
                   motor_current_a: float = 0.0) -> Dict:
        if not self.initialized:
            return {"status": "not_initialized"}
        if abs(math.cos(pitch)) < 0.1:
            return {"status": "skipped_attitude"}
        mh = self.mag_heading(self.calibrate(raw_ut, motor_current_a), roll, pitch)
        c = self.cal
        out = {"field_ut": mh["field_ut"], "dip_deg": math.degrees(mh["dip_rad"])}
        if abs(mh["field_ut"] - c.field_ut) > c.field_tol_frac * c.field_ut:
            self.n_rejected += 1
            self.n_rejected_field += 1
            out["status"] = "rejected_field"
            return out
        if abs(mh["dip_rad"] - c.dip_rad) > c.dip_tol_rad:
            self.n_rejected += 1
            self.n_rejected_dip += 1
            out["status"] = "rejected_dip"
            return out
        # Anomaly evidence: a field-magnitude or dip deviation means the field is disturbed
        # by at least that much, which can rotate the horizontal field by about
        # dB / B_horizontal. This inflates R (it is evidence, not a bound).
        h_exp = c.field_ut * math.cos(c.dip_rad)
        mh["R"] += ((mh["field_ut"] - c.field_ut) / max(h_exp, 1e-6)) ** 2 + (mh["dip_rad"] - c.dip_rad) ** 2
        y = wrap_pi(mh["yaw"] - self.x[0])
        S = self.P[0, 0] + mh["R"] + c.heading_floor_rad ** 2
        nis = y * y / S
        out.update({"innovation_deg": math.degrees(y), "nis": nis})
        if nis > self.gate_chi2:
            self.n_rejected += 1
            self.n_rejected_innovation += 1
            self._consec_innov_reject += 1
            out["status"] = "rejected_innovation"
            if self.n_accepted == 0 and self._consec_innov_reject >= self.reacquire_after:
                self.P[0, 0] += y * y
                self.n_reacquire += 1
                self._consec_innov_reject = 0
                out["status"] = "rejected_innovation_reacquire"
            return out
        self._consec_innov_reject = 0
        H = np.array([1.0, 0.0])
        K = self.P @ H / (self.P[0, 0] + mh["R"])
        self.x = self.x + K * y
        self.x[0] = wrap_pi(self.x[0])
        IKH = np.eye(2) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * mh["R"]
        self.P = 0.5 * (self.P + self.P.T)
        self.n_accepted += 1
        out["status"] = "accepted"
        return out

    def update_yaw(self, yaw_meas: float, r_var: float) -> Dict:
        """
        Direct yaw measurement (e.g. bearing to a KNOWN landmark), chi-square gated with
        the innovation wrapped. Independent of the magnetometer, so it keeps heading honest
        while the magnetometer is rejected. Re-acquires after LM_REACQUIRE_AFTER
        consecutive rejections (see class docstring). Landmark identity is assumed correct.
        """
        if not self.initialized:
            return {"status": "not_initialized"}
        y = wrap_pi(yaw_meas - self.x[0])
        S = self.P[0, 0] + r_var
        nis = y * y / S
        if nis > self.gate_chi2:
            self.n_lm_rejected += 1
            self._consec_lm_reject += 1
            if self._consec_lm_reject < self.lm_reacquire_after:
                return {"status": "rejected_innovation", "innovation_deg": math.degrees(y), "nis": nis}
            # re-acquire: inflate the yaw variance by the innovation^2 and fuse this bearing
            # (otherwise the next magnetometer reading would pull the state straight back)
            self.P[0, 0] += y * y
            self.n_lm_reacquire += 1
            S = self.P[0, 0] + r_var
            nis = y * y / S
            self.n_lm_rejected -= 1           # this bearing is fused, count it as accepted
            reacq = True
        else:
            reacq = False
        self._consec_lm_reject = 0
        H = np.array([1.0, 0.0])
        K = self.P @ H / S
        self.x = self.x + K * y
        self.x[0] = wrap_pi(self.x[0])
        IKH = np.eye(2) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * r_var
        self.P = 0.5 * (self.P + self.P.T)
        self.n_lm_accepted += 1
        return {"status": "accepted_reacquire" if reacq else "accepted",
                "innovation_deg": math.degrees(y), "nis": nis}

    def yaw_at(self, dt: float) -> Tuple[float, float]:
        """
        Yaw and its reported sigma dt seconds after the filter time (constant-rate
        extrapolation). Once the magnetometer has been used, the calibration floor is
        included because it does not average down.
        """
        yaw = wrap_pi(self.x[0] + self.last_rate * dt)
        var = self.P[0, 0] + self.gyro_arw**2 * abs(dt)
        if self.n_accepted > 0:
            var += self.cal.heading_floor_rad ** 2
        return yaw, math.sqrt(max(var, 0.0))


# ===========================================================================
# Obstacle map (unknown obstacles go here, never into the pose)
# ===========================================================================
@dataclass
class MapObstacle:
    pos_enu: np.ndarray          # world position (built from the pose estimate)
    cov: np.ndarray              # sensor-only covariance; pose sigma is reported separately
    last_seen_s: float
    hits: int = 1
    ir_consistent_hits: int = 0


class ObstacleMap:
    """
    Short-memory point map. Each accepted echo becomes a world point
    p_hat(t_echo) + r * u. Points within the merge gate (Euclidean or
    Mahalanobis) are fused (static-point
    Kalman update, with covariance inflated for pose drift since last seen);
    points that are inside the sonar's view but not re-seen for timeout_s are
    dropped; points outside the view (e.g. beside the drone) are kept for
    memory_s, so a forward-looking sonar does not forget an obstacle the moment
    it leaves the beam. Relative geometry (what guidance
    uses) is good to sensor accuracy; absolute positions also carry the pose sigma.
    """

    def __init__(self, timeout_s: float = MAP_TIMEOUT_S, merge_gate_m: float = 0.3,
                 drift_mps: float = 0.2, max_items: int = 256, memory_s: float = MAP_MEMORY_S):
        self.timeout_s = timeout_s
        self.memory_s = memory_s
        self.merge_gate_m = merge_gate_m
        self.drift_mps = drift_mps       # ASSUMED pose-estimate drift rate for inflation
        self.max_items = max_items
        self.items: List[MapObstacle] = []

    def update(self, pos: np.ndarray, cov: np.ndarray, t: float, ir_consistent: bool = False) -> MapObstacle:
        best, best_d = None, float("inf")
        for ob in self.items:
            diff = ob.pos_enu - pos
            d = float(np.linalg.norm(diff))
            # Mahalanobis gate (chi-square 3 dof, 99%), so a large LATERAL sigma (e.g. from
            # heading uncertainty) does not merge points that are well separated in range;
            # points closer than merge_gate_m always merge.
            m2 = float(diff @ np.linalg.solve(ob.cov + cov + np.eye(3) * 1e-6, diff))
            if (d < self.merge_gate_m or m2 < 11.34) and d < best_d:
                best, best_d = ob, d
        if best is None:
            ob = MapObstacle(np.asarray(pos, float).copy(), np.asarray(cov, float).copy(), t,
                             1, int(ir_consistent))
            self.items.append(ob)
            if len(self.items) > self.max_items:
                self.items.sort(key=lambda o: o.last_seen_s)
                self.items = self.items[-self.max_items:]
            return ob
        C = best.cov + np.eye(3) * (self.drift_mps * max(t - best.last_seen_s, 0.0)) ** 2
        K = C @ np.linalg.inv(C + cov)
        best.pos_enu = best.pos_enu + K @ (pos - best.pos_enu)
        best.cov = (np.eye(3) - K) @ C
        best.cov = 0.5 * (best.cov + best.cov.T)
        best.last_seen_s = max(best.last_seen_s, t)
        best.hits += 1
        best.ir_consistent_hits += int(ir_consistent)
        return best

    def prune(self, t: float, in_view=None) -> None:
        """in_view(obstacle) -> bool says whether the sonar would have seen it this tick."""
        keep = []
        for o in self.items:
            age = t - o.last_seen_s
            visible = True if in_view is None else bool(in_view(o))
            if age <= (self.timeout_s if visible else self.memory_s):
                keep.append(o)
        self.items = keep

    def active(self, t: float) -> List[MapObstacle]:
        return list(self.items)


# ===========================================================================
# Core
# ===========================================================================
FLIGHT_CRITICAL_LOADS = ("motors_hover", "avionics")   # never scaled
SAFETY_SENSOR_LOADS = ("acoustic",)                    # not shed while airborne


class AutonomousDroneCore:
    RHO_AIR = 1.225

    def __init__(
        self,
        mass_kg: float = 1.45,
        drag_k: float = 0.18,
        temperature_c: float = 15.0,
        humidity_pct: float = 50.0,
        battery: Optional[BatteryState] = None,
        min_safe_soc: float = 0.15,        # ASSUMED: LAND at or below
        return_home_soc: float = 0.25,     # ASSUMED: RETURN_HOME at or below
        kf_jerk_psd: float = 25.0,         # ASSUMED
        a_plan_mps2: float = A_PLAN_MPS2,
        a_emergency_mps2: float = A_EMERGENCY_MPS2,
        t_react_s: float = T_REACT_S,
        drone_radius_m: float = DRONE_RADIUS_M,
        standoff_m: float = STANDOFF_M,
        landmarks: Optional[Dict[str, Sequence[float]]] = None,
        mag_calibration: Optional[MagCalibration] = None,
        heading_init_sigma_rad: float = HEADING_INIT_SIGMA_RAD,
    ):
        self.mass = float(mass_kg)
        self.drag_k = float(drag_k)
        self.temperature_c = float(temperature_c)
        self.humidity_pct = float(humidity_pct)
        self.c_sound = self._sound_speed(temperature_c, humidity_pct)
        self.battery = battery or BatteryState(capacity_wh=45.0, remaining_wh=45.0)
        self.min_safe_soc = float(min_safe_soc)
        self.return_home_soc = float(return_home_soc)
        self.a_plan = float(a_plan_mps2)
        self.a_emergency = float(a_emergency_mps2)
        self.t_react = float(t_react_s)
        self.drone_radius = float(drone_radius_m)
        self.standoff = float(standoff_m)
        self.kf = DragConsistentKF(drag_k=self.drag_k, mass_kg=self.mass, jerk_psd=kf_jerk_psd)
        self._last_update_t: Optional[float] = None
        self._last_sonar_data_t: Optional[float] = None
        self.stale_telemetry_dropped = 0
        self.ranger = AcousticRanger(self.c_sound)
        self.ir_bias = IRBiasEstimator()
        self.map = ObstacleMap()
        self.landmarks = {k: np.asarray(v, float) for k, v in (landmarks or {}).items()}
        self.heading = HeadingEstimator(mag_calibration)
        self.heading_init_sigma = float(heading_init_sigma_rad)

    # -- environment --------------------------------------------------------
    @staticmethod
    def _sound_speed(temperature_c: float, humidity_pct: float = 0.0) -> float:
        """Speed of sound from Cramer (1993) at sea-level pressure."""
        return sound_speed_cramer(temperature_c, humidity_pct)

    def set_environment(self, temperature_c: float, humidity_pct: float) -> None:
        self.temperature_c = float(temperature_c)
        self.humidity_pct = float(humidity_pct)
        self.c_sound = self._sound_speed(self.temperature_c, self.humidity_pct)
        self.ranger.c_sound = self.c_sound

    @staticmethod
    def expected_echo_loss_db(range_m: float) -> float:
        """Two-way absorption with the ASSUMED (unverified) coefficient. Informational only."""
        return 2.0 * ULTRASONIC_ABSORPTION_DB_PER_M * max(range_m, 0.0)

    # -- predictive sensor scheduling --------------------------------------
    def acoustic_pulse_interval_s(self, base_interval_s: float = 0.05) -> float:
        """
        Shorten the pulse interval when the filter is uncertain or the drone is
        fast: interval = base / (1 + 4 sigma_v + 0.1 speed), at least base/4 and
        never below the round-trip time at max range (no ping overlap). The
        urgency factor is >= 1, so the interval never exceeds base.
        """
        min_interval = 2.0 * self.ranger.max_range_m / self.c_sound
        if not self.kf.initialized:
            return max(base_interval_s, min_interval)
        sigma_v = self.kf.sigmas()["vel_sigma_mps"]
        speed = float(np.linalg.norm(self.kf.x[3:6]))
        urgency = 1.0 + 4.0 * sigma_v + 0.1 * speed
        interval = max(base_interval_s / urgency, base_interval_s / 4.0)
        return max(interval, min_interval)

    # -- 1. Latency compensation -------------------------------------------
    def compensate_latency(self, telemetry: AirframeState, tau_lag_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fuse telemetry (stamped at the time it describes) and extrapolate to
        now = timestamp + tau. Late (out-of-order) packets are NOT fused and do
        not move the filter clock back; the filter is extrapolated to their 'now'.
        """
        if tau_lag_s < 0:
            raise ValueError("tau_lag_s must be nonnegative")
        ts = float(telemetry.timestamp_s)
        if not self.kf.initialized:
            self.kf.initialize(telemetry)
            self._last_update_t = ts
        elif ts < self._last_update_t - 1e-9:
            self.stale_telemetry_dropped += 1
        else:
            self.kf.predict(ts - self._last_update_t)
            self._last_update_t = ts
            self.kf.update_pos(telemetry.pos_m, R=np.diag(np.asarray(telemetry.pos_sigma_m, float) ** 2))
            self.kf.update_vel(telemetry.vel_mps, R=np.diag(np.asarray(telemetry.vel_sigma_mps, float) ** 2))
            self.kf.update_acc(telemetry.acc_mps2, R=np.diag(np.asarray(telemetry.acc_sigma_mps2, float) ** 2))
        dt_now = ts + tau_lag_s - self._last_update_t
        p_future, v_future, _ = self.kf.state_at(dt_now)
        return p_future, v_future, self.kf.trust(dt_now)

    # -- 1b. Heading ----------------------------------------------------------
    def _update_heading(self, tm: AirframeState, now_s: float, stale: bool,
                        landmark_bearings: Sequence[Tuple[str, float, float, float]] = ()) -> Tuple[Attitude, Dict]:
        """
        If the telemetry carries gyro rates, run the heading filter (initialised from the
        telemetry yaw with heading_init_sigma) and fuse the magnetometer when present. The
        attitude used for sonar rays is then roll/pitch from telemetry + the estimated yaw
        at 'now'. Without gyro data the telemetry attitude is used unchanged (v2 behaviour)
        and no heading sigma is claimed.
        Landmark bearings: the body-frame bearing is levelled with roll/pitch, giving the
        bearing in the yaw frame (beta); the world bearing from the filter position to the
        known landmark (theta) then gives yaw = theta - beta. Its variance includes the
        bearing sigma, tilt error and the cross-track position sigma / distance (the pose
        and heading filters are separate, so this keeps it conservative).
        """
        roll, pitch, yaw_tm = tm.attitude.to_euler()
        hd = self.heading
        info: Dict = {"heading_source": "telemetry_attitude", "mag_status": "disabled", "mag_ok": False}
        if tm.gyro_rps is None and not hd.initialized:
            info.update({"yaw_enu_deg": round(math.degrees(yaw_tm), 3), "heading_sigma_deg": None})
            info["compass_heading_deg"] = round((90.0 - math.degrees(yaw_tm)) % 360.0, 3)
            return tm.attitude, info
        ts = float(tm.timestamp_s)
        if not hd.initialized:
            hd.initialize(yaw_tm, self.heading_init_sigma, ts)
        elif not stale and tm.gyro_rps is not None and ts > hd.t:
            hd.predict(ts - hd.t, np.asarray(tm.gyro_rps, float), roll, pitch)
        lm_status = []
        if not stale:
            p = self.kf.x[0:3]
            Pp = self.kf.P[0:3, 0:3]
            for lid, az_b, el_b, sig in landmark_bearings:
                if lid not in self.landmarks:
                    continue
                d = self.landmarks[lid] - p
                dist_h = float(math.hypot(d[0], d[1]))
                if dist_h < 1.0:
                    continue
                u_l = _rot_y(pitch) @ _rot_x(roll) @ body_ray(az_b, el_b)
                beta = math.atan2(u_l[1], u_l[0])
                theta = math.atan2(d[1], d[0])
                n_perp = np.array([-d[1], d[0], 0.0]) / dist_h
                pos_var = float(n_perp @ Pp @ n_perp) / dist_h**2
                r_var = sig**2 + hd.cal.tilt_sigma_rad**2 + pos_var
                lm_status.append(hd.update_yaw(wrap_pi(theta - beta), r_var)["status"])
        status = "no_reading"
        mres: Dict = {}
        if tm.mag_body_ut is not None and not stale:
            mres = hd.update_mag(tm.mag_body_ut, roll, pitch, tm.motor_current_a)
            status = mres["status"]
        yaw_now, sig_now = hd.yaw_at(now_s - hd.t)
        info = {
            "heading_source": ("gyro+mag" if tm.mag_body_ut is not None or hd.n_accepted else "gyro_only")
                              + ("+landmarks" if hd.n_lm_accepted else ""),
            "yaw_enu_deg": round(math.degrees(yaw_now), 3),
            "compass_heading_deg": round((90.0 - math.degrees(yaw_now)) % 360.0, 3),
            "heading_sigma_deg": round(math.degrees(sig_now), 4),
            "gyro_z_bias_dps": round(math.degrees(hd.x[1]), 4),
            "mag_status": status,
            "mag_ok": status == "accepted",
            "mag_field_ut": None if "field_ut" not in mres else round(mres["field_ut"], 3),
            "mag_dip_deg": None if "dip_deg" not in mres else round(mres["dip_deg"], 3),
            "mag_accepted": hd.n_accepted,
            "mag_rejected": hd.n_rejected,
            "mag_rejected_field": hd.n_rejected_field,
            "mag_rejected_dip": hd.n_rejected_dip,
            "mag_rejected_innovation": hd.n_rejected_innovation,
            "heading_reacquired": hd.n_reacquire,
            "landmark_reacquired": hd.n_lm_reacquire,
            "landmark_bearing_status": lm_status,
            "landmark_bearings_accepted": hd.n_lm_accepted,
            "landmark_bearings_rejected": hd.n_lm_rejected,
        }
        return Attitude.from_euler(roll, pitch, yaw_now), info

    # -- 2. Acoustic returns -> map points, IR cross-check ----------------
    @staticmethod
    def _acoustic_confidence(echo: AcousticSensorReturn) -> float:
        """Heuristic 0..1 from measured SNR and amplitude (absorption is already in the measured SNR)."""
        snr_term = 1.0 - math.exp(-max(echo.snr_db, 0.0) / 10.0)
        amp_term = float(np.clip(echo.echo_amplitude / 0.85, 0.0, 1.0))
        return float(np.clip(0.6 * snr_term + 0.4 * amp_term, 0.05, 0.99))

    @staticmethod
    def _ir_gate_by_prediction(ir: InfraredReturn) -> float:
        """IR confidence, discounted for grazing incidence when the surface angle is known."""
        base = float(np.clip(ir.confidence, 0.05, 0.99))
        if ir.surface_angle_rad is not None:
            base *= max(0.2, math.cos(min(abs(ir.surface_angle_rad), math.pi / 2)))
        return float(np.clip(base, 0.05, 0.99))

    @staticmethod
    def _pair_ir(reps: List[Optional[AcousticSensorReturn]], irs: Sequence[InfraredReturn]) -> Dict[int, InfraredReturn]:
        """Pair each IR return with at most one acoustic group by bearing and time."""
        pairs: Dict[int, InfraredReturn] = {}
        for ir in irs:
            best, best_ang = None, IR_PAIR_MAX_ANGLE_RAD
            for i, rep in enumerate(reps):
                if rep is None or i in pairs:
                    continue
                if abs(rep.timestamp_s - ir.timestamp_s) > IR_PAIR_MAX_DT_S:
                    continue
                ang = math.acos(float(np.clip(np.dot(body_ray(rep.azimuth_rad, rep.elevation_rad),
                                                       body_ray(ir.azimuth_rad, ir.elevation_rad)), -1, 1)))
                if ang <= best_ang:
                    best, best_ang = i, ang
            if best is not None:
                pairs[best] = ir
        return pairs

    def resolve_acousto_ir_voxel(self, est: Dict, attitude: Attitude, now_s: float,
                                 ir: Optional[InfraredReturn] = None,
                                 heading_sigma_rad: Optional[float] = None) -> Dict:
        """
        One accepted echo group -> obstacle point.
        Round-trip motion compensation: the drone moves v*tof during the flight
        of the pulse, so the range at reception is r = (c - v.u) * tof / 2
        (v.u > 0 when moving toward the obstacle). The point is anchored at the
        filter's position at reception time, then expressed relative to the
        latency-compensated position 'now'.
        heading_sigma_rad: if the heading is estimated, its uncertainty is propagated
        into the point covariance: d(point)/d(yaw) = z x (r u), so the lateral sigma
        grows as range * heading sigma.
        """
        rep = est["representative"]
        t0 = self._last_update_t
        p_recv, v_recv, _ = self.kf.state_at(rep.timestamp_s - t0)
        p_now, _, _ = self.kf.state_at(now_s - t0)
        u_body = body_ray(rep.azimuth_rad, rep.elevation_rad)
        u_enu = attitude.rotate(u_body)
        u_enu /= max(np.linalg.norm(u_enu), 1e-9)
        r_raw = 0.5 * rep.tof_seconds * self.c_sound
        v_toward = float(np.dot(v_recv, u_enu))
        r = 0.5 * rep.tof_seconds * (self.c_sound - v_toward)
        world = p_recv + r * u_enu
        rel_now = world - p_now
        az, el = rep.azimuth_rad, rep.elevation_rad
        saz, caz, sel, cel = math.sin(az), math.cos(az), math.sin(el), math.cos(el)
        J_body = np.array([[cel * caz, -r * cel * saz, -r * sel * caz],
                           [cel * saz,  r * cel * caz, -r * sel * saz],
                           [sel,        0.0,            r * cel]])
        J = attitude.to_matrix() @ J_body
        cov = J @ np.diag([est["range_sigma_m"]**2, est["az_sigma_rad"]**2, est["el_sigma_rad"]**2]) @ J.T
        heading_lat = 0.0
        if heading_sigma_rad is not None:
            d = r * u_enu
            j_yaw = np.array([-d[1], d[0], 0.0])
            cov = cov + np.outer(j_yaw, j_yaw) * heading_sigma_rad**2
            heading_lat = float(np.linalg.norm(j_yaw)) * heading_sigma_rad

        c_a = self._acoustic_confidence(rep)
        out = {
            "voxel_pos_enu_m": [round(v, 3) for v in world.tolist()],
            "voxel_rel_now_m": [round(v, 3) for v in rel_now.tolist()],
            "voxel_pos_body_m": [round(float(r * c), 3) for c in u_body],
            "acoustic_range_m": round(float(r), 3),
            "acoustic_range_raw_m": round(float(r_raw), 3),
            "range_sigma_m": round(float(est["range_sigma_m"]), 4),
            "lateral_sigma_from_heading_m": round(heading_lat, 4),
            "single_echo_floor_sigma": bool(est["single_echo"]),
            "acoustic_conf": round(c_a, 3),
            "expected_two_way_absorption_db": round(self.expected_echo_loss_db(r), 2),
        }
        ir_ok = False
        agree = 0.85   # ASSUMED factor when no IR is paired (acoustic alone)
        if ir is not None:
            ir_conf = self._ir_gate_by_prediction(ir)
            ir_corr = self.ir_bias.correct(ir.depth_m)
            sig = math.sqrt(est["range_sigma_m"]**2 + self.ir_bias.ir_sigma_floor_m**2)
            dis = float(r - ir_corr)
            ir_ok = abs(dis) <= 3.0 * sig and ir_conf > 0.3
            agree = 1.0 if ir_ok else 0.6
            self.ir_bias.update(r, est["range_sigma_m"], ir.depth_m, ir_conf)
            out.update({
                "infrared_surface_raw_m": round(float(ir.depth_m), 3),
                "infrared_surface_corrected_m": round(float(ir_corr), 3),
                "ir_bias_estimate_m": round(float(self.ir_bias.bias_m), 4),
                "ir_conf": round(ir_conf, 3),
                "acoustic_minus_ir_m": round(dis, 3),
                "acoustic_ir_consistent": bool(ir_ok),
            })
        out["voxel_conf"] = round(float(np.clip(c_a * agree * (1.0 - 0.5 * est["outlier_ratio"]), 0.0, 0.95)), 3)
        out["_world"] = world
        out["_rel_now"] = rel_now
        out["_cov"] = cov
        out["_ir_ok"] = ir_ok
        return out

    # -- 3. Power budget ----------------------------------------------------
    def allocate_duty_cycle(self, requested: Dict[str, float], horizon_s: float = 0.1) -> Dict:
        """
        Energy budget for the horizon = sum(power_i * duty_i) * horizon.
        Flight-critical loads (motors, avionics) are always granted in full; the
        acoustic safety sensor is kept while airborne; optional loads are shed
        when at/below the return threshold or when the horizon would cross the
        reserve. The battery is NOT drained here (use BatteryState.consume with
        measured/simulated energy) and is never floored at the reserve.
        """
        nominal = self.battery.draw_w
        req = {k: float(np.clip(v, 0.0, 1.0)) for k, v in requested.items()}
        keep = set(FLIGHT_CRITICAL_LOADS) | set(SAFETY_SENSOR_LOADS)
        crit_w = sum(nominal.get(k, 0.0) * f for k, f in req.items() if k in keep)
        opt_w = sum(nominal.get(k, 0.0) * f for k, f in req.items() if k not in keep)
        soc = self.battery.soc
        usable_wh = self.battery.remaining_wh - self.battery.capacity_wh * self.min_safe_soc
        need_all_wh = (crit_w + opt_w) * horizon_s / 3600.0
        granted = dict(req)
        shed: List[str] = []
        if soc <= self.return_home_soc or need_all_wh > max(usable_wh, 0.0):
            for k in req:
                if k not in keep and req[k] > 0.0:
                    granted[k] = 0.0
                    shed.append(k)
        need_wh = sum(nominal.get(k, 0.0) * f for k, f in granted.items()) * horizon_s / 3600.0
        if soc <= self.min_safe_soc or crit_w * horizon_s / 3600.0 > max(usable_wh, 0.0):
            action = "LAND"
        elif soc <= self.return_home_soc:
            action = "RETURN_HOME"
        else:
            action = "NOMINAL"
        endurance_s = max(usable_wh, 0.0) * 3600.0 / crit_w if crit_w > 0 else float("inf")
        span = max(self.return_home_soc - self.min_safe_soc, 1e-6)
        power_tp = float(np.clip((soc - self.min_safe_soc) / span, 0.0, 1.0))
        return {
            "granted_fractions": {k: round(v, 3) for k, v in granted.items()},
            "shed_loads": shed,
            "flight_critical_scaled": False,
            "energy_needed_wh": round(need_wh, 5),
            "budget_basis": "sum(power*duty)*horizon",
            "usable_above_reserve_wh": round(usable_wh, 4),
            "endurance_to_reserve_s": round(endurance_s, 1) if math.isfinite(endurance_s) else None,
            "soc": round(soc, 4),
            "mission_action": action,
            "power_tp": round(power_tp, 3),
        }

    # -- 4. Guidance: CPA, time to contact, stopping distance ---------------
    def stopping_distance_m(self, closing_speed_mps: float) -> float:
        """Reaction distance + braking distance v^2 / (2 a_plan)."""
        v = max(closing_speed_mps, 0.0)
        return v * self.t_react + v * v / (2.0 * self.a_plan)

    def safe_speed_for_distance(self, free_m: float) -> float:
        """Largest speed that can stop within free_m: v*t_react + v^2/(2a) = free_m."""
        a, t = self.a_plan, self.t_react
        return a * (-t + math.sqrt(t * t + 2.0 * max(free_m, 0.0) / a))

    def max_safe_speed(self) -> float:
        """Sensor-range speed limit: an obstacle first seen at max range must be stoppable."""
        return self.safe_speed_for_distance(self.ranger.max_range_m - self.drone_radius - self.standoff)

    def _velocity_bound(self, free_m: float) -> float:
        """Largest allowed velocity component toward an obstacle with free_m of room.
        Inside the standoff it is negative (back off at 0.5 m/s per m, max 0.5 m/s; ASSUMED)."""
        if free_m > 0.0:
            return self.safe_speed_for_distance(free_m)
        return -min(0.5, 0.5 * (-free_m))

    def constrain_velocity(self, v_des: np.ndarray, constraints: List[Tuple[np.ndarray, float]],
                           speed_limit: float) -> np.ndarray:
        """
        Scale the desired velocity down (same direction) until v.u_i <= s_i for
        every obstacle direction u_i with bound s_i; never redirect it sideways
        (projection would turn 'fly into the wall' into fast sliding along it).
        Bounds below zero (inside the standoff) add a back-off velocity, capped
        at 0.5 m/s. Finally |v| <= speed_limit.
        """
        v_des = np.asarray(v_des, float)
        scale = 1.0
        back = np.zeros(3)
        for u, s in constraints:
            c = float(v_des @ u)
            if c > 1e-9:
                scale = 0.0 if s <= 0.0 else min(scale, s / c)
            if s < 0.0:
                back = back + s * u
        nb = float(np.linalg.norm(back))
        if nb > 0.5:
            back *= 0.5 / nb
        v = v_des * scale + back
        n = float(np.linalg.norm(v))
        if speed_limit <= 0.0:
            return back
        return v * (speed_limit / n) if n > speed_limit else v

    def _predictive_avoidance(self, pred_vel: np.ndarray, obstacles_rel: List[np.ndarray],
                              blind: bool = False, desired_vel: Optional[np.ndarray] = None,
                              heading_uncertain: bool = False) -> Dict:
        """
        obstacles_rel: obstacle position minus the drone's CURRENT (latency-
        compensated) position. For a static obstacle, relative velocity is -v, so
        t_cpa = (r.v)/|v|^2 clamped >= 0 and d_cpa = |r - v t_cpa|.
        Each obstacle bounds the velocity component toward it by the speed that
        can still stop in the free distance (reaction + v^2/(2 a_plan)). If the
        current closing speed already exceeds that bound, brake along the
        obstacle direction with the deceleration needed to stop in the remaining
        room (at least a_plan, capped at a_emergency).
        """
        v = np.asarray(pred_vel, float)
        vv = float(v @ v)
        speed_limit = self.max_safe_speed()
        if blind:
            speed_limit = min(speed_limit, BLIND_SPEED_LIMIT_MPS)
        if heading_uncertain:     # sonar points cannot be placed reliably in the world frame
            speed_limit = min(speed_limit, HEADING_SLOW_SPEED_MPS)
        worst = None
        constraints: List[Tuple[np.ndarray, float]] = []
        for r in obstacles_rel:
            d = float(np.linalg.norm(r))
            if d < 1e-6:
                continue
            u = r / d
            v_c = float(v @ u)
            t_cpa = max(0.0, float(r @ v) / vv) if vv > 1e-9 else 0.0
            d_cpa = float(np.linalg.norm(r - v * t_cpa))
            free = d - self.drone_radius - self.standoff
            ttc = (d - self.drone_radius) / v_c if v_c > 1e-6 else float("inf")
            bound = self._velocity_bound(free)
            constraints.append((u, bound))
            room = free - max(v_c, 0.0) * self.t_react
            a_req = v_c * v_c / (2.0 * max(room, 0.05)) if v_c > 0 else 0.0
            excess = v_c - max(bound, 0.0)
            cand = dict(d=d, u=u, v_c=v_c, t_cpa=t_cpa, d_cpa=d_cpa, free=free, ttc=ttc,
                        a_req=a_req, excess=excess)
            if worst is None or (excess, -d) > (worst["excess"], -worst["d"]):
                worst = cand
        safe_v = None
        if desired_vel is not None:
            safe_v = self.constrain_velocity(desired_vel, constraints, speed_limit)
        if worst is None:
            return {"a_cmd": np.zeros(3), "action": "TRACKING_NOMINAL", "t_cpa": None, "d_cpa": None,
                    "ttc": None, "stop_dist": 0.0, "free": None, "speed_limit": speed_limit,
                    "closest": None, "safe_vel": safe_v}
        w = worst
        stop = self.stopping_distance_m(w["v_c"])
        a_cmd = np.zeros(3)
        if w["v_c"] > 0 and w["excess"] > 0.1:          # 0.1 m/s tolerance (ASSUMED)
            a_mag = min(max(w["a_req"], self.a_plan), self.a_emergency)
            a_cmd = -w["u"] * a_mag
            action = "COLLISION_LIKELY_MAX_BRAKE" if w["a_req"] > self.a_emergency else "COLLISION_AVOIDANCE_BRAKE"
        elif w["free"] <= 0.0:
            action = "HOLD_STANDOFF"
        elif w["v_c"] > 0 and w["free"] <= 2.0 * stop + 1.0:
            action = "CAUTION_SLOW"
        elif w["d_cpa"] < self.drone_radius + self.standoff and w["t_cpa"] < 3.0 and vv > 1e-6:
            action = "CAUTION_PASSING_CLOSE"
        else:
            action = "TRACKING_NOMINAL"
        closest = min(float(np.linalg.norm(r)) for r in obstacles_rel)
        return {"a_cmd": a_cmd, "action": action, "t_cpa": w["t_cpa"], "d_cpa": w["d_cpa"],
                "ttc": w["ttc"], "stop_dist": stop, "free": w["free"], "speed_limit": speed_limit,
                "closest": closest, "safe_vel": safe_v}

    # -- Main entry ---------------------------------------------------------
    def process_flight_tick(
        self,
        current_telemetry: AirframeState,
        tau_latency_s: float,
        acoustic_returns: Sequence[AcousticSensorReturn] | List[AcousticEchoGroup],
        infrared_distances: Optional[List[float]] = None,
        infrared_returns: Optional[List[InfraredReturn]] = None,
        duty_requests: Optional[Dict[str, float]] = None,
        horizon_s: float = 0.1,
        landmark_ranges: Optional[List[Tuple[str, float, float]]] = None,
        desired_velocity_mps: Optional[np.ndarray] = None,
        landmark_bearings: Optional[List[Tuple[str, float, float, float]]] = None,
    ) -> Dict:
        """
        acoustic_returns: echo groups (one per ping/sensor) or bare echoes. An
        empty group means 'pinged, no echo in range'; an empty list means no
        sonar data this tick (counts toward 'blind'). IR is optional and is
        paired by bearing/time; acoustic groups are processed with or without IR.
        landmark_ranges: (landmark_id, range_m, sigma_m) to KNOWN landmarks.
        landmark_bearings: (landmark_id, body_az_rad, body_el_rad, sigma_rad) to KNOWN
        landmarks, measured at the telemetry timestamp; they constrain the heading.
        desired_velocity_mps: if given, the output includes 'safe_velocity_mps',
        the desired velocity projected onto the obstacle and speed constraints.
        """
        if infrared_returns is None:
            infrared_returns = [InfraredReturn(d, timestamp_s=current_telemetry.timestamp_s + tau_latency_s)
                                for d in (infrared_distances or [])]
        if acoustic_returns and isinstance(acoustic_returns[0], AcousticEchoGroup):
            groups = list(acoustic_returns)
        else:
            groups = [AcousticEchoGroup(echoes=[e]) for e in acoustic_returns]

        duty_requests = duty_requests or {"avionics": 1.0, "acoustic": 1.0,
                                          "infrared": 1.0, "motors_hover": 1.0}
        power = self.allocate_duty_cycle(duty_requests, horizon_s=horizon_s)
        if power["granted_fractions"].get("infrared", 1.0) <= 0.0:
            infrared_returns = []

        n_stale = self.stale_telemetry_dropped
        pred_pos, pred_vel, kin_tp = self.compensate_latency(current_telemetry, tau_latency_s)
        now = current_telemetry.timestamp_s + tau_latency_s
        stale = self.stale_telemetry_dropped > n_stale

        landmark_nis = []
        for lid, rng_m, sig in (landmark_ranges or []):
            if lid in self.landmarks:
                landmark_nis.append(self.kf.update_landmark_range(self.landmarks[lid], rng_m, sig))
        if landmark_nis:
            dt_now = now - self._last_update_t
            pred_pos, pred_vel, _ = self.kf.state_at(dt_now)
            kin_tp = self.kf.trust(dt_now)
        att_used, heading_info = self._update_heading(current_telemetry, now, stale,
                                                      landmark_bearings or [])
        heading_sigma = (None if heading_info.get("heading_sigma_deg") is None
                         else math.radians(heading_info["heading_sigma_deg"]))
        heading_uncertain = heading_sigma is not None and heading_sigma > HEADING_SLOW_SIGMA_RAD

        if groups:
            self._last_sonar_data_t = now
        blind_s = float("inf") if self._last_sonar_data_t is None else now - self._last_sonar_data_t
        blind = blind_s > BLIND_AFTER_S

        ests = [self.ranger.estimate(g) for g in groups]
        reps = [e["representative"] if e else None for e in ests]
        pairs = self._pair_ir(reps, infrared_returns)
        point_cloud: List[Dict] = []
        stats = {"n_groups": len(groups), "n_no_echo_or_rejected": 0, "n_input_total": 0,
                 "n_kept_total": 0, "n_ir_paired": 0, "ir_bias_m": 0.0,
                 "pulse_interval_s": round(self.acoustic_pulse_interval_s(), 4)}
        for i, (g, est) in enumerate(zip(groups, ests)):
            stats["n_input_total"] += len(g.echoes)
            if est is None:
                stats["n_no_echo_or_rejected"] += 1
                continue
            stats["n_kept_total"] += est["n_kept"]
            ir = pairs.get(i)
            stats["n_ir_paired"] += int(ir is not None)
            vox = self.resolve_acousto_ir_voxel(est, att_used, now, ir, heading_sigma)
            self.map.update(vox.pop("_world"), vox.pop("_cov"), now, vox.pop("_ir_ok"))
            vox.pop("_rel_now")
            point_cloud.append(vox)
        stats["ir_bias_m"] = round(float(self.ir_bias.bias_m), 4)

        att = att_used
        max_r = self.ranger.max_range_m

        def in_view(o: MapObstacle) -> bool:
            if not groups:            # no ping this tick: no negative evidence
                return False
            b = att.inverse_rotate(o.pos_enu - pred_pos)
            rng_b = float(np.linalg.norm(b))
            az, el = math.atan2(b[1], b[0]), math.asin(float(np.clip(b[2] / max(rng_b, 1e-9), -1, 1)))
            return rng_b <= max_r and abs(az) <= SONAR_HALF_FOV_RAD and abs(el) <= SONAR_HALF_FOV_RAD

        self.map.prune(now, in_view)
        rel = [o.pos_enu - pred_pos for o in self.map.active(now)]
        g = self._predictive_avoidance(pred_vel, rel, blind=blind, desired_vel=desired_velocity_mps,
                                       heading_uncertain=heading_uncertain)

        sig = self.kf.sigmas(now - self._last_update_t)
        if point_cloud:
            map_tp = float(np.mean([p["voxel_conf"] for p in point_cloud]))
        elif groups:
            map_tp = 0.8   # ASSUMED: pinged, nothing in range (cannot rule out soft absorbers)
        else:
            map_tp = 0.0 if not math.isfinite(blind_s) else float(math.exp(-blind_s / 0.5))
        composite = 0.5 * kin_tp + 0.3 * map_tp + 0.2 * power["power_tp"]
        if blind:
            composite *= 0.5
        heading_tp = None
        if heading_sigma is not None:
            heading_tp = 1.0 / (1.0 + heading_sigma / HEADING_TRUST_REF_RAD)
            composite *= heading_tp
        return {
            "predictive_telemetry": {
                "compensated_tau_s": round(float(tau_latency_s), 4),
                "now_s": round(now, 4),
                "extrapolated_pos": np.round(pred_pos, 3).tolist(),
                "extrapolated_vel": np.round(pred_vel, 3).tolist(),
                "pos_sigma_m": round(sig["pos_sigma_m"], 4),
                "vel_sigma_mps": round(sig["vel_sigma_mps"], 4),
                "kinematic_tp": round(kin_tp, 3),
                "stale_telemetry_dropped": self.stale_telemetry_dropped,
                "landmark_updates": len([x for x in landmark_nis if x is not None]),
            },
            "heading": heading_info,
            "spatial_cloud_3d": point_cloud,
            "ranging_stats": stats,
            "obstacle_map": {"n_active": len(rel),
                             "note": "world points carry the pose sigma in addition to sensor sigma"},
            "flight_guidance": {
                "closest_obstacle_m": None if g["closest"] is None else round(g["closest"], 3),
                "time_to_cpa_s": None if g["t_cpa"] is None else round(g["t_cpa"], 4),
                "miss_distance_at_cpa_m": None if g["d_cpa"] is None else round(g["d_cpa"], 3),
                "time_to_contact_s": None if g["ttc"] is None or not math.isfinite(g["ttc"]) else round(g["ttc"], 3),
                "stopping_distance_m": round(g["stop_dist"], 3),
                "free_distance_m": None if g["free"] is None else round(g["free"], 3),
                "speed_limit_mps": round(g["speed_limit"], 3),
                "safe_velocity_mps": None if g["safe_vel"] is None else np.round(g["safe_vel"], 3).tolist(),
                "command_accel_mps2": np.round(g["a_cmd"], 3).tolist(),
                "action": g["action"],
                "mission_action": power["mission_action"],
                "sonar_blind": bool(blind),
                "heading_uncertain": bool(heading_uncertain),
                "heading_tp": None if heading_tp is None else round(heading_tp, 3),
                "map_tp": round(map_tp, 3),
                "power_tp": power["power_tp"],
                "system_tp": round(float(composite), 3),
            },
            "power_management": power,
        }


# ===========================================================================
# Self-tests (v1's checks, with the ones that encoded bugs inverted)
# ===========================================================================
def _self_test() -> None:
    core = AutonomousDroneCore(temperature_c=25.0, humidity_pct=60.0)
    assert abs(core.c_sound - 347.30) < 0.05            # Cramer 1993 reference value

    st = AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    p, v, tp = core.compensate_latency(st, 0.0)
    assert np.allclose(p, 0.0, atol=1e-3) and np.allclose(v, 0.0, atol=1e-3) and tp > 0.4

    core2 = AutonomousDroneCore()
    st = AirframeState(0.0, np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3), np.zeros(3))
    p1, _, _ = core2.compensate_latency(st, 0.01)
    p2, _, _ = core2.compensate_latency(st, 0.05)
    assert p2[0] > p1[0] > 0.0

    core3 = AutonomousDroneCore(temperature_c=15.0, humidity_pct=0.0)
    c_before = core3.c_sound
    core3.set_environment(35.0, 90.0)
    assert core3.c_sound > c_before

    q = Attitude(w=math.cos(math.pi / 8), x=0, y=0, z=math.sin(math.pi / 8))
    v_enu = q.rotate(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(q.inverse_rotate(v_enu), [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(q.to_matrix() @ [1.0, 0.0, 0.0], v_enu, atol=1e-12)

    ranger = AcousticRanger(c_sound_mps=343.0)
    good = AcousticSensorReturn(0.0, 0.0, 2.0 / 343.0, 0.9, 0.0, 25.0)
    outlier = AcousticSensorReturn(0.0, 0.0, 2.0 * 5.0 / 343.0, 0.9, 0.0, 25.0)
    est = ranger.estimate(AcousticEchoGroup(echoes=[good, good, good, outlier]))
    assert est is not None and abs(est["range_m"] - 1.0) < 0.05
    assert est["range_sigma_m"] >= RANGE_SIGMA_FLOOR_M   # identical echoes do not give sigma 0

    est_ir = IRBiasEstimator(alpha0=0.2)
    for _ in range(200):
        est_ir.update(acoustic_range_m=2.0, acoustic_sigma_m=0.01, ir_depth_m=2.05, ir_confidence=0.9)
    assert abs(est_ir.bias_m - 0.05) < 0.01

    # INVERTED (v1 asserted the pose moved toward the echo): an obstacle echo
    # must update the map, not the pose.
    core4 = AutonomousDroneCore()
    telem = AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    core4.compensate_latency(telem, 0.0)
    p_before = core4.kf.x[0:3].copy()
    echo = AcousticSensorReturn(0.0, 0.0, 2.0 / core4.c_sound, 0.9, 0.0, 25.0)
    out = core4.process_flight_tick(telem, 0.0, [AcousticEchoGroup([echo] * 3)])
    assert np.allclose(core4.kf.x[0:3], p_before, atol=1e-9)
    assert out["obstacle_map"]["n_active"] == 1

    # INVERTED (v1 gated on distance from the drone): gating uses the ranger's
    # own evidence. A far-but-valid echo is kept; an out-of-envelope echo is not.
    far = AcousticSensorReturn(0.0, 0.0, 2.0 * 12.0 / 343.0, 0.9, 0.0, 25.0)
    assert ranger.estimate(AcousticEchoGroup([far] * 3)) is not None
    absurd = AcousticSensorReturn(0.0, 0.0, 2.0 * 1e6 / 343.0, 0.9, 0.0, 25.0)
    assert ranger.estimate(AcousticEchoGroup([absurd])) is None

    # INVERTED (v1 asserted motors were throttled and the battery floored):
    # flight-critical loads are never scaled; low charge -> LAND; no floor.
    tiny = BatteryState(capacity_wh=1.0, remaining_wh=0.2)
    core5 = AutonomousDroneCore(battery=tiny)
    pwr = core5.allocate_duty_cycle({"motors_hover": 1.0, "infrared": 1.0}, horizon_s=60.0)
    assert pwr["granted_fractions"]["motors_hover"] == 1.0
    assert pwr["mission_action"] == "LAND" and "infrared" in pwr["shed_loads"]
    assert core5.battery.remaining_wh == 0.2

    core6 = AutonomousDroneCore()
    base = core6.acoustic_pulse_interval_s(0.2)
    core6.compensate_latency(telem, 0.0)
    core6.kf.P[3:6, 3:6] *= 100.0
    assert core6.acoustic_pulse_interval_s(0.2) < base
    # Never faster than the round trip at max range (v1 allowed 12.5 ms pings
    # while a 15 m echo takes ~88 ms to return).
    assert core6.acoustic_pulse_interval_s(0.05) >= 2 * 15.0 / core6.c_sound - 1e-12
    print("Self-tests passed.")


# ===========================================================================
# Demo (same inputs as the v1 demo)
# ===========================================================================
def demo() -> Dict:
    core = AutonomousDroneCore(temperature_c=15.0, humidity_pct=50.0)
    att = Attitude(w=math.cos(math.pi / 8), x=0, y=0, z=math.sin(math.pi / 8))  # 45 deg yaw
    telemetry = AirframeState(
        timestamp_s=24.500,
        pos_m=np.array([10.0, 5.0, -2.5]),
        vel_mps=np.array([12.0, 0.8, -0.2]),
        acc_mps2=np.array([-1.5, 0.2, 0.05]),
        jerk_mps3=np.array([0.2, -0.05, 0.0]),
        attitude=att,
        pos_sigma_m=np.array([0.02, 0.02, 0.03]),
        vel_sigma_mps=np.array([0.05, 0.05, 0.05]),
    )
    clean = AcousticEchoGroup(echoes=[
        AcousticSensorReturn(math.radians(10.0), math.radians(-2.0), 0.0160, 0.88, 24.498, 22.0),
        AcousticSensorReturn(math.radians(10.2), math.radians(-2.1), 0.0159, 0.90, 24.498, 24.0),
        AcousticSensorReturn(math.radians(9.9), math.radians(-1.9), 0.0161, 0.85, 24.498, 20.0),
    ])
    noisy = AcousticEchoGroup(echoes=[
        AcousticSensorReturn(math.radians(-30.0), math.radians(0.0), 0.0080, 0.80, 24.499, 18.0),
        AcousticSensorReturn(math.radians(-30.1), math.radians(0.1), 0.0081, 0.82, 24.499, 19.0),
        AcousticSensorReturn(math.radians(-45.0), math.radians(3.0), 0.0300, 0.30, 24.499, 12.0),
    ])
    irs = [
        InfraredReturn(depth_m=2.40, timestamp_s=24.499, confidence=0.85,
                       surface_angle_rad=math.radians(15.0),
                       azimuth_rad=math.radians(10.0), elevation_rad=math.radians(-2.0)),
        InfraredReturn(depth_m=1.30, timestamp_s=24.499, confidence=0.80,
                       surface_angle_rad=math.radians(40.0),
                       azimuth_rad=math.radians(-30.0), elevation_rad=0.0),
    ]
    return core.process_flight_tick(
        current_telemetry=telemetry,
        tau_latency_s=0.042,
        acoustic_returns=[clean, noisy],
        infrared_returns=irs,
        duty_requests={"avionics": 1.0, "acoustic": 1.0, "infrared": 0.5, "motors_hover": 0.8},
        horizon_s=0.1,
    )


if __name__ == "__main__":
    _self_test()
    print(json.dumps(demo(), indent=2))
    print("\nClosed-loop Monte Carlo: run mc_v2.py")

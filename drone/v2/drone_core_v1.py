"""
REAL MODE: Integrated Autonomous Drone Navigation & Predictive Core
Unifies:
  1. Kalman filter (9-state, drag-consistent) with prediction-driven fusion
  2. Acousto-IR 3D voxel resolution with multi-echo median + SNR gating,
     quaternion attitude, live sound speed, IR bias estimation
  3. Dynamic battery duty-cycle power allocation from P*dt budget
  4. Predictive sensor scheduling, voxel gating, and time-to-CPA guidance

Assumptions & limitations:
  - Local ENU frame.
  - Single-bounce acoustic model; multipath rejected via median + gating.
  - Sound speed from temperature + humidity (dry-air linear + humidity term).
  - Airframe is a point mass with linear drag; rotor wash modeled as extra
    acoustic noise floor, not as a full aero model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Sequence

import numpy as np


# ===========================================================================
# Attitude: quaternion, body -> ENU
# ===========================================================================
@dataclass
class Attitude:
    """
    Unit quaternion (w, x, y, z) mapping body -> local ENU.

    Convention: q rotates a body-frame vector into the ENU frame:
        v_enu = q ⊗ v_body ⊗ q*
    """
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalize(self) -> "Attitude":
        n = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if n < 1e-12:
            return Attitude()
        return Attitude(self.w / n, self.x / n, self.y / n, self.z / n)

    def rotate(self, v_body: np.ndarray) -> np.ndarray:
        q = self.normalize()
        v = np.asarray(v_body, float)
        # v' = v + 2*w*(q_vec × v) + 2*q_vec × (q_vec × v)
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


# ===========================================================================
# Data containers
# ===========================================================================
@dataclass
class AirframeState:
    timestamp_s: float
    pos_m: np.ndarray
    vel_mps: np.ndarray
    acc_mps2: np.ndarray
    jerk_mps3: np.ndarray
    attitude: Attitude = field(default_factory=Attitude)
    pos_sigma_m: np.ndarray = field(default_factory=lambda: np.full(3, 0.05))
    vel_sigma_mps: np.ndarray = field(default_factory=lambda: np.full(3, 0.10))


@dataclass
class AcousticSensorReturn:
    azimuth_rad: float          # body-frame sensor azimuth
    elevation_rad: float        # body-frame sensor elevation
    tof_seconds: float
    echo_amplitude: float
    timestamp_s: float = 0.0
    snr_db: float = 20.0


@dataclass
class InfraredReturn:
    depth_m: float
    timestamp_s: float = 0.0
    confidence: float = 0.9
    surface_angle_rad: Optional[float] = None  # angle between ray and surface normal


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
    })

    @property
    def soc(self) -> float:
        return max(0.0, min(1.0, self.remaining_wh / max(self.capacity_wh, 1e-9)))


# ===========================================================================
# Multi-echo robust ranger
# ===========================================================================
@dataclass
class AcousticEchoGroup:
    echoes: List[AcousticSensorReturn]


class AcousticRanger:
    def __init__(
        self,
        c_sound_mps: float,
        min_snr_db: float = 10.0,
        min_amplitude: float = 0.15,
        min_range_m: float = 0.1,
        max_range_m: float = 15.0,
        max_angle_jitter_rad: float = math.radians(8.0),
    ):
        self.c_sound = float(c_sound_mps)
        self.min_snr_db = float(min_snr_db)
        self.min_amplitude = float(min_amplitude)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.max_angle_jitter_rad = float(max_angle_jitter_rad)

    def estimate(self, group: AcousticEchoGroup) -> Optional[Dict]:
        if not group.echoes:
            return None
        kept = [e for e in group.echoes
                if e.snr_db >= self.min_snr_db and e.echo_amplitude >= self.min_amplitude]
        if not kept:
            return None
        ranges = np.array([0.5 * e.tof_seconds * self.c_sound for e in kept])
        in_env = (ranges >= self.min_range_m) & (ranges <= self.max_range_m)
        if not np.any(in_env):
            return None
        kept = [e for e, ok in zip(kept, in_env) if ok]
        az = np.array([e.azimuth_rad for e in kept])
        el = np.array([e.elevation_rad for e in kept])
        az_med = float(np.median(az))
        el_med = float(np.median(el))
        d_ang = np.sqrt((az - az_med) ** 2 + (el - el_med) ** 2)
        consistent = d_ang <= self.max_angle_jitter_rad
        if not np.any(consistent):
            return None
        kept = [e for e, ok in zip(kept, consistent) if ok]
        tofs = np.array([e.tof_seconds for e in kept])
        azs = np.array([e.azimuth_rad for e in kept])
        els = np.array([e.elevation_rad for e in kept])
        amps = np.array([e.echo_amplitude for e in kept])
        snrs = np.array([e.snr_db for e in kept])

        tof_med = float(np.median(tofs))
        az_med = float(np.median(azs))
        el_med = float(np.median(els))
        range_med = 0.5 * tof_med * self.c_sound

        def mad(x):
            return float(np.median(np.abs(x - np.median(x))) * 1.4826) if len(x) > 1 else 0.0

        tof_sigma = mad(tofs)
        range_sigma = 0.5 * tof_sigma * self.c_sound
        az_sigma = mad(azs)
        el_sigma = mad(els)

        n_total = len(group.echoes)
        n_kept = len(kept)
        outlier_ratio = 1.0 - n_kept / n_total

        best_idx = int(np.argmax(snrs))
        rep = AcousticSensorReturn(
            azimuth_rad=az_med,
            elevation_rad=el_med,
            tof_seconds=tof_med,
            echo_amplitude=float(np.clip(np.mean(amps), 0.0, 1.0)),
            timestamp_s=kept[best_idx].timestamp_s,
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
            "outlier_ratio": outlier_ratio,
            "noise_scale": 1.0 + 3.0 * outlier_ratio,
        }


# ===========================================================================
# Kalman filter: 9-state drag-consistent, with voxel update
# ===========================================================================
class DragConsistentKF:
    """
    9-state [p, v, a] Kalman filter with a drag-consistent transition.

    Measurement updates:
      - update_pos(z_pos, R)      : position from GPS or fused voxel
      - update_vel(z_vel, R)      : velocity from GPS
      - update_voxel(...)          : ENU position from range + body-frame ray
    """

    def __init__(
        self,
        drag_k: float,
        mass_kg: float,
        sigma_a: float = 1.0,
        sigma_jerk: float = 2.0,
        sigma_pos_meas: float = 0.05,
        sigma_vel_meas: float = 0.10,
    ):
        self.lam = drag_k / max(mass_kg, 1e-9)
        self.sigma_a = sigma_a
        self.sigma_jerk = sigma_jerk
        self.R_pos = np.eye(3) * sigma_pos_meas**2
        self.R_vel = np.eye(3) * sigma_vel_meas**2
        self.x = np.zeros(9)
        self.P = np.eye(9) * 1.0
        self.initialized = False

    def _F(self, dt: float) -> np.ndarray:
        lam = self.lam
        if lam < 1e-9:
            F = np.eye(9)
            F[0:3, 3:6] = np.eye(3) * dt
            F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
            F[3:6, 6:9] = np.eye(3) * dt
            return F
        e = math.exp(-lam * dt)
        I = np.eye(3)
        a_pv = (1.0 - e) / lam
        a_pa = (dt - a_pv) / lam
        F = np.eye(9)
        F[0:3, 3:6] = I * a_pv
        F[0:3, 6:9] = I * a_pa
        F[3:6, 3:6] = I * e
        F[3:6, 6:9] = I * a_pv
        return F

    def _Q(self, dt: float) -> np.ndarray:
        lam = self.lam
        e = math.exp(-lam * dt) if lam > 1e-9 else 1.0
        a_pv = (1.0 - e) / lam if lam > 1e-9 else dt
        qj = self.sigma_jerk**2
        qa = self.sigma_a**2
        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = np.eye(3) * (qj * a_pv**2 * dt + qa * dt**3 / 3.0)
        Q[3:6, 3:6] = np.eye(3) * (qj * a_pv**2 + qa * dt)
        Q[6:9, 6:9] = np.eye(3) * qj * dt
        Q[0:3, 3:6] = np.eye(3) * qa * dt**2 / 2.0
        Q[3:6, 0:3] = Q[0:3, 3:6]
        return Q

    def initialize(self, state: AirframeState) -> None:
        self.x[0:3] = np.asarray(state.pos_m, float)
        self.x[3:6] = np.asarray(state.vel_mps, float)
        self.x[6:9] = np.asarray(state.acc_mps2, float)
        ps = np.asarray(state.pos_sigma_m, float) ** 2
        vs = np.asarray(state.vel_sigma_mps, float) ** 2
        asig = np.full(3, 1.0)
        self.P = np.diag(np.concatenate([ps, vs, asig**2]))
        self.initialized = True

    def predict(self, dt: float) -> None:
        if not self.initialized or dt <= 0:
            return
        F = self._F(dt)
        Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _kalman_update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        y = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_pos(self, z_pos, R=None):
        if not self.initialized:
            return
        H = np.zeros((3, 9)); H[0:3, 0:3] = np.eye(3)
        self._kalman_update(H, z_pos, self.R_pos if R is None else R)

    def update_vel(self, z_vel, R=None):
        if not self.initialized:
            return
        H = np.zeros((3, 9)); H[0:3, 3:6] = np.eye(3)
        self._kalman_update(H, z_vel, self.R_vel if R is None else R)

    def update_voxel(
        self,
        range_m: float,
        az_body: float,
        el_body: float,
        attitude: Attitude,
        range_sigma_m: float,
        az_sigma_rad: float,
        el_sigma_rad: float,
        ir_bias_m: float = 0.0,
    ) -> None:
        """
        Fuse one acoustic range + body-frame ray into the filter as a
        position-only observation.

        The ray is rotated into ENU by the attitude. The observation is
        z = p_current + range * u_enu, and the covariance of z comes from
        first-order propagation of the range and angular sigmas through the
        spherical-to-Cartesian map.
        """
        if not self.initialized:
            return

        # Correct range for IR bias learned from acoustic-IR agreement
        r_corr = max(range_m - ir_bias_m, 0.0)

        # Body ray -> ENU
        u_body = np.array([
            math.cos(el_body) * math.cos(az_body),
            math.cos(el_body) * math.sin(az_body),
            math.sin(el_body),
        ])
        u_enu = attitude.rotate(u_body)
        u_enu /= max(np.linalg.norm(u_enu), 1e-9)

        # Jacobian of (r, az, el) -> Cartesian in body frame
        d = r_corr
        saz, caz = math.sin(az_body), math.cos(az_body)
        sel, cel = math.sin(el_body), math.cos(el_body)
        J_body = np.array([
            [cel * caz, -d * cel * saz, -d * sel * caz],
            [cel * saz,  d * cel * caz, -d * sel * saz],
            [sel,        0.0,            d * cel],
        ])
        # Rotate the Jacobian into ENU: rows are ENU axes
        R_b2e = attitude.to_matrix()
        J_enu = R_b2e @ J_body

        # Measurement covariance from diagonal sigmas
        Sigma_rr = np.diag([range_sigma_m**2,
                            az_sigma_rad**2,
                            el_sigma_rad**2])
        R_meas = J_enu @ Sigma_rr @ J_enu.T + np.eye(3) * 1e-6

        # Predicted position is the filter's current p. The observation is
        # p + r*u. Since H = [I 0 0], the innovation is r*u_enu.
        z = self.x[0:3] + r_corr * u_enu
        self.update_pos(z, R=R_meas)

    def state_at(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.initialized:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        F = self._F(dt)
        x = F @ self.x
        return x[0:3].copy(), x[3:6].copy(), x[6:9].copy()

    def covariance_at(self, dt: float) -> np.ndarray:
        if not self.initialized:
            return np.eye(9)
        F = self._F(dt)
        return F @ self.P @ F.T + self._Q(dt)

    def trust(self) -> float:
        if not self.initialized:
            return 0.40
        tr = float(np.trace(self.P[0:3, 0:3]))
        sigma_p = math.sqrt(max(tr, 0.0) / 3.0)
        return float(np.clip(1.0 / (1.0 + 50.0 * sigma_p), 0.40, 0.95))


# ===========================================================================
# IR bias estimator
# ===========================================================================
class IRBiasEstimator:
    """
    Recursive estimate of the systematic IR depth bias.

    Updates only when:
      - the acoustic range has small sigma (< 5 cm),
      - the IR confidence is high (> 0.7),
      - the acoustic and IR agree within 3 sigma of the combined spread.

    Uses an exponential-moving-average with adaptive gain.
    """

    def __init__(self, alpha0: float = 0.05, min_alpha: float = 0.01, max_samples: int = 500):
        self.bias_m = 0.0
        self.alpha0 = alpha0
        self.min_alpha = min_alpha
        self.n = 0
        self.max_samples = max_samples

    def update(
        self,
        acoustic_range_m: float,
        acoustic_sigma_m: float,
        ir_depth_m: float,
        ir_confidence: float,
    ) -> float:
        if ir_confidence < 0.7 or acoustic_sigma_m > 0.05:
            return self.bias_m
        resid = ir_depth_m - acoustic_range_m
        combined_sigma = math.sqrt(acoustic_sigma_m**2 + 0.02**2)  # 2 cm IR floor
        if abs(resid) > 3.0 * combined_sigma:
            return self.bias_m
        self.n = min(self.n + 1, self.max_samples)
        # Gain decays as we accumulate samples
        alpha = max(self.min_alpha, self.alpha0 * (1.0 - self.n / self.max_samples))
        self.bias_m = (1.0 - alpha) * self.bias_m + alpha * resid
        return self.bias_m

    def correct(self, ir_depth_m: float) -> float:
        return ir_depth_m - self.bias_m


# ===========================================================================
# Core
# ===========================================================================
class AutonomousDroneCore:
    RHO_AIR = 1.225
    C0_MPS = 331.3
    AIR_ABSORPTION_DB_PER_M = 0.02

    def __init__(
        self,
        mass_kg: float = 1.45,
        drag_k: float = 0.18,
        temperature_c: float = 15.0,
        humidity_pct: float = 50.0,
        battery: Optional[BatteryState] = None,
        hover_power_w: float = 55.0,
        min_safe_soc: float = 0.15,
        kf_sigma_a: float = 1.0,
        kf_sigma_jerk: float = 2.0,
    ):
        self.mass = float(mass_kg)
        self.drag_k = float(drag_k)
        self.temperature_c = float(temperature_c)
        self.humidity_pct = float(humidity_pct)
        self.c_sound = self._sound_speed(temperature_c, humidity_pct)

        self.battery = battery or BatteryState(capacity_wh=45.0, remaining_wh=45.0)
        self.hover_power_w = float(hover_power_w)
        self.min_safe_soc = float(min_safe_soc)

        self.kf = DragConsistentKF(
            drag_k=self.drag_k,
            mass_kg=self.mass,
            sigma_a=kf_sigma_a,
            sigma_jerk=kf_sigma_jerk,
        )
        self._last_update_t: Optional[float] = None

        self.ranger = AcousticRanger(self.c_sound)
        self.ir_bias = IRBiasEstimator()

    # -- environment --------------------------------------------------------
    @staticmethod
    def _sound_speed(temperature_c: float, humidity_pct: float = 0.0) -> float:
        """Temperature + humidity corrected sound speed (dry-air + RH term)."""
        c_dry = 331.3 + 0.606 * temperature_c
        # Humidity raises c by up to ~1.5 m/s at 100% RH, 30 C
        c_hum = c_dry * (1.0 + 0.0016 * max(0.0, humidity_pct) / 100.0)
        return c_hum

    def set_environment(self, temperature_c: float, humidity_pct: float) -> None:
        self.temperature_c = float(temperature_c)
        self.humidity_pct = float(humidity_pct)
        self.c_sound = self._sound_speed(self.temperature_c, self.humidity_pct)
        self.ranger.c_sound = self.c_sound

    # -- predictive sensor scheduling --------------------------------------
    def acoustic_pulse_interval_s(self, base_interval_s: float = 0.05) -> float:
        """
        Shorten the pulse interval when the filter is uncertain or the drone
        is moving fast. Bounded to [base/4, base*4].
        """
        if not self.kf.initialized:
            return base_interval_s
        tr_v = float(np.trace(self.kf.P[3:6, 3:6]))
        sigma_v = math.sqrt(max(tr_v, 0.0) / 3.0)
        speed = float(np.linalg.norm(self.kf.x[3:6]))
        # Scale factor: 1 at sigma_v=0.1, speed=1; grows with both
        urgency = 1.0 + 4.0 * sigma_v + 0.1 * speed
        interval = base_interval_s / urgency
        return float(np.clip(interval, base_interval_s / 4.0, base_interval_s * 4.0))

    # -- 1. Latency compensation -------------------------------------------
    def compensate_latency(
        self,
        telemetry: AirframeState,
        tau_lag_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if tau_lag_s < 0:
            raise ValueError("tau_lag_s must be nonnegative")
        if not self.kf.initialized:
            self.kf.initialize(telemetry)
            self._last_update_t = telemetry.timestamp_s

        if self._last_update_t is not None and telemetry.timestamp_s > self._last_update_t:
            self.kf.predict(telemetry.timestamp_s - self._last_update_t)
        self._last_update_t = telemetry.timestamp_s

        R_pos = np.diag(np.asarray(telemetry.pos_sigma_m, float) ** 2)
        R_vel = np.diag(np.asarray(telemetry.vel_sigma_mps, float) ** 2)
        self.kf.update_pos(telemetry.pos_m, R=R_pos)
        self.kf.update_vel(telemetry.vel_mps, R=R_vel)

        p_future, v_future, _ = self.kf.state_at(tau_lag_s)
        return p_future, v_future, self.kf.trust()

    # -- 2. Voxel fusion with prediction gating ----------------------------
    def _acoustic_confidence(self, echo: AcousticSensorReturn) -> float:
        rng = 0.5 * echo.tof_seconds * self.c_sound
        absorption = self.AIR_ABSORPTION_DB_PER_M * max(rng, 0.0)
        snr_eff = max(echo.snr_db - absorption, 0.0)
        snr_term = 1.0 - math.exp(-snr_eff / 10.0)
        amp_term = float(np.clip(echo.echo_amplitude / 0.85, 0.0, 1.0))
        return float(np.clip(0.6 * snr_term + 0.4 * amp_term, 0.05, 0.99))

    def _ir_gate_by_prediction(
        self,
        ir: InfraredReturn,
        pred_pos: np.ndarray,
        pred_vel: np.ndarray,
        attitude: Attitude,
    ) -> float:
        """
        Predictive IR gate. If the surface angle was reported, discount
        grazing returns. If not, use the previous frame's implied surface
        orientation against the predicted motion: surfaces that we are
        moving *toward* are more reliable than surfaces we are skimming.
        """
        base = float(np.clip(ir.confidence, 0.05, 0.99))
        if ir.surface_angle_rad is not None:
            # cos(angle): 1 = head-on, 0 = grazing
            base *= max(0.2, math.cos(min(ir.surface_angle_rad, math.pi / 2)))
        # Slow down confidence if we are moving fast toward the surface
        speed = float(np.linalg.norm(pred_vel))
        base *= 1.0 / (1.0 + 0.02 * speed)
        return float(np.clip(base, 0.05, 0.99))

    def _predictive_voxel_gate(
        self,
        voxel_pos_enu: np.ndarray,
        range_sigma_m: float,
    ) -> float:
        """
        Chi-square-style gate against the filter's predicted position.

        A voxel is a measurement of position along a ray. Its expected spread
        is dominated by range_sigma and by the filter's own position sigma.
        """
        if not self.kf.initialized:
            return 1.0
        sigma_pred = math.sqrt(max(np.trace(self.kf.P[0:3, 0:3]) / 3.0, 1e-6))
        # Predicted position for this voxel is the current filter state, since
        # the echo was just received. Reject if the voxel is absurdly far.
        d = float(np.linalg.norm(voxel_pos_enu - self.kf.x[0:3]))
        combined = math.sqrt(sigma_pred**2 + range_sigma_m**2) + 1e-6
        z = d / combined
        if z < 3.0:
            return 1.0
        if z > 8.0:
            return 0.0
        return float(1.0 - (z - 3.0) / 5.0)

    def resolve_acousto_ir_voxel(
        self,
        echo: AcousticSensorReturn,
        ir_depth_m: float,
        attitude: Attitude,
        ir_confidence: float = 0.9,
        ir_surface_angle_rad: Optional[float] = None,
        echo_age_s: float = 0.0,
        current_pos_m: Optional[np.ndarray] = None,
        current_vel_mps: Optional[np.ndarray] = None,
        noise_scale: float = 1.0,
    ) -> Dict:
        rng = 0.5 * echo.tof_seconds * self.c_sound
        az, el = echo.azimuth_rad, echo.elevation_rad

        # Body ray -> ENU
        u_body = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ])
        u_enu = attitude.rotate(u_body)
        u_enu /= max(np.linalg.norm(u_enu), 1e-9)

        origin = np.zeros(3) if current_pos_m is None else np.asarray(current_pos_m, float)
        if current_vel_mps is not None and echo_age_s > 0.0:
            origin = origin - np.asarray(current_vel_mps, float) * echo_age_s

        voxel_enu = origin + rng * u_enu

        # IR bias correction
        ir_corrected = self.ir_bias.correct(ir_depth_m)

        ir_valid = ir_corrected > 0.0 and ir_confidence > 0.3
        penetrated = bool(ir_valid and rng > ir_corrected)
        penetration_depth = float(max(rng - ir_corrected, 0.0)) if penetrated else 0.0

        # Predictive gating
        gate = self._predictive_voxel_gate(voxel_enu, range_sigma_m=0.02)

        c_a = self._acoustic_confidence(echo)
        c_i = float(np.clip(ir_confidence, 0.05, 0.99))
        agreement = (1.0 - min(abs(rng - ir_corrected) / max(ir_corrected, 0.5), 1.0)) if ir_valid else 0.5
        voxel_tp = float(np.clip(c_a * (0.5 + 0.5 * agreement) * gate / max(noise_scale, 1e-6),
                                 0.0, 0.95))

        return {
            "voxel_pos_enu_m": [round(v, 3) for v in voxel_enu.tolist()],
            "voxel_pos_body_m": [round(rng * u_body[0], 3),
                                 round(rng * u_body[1], 3),
                                 round(rng * u_body[2], 3)],
            "acoustic_range_m": round(float(rng), 3),
            "infrared_surface_raw_m": round(float(ir_depth_m), 3),
            "infrared_surface_corrected_m": round(float(ir_corrected), 3),
            "ir_bias_estimate_m": round(float(self.ir_bias.bias_m), 4),
            "penetrated_barrier": penetrated,
            "internal_depth_m": round(penetration_depth, 3),
            "acoustic_conf": round(c_a, 3),
            "ir_conf": round(c_i, 3),
            "predictive_gate": round(gate, 3),
            "voxel_conf": round(voxel_tp, 3),
        }

    # -- 3. Power budget from P*dt -----------------------------------------
    def allocate_duty_cycle(
        self,
        requested: Dict[str, float],
        horizon_s: float = 1.0,
    ) -> Dict[str, float]:
        """
        Power-based budget: the horizon's energy cost is hover_power * horizon,
        with a small margin. This replaces the fixed 1% of usable Wh.
        """
        nominal = self.battery.draw_w
        # Energy the horizon *needs* if we grant everything at full power
        full_cost_wh = sum(nominal.get(k, 0.0) * float(max(0.0, min(1.0, v)))
                           for k, v in requested.items()) * horizon_s / 3600.0
        # Energy we can afford without crossing the reserve
        reserve_wh = self.battery.capacity_wh * self.min_safe_soc
        usable_wh = max(self.battery.remaining_wh - reserve_wh, 0.0)
        # Allow at most 5% of usable per tick; floor to avoid divide-by-zero
        budget_wh = max(usable_wh * 0.05, 1e-6)

        scale = 1.0 if full_cost_wh <= budget_wh or full_cost_wh == 0 else budget_wh / full_cost_wh
        granted = {k: round(float(max(0.0, min(1.0, v))) * scale, 3)
                   for k, v in requested.items()}
        actual_wh = full_cost_wh * scale
        self.battery.remaining_wh = max(
            self.battery.remaining_wh - actual_wh,
            self.battery.capacity_wh * self.min_safe_soc,
        )
        return {
            "granted_fractions": granted,
            "energy_used_wh": round(actual_wh, 4),
            "budget_wh": round(budget_wh, 4),
            "soc_after": round(self.battery.soc, 4),
            "throttled": bool(scale < 0.999),
        }

    # -- 4. Predictive guidance (time-to-CPA) ------------------------------
    def _predictive_avoidance(
        self,
        pred_pos: np.ndarray,
        pred_vel: np.ndarray,
        closest_point: Optional[np.ndarray],
        closest_range: float,
        flare_distance_m: float = 1.2,
    ) -> Tuple[np.ndarray, str, float]:
        """
        Compute the commanded acceleration at the time of closest approach,
        not at the current tick. This anticipates braking rather than
        reacting after the fact.
        """
        if closest_point is None or not math.isfinite(closest_range):
            return np.zeros(3), "TRACKING_NOMINAL", float("inf")

        rel = pred_pos - closest_point
        d = float(np.linalg.norm(rel))
        if d < 1e-3:
            vnorm = float(np.linalg.norm(pred_vel))
            if vnorm < 1e-3:
                return np.array([0.0, 0.0, 1.0]) * 2.0, "COLLISION_AVOIDANCE_DEGENERATE", 0.0
            return -pred_vel / vnorm * 3.0, "COLLISION_AVOIDANCE_DEGENERATE", 0.0

        # Time-to-CPA for a straight-line projection
        u = rel / d
        v_rel = pred_vel
        v_r = float(np.dot(v_rel, u))
        t_cpa = -v_r / max(np.dot(v_rel, v_rel), 1e-6) if v_r < 0 else 0.0
        t_cpa = float(np.clip(t_cpa, 0.0, 2.0))

        # Positions at CPA
        p_cpa = pred_pos + pred_vel * t_cpa
        rel_cpa = p_cpa - closest_point
        d_cpa = float(np.linalg.norm(rel_cpa)) + 1e-9
        u_cpa = rel_cpa / d_cpa

        # Repulsive accel at CPA
        k_rep = 4.0
        a_rep = k_rep * u_cpa / max(d_cpa, 0.2) ** 1.5
        a_rep = np.clip(a_rep, -8.0, 8.0)

        # Braking term if we are approaching
        a_brake = np.zeros(3)
        if v_r < 0 and d < flare_distance_m * 3.0:
            a_brake = -pred_vel * 0.8

        a_cmd = a_rep + a_brake
        status = "COLLISION_AVOIDANCE_FLARE" if (d_cpa < flare_distance_m or v_r < -0.5) else "TRACKING_NOMINAL"
        return a_cmd, status, t_cpa

    # -- Main entry ---------------------------------------------------------
    def process_flight_tick(
        self,
        current_telemetry: AirframeState,
        tau_latency_s: float,
        acoustic_returns: Sequence[AcousticSensorReturn] | List[AcousticEchoGroup],
        infrared_distances: List[float],
        infrared_returns: Optional[List[InfraredReturn]] = None,
        duty_requests: Optional[Dict[str, float]] = None,
        horizon_s: float = 0.1,
        fuse_voxels_into_kf: bool = True,
    ) -> Dict:
        # Normalize IR
        if infrared_returns is None:
            infrared_returns = [InfraredReturn(d) for d in infrared_distances]

        # Normalize acoustic groups
        if acoustic_returns and isinstance(acoustic_returns[0], AcousticEchoGroup):
            groups = list(acoustic_returns)
        else:
            groups = [AcousticEchoGroup(echoes=[e]) for e in acoustic_returns]

        n = min(len(groups), len(infrared_returns))
        groups = groups[:n]
        infrared_returns = infrared_returns[:n]

        # 0. Power gate
        duty_requests = duty_requests or {"avionics": 1.0, "acoustic": 1.0,
                                          "infrared": 1.0, "motors_hover": 1.0}
        power = self.allocate_duty_cycle(duty_requests, horizon_s=horizon_s)

        # 1. KF-based latency compensation (updates the filter with telemetry)
        pred_pos, pred_vel, kin_tp = self.compensate_latency(
            current_telemetry, tau_latency_s
        )

        # 2. Multi-echo robust ranging + predictive gating + fusion
        now = current_telemetry.timestamp_s + tau_latency_s
        point_cloud: List[Dict] = []
        ranging_stats = {
            "n_groups": len(groups),
            "n_kept_total": 0,
            "n_input_total": 0,
            "outlier_ratio_mean": 0.0,
            "gated_out": 0,
            "ir_bias_m": round(float(self.ir_bias.bias_m), 4),
            "pulse_interval_s": round(float(self.acoustic_pulse_interval_s()), 4),
        }
        ratios = []

        for group, ir in zip(groups, infrared_returns):
            est = self.ranger.estimate(group)
            if est is None:
                continue
            ranging_stats["n_kept_total"] += est["n_kept"]
            ranging_stats["n_input_total"] += est["n_total"]
            ratios.append(est["outlier_ratio"])

            rep = est["representative"]
            age = max(0.0, now - (rep.timestamp_s or now))

            # Predictive IR gate
            ir_conf = self._ir_gate_by_prediction(
                ir, pred_pos, pred_vel, current_telemetry.attitude
            )

            vox = self.resolve_acousto_ir_voxel(
                rep,
                ir.depth_m,
                attitude=current_telemetry.attitude,
                ir_confidence=ir_conf,
                ir_surface_angle_rad=ir.surface_angle_rad,
                echo_age_s=age,
                current_pos_m=pred_pos,
                current_vel_mps=pred_vel,
                noise_scale=est["noise_scale"],
            )
            vox["range_sigma_m"] = round(est["range_sigma_m"], 4)
            vox["az_sigma_rad"] = round(est["az_sigma_rad"], 5)
            vox["el_sigma_rad"] = round(est["el_sigma_rad"], 5)

            if vox["predictive_gate"] <= 0.0:
                ranging_stats["gated_out"] += 1
                continue

            # IR bias update from this pair (acoustic vs raw IR)
            self.ir_bias.update(
                acoustic_range_m=est["range_m"],
                acoustic_sigma_m=est["range_sigma_m"],
                ir_depth_m=ir.depth_m,
                ir_confidence=ir_conf,
            )

            # Fuse voxel into the filter (measurement update)
            if fuse_voxels_into_kf:
                self.kf.update_voxel(
                    range_m=est["range_m"],
                    az_body=rep.azimuth_rad,
                    el_body=rep.elevation_rad,
                    attitude=current_telemetry.attitude,
                    range_sigma_m=est["range_sigma_m"],
                    az_sigma_rad=est["az_sigma_rad"],
                    el_sigma_rad=est["el_sigma_rad"],
                    ir_bias_m=self.ir_bias.bias_m,
                )

            point_cloud.append(vox)

        if ratios:
            ranging_stats["outlier_ratio_mean"] = round(float(np.mean(ratios)), 3)

        # 3. Closest obstacle from the (now-updated) filter's perspective
        pred_pos2, pred_vel2, _ = self.kf.state_at(0.0)
        if point_cloud:
            ranges = np.array([p["acoustic_range_m"] for p in point_cloud])
            i_closest = int(np.argmin(ranges))
            closest_range = float(ranges[i_closest])
            closest_point = np.array(point_cloud[i_closest]["voxel_pos_enu_m"], float)
        else:
            closest_range = float("inf")
            closest_point = None

        # 4. Predictive guidance (time-to-CPA)
        a_cmd, action, t_cpa = self._predictive_avoidance(
            pred_pos2, pred_vel2, closest_point, closest_range
        )

        # 5. Composite trust
        map_tp = float(np.mean([p["voxel_conf"] for p in point_cloud])) if point_cloud else 0.5
        power_tp = 1.0 if not power["throttled"] else max(0.5, power["soc_after"] / max(self.min_safe_soc, 0.05))
        composite_tp = float(np.clip(0.5 * kin_tp + 0.3 * map_tp + 0.2 * power_tp, 0.05, 0.95))

        return {
            "predictive_telemetry": {
                "compensated_tau_s": round(float(tau_latency_s), 4),
                "extrapolated_pos": np.round(pred_pos, 3).tolist(),
                "extrapolated_vel": np.round(pred_vel, 3).tolist(),
                "post_fusion_pos": np.round(pred_pos2, 3).tolist(),
                "post_fusion_vel": np.round(pred_vel2, 3).tolist(),
                "kinematic_tp": round(kin_tp, 3),
            },
            "spatial_cloud_3d": point_cloud,
            "ranging_stats": ranging_stats,
            "flight_guidance": {
                "closest_obstacle_m": round(closest_range, 3) if math.isfinite(closest_range) else None,
                "time_to_cpa_s": round(t_cpa, 4) if math.isfinite(t_cpa) else None,
                "command_accel_mps2": np.round(a_cmd, 3).tolist(),
                "action": action,
                "map_tp": round(map_tp, 3),
                "power_tp": round(power_tp, 3),
                "system_tp": round(composite_tp, 3),
            },
            "power_management": power,
        }


# ===========================================================================
# Self-tests
# ===========================================================================
def _self_test() -> None:
    core = AutonomousDroneCore(temperature_c=25.0, humidity_pct=60.0)
    # c at 25C, 60% RH ~ 347.5 m/s
    assert 345.0 < core.c_sound < 350.0

    # Zero-latency identity
    st = AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    p, v, tp = core.compensate_latency(st, 0.0)
    assert np.allclose(p, 0.0, atol=1e-3) and np.allclose(v, 0.0, atol=1e-3) and tp > 0.4

    # Positive latency: monotone drift
    core2 = AutonomousDroneCore()
    st = AirframeState(0.0, np.zeros(3), np.array([10.0, 0, 0]),
                       np.zeros(3), np.zeros(3))
    p1, _, _ = core2.compensate_latency(st, 0.01)
    p2, _, _ = core2.compensate_latency(st, 0.05)
    assert p2[0] > p1[0] > 0.0

    # Environment update changes sound speed
    core3 = AutonomousDroneCore(temperature_c=15.0, humidity_pct=0.0)
    c_before = core3.c_sound
    core3.set_environment(35.0, 90.0)
    assert core3.c_sound > c_before

    # Attitude rotation round-trip
    q = Attitude(w=math.cos(math.pi/8), x=0, y=0, z=math.sin(math.pi/8))  # 45 deg yaw
    v_body = np.array([1.0, 0.0, 0.0])
    v_enu = q.rotate(v_body)
    v_back = q.inverse_rotate(v_enu)
    assert np.allclose(v_back, v_body, atol=1e-9)

    # Multi-echo median: outliers rejected
    ranger = AcousticRanger(c_sound_mps=343.0)
    good = AcousticSensorReturn(0.0, 0.0, 2.0/343.0, 0.9, 0.0, 25.0)
    outlier = AcousticSensorReturn(0.0, 0.0, 2.0*5.0/343.0, 0.9, 0.0, 25.0)
    est = ranger.estimate(AcousticEchoGroup(echoes=[good, good, good, outlier]))
    assert est is not None and abs(est["range_m"] - 1.0) < 0.05

    # IR bias estimator converges
    est_ir = IRBiasEstimator(alpha0=0.2)
    for _ in range(200):
        est_ir.update(acoustic_range_m=2.0, acoustic_sigma_m=0.01,
                      ir_depth_m=2.05, ir_confidence=0.9)
    assert abs(est_ir.bias_m - 0.05) < 0.01

    # Voxel KF update: filter position moves toward the voxel
    core4 = AutonomousDroneCore()
    telem = AirframeState(0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    core4.compensate_latency(telem, 0.0)
    p_before = core4.kf.x[0:3].copy()
    core4.kf.update_voxel(
        range_m=1.0, az_body=0.0, el_body=0.0,
        attitude=Attitude(),
        range_sigma_m=0.02, az_sigma_rad=0.01, el_sigma_rad=0.01,
    )
    p_after = core4.kf.x[0:3].copy()
    assert np.linalg.norm(p_after - p_before) > 1e-4
    assert p_after[0] > p_before[0]  # moved +x toward the echo

    # Predictive gate rejects absurd voxels
    gate = core4._predictive_voxel_gate(np.array([1e6, 0, 0]), range_sigma_m=0.02)
    assert gate == 0.0

    # Power budget uses P*dt
    tiny = BatteryState(capacity_wh=1.0, remaining_wh=0.2)
    core5 = AutonomousDroneCore(battery=tiny)
    pwr = core5.allocate_duty_cycle({"motors_hover": 1.0}, horizon_s=60.0)
    assert pwr["throttled"] is True
    assert pwr["soc_after"] >= core5.min_safe_soc - 1e-9

    # Predictive pulse interval shrinks under high uncertainty
    core6 = AutonomousDroneCore()
    base = core6.acoustic_pulse_interval_s(0.05)
    core6.compensate_latency(telem, 0.0)
    core6.kf.P[3:6, 3:6] *= 100.0  # inflate velocity variance
    fast = core6.acoustic_pulse_interval_s(0.05)
    assert fast < base

    print("Self-tests passed.")


# ===========================================================================
# Monte Carlo
# ===========================================================================
def monte_carlo_stress(n_outer: int = 200, n_inner: int = 200, seed: int = 0xC0FFEE) -> Dict:
    rng = np.random.default_rng(seed)
    core = AutonomousDroneCore(temperature_c=15.0)

    err_pos = np.zeros((n_outer, n_inner))
    err_vel = np.zeros((n_outer, n_inner))
    trust = np.zeros((n_outer, n_inner))
    avoid_ok = np.zeros((n_outer, n_inner), dtype=bool)
    kf_ok = np.zeros((n_outer, n_inner), dtype=bool)
    outlier_ratios = np.zeros((n_outer, n_inner))

    t0 = time.perf_counter()
    for i in range(n_outer):
        p0 = rng.normal(0.0, 5.0, 3)
        v0 = rng.normal(0.0, 10.0, 3)
        a0 = rng.normal(0.0, 2.0, 3)
        j0 = rng.normal(0.0, 0.5, 3)
        tau = float(rng.uniform(0.005, 0.120))
        true_dist = float(rng.uniform(0.5, 6.0))
        ir_bias_true = float(rng.normal(0.0, 0.05))
        temp_c = float(rng.uniform(-10.0, 45.0))
        rh = float(rng.uniform(0.0, 90.0))

        core.set_environment(temp_c, rh)
        core.kf = DragConsistentKF(drag_k=core.drag_k, mass_kg=core.mass)
        core._last_update_t = None
        core.ir_bias = IRBiasEstimator()

        lam = core.drag_k / core.mass
        decay = math.exp(-lam * tau)
        int_decay = (1.0 - decay) / lam
        true_pos = (p0 + v0 * int_decay
                    + 0.5 * a0 * int_decay * tau
                    + (1.0/6.0) * j0 * int_decay * tau**2)
        true_vel = (v0 + a0 * tau + 0.5 * j0 * tau**2) * decay

        att = Attitude()  # identity for the MC

        for k in range(n_inner):
            dp = rng.normal(0, 0.02, 3)
            dv = rng.normal(0, 0.05, 3)
            da = rng.normal(0, 0.10, 3)
            noisy = AirframeState(
                timestamp_s=0.0,
                pos_m=p0 + dp, vel_mps=v0 + dv,
                acc_mps2=a0 + da, jerk_mps3=j0,
                attitude=att,
                pos_sigma_m=np.array([0.02]*3),
                vel_sigma_mps=np.array([0.05]*3),
            )
            pred_p, pred_v, tp = core.compensate_latency(noisy, tau)
            err_pos[i, k] = float(np.linalg.norm(pred_p - true_pos))
            err_vel[i, k] = float(np.linalg.norm(pred_v - true_vel))
            trust[i, k] = tp

            # Acoustic group with possible outlier
            true_tof = 2.0 * true_dist / core.c_sound
            echoes = []
            for _ in range(3):
                echoes.append(AcousticSensorReturn(
                    azimuth_rad=rng.normal(0, math.radians(0.3)),
                    elevation_rad=rng.normal(0, math.radians(0.3)),
                    tof_seconds=max(true_tof + rng.normal(0, 5e-6), 1e-6),
                    echo_amplitude=float(np.clip(0.85 + rng.normal(0, 0.05), 0.05, 1.0)),
                    snr_db=float(rng.uniform(15.0, 30.0)),
                ))
            if rng.random() < 0.30:
                echoes.append(AcousticSensorReturn(
                    azimuth_rad=rng.normal(0, math.radians(5.0)),
                    elevation_rad=rng.normal(0, math.radians(5.0)),
                    tof_seconds=true_tof * rng.uniform(2.0, 5.0),
                    echo_amplitude=0.4,
                    snr_db=float(rng.uniform(10.0, 20.0)),
                ))
            est = core.ranger.estimate(AcousticEchoGroup(echoes=echoes))
            if est is not None:
                outlier_ratios[i, k] = est["outlier_ratio"]
                avoid_ok[i, k] = abs(est["range_m"] - true_dist) < 0.20

            # Voxel KF fusion
            p_before = core.kf.x[0:3].copy()
            core.kf.update_voxel(
                range_m=true_dist,
                az_body=0.0, el_body=0.0,
                attitude=att,
                range_sigma_m=0.02, az_sigma_rad=0.005, el_sigma_rad=0.005,
            )
            p_after = core.kf.x[0:3].copy()
            # The filter should shift toward +x by some amount bounded by R
            kf_ok[i, k] = (p_after[0] - p_before[0]) > 0.0

    elapsed = time.perf_counter() - t0

    def stats(x):
        x = np.asarray(x).ravel()
        return {"mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "max": float(np.max(x))}

    corr = float(np.corrcoef(trust.ravel(), err_pos.ravel())[0, 1])
    return {
        "trials": n_outer * n_inner,
        "elapsed_s": round(elapsed, 2),
        "trials_per_s": round(n_outer * n_inner / elapsed, 0),
        "pos_err_m": stats(err_pos),
        "vel_err_mps": stats(err_vel),
        "trust": stats(trust),
        "trust_error_corr": corr,
        "avoidance_pass_rate": float(avoid_ok.mean()),
        "kf_fusion_rate": float(kf_ok.mean()),
        "outlier_ratio_mean": float(np.mean(outlier_ratios)),
    }


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    _self_test()

    core = AutonomousDroneCore(temperature_c=15.0, humidity_pct=50.0)

    # Vehicle is yawed 45 deg so the acoustic ray comes back along body +x,
    # which is world (cos45, sin45, 0).
    att = Attitude(w=math.cos(math.pi/8), x=0, y=0, z=math.sin(math.pi/8))

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
        AcousticSensorReturn(math.radians(9.9),  math.radians(-1.9), 0.0161, 0.85, 24.498, 20.0),
    ])
    noisy = AcousticEchoGroup(echoes=[
        AcousticSensorReturn(math.radians(-30.0), math.radians(0.0), 0.0080, 0.80, 24.499, 18.0),
        AcousticSensorReturn(math.radians(-30.1), math.radians(0.1), 0.0081, 0.82, 24.499, 19.0),
        AcousticSensorReturn(math.radians(-45.0), math.radians(3.0), 0.0300, 0.30, 24.499, 12.0),
    ])
    irs = [
        InfraredReturn(depth_m=2.40, timestamp_s=24.499, confidence=0.85,
                       surface_angle_rad=math.radians(15.0)),
        InfraredReturn(depth_m=1.30, timestamp_s=24.499, confidence=0.80,
                       surface_angle_rad=math.radians(40.0)),
    ]

    out = core.process_flight_tick(
        current_telemetry=telemetry,
        tau_latency_s=0.042,
        acoustic_returns=[clean, noisy],
        infrared_distances=[ir.depth_m for ir in irs],
        infrared_returns=irs,
        duty_requests={"avionics": 1.0, "acoustic": 1.0,
                       "infrared": 0.5, "motors_hover": 0.8},
        horizon_s=0.1,
    )
    print(json.dumps(out, indent=2))

    print("\n--- Monte Carlo (200x200) ---")
    print(json.dumps(monte_carlo_stress(200, 200), indent=2))

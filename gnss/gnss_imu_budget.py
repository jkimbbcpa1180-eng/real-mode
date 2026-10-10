"""
Drone State-Estimation Error Budget
===================================

Decomposes the horizontal/vertical position error of a small multirotor into
per-source contributions and computes the reduction achieved by each class
of space asset:

  - GPS L1 C/A                  (baseline)
  - GPS L1 + L5 dual-frequency  (ionosphere cancellation)
  - SBAS GEO corrections        (WAAS/EGNOS/MSAS; Korea falls under MSAS)
  - PPP / SSR corrections       (precise orbit + clock + phase bias)
  - Space-weather-aware weighting (F10.7, Ap, GOES X-ray)

References for the numbers (order-of-magnitude, tune per platform):
  - RTCA DO-229E, MOPS for GPS/SBAS airborne equipment
  - ICAO Annex 10, GNSS SARPs
  - Klobuchar 1987 ionospheric model
  - NRLMSISE-00 density envelope
  - IGNAV / IGS PPP convergence studies (2018-2024)

Run:
    python gnss_imu_budget.py
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np


# ===========================================================================
# 1. Space-asset models
# ===========================================================================
@dataclass
class SpaceWeather:
    f107: float = 150.0          # sfu
    ap: float = 4.0              # 0-400
    xray_wm2: float = 1e-9       # GOES 0.1-0.8 nm

    @property
    def iono_activity(self) -> float:
        """0..1 scalar for ionospheric disturbance."""
        base = np.clip((self.f107 - 70.0) / 180.0, 0.0, 1.0)
        geomag = np.clip(self.ap / 100.0, 0.0, 1.0)
        flare = np.clip(math.log10(1.0 + self.xray_wm2 / 1e-6) / 3.0, 0.0, 1.0)
        return float(np.clip(0.5 * base + 0.3 * geomag + 0.2 * flare, 0.0, 1.0))


@dataclass
class GNSSConfig:
    """Which corrections are available on the drone."""
    l5_available: bool = False       # dual-frequency
    sbas_available: bool = True      # MSAS over Korea
    ppp_available: bool = False      # requires a corrections link
    n_sats: int = 10                 # tracked satellites
    hdop: float = 1.2                # horizontal dilution
    vdop: float = 1.8                # vertical dilution
    sbas_geo_elevation_deg: float = 45.0
    sbas_geo_azimuth_deg: float = 220.0  # MTSAT/MSAS toward SW from Korea


# ===========================================================================
# 2. Per-source error models (1-sigma, metres)
# ===========================================================================
class ErrorSources:
    """
    All outputs are 1-sigma in metres for the *pseudorange* domain unless
    labelled otherwise. Multiply by DOP to get position error.
    """

    # --- ionosphere -------------------------------------------------------
    @staticmethod
    def iono_delay_zenith_m(sw: SpaceWeather) -> float:
        """Single-frequency L1 zenith ionospheric delay."""
        # Quiet ~2 m, storm ~20 m. Klobuchar envelope.
        return 1.5 + 8.0 * sw.iono_activity + 12.0 * sw.iono_activity**2

    @staticmethod
    def iono_residual_l1(sw: SpaceWeather) -> float:
        """Broadcast Klobuchar model leaves ~30-50% of the delay."""
        return 0.4 * ErrorSources.iono_delay_zenith_m(sw)

    @staticmethod
    def iono_residual_l5(sw: SpaceWeather) -> float:
        """
        Dual-frequency (L1/L5) removes ~99% of first-order delay. Residual
        is second/third-order + inter-frequency bias, ~1-3 cm quiet, up to
        15 cm during X-class flares.
        """
        return 0.02 + 0.15 * sw.iono_activity

    # --- troposphere ------------------------------------------------------
    @staticmethod
    def tropo_zenith_m(altitude_m: float = 0.0) -> float:
        """Zenith wet+dry delay ~2.3 m at sea level, thin with altitude."""
        return 2.3 * math.exp(-altitude_m / 8000.0)

    @staticmethod
    def tropo_residual_after_model_m(elevation_deg: float = 90.0) -> float:
        """Saastamoinen + mapping function leaves ~3-5 cm."""
        obliquity = 1.0 / max(math.sin(math.radians(elevation_deg)), 0.1)
        return 0.04 * obliquity

    # --- satellite clock & orbit -----------------------------------------
    @staticmethod
    def broadcast_clock_m() -> float:
        return 1.5

    @staticmethod
    def broadcast_orbit_m() -> float:
        return 2.0

    @staticmethod
    def sbas_corrected_clock_m() -> float:
        return 0.4

    @staticmethod
    def sbas_corrected_orbit_m() -> float:
        return 0.3

    @staticmethod
    def ppp_corrected_clock_m() -> float:
        return 0.05

    @staticmethod
    def ppp_corrected_orbit_m() -> float:
        return 0.03

    # --- multipath & receiver noise --------------------------------------
    @staticmethod
    def multipath_m(environment: str = "open") -> float:
        env = {
            "open": 0.3,
            "suburban": 0.8,
            "urban": 2.0,
            "urban_canyon": 4.5,
        }
        return env.get(environment, 0.8)

    @staticmethod
    def receiver_noise_m(bandwidth_hz: float = 2e6) -> float:
        # C/N0 ~ 45 dB-Hz typical
        cn0_db = 45.0
        return 0.3 * math.sqrt(2e6 / max(bandwidth_hz, 1e3)) * math.sqrt(
            10 ** (-cn0_db / 20.0) / 10 ** (-45.0 / 20.0)
        ) + 0.15

    # --- SBAS GEO link budget --------------------------------------------
    @staticmethod
    def sbas_geo_range_error_m(elevation_deg: float,
                              range_sigma_m: float = 0.5) -> float:
        """
        The GEO correction itself carries residual error. It grows at low
        elevation because the GEO-to-drone path shares less atmosphere with
        the GPS-to-GEO path.
        """
        obliq = 1.0 / max(math.sin(math.radians(elevation_deg)), 0.2)
        return range_sigma_m * obliq


# ===========================================================================
# 3. Pseudorange covariance builder
# ===========================================================================
def pseudorange_sigma_m(
    sw: SpaceWeather,
    cfg: GNSSConfig,
    environment: str = "suburban",
    sat_elevation_deg: float = 45.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Total per-satellite pseudorange 1-sigma (m) and the per-term breakdown.
    """
    terms: Dict[str, float] = {}

    # Ionosphere
    if cfg.l5_available:
        terms["iono"] = ErrorSources.iono_residual_l5(sw)
    elif cfg.sbas_available:
        terms["iono"] = 0.5 * ErrorSources.iono_residual_l1(sw)  # SBAS halves it
    else:
        terms["iono"] = ErrorSources.iono_residual_l1(sw)

    # Troposphere
    terms["tropo"] = ErrorSources.tropo_residual_after_model_m(sat_elevation_deg)

    # Clock + orbit
    if cfg.ppp_available:
        terms["clock"] = ErrorSources.ppp_corrected_clock_m()
        terms["orbit"] = ErrorSources.ppp_corrected_orbit_m()
    elif cfg.sbas_available:
        terms["clock"] = ErrorSources.sbas_corrected_clock_m()
        terms["orbit"] = ErrorSources.sbas_corrected_orbit_m()
    else:
        terms["clock"] = ErrorSources.broadcast_clock_m()
        terms["orbit"] = ErrorSources.broadcast_orbit_m()

    # Local
    terms["multipath"] = ErrorSources.multipath_m(environment)
    terms["receiver"] = ErrorSources.receiver_noise_m()

    # SBAS GEO residual (only if SBAS is used)
    if cfg.sbas_available:
        terms["sbas_geo"] = ErrorSources.sbas_geo_range_error_m(
            cfg.sbas_geo_elevation_deg
        )

    # Combine in quadrature
    var = sum(v * v for v in terms.values())
    return math.sqrt(var), {k: round(v, 4) for k, v in terms.items()}


# ===========================================================================
# 4. Position covariance from pseudorange + DOP
# ===========================================================================
@dataclass
class PositionError:
    horizontal_m: float
    vertical_m: float
    horizontal_cep50_m: float
    vertical_cep50_m: float
    terms_h_m: Dict[str, float]
    terms_v_m: Dict[str, float]


def position_error_from_pseudorange(
    pr_sigma_m: float,
    cfg: GNSSConfig,
    terms: Dict[str, float],
    sw: SpaceWeather,
) -> PositionError:
    """
    Apply DOP scaling per axis. SBAS/PPP reduce the *pseudorange* sigma, not
    DOP, but DOP is modulated by ionospheric gradient in single-frequency
    mode — we approximate that by inflating HDOP/VDOP as iono activity rises.
    """
    iono_inflate = 1.0 + 0.4 * sw.iono_activity if not cfg.l5_available else 1.0
    hdop = cfg.hdop * iono_inflate
    vdop = cfg.vdop * iono_inflate

    h_sigma = pr_sigma_m * hdop
    v_sigma = pr_sigma_m * vdop

    # Per-term horizontal/vertical contributions (DOP-weighted)
    terms_h = {k: round(v * hdop, 4) for k, v in terms.items()}
    terms_v = {k: round(v * vdop, 4) for k, v in terms.items()}

    # CEP50 for 2D Rayleigh: ~1.177 * sigma
    return PositionError(
        horizontal_m=round(h_sigma, 3),
        vertical_m=round(v_sigma, 3),
        horizontal_cep50_m=round(1.177 * h_sigma, 3),
        vertical_cep50_m=round(0.674 * v_sigma, 3),
        terms_h_m=terms_h,
        terms_v_m=terms_v,
    )


# ===========================================================================
# 5. IMU error model + complementary filter
# ===========================================================================
@dataclass
class IMUConfig:
    accel_bias_mps2: float = 0.02       # residual after calibration
    accel_noise_mps2_per_sqrt_hz: float = 0.05
    gyro_bias_dps: float = 0.05
    gyro_noise_dps_per_sqrt_hz: float = 0.02
    sample_rate_hz: float = 200.0


def imu_position_drift_m(
    imu: IMUConfig,
    time_s: float,
    integration_order: int = 2,
) -> float:
    """
    1-sigma position drift from a pure IMU integration over `time_s`.

    Bias term grows as t^2/2, noise term grows as t^1.5 (random walk).
    """
    bias = imu.accel_bias_mps2 * time_s**2 / 2.0
    noise = (imu.accel_noise_mps2_per_sqrt_hz *
             math.sqrt(imu.sample_rate_hz) *
             time_s**1.5 / math.sqrt(3.0))
    return math.sqrt(bias**2 + noise**2)


def gnss_imu_fusion_sigma(
    gnss_sigma_m: float,
    imu: IMUConfig,
    outage_s: float,
    process_noise_mps2: float = 0.5,
) -> float:
    """
    Steady-state GNSS/IMU fusion: during GNSS outage, position sigma grows
    as a random walk driven by the process noise plus IMU bias.
    """
    imu_drift = imu_position_drift_m(imu, outage_s)
    rw = process_noise_mps2 * 0.5 * outage_s**2 / math.sqrt(3.0)
    return math.sqrt(gnss_sigma_m**2 + imu_drift**2 + rw**2)


# ===========================================================================
# 6. Top-level budget
# ===========================================================================
@dataclass
class BudgetScenario:
    name: str
    cfg: GNSSConfig


def compute_budget(
    sw: SpaceWeather,
    scenario: BudgetScenario,
    environment: str = "suburban",
    sat_elevation_deg: float = 45.0,
    outage_s: float = 0.0,
    imu: IMUConfig = field(default_factory=IMUConfig),
) -> Dict:
    pr_sigma, terms = pseudorange_sigma_m(
        sw, scenario.cfg, environment, sat_elevation_deg
    )
    pos = position_error_from_pseudorange(pr_sigma, scenario.cfg, terms, sw)
    fused_h = gnss_imu_fusion_sigma(pos.horizontal_m, imu, outage_s)
    fused_v = gnss_imu_fusion_sigma(pos.vertical_m, imu, outage_s)

    return {
        "scenario": scenario.name,
        "space_weather": asdict(sw),
        "iono_activity": round(sw.iono_activity, 3),
        "pseudorange_sigma_m": round(pr_sigma, 3),
        "terms_pr_m": terms,
        "position_gnss_only": asdict(pos),
        "position_fused": {
            "horizontal_m": round(fused_h, 3),
            "vertical_m": round(fused_v, 3),
            "horizontal_cep50_m": round(1.177 * fused_h, 3),
            "vertical_cep50_m": round(0.674 * fused_v, 3),
            "outage_s": outage_s,
        },
    }


# ===========================================================================
# 7. Space-asset marginal value (the actual gain)
# ===========================================================================
def marginal_value_table(
    sw: SpaceWeather,
    environment: str = "suburban",
    outage_s: float = 2.0,
) -> Dict:
    """
    Start from a bare single-frequency GPS fix and progressively add each
    space asset. Report the incremental reduction in fused horizontal error.
    """
    scenarios: List[BudgetScenario] = [
        BudgetScenario("L1 only (bare)", GNSSConfig(
            l5_available=False, sbas_available=False, ppp_available=False)),
        BudgetScenario("L1 + SBAS (MSAS)", GNSSConfig(
            l5_available=False, sbas_available=True, ppp_available=False)),
        BudgetScenario("L1 + L5 dual-freq", GNSSConfig(
            l5_available=True, sbas_available=False, ppp_available=False)),
        BudgetScenario("L1 + L5 + SBAS", GNSSConfig(
            l5_available=True, sbas_available=True, ppp_available=False)),
        BudgetScenario("L1 + L5 + PPP", GNSSConfig(
            l5_available=True, sbas_available=False, ppp_available=True)),
        BudgetScenario("L1 + L5 + SBAS + PPP", GNSSConfig(
            l5_available=True, sbas_available=True, ppp_available=True)),
    ]

    rows = []
    baseline_h = None
    for sc in scenarios:
        b = compute_budget(sw, sc, environment=environment, outage_s=outage_s)
        h = b["position_fused"]["horizontal_m"]
        if baseline_h is None:
            baseline_h = h
        rows.append({
            "scenario": sc.name,
            "pr_sigma_m": b["pseudorange_sigma_m"],
            "gnss_h_m": b["position_gnss_only"]["horizontal_m"],
            "fused_h_m": h,
            "fused_v_m": b["position_fused"]["vertical_m"],
            "h_reduction_pct": round(100.0 * (baseline_h - h) / baseline_h, 2),
        })

    return {
        "environment": environment,
        "outage_s": outage_s,
        "iono_activity": round(sw.iono_activity, 3),
        "rows": rows,
    }


# ===========================================================================
# 8. Demo
# ===========================================================================
def _print_table(rows: List[Dict], headers: List[str]) -> None:
    widths = [max(len(str(r[h])) for r in rows + [{h: h}]) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(headers) - 1)))
    for r in rows:
        print(fmt.format(*[str(r[h]) for h in headers]))


def main() -> None:
    print("=" * 78)
    print("DRONE STATE-ESTIMATION ERROR BUDGET")
    print("Gyeonggi-do, South Korea (example mid-latitude site)")
    print("=" * 78)

    # --- Quiet vs stormy space weather -----------------------------------
    for label, sw in [
        ("QUIET  (F10.7=100, Ap=4,  no flare)", SpaceWeather(f107=100, ap=4, xray_wm2=1e-9)),
        ("ACTIVE (F10.7=180, Ap=30, M5 flare)", SpaceWeather(f107=180, ap=30, xray_wm2=5e-5)),
        ("STORM  (F10.7=250, Ap=120, X2 flare)", SpaceWeather(f107=250, ap=120, xray_wm2=2e-4)),
    ]:
        print(f"\n{label}")
        print("-" * len(label))
        table = marginal_value_table(sw, environment="suburban", outage_s=2.0)
        _print_table(
            table["rows"],
            ["scenario", "pr_sigma_m", "gnss_h_m", "fused_h_m",
             "fused_v_m", "h_reduction_pct"],
        )

    # --- Urban canyon sensitivity ----------------------------------------
    print("\n" + "=" * 78)
    print("ENVIRONMENT SENSITIVITY  (STORM weather, L1+L5+SBAS+PPP)")
    print("=" * 78)
    cfg = GNSSConfig(l5_available=True, sbas_available=True, ppp_available=True)
    sc = BudgetScenario("full-space-assets", cfg)
    sw_storm = SpaceWeather(f107=250, ap=120, xray_wm2=2e-4)
    rows = []
    for env in ["open", "suburban", "urban", "urban_canyon"]:
        b = compute_budget(sw_storm, sc, environment=env, outage_s=2.0)
        rows.append({
            "environment": env,
            "pr_sigma_m": b["pseudorange_sigma_m"],
            "gnss_h_m": b["position_gnss_only"]["horizontal_m"],
            "fused_h_m": b["position_fused"]["horizontal_m"],
            "fused_v_m": b["position_fused"]["vertical_m"],
        })
    _print_table(rows, ["environment", "pr_sigma_m", "gnss_h_m",
                        "fused_h_m", "fused_v_m"])

    # --- Outage sensitivity ----------------------------------------------
    print("\n" + "=" * 78)
    print("GNSS OUTAGE SENSITIVITY  (STORM, suburban, full space assets)")
    print("=" * 78)
    rows = []
    for outage in [0.0, 1.0, 5.0, 15.0, 30.0]:
        b = compute_budget(sw_storm, sc, environment="suburban", outage_s=outage)
        rows.append({
            "outage_s": outage,
            "fused_h_m": b["position_fused"]["horizontal_m"],
            "fused_v_m": b["position_fused"]["vertical_m"],
            "h_cep50_m": b["position_fused"]["horizontal_cep50_m"],
        })
    _print_table(rows, ["outage_s", "fused_h_m", "fused_v_m", "h_cep50_m"])

    # --- Marginal value: space-asset stack ---------------------------------
    print("\n" + "=" * 78)
    print("SPACE-ASSET MARGINAL VALUE  (STORM, suburban, 2 s outage)")
    print("=" * 78)
    mv = marginal_value_table(sw_storm, environment="suburban", outage_s=2.0)
    for row in mv["rows"]:
        print(f"  {row['scenario']:<28} "
              f"pr={row['pr_sigma_m']:>6.3f} m  "
              f"fused_h={row['fused_h_m']:>6.3f} m  "
              f"({row['h_reduction_pct']:>6.2f}% vs L1)")


if __name__ == "__main__":
    main()

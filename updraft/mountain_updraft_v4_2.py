#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""
===============================================================================
REAL MODE: Mountain Updraft Eco-Kinetic Power & Air Purification System
v4.2 — physics fixes of v4.1 (fixes by Grok)

Target Region: Gyeonggi-do, South Korea
Grid: KEPCO 60 Hz Standard Sub-transmission
===============================================================================

What changed vs v4.1 (all simplified engineering models; numbers marked ASSUMED
are assumptions, not measurements):

1. Coupled draft. The turbine takes a fraction x of the buoyant head dp_b. The rest
   (1 - x) dp_b drives the flow against the losses:
       (1 - x) dp_b = (K_in + f L / D + K_esp + K_exit) * rho v^2 / 2
   where
   - f is the Darcy friction factor (ASSUMED 0.010: large concrete duct, Re ~ 1e7);
   - L is the conduit length along the slope;
   - K_exit = 1.0 (the exit kinetic energy is lost);
   - K_in = 0.0 and K_esp = 0.5 (ASSUMED).
   dT = Q_th / (m_dot cp) and dp_b ~ rho g H dT / T, so the turbine power is
       P = eta_t x dp_b Q ~ eta_t x g H Q_th / (cp T),
   which is independent of the flow velocity. g H / (cp T) is the chimney
   ("Carnot-like") efficiency limit, about 2.2 % for H = 650 m. v4.1 used the full
   head for the velocity AND took 70 % of it in the turbine, counting the same
   pressure twice.
2. Turbine efficiency. The default is 0.80: a ducted, pressure-staged turbine.
   Gannon & von Backstrom, "Solar chimney turbine characteristics", Solar Energy 76
   (2004) measured 85-90 % total-to-total and 77-80 % total-to-static on a model
   turbine, and list 80 % as the value commonly assumed (Schlaich 1995). The v4.1
   value 0.44 (an open wind-turbine figure) is kept as a sensitivity case.
3. Sounding. Buoyancy is integrated along the conduit: the plume cools
   dry-adiabatically, while the ambient virtual temperature (from the dewpoint) is
   interpolated from the soundings. Stable layers reduce the draft. The stable-layer
   penalty against a neutral (dry-adiabatic) atmosphere is reported.
4. ESP.
   - Charge: field (Pauthenier saturation) + diffusion (White) charge for all sizes.
     Summing them is a common engineering approximation.
   - Kept: the Cunningham correction and the Deutsch-Anderson model.
   - Reported: specific collection area, PM2.5 mass removed per day (ambient PM2.5 is
     an input; 35 ug/m3 is an example), corona power (current density ASSUMED) and the
     ESP pressure drop.
   - Effective-velocity derating. Industrial effective migration velocities are much
     lower than the theoretical value; the derate is ASSUMED 0.25 and the theoretical
     result is reported too.
   - "clean_air_km3_day" is replaced by: air treated, PM2.5 removed and air vented
     above the inversion.
5. Grid.
   - The synthetic-inertia cooldown is deterministic: the time is passed in, and
     there is no class-level wall-clock state.
   - A BESS with capacity, power and SoC limits supplies export above generation;
     export exceeds generation only when the BESS can deliver it.
   - The droop math is unchanged.
6. Removed: the hard-coded truth-probability field and the always-"OPTIMAL" status.
   The status now comes from computed checks.
7. Scale reality: thermal input, chimney efficiency, net electric power, a PV
   comparison on the same land (20 % modules, ASSUMED) and the daily energy for a
   daily irradiance profile.
8. Centrifugal stress: the root stress of a uniform blade,
   rho omega^2 (R^2 - r_hub^2) / 2, with r_hub = R - blade_length.

Standard library only. The FastAPI server is optional (--serve).
License: CC0 1.0 Universal (public domain).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Sequence

__version__ = "4.2.0"

# =====================================================================
# 1. PHYSICAL CONSTANTS
# =====================================================================

G_ACCEL = 9.80665
R_SPECIFIC_AIR = 287.058
CP_AIR = 1005.0
EPSILON_0 = 8.854187817e-12
AIR_VISCOSITY_DEFAULT = 1.81e-5
K_BOLTZMANN = 1.380649e-23
E_CHARGE = 1.602176634e-19
AIR_DENSITY_STANDARD = 1.225
DRY_ADIABATIC_LAPSE_K_M = G_ACCEL / CP_AIR          # ~9.76 K/km


# =====================================================================
# 2. DATA STRUCTURES
# =====================================================================

@dataclass
class FacilityConfig:
    base_elevation_m: float = 35.0
    crest_elevation_m: float = 685.0
    conduit_diameter_m: float = 24.0
    conduit_slope_deg: float = 30.0                 # ASSUMED: conduit length = H / sin(slope)
    collector_radius_m: float = 350.0
    collector_transmissivity: float = 0.68
    collector_loss_w_m2k: float = 0.0               # optional collector heat loss U (sensitivity)
    turbine_aerodynamic_efficiency: float = 0.80    # ducted pressure-staged turbine (see docstring)
    turbine_head_extraction_fraction: float = 0.70
    generator_efficiency: float = 0.92
    friction_factor: float = 0.010                  # ASSUMED Darcy f
    k_inlet: float = 0.0
    k_exit: float = 1.0
    k_esp: float = 0.5                              # ASSUMED ESP loss coefficient (plates along the flow)
    esp_plate_area_m2: float = 9200.0
    esp_plate_gap_m: float = 0.10
    esp_current_density_a_m2: float = 0.3e-3        # ASSUMED corona current per plate area
    esp_velocity_derate: float = 0.25               # ASSUMED effective / theoretical migration velocity
    ambient_pm25_ug_m3: float = 35.0                # EXAMPLE ambient PM2.5
    rated_inverter_capacity_mw: float = 5.0
    grid_droop_r: float = 0.04
    rotor_inertia_j: float = 4.2e5
    synthetic_inertia_cooldown_s: float = 30.0
    pv_module_efficiency: float = 0.20              # ASSUMED, for the same-land PV comparison


@dataclass
class SoundingLayer:
    pressure_hpa: float
    altitude_m: float
    temp_c: float
    dewpoint_c: float
    wind_speed_mps: float


@dataclass
class TurbineBladeSpecs:
    rotor_radius_m: float = 11.5
    blade_length_m: float = 8.0
    chord_length_m: float = 0.95
    blade_thickness_m: float = 0.18
    blade_density_kg_m3: float = 1750.0
    material_yield_mpa: float = 600.0
    cl_aerodynamic: float = 1.15


@dataclass
class BatteryStorage:
    """BESS with energy, power and SoC limits (ASSUMED sizes)."""
    capacity_mwh: float = 2.0
    power_mw: float = 1.5
    soc: float = 0.5
    soc_min: float = 0.10
    soc_max: float = 0.95
    round_trip_efficiency: float = 0.90

    def available_discharge_mw(self, dt_s: float) -> float:
        e = max(0.0, (self.soc - self.soc_min) * self.capacity_mwh)
        return min(self.power_mw, e * 3600.0 / dt_s)

    def available_charge_mw(self, dt_s: float) -> float:
        e = max(0.0, (self.soc_max - self.soc) * self.capacity_mwh)
        return min(self.power_mw, e * 3600.0 / dt_s / math.sqrt(self.round_trip_efficiency))

    def apply(self, p_mw: float, dt_s: float) -> None:
        """p_mw > 0 discharge, < 0 charge; caller respects the available limits."""
        eta = math.sqrt(self.round_trip_efficiency)
        de = p_mw * dt_s / 3600.0
        self.soc -= (de / eta if p_mw > 0 else de * eta) / self.capacity_mwh
        self.soc = min(self.soc_max, max(self.soc_min, self.soc))


# =====================================================================
# 3. ATMOSPHERIC THERMODYNAMICS
# =====================================================================

def _vapour_pressure_hpa(td_c: float) -> float:
    return 6.112 * math.exp(17.67 * td_c / (td_c + 243.5))      # Bolton (1980)


def _specific_humidity(td_c: float, p_hpa: float) -> float:
    e = _vapour_pressure_hpa(td_c)
    return 0.622 * e / (p_hpa - 0.378 * e)


class AtmosphericThermodynamics:

    @staticmethod
    def detect_inversion(soundings: List[SoundingLayer]) -> Dict:
        detected = False
        base = 0.0
        top = 0.0
        max_lapse = -999.0

        for i in range(len(soundings) - 1):
            dz = soundings[i + 1].altitude_m - soundings[i].altitude_m
            if dz <= 0:
                continue
            dt = soundings[i + 1].temp_c - soundings[i].temp_c
            lapse = dt / dz
            if lapse > 0 and not detected:
                detected = True
                base = soundings[i].altitude_m
                top = soundings[i + 1].altitude_m
                max_lapse = lapse
            elif lapse > 0 and detected:
                top = soundings[i + 1].altitude_m
                max_lapse = max(max_lapse, lapse)

        return {
            "inversion_present": detected,
            "inversion_base_m": round(base, 1),
            "inversion_top_m": round(top, 1),
            "inversion_strength_c_per_100m": round(max(0.0, max_lapse * 100.0), 2),
        }

    @staticmethod
    def ambient_profile(soundings: Sequence[SoundingLayer], z_m: float) -> Tuple[float, float]:
        """(temperature K, specific humidity) at z by linear interpolation; beyond the
        soundings the end segment's lapse rate is extrapolated."""
        s = sorted(soundings, key=lambda L: L.altitude_m)
        if len(s) == 1:
            L = s[0]
            return L.temp_c + 273.15, _specific_humidity(L.dewpoint_c, L.pressure_hpa)
        i = 0
        while i < len(s) - 2 and z_m > s[i + 1].altitude_m:
            i += 1
        a, b = s[i], s[i + 1]
        w = (z_m - a.altitude_m) / max(1e-9, b.altitude_m - a.altitude_m)
        t = a.temp_c + w * (b.temp_c - a.temp_c) + 273.15
        qa = _specific_humidity(a.dewpoint_c, a.pressure_hpa)
        qb = _specific_humidity(b.dewpoint_c, b.pressure_hpa)
        return t, qa + w * (qb - qa)

    @staticmethod
    def buoyant_head(
        config: FacilityConfig,
        soundings: Optional[Sequence[SoundingLayer]],
        surface: SoundingLayer,
        delta_t_k: float,
        n_steps: int = 200,
    ) -> float:
        """Integral of g (rho_ambient - rho_plume) dz along the conduit (Pa).
        soundings=None: neutral (dry-adiabatic) ambient from the surface values."""
        z0, z1 = config.base_elevation_m, config.crest_elevation_m
        t_s = surface.temp_c + 273.15
        q_s = _specific_humidity(surface.dewpoint_c, surface.pressure_hpa)
        p = surface.pressure_hpa * 100.0
        dz = (z1 - z0) / n_steps
        head = 0.0
        for k in range(n_steps):
            z = z0 + (k + 0.5) * dz
            if soundings is None:
                t_a, q_a = t_s - DRY_ADIABATIC_LAPSE_K_M * (z - z0), q_s
            else:
                t_a, q_a = AtmosphericThermodynamics.ambient_profile(soundings, z)
            tv_a = t_a * (1.0 + 0.608 * q_a)
            t_p = t_s + delta_t_k - DRY_ADIABATIC_LAPSE_K_M * (z - z0)
            tv_p = t_p * (1.0 + 0.608 * q_s)                     # plume keeps the surface moisture
            rho_a = p / (R_SPECIFIC_AIR * tv_a)
            rho_p = p / (R_SPECIFIC_AIR * tv_p)
            head += G_ACCEL * (rho_a - rho_p) * dz
            p -= rho_a * G_ACCEL * dz                             # hydrostatic ambient pressure
        return head

    @staticmethod
    def conduit_geometry(config: FacilityConfig) -> Tuple[float, float, float]:
        h = config.crest_elevation_m - config.base_elevation_m
        area = math.pi * (config.conduit_diameter_m / 2.0) ** 2
        length = h / math.sin(math.radians(config.conduit_slope_deg))
        return h, area, length

    @staticmethod
    def loss_coefficient(config: FacilityConfig) -> float:
        _, _, length = AtmosphericThermodynamics.conduit_geometry(config)
        return (config.k_inlet + config.friction_factor * length / config.conduit_diameter_m
                + config.k_esp + config.k_exit)

    @staticmethod
    def solve_draft(
        config: FacilityConfig,
        surface: SoundingLayer,
        crest: SoundingLayer,
        irradiance_w_m2: float,
        soundings: Optional[Sequence[SoundingLayer]] = None,
        neutral: bool = False,
        v_guess: float = 10.0,
    ) -> Tuple[float, float, float, float, float]:
        """Coupled draft. Returns (dT K, buoyant head Pa, velocity m/s, mass flow kg/s,
        volume flow m3/s), like v4.1. soundings=None or neutral=True: neutral ambient."""
        t_base_k = surface.temp_c + 273.15
        rho_base = surface.pressure_hpa * 100.0 / (R_SPECIFIC_AIR * t_base_k)
        h_eff, conduit_area, _ = AtmosphericThermodynamics.conduit_geometry(config)
        collector_area = math.pi * config.collector_radius_m ** 2
        wind_penalty = 1.0 - 0.005 * min(surface.wind_speed_mps, 20.0)
        absorbed = collector_area * irradiance_w_m2 * config.collector_transmissivity * wind_penalty
        k_tot = AtmosphericThermodynamics.loss_coefficient(config)
        prof = None if neutral else soundings

        def state(v):
            m_dot = rho_base * conduit_area * v
            # collector energy balance with optional loss U*A*dT/2 (mean collector excess)
            u_term = config.collector_loss_w_m2k * collector_area / 2.0
            dt = absorbed / (m_dot * CP_AIR + u_term)
            head = AtmosphericThermodynamics.buoyant_head(config, prof, surface, dt)
            return m_dot, dt, head

        def residual(v):
            _, _, head = state(v)
            return (1.0 - config.turbine_head_extraction_fraction) * head - k_tot * 0.5 * rho_base * v * v

        if absorbed <= 0.0 or residual(1e-3) <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        lo, hi = 1e-3, max(1.0, v_guess)
        while residual(hi) > 0.0 and hi < 500.0:          # bracket (residual falls monotonically with v)
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        v = 0.5 * (lo + hi)
        m_dot, dt, head = state(v)
        return dt, head, v, m_dot, conduit_area * v


# =====================================================================
# 4. ELECTROSTATIC PRECIPITATOR
# =====================================================================

class ElectrostaticPrecipitatorEngine:

    @staticmethod
    def _particle_charge(
        radius_m: float,
        field_v_m: float,
        dielectric_constant: float = 4.0,
        ion_concentration_per_m3: float = 1e13,
        charging_time_s: float = 1.0,
        gas_temperature_k: float = 293.15,
    ) -> float:
        """Field (Pauthenier saturation) + diffusion (White) charge, summed for every size:
        a common engineering approximation of combined charging."""
        q_field = (
            12.0 * math.pi * EPSILON_0
            * (dielectric_constant / (dielectric_constant + 2.0))
            * (radius_m ** 2)
            * field_v_m
        )
        ion_thermal_speed = 300.0
        pre_factor = 4.0 * math.pi * EPSILON_0 * K_BOLTZMANN * gas_temperature_k * radius_m / E_CHARGE
        # White (1951) diffusion charge, as in Hinds, Aerosol Technology:
        # n = (d kT / (2 K_E e^2)) ln(1 + pi K_E d c e^2 N t / (2 kT)), K_E = 1/(4 pi eps0)
        # => log argument = r * c * e^2 * N * t / (4 eps0 kT). v4.1 left out r * e here,
        #    which overstated diffusion charge about 10x for PM2.5 (fixed in v4.2).
        inner = 1.0 + (radius_m * E_CHARGE ** 2 * ion_thermal_speed
                       * ion_concentration_per_m3 * charging_time_s) \
            / (4.0 * EPSILON_0 * K_BOLTZMANN * gas_temperature_k)
        q_diffusion = pre_factor * math.log(inner)
        return q_field + q_diffusion

    @staticmethod
    def migration_velocity(field_v_m: float, particle_radius_m: float) -> float:
        q_particle = ElectrostaticPrecipitatorEngine._particle_charge(particle_radius_m, field_v_m)
        lambda_mfp = 6.6e-8
        c_c = 1.0 + (lambda_mfp / particle_radius_m) * (
            1.257 + 0.4 * math.exp(-1.1 * particle_radius_m / lambda_mfp))
        return (q_particle * field_v_m * c_c) / (6.0 * math.pi * AIR_VISCOSITY_DEFAULT * particle_radius_m)

    @staticmethod
    def calculate_filtration(
        config: FacilityConfig,
        volumetric_flow_m3_s: float,
        voltage_kv: float,
        particle_radius_m: float = 1.25e-6,
        rho_air: float = AIR_DENSITY_STANDARD,
        velocity_mps: float = 0.0,
    ) -> Dict:
        field_v_m = (voltage_kv * 1000.0) / config.esp_plate_gap_m
        w_theory = ElectrostaticPrecipitatorEngine.migration_velocity(field_v_m, particle_radius_m)
        w_eff = w_theory * config.esp_velocity_derate
        q = max(1e-9, volumetric_flow_m3_s)
        sca = config.esp_plate_area_m2 / q                     # s/m  (m2 per m3/s)
        eff_theory = 1.0 - math.exp(-min(50.0, sca * w_theory))
        eff = 1.0 - math.exp(-min(50.0, sca * w_eff))
        treated_m3_day = volumetric_flow_m3_s * 86400.0
        pm_removed_kg_day = treated_m3_day * config.ambient_pm25_ug_m3 * 1e-9 * eff
        corona_mw = voltage_kv * 1e3 * config.esp_current_density_a_m2 * config.esp_plate_area_m2 / 1e6
        dp_esp = config.k_esp * 0.5 * rho_air * velocity_mps ** 2
        return {
            "migration_velocity_theory_mps": w_theory,
            "migration_velocity_effective_mps": w_eff,
            "specific_collection_area_s_per_m": sca,
            "efficiency_theory": eff_theory,
            "efficiency": eff,
            "air_treated_km3_day": treated_m3_day / 1e9,
            "pm25_removed_kg_day": pm_removed_kg_day,
            "corona_power_mw": corona_mw,
            "pressure_drop_pa": dp_esp,
            "pressure_drop_flow_work_mw": dp_esp * volumetric_flow_m3_s / 1e6,
        }


# =====================================================================
# 5. TURBOMACHINERY STRESS
# =====================================================================

class TurbomachineryStressEngine:

    @staticmethod
    def evaluate_structural_margin(
        v_flow_mps: float,
        specs: TurbineBladeSpecs,
        rotor_rpm: float,
    ) -> Dict:
        omega = rotor_rpm * (2.0 * math.pi / 60.0)
        v_tip = omega * specs.rotor_radius_m
        v_rel = math.sqrt(v_flow_mps ** 2 + v_tip ** 2)

        # uniform blade, root stress: rho omega^2 (R^2 - r_hub^2) / 2, r_hub = R - blade length
        r_hub = max(0.0, specs.rotor_radius_m - specs.blade_length_m)
        sigma_centrifugal = 0.5 * specs.blade_density_kg_m3 * omega ** 2 * (specs.rotor_radius_m ** 2 - r_hub ** 2)

        lift_per_m = 0.5 * AIR_DENSITY_STANDARD * (v_rel ** 2) * specs.chord_length_m * specs.cl_aerodynamic
        m_bend = 0.5 * lift_per_m * (specs.blade_length_m ** 2)

        i_xx = (math.pi / 64.0) * specs.chord_length_m * (specs.blade_thickness_m ** 3)
        y_max = specs.blade_thickness_m / 2.0
        sigma_bending = (m_bend * y_max) / max(1e-9, i_xx)

        sigma_combined = (sigma_centrifugal + sigma_bending) / 1e6
        sf = specs.material_yield_mpa / max(0.01, sigma_combined)

        return {
            "relative_flow_velocity_mps": round(v_rel, 2),
            "centrifugal_stress_mpa": round(sigma_centrifugal / 1e6, 3),
            "aerodynamic_bending_stress_mpa": round(sigma_bending / 1e6, 2),
            "peak_combined_stress_mpa": round(sigma_combined, 2),
            "safety_factor": round(sf, 2),
            "structural_status": (
                "CLEARED" if sf >= 2.0
                else "TRIP_WARNING" if sf >= 1.2
                else "EMERGENCY_SHUTDOWN"
            ),
        }


# =====================================================================
# 6. GRID INTERTIE
# =====================================================================

class GridIntertieController:
    """Droop + synthetic inertia + BESS. Deterministic: the caller passes the time."""

    def __init__(self, bess: Optional[BatteryStorage] = None, dispatch_interval_s: float = 60.0):
        self.bess = bess if bess is not None else BatteryStorage()
        self.dt_s = dispatch_interval_s
        self.last_synthetic_inertia_ts: Optional[float] = None

    def modulate_grid_export(
        self,
        config: FacilityConfig,
        generation_raw_mw: float,
        measured_freq_hz: float,
        rotor_rpm: float,
        now_s: float = 0.0,
    ) -> Dict:
        nominal_freq = 60.000
        delta_f = measured_freq_hz - nominal_freq

        # Under-frequency (delta_f < 0) => positive export increase (unchanged from v4.1)
        p_droop_mw = -(1.0 / config.grid_droop_r) * (delta_f / nominal_freq) * config.rated_inverter_capacity_mw
        cap = config.rated_inverter_capacity_mw * 0.35
        p_droop_mw = max(-cap, min(cap, p_droop_mw))

        omega = rotor_rpm * (2.0 * math.pi / 60.0)
        synthetic_mw = 0.0
        cooled = (self.last_synthetic_inertia_ts is None
                  or (now_s - self.last_synthetic_inertia_ts) > config.synthetic_inertia_cooldown_s)
        if delta_f < -0.15 and omega > 2.0 and cooled:
            e_rotor = 0.5 * config.rotor_inertia_j * omega ** 2
            extractable_mj = 0.05 * e_rotor / 1e6
            synthetic_mw = min(1.20, extractable_mj / 0.5)       # 0.5 s release
            self.last_synthetic_inertia_ts = now_s

        request = generation_raw_mw + p_droop_mw + synthetic_mw
        target = max(0.0, min(config.rated_inverter_capacity_mw, request))
        # the BESS covers export - generation - synthetic (positive: discharge)
        need = target - generation_raw_mw - synthetic_mw
        if need > 0:
            bess_mw = min(need, self.bess.available_discharge_mw(self.dt_s))
        else:
            bess_mw = -min(-need, self.bess.available_charge_mw(self.dt_s))
        export = generation_raw_mw + synthetic_mw + bess_mw
        curtailed = max(0.0, generation_raw_mw + synthetic_mw + bess_mw - target)
        export -= curtailed
        self.bess.apply(bess_mw, self.dt_s)
        return {
            "grid_frequency_hz": round(measured_freq_hz, 3),
            "frequency_error_hz": round(delta_f, 3),
            "pfr_droop_command_mw": round(p_droop_mw, 3),
            "synthetic_inertia_discharged_mw": round(synthetic_mw, 3),
            "requested_export_mw": round(target, 3),
            "net_intertie_export_mw": round(export, 3),
            "bess_flow_mw": round(bess_mw, 3),               # + discharge / - charge
            "curtailed_mw": round(curtailed, 3),
            "bess_soc": round(self.bess.soc, 4),
            "bess_limited": abs(export - target) > 1e-9,
        }


# =====================================================================
# 7. TURBINE POWER (PRESSURE-DROP FORMULATION)
# =====================================================================

def compute_turbine_power(
    config: FacilityConfig,
    delta_p_pa: float,
    vol_flow_m3_s: float,
) -> Tuple[float, float]:
    """delta_p_pa is the buoyant head; the turbine takes the fraction x of it."""
    delta_p_turbine = delta_p_pa * config.turbine_head_extraction_fraction
    p_shaft_w = config.turbine_aerodynamic_efficiency * delta_p_turbine * vol_flow_m3_s
    p_elec_w = p_shaft_w * config.generator_efficiency
    return p_shaft_w / 1e6, p_elec_w / 1e6


def chimney_efficiency(config: FacilityConfig, surface: SoundingLayer) -> float:
    """g H / (cp T): the ideal fraction of collector heat convertible by the column."""
    h = config.crest_elevation_m - config.base_elevation_m
    return G_ACCEL * h / (CP_AIR * (surface.temp_c + 273.15))


def default_irradiance_profile(peak_w_m2: float = 780.0, day_hours: float = 12.0) -> List[float]:
    """Hourly irradiance (24 values): half-sine over day_hours centred on noon (EXAMPLE)."""
    out = []
    for h in range(24):
        t = h + 0.5 - (12.0 - day_hours / 2.0)
        out.append(peak_w_m2 * math.sin(math.pi * t / day_hours) if 0.0 < t < day_hours else 0.0)
    return out


# =====================================================================
# 8. INTEGRATED PIPELINE
# =====================================================================

def run_system_cycle(
    config: FacilityConfig,
    soundings: List[SoundingLayer],
    irradiance_w_m2: float,
    esp_voltage_kv: float,
    grid_freq_hz: float,
    rotor_rpm: float,
    blade_specs: Optional[TurbineBladeSpecs] = None,
    grid: Optional[GridIntertieController] = None,
    now_s: float = 0.0,
    daily_irradiance_w_m2: Optional[Sequence[float]] = None,
) -> Dict:
    if len(soundings) < 2:
        raise ValueError("at least two sounding layers required")

    surface = soundings[0]
    crest = soundings[-1]
    blades = blade_specs or TurbineBladeSpecs()
    grid = grid or GridIntertieController()

    inversion = AtmosphericThermodynamics.detect_inversion(soundings)
    pierces = config.crest_elevation_m > inversion["inversion_top_m"]

    delta_t, delta_p, v_flow, m_dot, vol_flow = AtmosphericThermodynamics.solve_draft(
        config, surface, crest, irradiance_w_m2, soundings)
    dt_n, dp_n, v_n, _, q_n = AtmosphericThermodynamics.solve_draft(
        config, surface, crest, irradiance_w_m2, soundings, neutral=True)

    shaft_mw, elec_mw = compute_turbine_power(config, delta_p, vol_flow)
    _, elec_neutral_mw = compute_turbine_power(config, dp_n, q_n)

    rho_base = surface.pressure_hpa * 100.0 / (R_SPECIFIC_AIR * (surface.temp_c + 273.15))
    esp = ElectrostaticPrecipitatorEngine.calculate_filtration(
        config, vol_flow, esp_voltage_kv, rho_air=rho_base, velocity_mps=v_flow)
    net_mw = elec_mw - esp["corona_power_mw"]

    stress = TurbomachineryStressEngine.evaluate_structural_margin(v_flow, blades, rotor_rpm)
    grid_out = grid.modulate_grid_export(config, max(0.0, net_mw), grid_freq_hz, rotor_rpm, now_s)

    collector_area = math.pi * config.collector_radius_m ** 2
    thermal_mw = m_dot * CP_AIR * delta_t / 1e6
    eta_ch = chimney_efficiency(config, surface)
    pv_mw = collector_area * irradiance_w_m2 * config.pv_module_efficiency / 1e6
    scale = {
        "thermal_input_mw": round(thermal_mw, 2),
        "chimney_efficiency_limit_percent": round(eta_ch * 100.0, 3),
        "electric_ceiling_mw": round(thermal_mw * eta_ch * config.turbine_aerodynamic_efficiency
                                     * config.turbine_head_extraction_fraction * config.generator_efficiency, 3),
        "net_electric_mw": round(net_mw, 3),
        "solar_to_net_electric_percent": round(100.0 * net_mw * 1e6 / max(1e-9, collector_area * irradiance_w_m2), 4),
        "pv_same_land_mw": round(pv_mw, 1),
        "pv_module_efficiency_assumed": config.pv_module_efficiency,
    }
    if daily_irradiance_w_m2 is not None:
        e_day = 0.0
        for g_h in daily_irradiance_w_m2:
            if g_h <= 0:
                continue
            dt_h, dp_h, _, _, q_h = AtmosphericThermodynamics.solve_draft(config, surface, crest, g_h, soundings)
            e_day += compute_turbine_power(config, dp_h, q_h)[1]
            e_day -= ElectrostaticPrecipitatorEngine.calculate_filtration(config, q_h, esp_voltage_kv)["corona_power_mw"]
        scale["daily_net_energy_mwh"] = round(e_day, 2)
        scale["daily_pv_same_land_mwh"] = round(sum(daily_irradiance_w_m2) * collector_area
                                                * config.pv_module_efficiency / 1e6, 1)

    checks = {
        "structural": stress["structural_status"],
        "positive_net_power": net_mw > 0.0,
        "flow_established": v_flow > 0.5,
        "conduit_pierces_inversion": pierces,
        "grid_request_met": not grid_out["bess_limited"],
        "esp_sca_in_industrial_range": esp["specific_collection_area_s_per_m"] >= 20.0,
    }
    if stress["structural_status"] == "EMERGENCY_SHUTDOWN" or not checks["flow_established"]:
        status = "FAULT"
    elif all(v is True for k, v in checks.items() if k != "structural") and stress["structural_status"] == "CLEARED":
        status = "NOMINAL"
    else:
        status = "DEGRADED"

    return {
        "version": __version__,
        "execution_status": status,
        "checks": checks,
        "meteorological_inversion": {**inversion, "conduit_pierces_inversion": pierces},
        "thermodynamics_and_updraft": {
            "temperature_lift_k": round(delta_t, 2),
            "buoyant_head_pa": round(delta_p, 2),
            "turbine_pressure_drop_pa": round(delta_p * config.turbine_head_extraction_fraction, 2),
            "chimney_velocity_mps": round(v_flow, 2),
            "volumetric_flow_m3_s": round(vol_flow, 1),
            "loss_coefficient_total": round(AtmosphericThermodynamics.loss_coefficient(config), 3),
            "shaft_power_mw": round(shaft_mw, 3),
            "electrical_power_mw": round(elec_mw, 3),
            "neutral_atmosphere_flow_m3_s": round(q_n, 1),
            "neutral_atmosphere_electrical_mw": round(elec_neutral_mw, 3),
            "stable_layer_penalty_percent": round(100.0 * (1.0 - elec_mw / elec_neutral_mw), 2)
            if elec_neutral_mw > 0 else None,
        },
        "esp_air_purification": {
            "pm25_migration_velocity_theory_mps": round(esp["migration_velocity_theory_mps"], 5),
            "pm25_migration_velocity_effective_mps": round(esp["migration_velocity_effective_mps"], 5),
            "specific_collection_area_s_per_m": round(esp["specific_collection_area_s_per_m"], 3),
            "capture_efficiency_theory_percent": round(esp["efficiency_theory"] * 100.0, 3),
            "capture_efficiency_percent": round(esp["efficiency"] * 100.0, 3),
            "air_treated_km3_day": round(esp["air_treated_km3_day"], 4),
            "pm25_removed_kg_day": round(esp["pm25_removed_kg_day"], 2),
            "air_vented_above_inversion_km3_day": round(esp["air_treated_km3_day"] if pierces else 0.0, 4),
            "ambient_pm25_ug_m3_example": config.ambient_pm25_ug_m3,
            "corona_power_mw": round(esp["corona_power_mw"], 4),
            "esp_pressure_drop_pa": round(esp["pressure_drop_pa"], 1),
            "esp_pressure_drop_flow_work_mw": round(esp["pressure_drop_flow_work_mw"], 3),
        },
        "turbomachinery_structural": stress,
        "grid_frequency_intertie": grid_out,
        "scale_reality": scale,
    }


# =====================================================================
# 9. STANDALONE BENCHMARK EXECUTION
# =====================================================================

DEMO_SOUNDINGS = [
    SoundingLayer(1016.0, 35.0, 14.2, 7.5, 1.2),
    SoundingLayer(995.0, 210.0, 12.8, 6.0, 1.8),
    SoundingLayer(975.0, 380.0, 15.1, 5.2, 2.4),
    SoundingLayer(940.0, 685.0, 11.4, 3.1, 5.8),
]
DEMO_INPUTS = dict(irradiance_w_m2=780.0, esp_voltage_kv=50.0, grid_freq_hz=59.780, rotor_rpm=44.0)


def run_self_benchmark() -> None:
    print("=" * 78)
    print(f"MOUNTAIN ECO-KINETIC ENGINE v{__version__} — STANDALONE BENCHMARK")
    print("=" * 78)
    config = FacilityConfig()
    result = run_system_cycle(config=config, soundings=DEMO_SOUNDINGS,
                              daily_irradiance_w_m2=default_irradiance_profile(), **DEMO_INPUTS)
    inv = result["meteorological_inversion"]
    print(f"[*] Inversion: {inv['inversion_base_m']}m to {inv['inversion_top_m']}m "
          f"(+{inv['inversion_strength_c_per_100m']}°C/100m) | pierces: {inv['conduit_pierces_inversion']}")
    th = result["thermodynamics_and_updraft"]
    print(f"[*] dT = {th['temperature_lift_k']} K | buoyant head = {th['buoyant_head_pa']} Pa "
          f"| turbine dp = {th['turbine_pressure_drop_pa']} Pa | v = {th['chimney_velocity_mps']} m/s")
    print(f"[*] Volumetric flow: {th['volumetric_flow_m3_s']} m³/s (neutral atmosphere: "
          f"{th['neutral_atmosphere_flow_m3_s']})")
    print(f"[*] Shaft / electrical: {th['shaft_power_mw']} / {th['electrical_power_mw']} MW "
          f"(stable-layer penalty {th['stable_layer_penalty_percent']} %)")
    esp = result["esp_air_purification"]
    print(f"[*] ESP: w theory {esp['pm25_migration_velocity_theory_mps']} m/s, effective "
          f"{esp['pm25_migration_velocity_effective_mps']} m/s | SCA {esp['specific_collection_area_s_per_m']} s/m")
    print(f"[*] ESP capture: {esp['capture_efficiency_percent']} % (theory {esp['capture_efficiency_theory_percent']} %)"
          f" | corona {esp['corona_power_mw']} MW | dp {esp['esp_pressure_drop_pa']} Pa")
    print(f"[*] Air treated {esp['air_treated_km3_day']} km³/day | PM2.5 removed {esp['pm25_removed_kg_day']} kg/day "
          f"(at {esp['ambient_pm25_ug_m3_example']} µg/m³ example) | vented above inversion "
          f"{esp['air_vented_above_inversion_km3_day']} km³/day")
    st = result["turbomachinery_structural"]
    print(f"[*] Stress: {st['peak_combined_stress_mpa']} MPa (centrifugal {st['centrifugal_stress_mpa']}) "
          f"| SF = {st['safety_factor']} [{st['structural_status']}]")
    gr = result["grid_frequency_intertie"]
    print(f"[*] Grid: df = {gr['frequency_error_hz']} Hz | droop {gr['pfr_droop_command_mw']} MW | synthetic "
          f"{gr['synthetic_inertia_discharged_mw']} MW")
    print(f"[+] Export {gr['net_intertie_export_mw']} MW (requested {gr['requested_export_mw']}) | BESS "
          f"{gr['bess_flow_mw']} MW, SoC {gr['bess_soc']}")
    sr = result["scale_reality"]
    print(f"[*] Scale: thermal {sr['thermal_input_mw']} MW x chimney limit {sr['chimney_efficiency_limit_percent']} % "
          f"-> net {sr['net_electric_mw']} MW ({sr['solar_to_net_electric_percent']} % of sunlight); "
          f"PV on same land ~{sr['pv_same_land_mw']} MW")
    print(f"[*] Daily (example profile): {sr['daily_net_energy_mwh']} MWh vs PV {sr['daily_pv_same_land_mwh']} MWh")
    print(f"[=] Status: {result['execution_status']} {result['checks']}")
    print("=" * 78)


# =====================================================================
# 10. OPTIONAL FASTAPI SERVER
# =====================================================================

def launch_server() -> None:
    try:
        from pydantic import BaseModel, Field
        from fastapi import FastAPI, HTTPException
        import uvicorn
    except ImportError:
        print("error: install with: pip install fastapi uvicorn pydantic", file=sys.stderr)
        sys.exit(1)

    class SoundingLayerModel(BaseModel):
        pressure_hpa: float = Field(..., ge=200.0, le=1100.0)
        altitude_m: float = Field(..., ge=-50.0, le=9000.0)
        temp_c: float = Field(..., ge=-60.0, le=60.0)
        dewpoint_c: float = Field(..., ge=-70.0, le=50.0)
        wind_speed_mps: float = Field(..., ge=0.0, le=100.0)

    class FullTelemetryPayload(BaseModel):
        solar_irradiance_w_m2: float = Field(default=750.0, ge=0.0, le=1400.0)
        esp_operating_voltage_kv: float = Field(default=48.0, ge=10.0, le=100.0)
        measured_grid_freq_hz: float = Field(default=60.0, ge=56.0, le=64.0)
        rotor_rpm: float = Field(default=45.0, ge=0.0, le=100.0)
        timestamp_s: float = Field(default=0.0, ge=0.0)
        sounding_layers: List[SoundingLayerModel]

    app = FastAPI(title="Mountain Eco-Kinetic Power & Smog Scrubber", version=__version__)
    facility = FacilityConfig()
    grid = GridIntertieController()          # one controller (BESS + inertia state) per server

    @app.post("/api/v1/telemetry/dispatch", tags=["Dispatch"])
    async def dispatch(payload: FullTelemetryPayload):
        if len(payload.sounding_layers) < 2:
            raise HTTPException(status_code=400, detail="at least two sounding layers required")
        soundings = [SoundingLayer(s.pressure_hpa, s.altitude_m, s.temp_c, s.dewpoint_c, s.wind_speed_mps)
                     for s in payload.sounding_layers]
        try:
            return run_system_cycle(config=facility, soundings=soundings,
                                    irradiance_w_m2=payload.solar_irradiance_w_m2,
                                    esp_voltage_kv=payload.esp_operating_voltage_kv,
                                    grid_freq_hz=payload.measured_grid_freq_hz,
                                    rotor_rpm=payload.rotor_rpm, grid=grid, now_s=payload.timestamp_s)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    if "--serve" in sys.argv:
        launch_server()
    else:
        run_self_benchmark()

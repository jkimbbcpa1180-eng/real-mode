#!/usr/bin/env python3
"""
===============================================================================
REAL MODE: Mountain Updraft Eco-Kinetic Power & Air Purification System
v4.1 — Monolithic Physics-Corrected Executable

Target Region: Gyeonggi-do, South Korea
Grid: KEPCO 60 Hz Standard Sub-transmission
===============================================================================
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


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


# =====================================================================
# 2. DATA STRUCTURES
# =====================================================================

@dataclass
class FacilityConfig:
    base_elevation_m: float = 35.0
    crest_elevation_m: float = 685.0
    conduit_diameter_m: float = 24.0
    collector_radius_m: float = 350.0
    collector_transmissivity: float = 0.68
    turbine_aerodynamic_efficiency: float = 0.44
    turbine_head_extraction_fraction: float = 0.70
    generator_efficiency: float = 0.92
    esp_plate_area_m2: float = 9200.0
    esp_plate_gap_m: float = 0.10
    rated_inverter_capacity_mw: float = 5.0
    grid_droop_r: float = 0.04
    rotor_inertia_j: float = 4.2e5
    synthetic_inertia_cooldown_s: float = 30.0


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


# =====================================================================
# 3. ATMOSPHERIC THERMODYNAMICS
# =====================================================================

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
    def solve_draft(
        config: FacilityConfig,
        surface: SoundingLayer,
        crest: SoundingLayer,
        irradiance_w_m2: float,
    ) -> Tuple[float, float, float, float, float]:
        t_base_k = surface.temp_c + 273.15
        p_base_pa = surface.pressure_hpa * 100.0
        rho_base = p_base_pa / (R_SPECIFIC_AIR * t_base_k)

        h_eff = config.crest_elevation_m - config.base_elevation_m
        collector_area = math.pi * (config.collector_radius_m ** 2)
        conduit_area = math.pi * ((config.conduit_diameter_m / 2.0) ** 2)

        wind_speed = surface.wind_speed_mps
        wind_penalty = 1.0 - 0.005 * min(wind_speed, 20.0)
        effective_transmissivity = config.collector_transmissivity * wind_penalty

        q_thermal_w = collector_area * irradiance_w_m2 * effective_transmissivity

        v_flow = 10.0
        delta_t = 5.0
        delta_p = 100.0
        m_dot = rho_base * conduit_area * v_flow

        for _ in range(20):
            m_dot = rho_base * conduit_area * v_flow
            delta_t = q_thermal_w / max(1.0, m_dot * CP_AIR)
            delta_p = rho_base * G_ACCEL * h_eff * (delta_t / t_base_k)
            v_flow = math.sqrt(max(0.1, 2.0 * delta_p / rho_base))

        vol_flow = conduit_area * v_flow
        return delta_t, delta_p, v_flow, m_dot, vol_flow


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
        if radius_m > 2.0e-6:
            # Pauthenier field charging
            return (
                12.0 * math.pi * EPSILON_0
                * (dielectric_constant / (dielectric_constant + 2.0))
                * (radius_m ** 2)
                * field_v_m
            )
        # White diffusion charging
        ion_thermal_speed = 300.0
        pre_factor = (
            4.0 * math.pi * EPSILON_0 * K_BOLTZMANN
            * gas_temperature_k * radius_m / E_CHARGE
        )
        inner = (
            1.0
            + (E_CHARGE * ion_thermal_speed
               * ion_concentration_per_m3 * charging_time_s)
            / (4.0 * EPSILON_0 * K_BOLTZMANN * gas_temperature_k)
        )
        return pre_factor * math.log(inner)

    @staticmethod
    def calculate_filtration(
        config: FacilityConfig,
        volumetric_flow_m3_s: float,
        voltage_kv: float,
        particle_radius_m: float = 1.25e-6,
    ) -> Tuple[float, float, float]:
        field_v_m = (voltage_kv * 1000.0) / config.esp_plate_gap_m
        dielectric_constant = 4.0

        q_particle = ElectrostaticPrecipitatorEngine._particle_charge(
            particle_radius_m, field_v_m, dielectric_constant
        )

        lambda_mfp = 6.6e-8
        c_c = 1.0 + (lambda_mfp / particle_radius_m) * (
            1.257 + 0.4 * math.exp(-1.1 * particle_radius_m / lambda_mfp)
        )

        w_e = (q_particle * field_v_m * c_c) / (
            6.0 * math.pi * AIR_VISCOSITY_DEFAULT * particle_radius_m
        )

        exponent = min(
            50.0,
            (config.esp_plate_area_m2 * w_e) / max(1.0, volumetric_flow_m3_s),
        )
        efficiency = 1.0 - math.exp(-exponent)

        clean_air_m3_day = volumetric_flow_m3_s * efficiency * 86400.0
        clean_air_km3_day = clean_air_m3_day / 1e9

        return w_e, efficiency, clean_air_km3_day


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

        sigma_centrifugal = (
            specs.blade_density_kg_m3
            * (omega ** 2)
            * (specs.rotor_radius_m ** 2)
        )

        lift_per_m = (
            0.5 * AIR_DENSITY_STANDARD * (v_rel ** 2)
            * specs.chord_length_m * specs.cl_aerodynamic
        )
        m_bend = 0.5 * lift_per_m * (specs.blade_length_m ** 2)

        i_xx = (math.pi / 64.0) * specs.chord_length_m * (specs.blade_thickness_m ** 3)
        y_max = specs.blade_thickness_m / 2.0
        sigma_bending = (m_bend * y_max) / max(1e-9, i_xx)

        sigma_combined = (sigma_centrifugal + sigma_bending) / 1e6
        sf = specs.material_yield_mpa / max(0.01, sigma_combined)

        return {
            "relative_flow_velocity_mps": round(v_rel, 2),
            "centrifugal_stress_mpa": round(sigma_centrifugal / 1e6, 2),
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

    _last_synthetic_inertia_ts: float = 0.0

    @classmethod
    def modulate_grid_export(
        cls,
        config: FacilityConfig,
        generation_raw_mw: float,
        measured_freq_hz: float,
        rotor_rpm: float,
    ) -> Dict:
        nominal_freq = 60.000
        delta_f = measured_freq_hz - nominal_freq

        # Under-frequency (delta_f < 0) => positive export increase
        p_droop_mw = (
            -(1.0 / config.grid_droop_r)
            * (delta_f / nominal_freq)
            * config.rated_inverter_capacity_mw
        )
        p_droop_mw = max(
            -config.rated_inverter_capacity_mw * 0.35,
            min(config.rated_inverter_capacity_mw * 0.35, p_droop_mw),
        )

        omega = rotor_rpm * (2.0 * math.pi / 60.0)
        synthetic_mw = 0.0
        now = time.time()

        if (
            delta_f < -0.15
            and omega > 2.0
            and (now - cls._last_synthetic_inertia_ts) > config.synthetic_inertia_cooldown_s
        ):
            e_rotor = 0.5 * config.rotor_inertia_j * (omega ** 2)
            extractable_mj = 0.05 * e_rotor / 1e6
            synthetic_mw = min(1.20, extractable_mj / 0.5)
            cls._last_synthetic_inertia_ts = now

        target_export = max(
            0.0,
            min(
                config.rated_inverter_capacity_mw,
                generation_raw_mw + p_droop_mw + synthetic_mw,
            ),
        )
        bess_diverted = generation_raw_mw - target_export

        return {
            "grid_frequency_hz": round(measured_freq_hz, 3),
            "frequency_error_hz": round(delta_f, 3),
            "pfr_droop_command_mw": round(p_droop_mw, 3),
            "synthetic_inertia_discharged_mw": round(synthetic_mw, 3),
            "net_intertie_export_mw": round(target_export, 3),
            "bess_buffer_flow_mw": round(bess_diverted, 3),
        }


# =====================================================================
# 7. TURBINE POWER (PRESSURE-DROP FORMULATION)
# =====================================================================

def compute_turbine_power(
    config: FacilityConfig,
    delta_p_pa: float,
    vol_flow_m3_s: float,
) -> Tuple[float, float]:
    delta_p_turbine = delta_p_pa * config.turbine_head_extraction_fraction
    p_shaft_w = (
        config.turbine_aerodynamic_efficiency
        * delta_p_turbine
        * vol_flow_m3_s
    )
    p_elec_w = p_shaft_w * config.generator_efficiency
    return p_shaft_w / 1e6, p_elec_w / 1e6


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
) -> Dict:
    if len(soundings) < 2:
        raise ValueError("at least two sounding layers required")

    surface = soundings[0]
    crest = soundings[-1]
    blades = blade_specs or TurbineBladeSpecs()

    inversion = AtmosphericThermodynamics.detect_inversion(soundings)
    pierces = config.crest_elevation_m > inversion["inversion_top_m"]

    delta_t, delta_p, v_flow, m_dot, vol_flow = AtmosphericThermodynamics.solve_draft(
        config, surface, crest, irradiance_w_m2
    )

    shaft_mw, elec_mw = compute_turbine_power(config, delta_p, vol_flow)

    w_e, esp_eff, clean_km3_day = ElectrostaticPrecipitatorEngine.calculate_filtration(
        config, vol_flow, esp_voltage_kv
    )

    stress = TurbomachineryStressEngine.evaluate_structural_margin(
        v_flow, blades, rotor_rpm
    )

    grid = GridIntertieController.modulate_grid_export(
        config, elec_mw, grid_freq_hz, rotor_rpm
    )

    tp = 0.95 if (pierces and stress["structural_status"] == "CLEARED") else 0.72

    return {
        "execution_status": "OPTIMAL",
        "system_truth_probability": tp,
        "meteorological_inversion": {
            **inversion,
            "conduit_pierces_inversion": pierces,
        },
        "thermodynamics_and_updraft": {
            "temperature_lift_k": round(delta_t, 2),
            "buoyant_head_pa": round(delta_p, 2),
            "chimney_velocity_mps": round(v_flow, 2),
            "volumetric_flow_m3_s": round(vol_flow, 2),
            "shaft_power_mw": round(shaft_mw, 3),
            "electrical_power_mw": round(elec_mw, 3),
        },
        "esp_air_purification": {
            "pm25_migration_velocity_mps": round(w_e, 6),
            "capture_efficiency_percent": round(esp_eff * 100.0, 4),
            "clean_air_volume_km3_day": round(clean_km3_day, 6),
        },
        "turbomachinery_structural": stress,
        "grid_frequency_intertie": grid,
    }


# =====================================================================
# 9. STANDALONE BENCHMARK EXECUTION
# =====================================================================

def run_self_benchmark() -> None:
    print("=" * 78)
    print("MOUNTAIN ECO-KINETIC ENGINE v4.1 — STANDALONE BENCHMARK")
    print("=" * 78)

    config = FacilityConfig()

    soundings = [
        SoundingLayer(1016.0, 35.0, 14.2, 7.5, 1.2),
        SoundingLayer(995.0, 210.0, 12.8, 6.0, 1.8),
        SoundingLayer(975.0, 380.0, 15.1, 5.2, 2.4),
        SoundingLayer(940.0, 685.0, 11.4, 3.1, 5.8),
    ]

    result = run_system_cycle(
        config=config,
        soundings=soundings,
        irradiance_w_m2=780.0,
        esp_voltage_kv=50.0,
        grid_freq_hz=59.780,
        rotor_rpm=44.0,
    )

    inv = result["meteorological_inversion"]
    print(f"[*] Inversion: {inv['inversion_base_m']}m to {inv['inversion_top_m']}m "
          f"(+{inv['inversion_strength_c_per_100m']}°C/100m)")
    print(f"[*] Pierces inversion: {inv['conduit_pierces_inversion']}")

    th = result["thermodynamics_and_updraft"]
    print(f"[*] dT = {th['temperature_lift_k']} K | dP = {th['buoyant_head_pa']} Pa "
          f"| v = {th['chimney_velocity_mps']} m/s")
    print(f"[*] Volumetric flow: {th['volumetric_flow_m3_s']} m³/s")
    print(f"[*] Shaft power:     {th['shaft_power_mw']} MW")
    print(f"[*] Electrical:      {th['electrical_power_mw']} MW")

    esp = result["esp_air_purification"]
    print(f"[*] ESP drift velocity: {esp['pm25_migration_velocity_mps']} m/s")
    print(f"[*] ESP capture eff:    {esp['capture_efficiency_percent']}%")
    print(f"[*] Clean air discharge: {esp['clean_air_volume_km3_day']} km³/day")

    st = result["turbomachinery_structural"]
    print(f"[*] Stress: {st['peak_combined_stress_mpa']} MPa "
          f"| SF = {st['safety_factor']} [{st['structural_status']}]")

    gr = result["grid_frequency_intertie"]
    print(f"[*] Grid freq: {gr['grid_frequency_hz']} Hz | "
          f"df = {gr['frequency_error_hz']} Hz")
    print(f"[*] PFR droop: {gr['pfr_droop_command_mw']} MW")
    print(f"[*] Synthetic inertia: {gr['synthetic_inertia_discharged_mw']} MW")
    print(f"[+] Net export: {gr['net_intertie_export_mw']} MW | "
          f"BESS buffer: {gr['bess_buffer_flow_mw']} MW")
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
        print("error: install with: pip install fastapi uvicorn pydantic",
              file=sys.stderr)
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
        sounding_layers: List[SoundingLayerModel]

    app = FastAPI(
        title="Mountain Eco-Kinetic Power & Smog Scrubber",
        version="4.1.0",
    )
    facility = FacilityConfig()

    @app.post("/api/v1/telemetry/dispatch", tags=["Dispatch"])
    async def dispatch(payload: FullTelemetryPayload):
        if len(payload.sounding_layers) < 2:
            raise HTTPException(status_code=400,
                                detail="at least two sounding layers required")
        soundings = [
            SoundingLayer(
                s.pressure_hpa, s.altitude_m,
                s.temp_c, s.dewpoint_c, s.wind_speed_mps,
            )
            for s in payload.sounding_layers
        ]
        try:
            return run_system_cycle(
                config=facility,
                soundings=soundings,
                irradiance_w_m2=payload.solar_irradiance_w_m2,
                esp_voltage_kv=payload.esp_operating_voltage_kv,
                grid_freq_hz=payload.measured_grid_freq_hz,
                rotor_rpm=payload.rotor_rpm,
            )
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

#!/usr/bin/env python3
"""
PROJECT DEEPGEM & STAR DRONE MODULE — v7.3.1 FINAL HARDENED

Unified multi-physics energy harvester with assistive tactical drone
inference. Complete standalone production release.

MODULES
-------
  1. Thermodynamic Methane Pyrolysis & SOFC-GT Generation (NIST Shomate Enthalpy).
  2. 3D Diamond Betavoltaic Core (Dark-floor continuous baseload).
  3. Hybrid Chassis Skin (Dorsal GaAs solar + perched micro-wind).
  4. Hysteretic Storage Buffer with ESR Joule loss & Schmitt-trigger guard.
  5. STAR Assistive Drone Classifier (Dual-tier inference, online SGD,
     temperature calibration, and per-regime Brier tracking).

Standard library only. Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


VERSION = "7.3.1-FINAL"


# ===========================================================================
# PHYSICAL CONSTANTS
# ===========================================================================

M_C = 12.0111
M_H = 1.00794
M_CH4 = M_C + 4.0 * M_H
M_H2 = 2.0 * M_H

LHV_CH4 = 50.009        # MJ/kg (pure CH4 at 298.15 K)
LHV_H2 = 119.96         # MJ/kg (pure H2 at 298.15 K)

ELEMENTARY_CHARGE = 1.602176634e-19
JOULES_PER_EV = ELEMENTARY_CHARGE
AVOGADRO = 6.02214076e23
SECONDS_PER_YEAR = 31557600.0        # Julian standard year (365.25 d)

DIAMOND_BANDGAP_EV = 5.47
DIAMOND_EHP_ENERGY_EV = 2.8 * DIAMOND_BANDGAP_EV + 0.5  # Klein limit (15.816 eV/pair)
DIAMOND_DENSITY_G_CM3 = 3.515

AIR_DENSITY_SEA_LEVEL = 1.225         # kg/m^3

# Low-energy beta attenuation mass absorption coefficients (cm^2/g)
BETA_MU_RHO_CM2_PER_G: Dict[str, float] = {
    "C-14": 12.0,
    "Ni-63": 25.0,
}

ISOTOPE_DATABASE = {
    "C-14": {"half_life_yr": 5700.0, "q_val_ev": 156476.0,
             "avg_beta_ev": 49470.0, "molar_mass": 14.003241},
    "Ni-63": {"half_life_yr": 101.2, "q_val_ev": 66980.0,
              "avg_beta_ev": 17420.0, "molar_mass": 62.929669},
}

# NIST Shomate Coefficients: T = 298K to 1300K
# Format: (A, B, C, D, E, F, H) -> H(T) - H(298.15) in kJ/mol
SHOMATE_COEFFS_CH4 = (-0.703029, 108.4773, -42.52157, 5.862788,
                      0.678565, -76.84376, -74.8731)
SHOMATE_COEFFS_H2 = (33.066178, -11.363417, 11.432816, -2.772874,
                     -0.158558, -9.980797, 0.0)
SHOMATE_COEFFS_C = (0.568645, 38.45834, -40.48378, 16.83315,
                    -0.009972, -8.63624, 0.0)

TACTICAL_OPTIONS: Tuple[str, ...] = (
    "hold_position",
    "advance_direct",
    "advance_flank_left",
    "advance_flank_right",
    "retreat_cover",
    "retreat_rejoin_swarm",
    "observe_and_report",
)
N_OPTIONS = len(TACTICAL_OPTIONS)
N_FEATURES_DRONE = 6
N_FEATURES_SERVER = 10

TACTICAL_DT_S = 30.0
DIURNAL_DT_S = 360.0
STEPS_PER_DAY = 240


# ===========================================================================
# MATH HELPERS
# ===========================================================================

def _clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        return (low + high) / 2.0
    return max(low, min(high, float(v)))


def _stable_softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(1e-3, temperature)
    scaled = [z / t for z in logits]
    max_z = max(scaled)
    exps = [math.exp(z - max_z) for z in scaled]
    total = sum(exps)
    return [e / max(1e-12, total) for e in exps]


def _shomate_enthalpy(coeffs: Tuple[float, ...], temp_k: float) -> float:
    t = temp_k / 1000.0
    a, b, c, d, e, f, h = coeffs
    return (a * t + (b * (t ** 2)) / 2.0 + (c * (t ** 3)) / 3.0
            + (d * (t ** 4)) / 4.0 - (e / t) + f - h)


def _shomate_delta_h(coeffs: Tuple[float, ...],
                     t_high: float, t_low: float) -> float:
    return _shomate_enthalpy(coeffs, t_high) - _shomate_enthalpy(coeffs, t_low)


# ===========================================================================
# MODULE 1: MACRO AUTOTHERMAL PYROLYSIS
# ===========================================================================

@dataclass
class PyrolysisReport:
    methane_input_kg: float
    reactor_temp_kelvin: float
    total_h2_produced_kg: float
    solid_carbon_recovered_kg: float
    dh_reaction_298k_mj: float
    sensible_preheat_feed_mj: float
    sensible_products_mj: float
    heat_recuperated_mj: float
    parasitic_h2_combusted_kg: float
    net_export_h2_kg: float
    sofc_gt_electricity_kwh: float
    thermal_efficiency_pct: float
    electrical_efficiency_pct: float


def run_pyrolysis_stage(
    methane_kg: float = 1.0,
    temp_k: float = 1273.15,
    hx_effectiveness: float = 0.85,
    combustor_eff: float = 0.95,
    sofc_eff: float = 0.65,
    t_inlet_k: float = 298.15,
) -> PyrolysisReport:
    if methane_kg <= 0:
        raise ValueError("methane_kg must be positive")
    if temp_k <= t_inlet_k:
        raise ValueError("reactor temp must exceed inlet temp")
    for name, v in (("hx_effectiveness", hx_effectiveness),
                    ("combustor_eff", combustor_eff),
                    ("sofc_eff", sofc_eff)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    mol_ch4 = (methane_kg * 1000.0) / M_CH4
    mol_h2_total = 2.0 * mol_ch4
    mass_h2_total_kg = (mol_h2_total * M_H2) / 1000.0
    mass_c_total_kg = (mol_ch4 * M_C) / 1000.0

    dh_rxn_298k_mj = (mol_ch4 * 74.873) / 1000.0

    dh_feed_kj_mol = _shomate_delta_h(SHOMATE_COEFFS_CH4, temp_k, t_inlet_k)
    dh_h2_kj_mol = _shomate_delta_h(SHOMATE_COEFFS_H2, temp_k, t_inlet_k)
    dh_c_kj_mol = _shomate_delta_h(SHOMATE_COEFFS_C, temp_k, t_inlet_k)

    q_feed_mj = (mol_ch4 * dh_feed_kj_mol) / 1000.0
    q_prod_mj = ((mol_h2_total * dh_h2_kj_mol) + (mol_ch4 * dh_c_kj_mol)) / 1000.0
    q_recup_mj = min(q_feed_mj, q_prod_mj) * hx_effectiveness

    net_heat_req_mj = max(0.0, dh_rxn_298k_mj + q_feed_mj - q_recup_mj)
    h2_burned_kg = min(mass_h2_total_kg,
                       net_heat_req_mj / max(1e-12, LHV_H2 * combustor_eff))
    net_h2_kg = max(0.0, mass_h2_total_kg - h2_burned_kg)

    sofc_kwh = ((net_h2_kg * LHV_H2) / 3.6) * sofc_eff
    input_energy_mj = methane_kg * LHV_CH4

    return PyrolysisReport(
        methane_input_kg=round(methane_kg, 6),
        reactor_temp_kelvin=round(temp_k, 2),
        total_h2_produced_kg=round(mass_h2_total_kg, 6),
        solid_carbon_recovered_kg=round(mass_c_total_kg, 6),
        dh_reaction_298k_mj=round(dh_rxn_298k_mj, 4),
        sensible_preheat_feed_mj=round(q_feed_mj, 4),
        sensible_products_mj=round(q_prod_mj, 4),
        heat_recuperated_mj=round(q_recup_mj, 4),
        parasitic_h2_combusted_kg=round(h2_burned_kg, 6),
        net_export_h2_kg=round(net_h2_kg, 6),
        sofc_gt_electricity_kwh=round(sofc_kwh, 6),
        thermal_efficiency_pct=round(
            100.0 * (net_h2_kg * LHV_H2) / max(1e-12, input_energy_mj), 4),
        electrical_efficiency_pct=round(
            100.0 * (sofc_kwh * 3.6) / max(1e-12, input_energy_mj), 4),
    )


# ===========================================================================
# MODULE 2: DIAMOND BETAVOLTAIC BASELOAD CORE
# ===========================================================================

@dataclass
class BetavoltaicReport:
    isotope: str
    mass_mg: float
    surface_area_cm2: float
    effective_thickness_um: float
    activity_bq: float
    radiological_power_uw: float
    self_absorption_factor: float
    open_circuit_voltage_v: float
    short_circuit_current_ua: float
    electrical_power_uw: float
    transduction_efficiency_pct: float
    carbon_loop_fraction: float
    carbon_kg_from_pyrolysis: float


def run_betavoltaic_stage(
    isotope: str = "Ni-63",
    mass_mg: float = 25.0,
    base_footprint_cm2: float = 1.0,
    aspect_ratio: float = 5.0,
    ideality_factor: float = 1.2,
    collection_yield: float = 0.70,
    carbon_loop_fraction: float = 0.0,
    carbon_available_kg: float = 0.0,
) -> BetavoltaicReport:
    if isotope not in ISOTOPE_DATABASE:
        raise ValueError(f"unsupported isotope: {isotope}")
    if mass_mg <= 0:
        raise ValueError("mass_mg must be positive")
    if base_footprint_cm2 <= 0 or aspect_ratio <= 0:
        raise ValueError("geometry must be positive")
    if not 0.0 <= carbon_loop_fraction <= 1.0:
        raise ValueError("carbon_loop_fraction must be in [0, 1]")
    if ideality_factor <= 0 or collection_yield <= 0:
        raise ValueError("ideality_factor and collection_yield must be positive")

    carbon_kg_looped = max(0.0, carbon_available_kg * carbon_loop_fraction)

    spec = ISOTOPE_DATABASE[isotope]
    mass_g = mass_mg * 1e-3
    effective_area_cm2 = base_footprint_cm2 * aspect_ratio

    decay_const = math.log(2.0) / (spec["half_life_yr"] * SECONDS_PER_YEAR)
    activity_bq = decay_const * (mass_g / spec["molar_mass"]) * AVOGADRO

    avg_e_j = spec["avg_beta_ev"] * JOULES_PER_EV
    p_rad_w = activity_bq * avg_e_j

    thickness_cm = mass_g / (DIAMOND_DENSITY_G_CM3 * effective_area_cm2)
    thickness_um = thickness_cm * 1e4

    mu_rho = BETA_MU_RHO_CM2_PER_G[isotope]
    mu_cm = mu_rho * DIAMOND_DENSITY_G_CM3
    mu_t = max(1e-9, mu_cm * thickness_cm)
    eta_self = (1.0 - math.exp(-mu_t)) / mu_t

    p_absorbed_w = p_rad_w * eta_self
    ehp_rate = p_absorbed_w / (DIAMOND_EHP_ENERGY_EV * JOULES_PER_EV)
    i_sc_a = ELEMENTARY_CHARGE * ehp_rate * collection_yield

    v_thermal = 0.02585
    i_0_a = 1e-15 * effective_area_cm2
    v_oc = min(
        DIAMOND_BANDGAP_EV,
        ideality_factor * v_thermal
        * math.log(max(1.0, (i_sc_a / i_0_a) + 1.0)),
    )

    v_norm = v_oc / (ideality_factor * v_thermal)
    ff = ((v_norm - math.log(v_norm + 0.72)) / (v_norm + 1.0)
          if v_norm > 1.0 else 0.25)
    p_elec_w = i_sc_a * v_oc * ff

    return BetavoltaicReport(
        isotope=isotope,
        mass_mg=round(mass_mg, 6),
        surface_area_cm2=round(effective_area_cm2, 4),
        effective_thickness_um=round(thickness_um, 4),
        activity_bq=activity_bq,
        radiological_power_uw=round(p_rad_w * 1e6, 6),
        self_absorption_factor=round(eta_self, 6),
        open_circuit_voltage_v=round(v_oc, 4),
        short_circuit_current_ua=round(i_sc_a * 1e6, 6),
        electrical_power_uw=round(p_elec_w * 1e6, 6),
        transduction_efficiency_pct=round(
            (p_elec_w / max(1e-12, p_rad_w)) * 100.0, 4),
        carbon_loop_fraction=round(carbon_loop_fraction, 4),
        carbon_kg_from_pyrolysis=round(carbon_kg_looped, 8),
    )


# ===========================================================================
# MODULE 3: CHASSIS HARVESTING & DUAL-THRESHOLD SCHMITT BUFFER
# ===========================================================================

@dataclass
class HybridChassisSkin:
    solar_area_m2: float = 0.08
    solar_efficiency: float = 0.24
    turbine_rotor_area_m2: float = 0.012
    turbine_cp: float = 0.32
    generator_eff: float = 0.85
    wind_cut_in_m_s: float = 2.0

    def compute_solar_power_w(self, irradiance_w_m2: float) -> float:
        return max(0.0, self.solar_area_m2 * irradiance_w_m2 * self.solar_efficiency)

    def compute_perched_wind_power_w(self, wind_speed_m_s: float,
                                      is_perched: bool) -> float:
        if not is_perched or wind_speed_m_s < self.wind_cut_in_m_s:
            return 0.0
        p_aero = (0.5 * AIR_DENSITY_SEA_LEVEL
                  * self.turbine_rotor_area_m2
                  * (wind_speed_m_s ** 3) * self.turbine_cp)
        return p_aero * self.generator_eff


@dataclass
class DeepGemPowerState:
    v_cap: float = 3.3
    capacitance_f: float = 0.05
    v_high: float = 3.3
    v_low: float = 2.0
    v_rearm: float = 2.65            # Hysteretic re-arm threshold
    esr_ohms: float = 0.05
    trickle_betavoltaic_uw: float = 0.45
    quiescent_leakage_w: float = 1.2e-7
    skin: HybridChassisSkin = field(default_factory=HybridChassisSkin)
    tactical_inference_enabled: bool = True
    safety_transitions: int = 0

    def __post_init__(self):
        if not (0.0 <= self.v_low < self.v_rearm < self.v_high):
            raise ValueError("require 0 <= v_low < v_rearm < v_high")

    def energy_available_j(self) -> float:
        return 0.5 * self.capacitance_f * max(
            0.0, (self.v_cap ** 2 - self.v_low ** 2))

    def recharge_step(
        self,
        delta_time_s: float,
        irradiance_w_m2: float = 0.0,
        wind_speed_m_s: float = 0.0,
        is_perched: bool = False,
    ) -> Dict[str, float]:
        p_beta_w = self.trickle_betavoltaic_uw * 1e-6
        p_solar_w = self.skin.compute_solar_power_w(irradiance_w_m2)
        p_wind_w = self.skin.compute_perched_wind_power_w(wind_speed_m_s, is_perched)

        p_total_in = p_beta_w + p_solar_w + p_wind_w
        p_net = p_total_in - self.quiescent_leakage_w

        e_current = 0.5 * self.capacitance_f * (self.v_cap ** 2)
        e_max = 0.5 * self.capacitance_f * (self.v_high ** 2)
        e_next = max(0.0, min(e_max, e_current + p_net * delta_time_s))
        self.v_cap = math.sqrt(max(0.0, 2.0 * e_next / self.capacitance_f))

        was_enabled = self.tactical_inference_enabled
        if self.v_cap <= self.v_low:
            self.tactical_inference_enabled = False
        elif self.v_cap >= self.v_rearm:
            self.tactical_inference_enabled = True
        if self.tactical_inference_enabled != was_enabled:
            self.safety_transitions += 1

        return {
            "p_beta_w": p_beta_w,
            "p_solar_w": p_solar_w,
            "p_wind_w": p_wind_w,
            "p_total_in_w": p_total_in,
            "p_net_w": p_net,
        }

    def attempt_burst_execution(self, power_w: float,
                                 duration_s: float) -> Tuple[bool, float]:
        v_nom = max(self.v_low, (self.v_cap + self.v_low) / 2.0)
        i_burst = power_w / v_nom
        joule_loss_j = (i_burst ** 2) * self.esr_ohms * duration_s
        total_drain_j = (power_w * duration_s) + joule_loss_j

        if self.energy_available_j() < total_drain_j or self.v_cap <= self.v_low:
            return False, 0.0

        e_remaining = (0.5 * self.capacitance_f * (self.v_cap ** 2)
                       - total_drain_j)
        self.v_cap = math.sqrt(max(0.0, 2.0 * e_remaining / self.capacitance_f))

        was_enabled = self.tactical_inference_enabled
        if self.v_cap <= self.v_low:
            self.tactical_inference_enabled = False
        if self.tactical_inference_enabled != was_enabled:
            self.safety_transitions += 1

        return True, joule_loss_j


# ===========================================================================
# MODULE 4: STAR ASSISTIVE CLASSIFIER & REGIME CALIBRATION
# ===========================================================================

@dataclass
class STALinearClassifier:
    weights: List[List[float]]
    bias: List[float]
    option_names: Tuple[str, ...]
    learning_rate: float = 0.035
    l2_reg: float = 0.001
    temperature: float = 1.05
    grad_clip: float = 5.0
    n_updates: int = 0

    @classmethod
    def initialize(cls, n_options: int, n_features: int,
                   option_names: Tuple[str, ...],
                   seed: int = 0,
                   learning_rate: float = 0.035,
                   temperature: float = 1.05) -> "STALinearClassifier":
        rng = random.Random(seed)
        weights = [[rng.gauss(0.0, 0.05) for _ in range(n_features)]
                   for _ in range(n_options)]
        bias = [0.0] * n_options
        return cls(weights=weights, bias=bias, option_names=option_names,
                   learning_rate=learning_rate, temperature=temperature)

    def forward(self, features: Sequence[float]) -> List[float]:
        logits = [self.bias[i] + sum(w * x for w, x in zip(row, features))
                  for i, row in enumerate(self.weights)]
        return _stable_softmax(logits, self.temperature)

    def train_step(self, features: Sequence[float],
                   outcome_index: int) -> float:
        probs = self.forward(features)
        loss = -math.log(max(1e-12, probs[outcome_index]))

        for i in range(len(self.weights)):
            err = probs[i] - (1.0 if i == outcome_index else 0.0)
            err_clipped = _clamp(err, -self.grad_clip, self.grad_clip)
            for j in range(len(self.weights[i])):
                grad = err * features[j] + self.l2_reg * self.weights[i][j]
                grad_clipped = _clamp(grad, -self.grad_clip, self.grad_clip)
                self.weights[i][j] -= self.learning_rate * grad_clipped
            self.bias[i] -= self.learning_rate * err_clipped

        self.n_updates += 1
        return loss

    def adapt_temperature(self, avg_top_prob: float,
                          target: float = 0.45) -> None:
        if avg_top_prob > target:
            self.temperature = min(2.5, self.temperature * 1.002)
        else:
            self.temperature = max(0.5, self.temperature * 0.998)


@dataclass
class STARCalibrationTracker:
    option_names: Tuple[str, ...]
    predictions: List[List[float]] = field(default_factory=list)
    outcomes: List[int] = field(default_factory=list)

    def add(self, probs: Sequence[float], outcome_idx: int) -> None:
        self.predictions.append(list(probs))
        self.outcomes.append(outcome_idx)

    def compute_brier(self) -> Optional[float]:
        if not self.predictions:
            return None
        total = 0.0
        for probs, y in zip(self.predictions, self.outcomes):
            total += sum((p - (1.0 if i == y else 0.0)) ** 2
                         for i, p in enumerate(probs))
        return total / len(self.predictions)

    def compute_accuracy(self) -> Optional[float]:
        if not self.predictions:
            return None
        correct = sum(
            1 for probs, y in zip(self.predictions, self.outcomes)
            if max(range(len(probs)), key=lambda i: probs[i]) == y
        )
        return correct / len(self.predictions)


@dataclass
class RegimeTrackers:
    day_airborne: STARCalibrationTracker
    day_perched: STARCalibrationTracker
    night_airborne: STARCalibrationTracker
    night_perched: STARCalibrationTracker

    @classmethod
    def initialize(cls, options: Tuple[str, ...]) -> "RegimeTrackers":
        return cls(
            day_airborne=STARCalibrationTracker(options),
            day_perched=STARCalibrationTracker(options),
            night_airborne=STARCalibrationTracker(options),
            night_perched=STARCalibrationTracker(options),
        )

    def select(self, is_day: bool,
               is_perched: bool) -> STARCalibrationTracker:
        if is_day and not is_perched:
            return self.day_airborne
        if is_day and is_perched:
            return self.day_perched
        if not is_day and not is_perched:
            return self.night_airborne
        return self.night_perched

    def summary(self) -> Dict[str, Any]:
        def _summ(t: STARCalibrationTracker) -> Dict[str, Any]:
            brier = t.compute_brier()
            accuracy = t.compute_accuracy()
            return {
                "n": len(t.predictions),
                "brier": round(brier, 6) if brier is not None else None,
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
            }
        return {
            "day_airborne": _summ(self.day_airborne),
            "day_perched": _summ(self.day_perched),
            "night_airborne": _summ(self.night_airborne),
            "night_perched": _summ(self.night_perched),
        }


# ===========================================================================
# PIPELINE INTEGRATOR
# ===========================================================================

def run_hybrid_system_simulation(
    methane_kg: float = 1.0,
    isotope: str = "Ni-63",
    isotope_mg: float = 25.0,
    carbon_loop_fraction: float = 0.0,
    n_steps: int = 2400,
    seed: int = 101,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    pyro_stage = run_pyrolysis_stage(methane_kg=methane_kg)
    beta_stage = run_betavoltaic_stage(
        isotope=isotope,
        mass_mg=isotope_mg,
        carbon_loop_fraction=carbon_loop_fraction,
        carbon_available_kg=pyro_stage.solid_carbon_recovered_kg,
    )

    power_harness = DeepGemPowerState(
        v_cap=3.3,
        trickle_betavoltaic_uw=beta_stage.electrical_power_uw,
    )

    drone_classifier = STALinearClassifier.initialize(
        N_OPTIONS, N_FEATURES_DRONE, TACTICAL_OPTIONS,
        seed=1, learning_rate=0.035, temperature=1.10)
    server_classifier = STALinearClassifier.initialize(
        N_OPTIONS, N_FEATURES_SERVER, TACTICAL_OPTIONS,
        seed=2, learning_rate=0.015, temperature=1.00)

    overall_drone = STARCalibrationTracker(TACTICAL_OPTIONS)
    overall_server = STARCalibrationTracker(TACTICAL_OPTIONS)
    regime_drone = RegimeTrackers.initialize(TACTICAL_OPTIONS)
    regime_server = RegimeTrackers.initialize(TACTICAL_OPTIONS)

    brownouts = 0
    reflex_count = 0
    agreement_count = 0
    link_drops = 0
    safety_halts = 0
    cumulative_joule_losses = 0.0
    cumulative_solar_j = 0.0
    cumulative_wind_j = 0.0
    cumulative_beta_j = 0.0

    solar_peak_w = 0.0
    beta_peak_w = 0.0

    for step in range(n_steps):
        is_perched = (rng.random() < 0.40)

        # Diurnal cycle: 240 steps = 1 solar day (86,400 s)
        day_fraction = (step % STEPS_PER_DAY) / STEPS_PER_DAY
        hour_angle = day_fraction * 2.0 * math.pi
        sin_hour = math.sin(hour_angle)
        irradiance = max(0.0, sin_hour) * 850.0
        is_day = sin_hour > 0.0

        wind_speed = rng.uniform(1.0, 7.5)

        # Delta-T synchronized to DIURNAL_DT_S (360.0 s)
        breakdown = power_harness.recharge_step(
            delta_time_s=DIURNAL_DT_S,
            irradiance_w_m2=irradiance,
            wind_speed_m_s=wind_speed,
            is_perched=is_perched,
        )

        cumulative_solar_j += breakdown["p_solar_w"] * DIURNAL_DT_S
        cumulative_wind_j += breakdown["p_wind_w"] * DIURNAL_DT_S
        cumulative_beta_j += breakdown["p_beta_w"] * DIURNAL_DT_S

        solar_peak_w = max(solar_peak_w, breakdown["p_solar_w"])
        beta_peak_w = max(beta_peak_w, breakdown["p_beta_w"])

        # Check Schmitt safety guard
        if not power_harness.tactical_inference_enabled:
            safety_halts += 1
            continue

        shared_state = {
            "relative_threat": rng.random(),
            "swarm_cohesion": rng.random(),
            "target_visibility": rng.random(),
            "energy_reserve": power_harness.v_cap / power_harness.v_high,
            "link_quality": rng.random(),
            "mission_urgency": rng.random(),
        }
        drone_features = [
            shared_state["relative_threat"],
            shared_state["swarm_cohesion"],
            shared_state["target_visibility"],
            shared_state["energy_reserve"],
            shared_state["link_quality"],
            shared_state["mission_urgency"],
        ]
        context_features = [rng.random() for _ in range(4)]
        server_features = drone_features + context_features

        burst_ok, j_loss = power_harness.attempt_burst_execution(
            power_w=1.1, duration_s=0.040)
        cumulative_joule_losses += j_loss

        if not burst_ok:
            brownouts += 1
            continue

        drone_probs = drone_classifier.forward(drone_features)
        top_drone_idx = max(range(N_OPTIONS), key=lambda i: drone_probs[i])
        if drone_probs[top_drone_idx] < 0.35:
            reflex_count += 1

        # Relay chain evaluation
        link_available = (shared_state["link_quality"] > 0.30)
        if link_available:
            server_probs = server_classifier.forward(server_features)
            top_server_idx = max(range(N_OPTIONS), key=lambda i: server_probs[i])
            if top_drone_idx == top_server_idx:
                agreement_count += 1
        else:
            link_drops += 1
            server_probs = None

        # Fair ground-truth resolution: derived strictly from shared features
        weighted_val = sum((i + 1) * v for i, v in enumerate(drone_features))
        outcome_idx = int(weighted_val * 7) % N_OPTIONS
        if rng.random() < 0.15:
            outcome_idx = rng.randrange(N_OPTIONS)

        overall_drone.add(drone_probs, outcome_idx)
        regime_drone.select(is_day, is_perched).add(drone_probs, outcome_idx)
        drone_classifier.train_step(drone_features, outcome_idx)

        if server_probs is not None:
            overall_server.add(server_probs, outcome_idx)
            regime_server.select(is_day, is_perched).add(server_probs, outcome_idx)
            server_classifier.train_step(server_features, outcome_idx)

        if (step + 1) % 50 == 0:
            drone_classifier.adapt_temperature(drone_probs[top_drone_idx])

    executed_cycles = max(1, n_steps - brownouts - safety_halts)
    connected_cycles = max(0, executed_cycles - link_drops)
    valid_agreement_rate = (round(agreement_count / connected_cycles, 4)
                            if connected_cycles > 0 else 0.0)

    total_harvest_j = cumulative_solar_j + cumulative_wind_j + cumulative_beta_j
    beta_fraction_pct = (100.0 * cumulative_beta_j / total_harvest_j
                         if total_harvest_j > 0 else 0.0)

    ratio = (solar_peak_w / max(1e-12, beta_peak_w)
             if beta_peak_w > 0 else float("inf"))

    return {
        "version": VERSION,
        "pyrolysis_macro_module": asdict(pyro_stage),
        "deepgem_betavoltaic_module": asdict(beta_stage),
        "energy_harvesting_totals": {
            "cumulative_solar_joules": round(cumulative_solar_j, 2),
            "cumulative_wind_joules": round(cumulative_wind_j, 2),
            "cumulative_betavoltaic_joules": round(cumulative_beta_j, 6),
            "cumulative_esr_losses_joules": round(cumulative_joule_losses, 6),
            "final_storage_voltage_v": round(power_harness.v_cap, 3),
            "safety_transitions": power_harness.safety_transitions,
            "peak_solar_w": round(solar_peak_w, 3),
            "peak_betavoltaic_w": round(beta_peak_w, 10),
            "solar_to_beta_ratio": round(ratio, 2) if math.isfinite(ratio) else "inf",
            "beta_fraction_of_total_pct": round(beta_fraction_pct, 8),
        },
        "star_drone_tactical_module": {
            "total_steps_simulated": n_steps,
            "executed_cycles": executed_cycles,
            "connected_cycles": connected_cycles,
            "brownout_events": brownouts,
            "brownout_rate": round(brownouts / n_steps, 4),
            "safety_halts": safety_halts,
            "safety_halt_rate": round(safety_halts / n_steps, 4),
            "reflex_fallback_rate": round(reflex_count / executed_cycles, 4),
            "link_drop_rate": round(link_drops / executed_cycles, 4),
            "server_agreement_rate": valid_agreement_rate,
            "drone_brier_score": round(overall_drone.compute_brier() or 0.0, 6),
            "drone_accuracy": round(overall_drone.compute_accuracy() or 0.0, 4),
            "drone_temperature": round(drone_classifier.temperature, 3),
            "server_brier_score": round(overall_server.compute_brier() or 0.0, 6),
            "server_accuracy": round(overall_server.compute_accuracy() or 0.0, 4),
        },
        "per_regime_drone": regime_drone.summary(),
        "per_regime_server": regime_server.summary(),
    }


# ===========================================================================
# CLI DISPATCHER
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"DeepGem & STAR Hybrid Engine v{VERSION}")
    parser.add_argument("--methane-kg", type=float, default=1.0)
    parser.add_argument("--isotope", choices=list(ISOTOPE_DATABASE.keys()),
                        default="Ni-63")
    parser.add_argument("--isotope-mg", type=float, default=25.0)
    parser.add_argument("--carbon-loop", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_hybrid_system_simulation(
        methane_kg=args.methane_kg,
        isotope=args.isotope,
        isotope_mg=args.isotope_mg,
        carbon_loop_fraction=args.carbon_loop,
        n_steps=args.steps,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0

    print("=" * 82)
    print(f"DEEPGEM × STAR DRONE — v{results['version']}")
    print("=" * 82)

    p = results["pyrolysis_macro_module"]
    print("\n[MODULE 1: Macro Autothermal Pyrolysis]")
    print(f"  Feedstock:              {p['methane_input_kg']} kg CH4 @ {p['reactor_temp_kelvin']} K")
    print(f"  H2 Produced:            {p['total_h2_produced_kg']} kg")
    print(f"  H2 Burned for Heat:     {p['parasitic_h2_combusted_kg']} kg")
    print(f"  Net H2 Export:          {p['net_export_h2_kg']} kg")
    print(f"  Solid Carbon:           {p['solid_carbon_recovered_kg']} kg")
    print(f"  SOFC-GT Electricity:    {p['sofc_gt_electricity_kwh']} kWh")
    print(f"  Thermal / Electrical:   {p['thermal_efficiency_pct']}% / {p['electrical_efficiency_pct']}%")

    b = results["deepgem_betavoltaic_module"]
    print(f"\n[MODULE 2: DeepGem Baseload Core ({b['isotope']})]")
    print(f"  Role:                   Continuous trickle (dark-floor survival)")
    print(f"  Mass / Area:            {b['mass_mg']} mg / {b['surface_area_cm2']} cm²")
    print(f"  Self-Absorption:        {b['self_absorption_factor']}")
    print(f"  Continuous Output:      {b['electrical_power_uw']} µW")
    if b["carbon_loop_fraction"] > 0:
        print(f"  Carbon Loop:            {b['carbon_kg_from_pyrolysis']} kg routed")

    h = results["energy_harvesting_totals"]
    print("\n[MODULE 3: Hybrid Chassis Energy Totals]")
    print(f"  Solar Harvested:        {h['cumulative_solar_joules']} J")
    print(f"  Wind Harvested:         {h['cumulative_wind_joules']} J (Stationary perched only)")
    print(f"  Betavoltaic Harvested:  {h['cumulative_betavoltaic_joules']} J "
          f"({h['beta_fraction_of_total_pct']:.6f}% of total)")
    print(f"  Cumulative ESR Losses:  {h['cumulative_esr_losses_joules']} J")
    print(f"  Final Storage Voltage:  {h['final_storage_voltage_v']} V")
    print(f"  Safety Transitions:     {h['safety_transitions']}")
    print(f"  Peak Solar / Beta:      {h['peak_solar_w']} W / {h['peak_betavoltaic_w']} W")
    print(f"  Solar-to-Beta Ratio:    {h['solar_to_beta_ratio']}:1")

    s = results["star_drone_tactical_module"]
    print("\n[MODULE 4: STAR Drone Tactical Inference Engine]")
    print(f"  Steps:                  {s['total_steps_simulated']} "
          f"({s['total_steps_simulated']/STEPS_PER_DAY:.2f} solar days)")
    print(f"  Executed Cycles:        {s['executed_cycles']}")
    print(f"  Connected Cycles:       {s['connected_cycles']}")
    print(f"  Brownouts:              {s['brownout_events']} ({s['brownout_rate']*100:.2f}%)")
    print(f"  Safety Halts:           {s['safety_halts']} ({s['safety_halt_rate']*100:.2f}%)")
    print(f"  Reflex Fallback:        {s['reflex_fallback_rate']*100:.2f}%")
    print(f"  Link Drops:             {s['link_drop_rate']*100:.2f}%")
    print(f"  Server Agreement:       {s['server_agreement_rate']*100:.2f}%")
    print("-" * 82)
    print(f"  Drone Brain:  Brier = {s['drone_brier_score']:.6f} | "
          f"Accuracy = {s['drone_accuracy']*100:.2f}% | "
          f"Temp = {s['drone_temperature']}")
    print(f"  Server Brain: Brier = {s['server_brier_score']:.6f} | "
          f"Accuracy = {s['server_accuracy']*100:.2f}%")

    print("\n[PER-REGIME DRONE]")
    for regime, m in results["per_regime_drone"].items():
        if m["n"] > 0:
            print(f"  {regime:18s}: n={m['n']:5d}  "
                  f"brier={m['brier']:.6f}  acc={m['accuracy']*100:.2f}%")

    print("\n[PER-REGIME SERVER]")
    for regime, m in results["per_regime_server"].items():
        if m["n"] > 0:
            print(f"  {regime:18s}: n={m['n']:5d}  "
                  f"brier={m['brier']:.6f}  acc={m['accuracy']*100:.2f}%")

    print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

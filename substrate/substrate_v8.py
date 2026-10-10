#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""
Substrate v8 — Mammalian Autarky + Demurrage + Municipal Ledger (fixed)
=======================================================================

Credit: humanity. Released to the public domain under CC0-1.0.

A single-file, standard-library Python reference implementation of a
proposed political economy. v8 is a corrected version of v7: see
README_v8.md for every change and the sources behind the pod model.

Layers:
  1. PodModel / HouseholdPod   parameterized cost, energy, water, calories
  2. PersonhoodRegistry        deduplicated vouches, genesis quorum, peppered
                               biometric hashes, exit and erasure
  3. LedgerClock + NeighborGraph + MeshConsensus
                               system-owned clock, mutual registered
                               neighbors, rolling-window and collusion caps
  4. AGIDataEngine             provenance-quality minting, per-person and
                               global per-period mint caps
  5. Demurrage                 threshold demurrage for every pod
                               (plus idle decay) -> bounded balances
  6. MunicipalZone / Ledger    ecological-fragility-scaled tolls
  7. UNRecognition + ladder    PROPOSED/FICTIONAL mechanism (not an existing
                               UN process); one severity source; reviewed
                               appeals
  8. TransparencyLog + Watcher pseudonymous IDs, aggregated activity,
                               crypto-shredding erasure
  9. Polity + Treasury         exit with settlement; quorum of distinct
                               registered members
 10. AuditorRegistry           HMAC-signed certificates, random assignment,
                               minimum-sample slashing rule
 11. SponsorPool / Sponsors    multi-sponsor attribution
 12. Federation                treaties; no personhood overwrite; fees go to
                               a federation treasury

Invariants (what the code now actually enforces):
  - Exit is implemented (Polity.exit); erasure via pseudonym shredding
  - Personhood is unique within a registry (biometric matching is a
    labeled simplification: real biometrics are fuzzy)
  - Balances are bounded for any bounded inflow (threshold demurrage)
  - Minting is capped per person and globally per period
  - The event log is public but pseudonymous by default

Run:
    python substrate_v8.py --test
    python substrate_v8.py --demo
    python substrate_v8.py --municipal-demo
    python substrate_v8.py --pod-report
"""

from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import math
import random
import secrets
import sys
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ===========================================================================
# 1a. PodModel — honest, parameterized cost / energy / water / calorie model
# ===========================================================================
# Every default below is either SOURCED (URL in SOURCES) or an ASSUMPTION
# (listed in ASSUMPTIONS). Nothing here is a measured result of a real pod.
SOURCES = {
    "pvgis_seoul": "PVGIS v5.2 API, 37.5N 127.0E, 1 kWp, 14% loss, horizontal "
                   "(default), ERA5 2005-2020: 3.27 kWh/kWp/day annual mean, "
                   "1.90 in Dec. https://re.jrc.ec.europa.eu/api/v5_2/PVcalc",
    "iea_pvps_kr": "IEA-PVPS National Survey Report Korea 2022: avg yield "
                   "1,261 kWh/kWp/yr; residential 5-10 kW rooftop "
                   "1,459.5 KRW/W excl. VAT. https://iea-pvps.org/wp-content/"
                   "uploads/2024/01/IEA-PVPS-National-Survey-Report-KOREA-2022.pdf",
    "irena_2023": "IRENA Renewable Power Generation Costs in 2023: battery "
                  "storage projects USD 273/kWh (2023, utility scale). "
                  "https://www.irena.org/Publications/2024/Sep/"
                  "Renewable-Power-Generation-Costs-in-2023",
    "nrel_atb_batt": "NREL ATB 2023 residential battery: pack $283/kWh, "
                     "bidirectional inverter $183/kWh (5 kW/12.5 kWh). "
                     "https://atb.nrel.gov/electricity/2023/index/"
                     "residential_battery_storage",
    "graamans_2018": "Graamans et al. 2018, Agric. Systems 160:31-43: plant "
                     "factory lettuce 247 kWh_e per kg dry weight (~17.3 "
                     "kWh/kg fresh weight at 7% DM; ~91% lighting). "
                     "https://doi.org/10.1016/j.agsy.2017.11.003",
    "nsf_pf_lettuce": "Plant-factory building energy model: 6.2-12.0 kWh per "
                      "kg fresh lettuce depending on design. "
                      "https://par.nsf.gov/servlets/purl/10409935",
    "kusuma_2020": "Kusuma, Pattison & Bugbee 2020, Hortic. Res. 7:56: fixtures "
                   "2.5-3.0 umol/J achieved; limits 3.4 (white+red) and 4.1 "
                   "(blue+red) umol/J. https://doi.org/10.1038/s41438-020-0283-7",
    "bugbee_1988": "Bugbee & Salisbury 1988, Plant Physiol. 88:869: wheat grain "
                   "60 g/m2/day at 150 mol/m2/day PPF (1,200 ppm CO2); "
                   "biomass PPF-use efficiency 1.32 -> 0.63 g/mol as light "
                   "rises. https://ntrs.nasa.gov/citations/20040112090",
    "kostat_rice_2024": "Statistics Korea, 2024 rice production: 514 kg milled "
                        "rice per 10a (5.14 t/ha). https://mods.go.kr/"
                        "boardDownload.es?bid=11712&list_no=434049&seq=4",
    "fao_potato_kr": "FAOSTAT via Our World in Data: South Korea potato yield "
                     "24.67 t/ha (2024). https://ourworldindata.org/grapher/"
                     "potato-yields",
    "usda_fdc": "USDA FoodData Central SR Legacy (read via a mirror): potato raw "
                "77 kcal/100 g (FDC 170026); green leaf lettuce raw 15 kcal/100 g "
                "(FDC 169249); white rice raw 365 kcal/100 g (FDC 169756).",
    "me_water_2023": "Ministry of Environment, 2023 waterworks statistics: "
                     "303.9 L/person/day; household use 3,665 of 5,862 million "
                     "m3 (62.5%) -> ~190 L/person/day household (derived). "
                     "https://www.waterjournal.co.kr/news/articleView.html?"
                     "idxno=79553",
}
ASSUMPTIONS = [
    "KRW/USD exchange rate 1,350 (assumption, not a quoted rate).",
    "Residential PV installed cost range USD 900-1,400/kW around the IEA-PVPS "
    "2022 Korean average (~USD 1,080/kW) — range width is an assumption.",
    "Residential battery installed cost USD 500-1,000/kWh (central 700): "
    "assumption bracketed by IRENA utility USD 273/kWh and NREL ATB pack + "
    "inverter USD 466/kWh before installation.",
    "Indoor grow system capex USD 300-1,500 per m2 canopy (central 800): "
    "assumption, no source found.",
    "Water tank + filtration + greywater system USD 1,500-5,000 per household "
    "(central 3,000): assumption.",
    "Lifetimes: PV 25 y, battery 12 y, grow system 10 y, water 15 y; O&M 1.5% "
    "of capex/yr; grow consumables USD 20/m2/yr: assumptions.",
    "Leafy-greens photon-to-calorie yield 1.06 kcal/mol: DERIVED from "
    "Graamans 2018 (17.3 kWh/kg FW, 91% lighting) with an ASSUMED 2.5 umol/J "
    "fixture efficacy and USDA 150 kcal/kg lettuce.",
    "Staple (wheat-like) 1.3 kcal/mol: DERIVED from Bugbee 1988 (0.40 g grain "
    "per mol) with an ASSUMED 330 kcal/100 g grain; 'optimistic' 2.6 kcal/mol "
    "assumes ~2x at moderate light (from the 1.32 vs 0.63 g/mol biomass LUE).",
    "Physics ceiling 14.3 kcal/mol photons: 686 kcal per mol glucose / 6 CO2 "
    "/ 8 photons minimum quantum requirement, before any respiration.",
    "HVAC/pumps overhead 1.2x lighting energy (assumption; Graamans: ~10%, "
    "other models higher).",
    "Household: 2 persons, non-food electricity 8 kWh/day, roof catchment "
    "60 m2, rainfall 1,300 mm/yr, 80% capture, 50% greywater reuse — all "
    "assumptions.",
    "Outdoor yields are one harvest per year at national-average yields.",
]

KCAL_PER_MOL_CEILING = 686.0 / 6.0 / 8.0          # 14.29 kcal/mol photons
CROP_KCAL_PER_MOL = {
    "leafy_greens": 1.06,          # derived, see ASSUMPTIONS
    "staple_grain": 1.32,          # derived from Bugbee 1988
    "staple_optimistic": 2.6,      # assumption
}
OUTDOOR_KCAL_PER_M2_YR = {
    "potato_kr": 24.67e3 * 770.0 / 1e4,     # 1,900 kcal/m2/yr
    "rice_kr": 5.14e3 * 3650.0 / 1e4,       # 1,876 kcal/m2/yr
}


@dataclass
class PodParams:
    persons: int = 2
    kcal_per_person_day: float = 2500.0       # within 2,000-2,800
    grow_area_m2: float = 10.0                # indoor canopy (all tiers)
    ppfd_umol_m2_s: float = 250.0
    photoperiod_h: float = 16.0
    led_efficacy_umol_j: float = 2.7          # Kusuma 2020 range 2.5-3.0
    hvac_overhead: float = 1.2
    crop: str = "leafy_greens"
    outdoor_area_m2: float = 0.0
    outdoor_crop: str = "potato_kr"
    pv_kwp: float = 6.0
    pv_kwh_per_kwp_day_mean: float = 3.27     # PVGIS
    pv_kwh_per_kwp_day_worst: float = 1.90    # PVGIS December
    battery_kwh: float = 15.0
    base_load_kwh_day: float = 8.0
    water_l_person_day: float = 190.0         # derived ME 2023
    greywater_reuse: float = 0.5
    roof_m2: float = 60.0
    rain_mm_yr: float = 1300.0
    rain_capture: float = 0.8
    grow_water_l_m2_day: float = 0.5          # assumption, closed loop
    # capex (USD), O&M
    pv_usd_per_kw: float = 1080.0
    battery_usd_per_kwh: float = 700.0
    grow_usd_per_m2: float = 800.0
    water_system_usd: float = 3000.0
    life_pv_y: float = 25.0
    life_batt_y: float = 12.0
    life_grow_y: float = 10.0
    life_water_y: float = 15.0
    om_frac: float = 0.015
    grow_consumables_usd_m2_yr: float = 20.0


class PodModel:
    """Steady-state annual model. Returns a dict; never invents results."""

    def __init__(self, params: Optional[PodParams] = None):
        self.p = params or PodParams()

    def dli(self) -> float:
        return self.p.ppfd_umol_m2_s * self.p.photoperiod_h * 3600 / 1e6

    def indoor_kcal_day(self) -> float:
        kpm = min(CROP_KCAL_PER_MOL[self.p.crop], KCAL_PER_MOL_CEILING)
        return self.p.grow_area_m2 * self.dli() * kpm

    def light_ceiling_kcal_day(self) -> float:
        return self.p.grow_area_m2 * self.dli() * KCAL_PER_MOL_CEILING

    def outdoor_kcal_day(self) -> float:
        return (self.p.outdoor_area_m2 *
                OUTDOOR_KCAL_PER_M2_YR[self.p.outdoor_crop] / 365.0)

    def grow_kwh_day(self) -> float:
        mol = self.p.grow_area_m2 * self.dli()
        joules = mol / (self.p.led_efficacy_umol_j * 1e-6)
        return joules / 3.6e6 * self.p.hvac_overhead

    def kwh_per_1000_kcal(self) -> float:
        k = self.indoor_kcal_day()
        return self.grow_kwh_day() / k * 1000.0 if k > 0 else math.inf

    def capex_usd(self) -> Dict[str, float]:
        p = self.p
        return {"pv": p.pv_kwp * p.pv_usd_per_kw,
                "battery": p.battery_kwh * p.battery_usd_per_kwh,
                "grow": p.grow_area_m2 * p.grow_usd_per_m2,
                "water": p.water_system_usd}

    def annualized_cost_usd(self) -> float:
        p = self.p
        c = self.capex_usd()
        dep = (c["pv"] / p.life_pv_y + c["battery"] / p.life_batt_y +
               c["grow"] / p.life_grow_y + c["water"] / p.life_water_y)
        om = p.om_frac * sum(c.values())
        return dep + om + p.grow_consumables_usd_m2_yr * p.grow_area_m2

    def water_balance_l_day(self) -> float:
        p = self.p
        rain = p.roof_m2 * p.rain_mm_yr * p.rain_capture / 365.0
        demand = (p.persons * p.water_l_person_day * (1 - p.greywater_reuse)
                  + p.grow_area_m2 * p.grow_water_l_m2_day)
        return rain - demand

    def report(self) -> Dict:
        p = self.p
        need_day = p.persons * p.kcal_per_person_day
        indoor = self.indoor_kcal_day()
        outdoor = self.outdoor_kcal_day()
        load = p.base_load_kwh_day + self.grow_kwh_day()
        pv_mean = p.pv_kwp * p.pv_kwh_per_kwp_day_mean
        pv_worst = p.pv_kwp * p.pv_kwh_per_kwp_day_worst
        night_need = load * 0.5          # assumption: half the load at night
        best_kpm = CROP_KCAL_PER_MOL["staple_optimistic"]
        full_mol = need_day / best_kpm
        full_kwh = full_mol / (p.led_efficacy_umol_j * 1e-6) / 3.6e6 * p.hvac_overhead
        full_out_m2 = need_day * 365 / OUTDOOR_KCAL_PER_M2_YR[p.outdoor_crop]
        energy_ok = pv_worst >= load and p.battery_kwh >= night_need
        water_ok = self.water_balance_l_day() >= 0
        food_ok = indoor + outdoor >= need_day
        ann = self.annualized_cost_usd()
        return {
            "persons": p.persons,
            "kcal_need_day": round(need_day, 1),
            "indoor_kcal_day": round(indoor, 1),
            "outdoor_kcal_day": round(outdoor, 1),
            "light_ceiling_kcal_day": round(self.light_ceiling_kcal_day(), 1),
            "calorie_share": round((indoor + outdoor) / need_day, 4),
            "grow_kwh_day": round(self.grow_kwh_day(), 2),
            "kwh_per_1000_kcal": round(self.kwh_per_1000_kcal(), 1),
            "load_kwh_day": round(load, 2),
            "pv_kwh_day_mean": round(pv_mean, 2),
            "pv_kwh_day_worst_month": round(pv_worst, 2),
            "pv_kwp_needed_worst_month": round(load / p.pv_kwh_per_kwp_day_worst, 2),
            "water_balance_l_day": round(self.water_balance_l_day(), 1),
            "capex_usd": {k: round(v) for k, v in self.capex_usd().items()},
            "capex_total_usd": round(sum(self.capex_usd().values())),
            "annualized_cost_usd": round(ann),
            "annualized_cost_per_person_usd": round(ann / p.persons),
            "full_indoor_kwh_day_best_case": round(full_kwh, 1),
            "full_indoor_pv_kwp_mean": round(full_kwh / p.pv_kwh_per_kwp_day_mean, 1),
            "full_outdoor_m2": round(full_out_m2),
            "energy_ok": energy_ok, "water_ok": water_ok,
            "food_self_sufficient": food_ok,
            "sustainable": energy_ok and water_ok and food_ok,
        }


# ===========================================================================
# 1b. HouseholdPod
# ===========================================================================
@dataclass
class HouseholdPod:
    node_id: str
    battery_capacity_kwh: float = 35.0
    battery_current_kwh: float = 28.0
    water_tank_capacity_l: float = 1200.0
    water_current_l: float = 900.0
    # v7 had a fixed 2800 kcal/day. v8: None -> computed from pod_model.
    aeroponic_daily_calories: Optional[float] = None
    transit_credit_balance: float = 0.0
    person_id: str = ""
    polity: str = ""
    neighbors: Set[str] = field(default_factory=set)   # informational only
    sponsors: Dict[str, float] = field(default_factory=dict)
    watcher_only: bool = False
    months_idle: int = 0
    monthly_demurrage_rate: float = 0.04       # idle decay (v7 behaviour)
    demurrage_threshold: float = 500.0         # v8: free holding allowance
    holding_demurrage_rate: float = 0.10       # v8: on balance above threshold
    greywater_reuse: float = 0.5               # v7 hard-coded 0.90
    pod_model: Optional[PodModel] = None

    def __post_init__(self):
        if self.pod_model is None:
            self.pod_model = PodModel()
        if self.aeroponic_daily_calories is None:
            self.aeroponic_daily_calories = self.pod_model.indoor_kcal_day()

    def credit(self, amount: float) -> None:
        self.transit_credit_balance += amount
        self.months_idle = 0

    def step_month(self, is_active: bool) -> Dict[str, float]:
        """Idle pods decay at monthly_demurrage_rate (v7). Every pod,
        active or not, also pays holding_demurrage_rate on the part of
        its balance above demurrage_threshold (v8), so no balance grows
        without limit under a bounded inflow."""
        initial = self.transit_credit_balance
        bal = initial
        if not is_active and bal > 0:
            self.months_idle += 1
            bal = bal * (1.0 - self.monthly_demurrage_rate)
        else:
            self.months_idle = 0
        excess = max(0.0, bal - self.demurrage_threshold)
        bal -= excess * self.holding_demurrage_rate
        self.transit_credit_balance = round(bal, 2)
        decayed = round(initial - self.transit_credit_balance, 2)
        return {
            "initial_balance": initial,
            "decayed_burned": decayed,
            "net_balance": self.transit_credit_balance,
            "months_idle": self.months_idle,
        }

    def balance_bound(self, max_monthly_inflow: float) -> float:
        """Upper bound on the balance under inflow <= max_monthly_inflow per
        month (credited before each step): fixed point of
        B' = T + (B + I - T)(1 - r)  ->  B* = T + I(1 - r)/r."""
        r = self.holding_demurrage_rate
        return self.demurrage_threshold + max_monthly_inflow * (1 - r) / r

    def cycle_day(self, solar_kwh: float, load_kwh: float,
                  rain_l: float, use_l: float,
                  yield_factor: float = 1.0) -> Dict:
        net_e = solar_kwh - load_kwh
        self.battery_current_kwh = min(self.battery_capacity_kwh,
                                       max(0.0, self.battery_current_kwh + net_e))
        net_w = rain_l + use_l * self.greywater_reuse - use_l
        self.water_current_l = min(self.water_tank_capacity_l,
                                   max(0.0, self.water_current_l + net_w))
        calories = self.aeroponic_daily_calories * yield_factor
        need = self.pod_model.p.kcal_per_person_day
        energy_ok = net_e >= 0 or self.battery_current_kwh > 0
        water_ok = net_w >= 0 or self.water_current_l > 0
        food_ok = calories >= need
        return {
            "battery_storage_kwh": round(self.battery_current_kwh, 2),
            "water_storage_l": round(self.water_current_l, 2),
            "calories_yield": round(calories, 1),
            "calorie_share_one_person": round(calories / need, 4),
            "cost_usd": round(self.pod_model.annualized_cost_usd() / 365.0, 2),
            "energy_ok": energy_ok, "water_ok": water_ok,
            "food_self_sufficient": food_ok,
            "sustainable": energy_ok and water_ok and food_ok,
        }

    def total_sponsor_share(self) -> float:
        return sum(self.sponsors.values())

# ===========================================================================
# 2. PersonhoodRegistry
# ===========================================================================
@dataclass
class PersonhoodAttestation:
    person_id: str
    biometric_hash: str
    sponsors: Set[str] = field(default_factory=set)
    vouched_in_person: bool = False
    registered_day: int = 0
    genesis: bool = False
    exited: bool = False


class PersonhoodRegistry:
    """One biometric -> one person -> at most one pod, within this registry.

    v8 changes: vouches are deduplicated and self-vouching is refused;
    the first members join only through register_genesis() as a founding
    quorum (no single-person bootstrap); biometric hashes are HMACs with a
    per-registry secret pepper. LIMITATION (out of scope, labeled): real
    biometrics are fuzzy; exact hashing of a template only works for this
    toy. A deployment needs a template-protection scheme (e.g. fuzzy
    extractors / cancellable biometrics)."""

    def __init__(self, vouches_required: int = 2,
                 bootstrap_sponsors: Optional[Set[str]] = None,
                 genesis_quorum: int = 3,
                 pepper: Optional[bytes] = None):
        self.vouches_required = vouches_required
        # kept for API compatibility; no longer used to admit anyone
        self.bootstrap_sponsors = bootstrap_sponsors or set()
        self.genesis_quorum = max(genesis_quorum, vouches_required + 1)
        self._pepper = pepper if pepper is not None else secrets.token_bytes(32)
        self._biometric_index: Dict[str, str] = {}
        self.persons: Dict[str, PersonhoodAttestation] = {}
        self._pod_bindings: Dict[str, str] = {}

    def _hash(self, b: str) -> str:
        return hmac.new(self._pepper, b.encode(), hashlib.sha256).hexdigest()[:32]

    def register_genesis(self, members: List[Tuple[str, str]],
                         day: int = 0) -> Dict:
        """Founding quorum: >= genesis_quorum distinct people, registered
        in person together, each vouched for by all the others."""
        if self.persons:
            return {"status": "FAILED", "reason": "genesis already done"}
        ids = [m[0] for m in members]
        bios = {self._hash(m[1]) for m in members}
        if len(set(ids)) != len(ids) or len(bios) != len(members):
            return {"status": "FAILED", "reason": "duplicate id or biometric"}
        if len(members) < self.genesis_quorum:
            return {"status": "FAILED",
                    "reason": f"genesis needs {self.genesis_quorum} members"}
        for pid, bio in members:
            bh = self._hash(bio)
            self._biometric_index[bh] = pid
            self.persons[pid] = PersonhoodAttestation(
                pid, bh, set(ids) - {pid}, True, day, genesis=True)
        return {"status": "GENESIS", "members": len(members)}

    def register(self, person_id: str, biometric: str,
                 sponsors: Optional[List[str]] = None,
                 in_person: bool = False, day: int = 0) -> Dict:
        bh = self._hash(biometric)
        if bh in self._biometric_index:
            return {"status": "FAILED",
                    "reason": "biometric already registered"}
        if person_id in self.persons:
            return {"status": "FAILED", "reason": "person_id taken"}
        if not self.persons:
            return {"status": "FAILED",
                    "reason": "empty registry: use register_genesis"}
        valid = {s for s in (sponsors or [])
                 if s in self.persons and s != person_id
                 and not self.persons[s].exited}
        if len(valid) < self.vouches_required:
            return {"status": "FAILED",
                    "reason": f"need {self.vouches_required} distinct "
                              f"sponsors, have {len(valid)}"}
        att = PersonhoodAttestation(person_id, bh, valid, in_person, day)
        self._biometric_index[bh] = person_id
        self.persons[person_id] = att
        return {"status": "REGISTERED", "person_id": person_id,
                "sponsors": len(att.sponsors)}

    def import_attestation(self, att: PersonhoodAttestation,
                           biometric: str) -> Dict:
        """Used by Federation: re-hash with THIS registry's pepper."""
        if att.person_id in self.persons:
            return {"status": "FAILED", "reason": "person_id exists in target"}
        bh = self._hash(biometric)
        if bh in self._biometric_index:
            return {"status": "FAILED", "reason": "already known to target"}
        self._biometric_index[bh] = att.person_id
        self.persons[att.person_id] = PersonhoodAttestation(
            att.person_id, bh, set(att.sponsors), att.vouched_in_person,
            att.registered_day)
        return {"status": "RECOGNIZED", "person_id": att.person_id}

    def bind_pod(self, person_id: str, pod_id: str) -> Dict:
        if person_id not in self.persons or self.persons[person_id].exited:
            return {"status": "FAILED", "reason": "person not registered"}
        if person_id in self._pod_bindings:
            return {"status": "FAILED",
                    "reason": f"person already bound to "
                              f"{self._pod_bindings[person_id]}"}
        self._pod_bindings[person_id] = pod_id
        return {"status": "BOUND", "person_id": person_id, "pod_id": pod_id}

    def unbind_pod(self, person_id: str) -> None:
        self._pod_bindings.pop(person_id, None)

    def pod_count(self) -> int:
        return len(self._pod_bindings)

    def is_active_member(self, person_id: str) -> bool:
        a = self.persons.get(person_id)
        return a is not None and not a.exited

    def erase(self, person_id: str) -> Dict:
        """Right to erasure on request. NOTE (residual risk): deleting the
        biometric hash means the same person could register again."""
        att = self.persons.pop(person_id, None)
        if att is None:
            return {"status": "FAILED", "reason": "unknown person"}
        self._biometric_index.pop(att.biometric_hash, None)
        self._pod_bindings.pop(person_id, None)
        for other in self.persons.values():
            other.sponsors.discard(person_id)
        return {"status": "ERASED"}


# ===========================================================================
# 3. LedgerClock + NeighborGraph + MeshConsensus
# ===========================================================================
class LedgerClock:
    """System-owned monotonic clock. Claimers cannot set it."""

    def __init__(self, start_hour: int = 0):
        self._hour = start_hour

    @property
    def hour(self) -> int:
        return self._hour

    def advance(self, hours: int = 1) -> int:
        if hours < 0:
            raise ValueError("clock is monotonic")
        self._hour += hours
        return self._hour


class NeighborGraph:
    """Neighbor edges exist only when both registered pods agree."""

    def __init__(self):
        self.registered: Set[str] = set()
        self._proposals: Set[Tuple[str, str]] = set()
        self._edges: Set[FrozenSet[str]] = set()

    def register_pod(self, pod_id: str) -> None:
        self.registered.add(pod_id)

    def propose(self, a: str, b: str) -> Dict:
        if a == b or a not in self.registered or b not in self.registered:
            return {"status": "FAILED", "reason": "both pods must be registered"}
        if (b, a) in self._proposals:
            self._proposals.discard((b, a))
            self._edges.add(frozenset((a, b)))
            return {"status": "MUTUAL"}
        self._proposals.add((a, b))
        return {"status": "PENDING"}

    def link(self, a: str, b: str) -> None:
        """Convenience: both sides propose."""
        self.propose(a, b)
        self.propose(b, a)

    def are_neighbors(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self._edges


class MeshConsensus:
    """Neighbor attestation. v8: time comes from the LedgerClock, rate
    limits use a rolling window, neighbors come from the NeighborGraph,
    and two collusion caps apply per period: per (claimer, attestor) pair
    and per exact attestor set. RESIDUAL RISK: a ring of genuinely
    registered, mutually linked neighbors can still collude up to these
    caps; the caps bound the rate of fake minting, they do not detect it."""

    def __init__(self, quorum: int = 3,
                 rate_limit_per_hour: int = 2,
                 neighbor_required: bool = True,
                 clock: Optional[LedgerClock] = None,
                 graph: Optional[NeighborGraph] = None,
                 window_hours: int = 1,
                 pair_cap_per_period: int = 4,
                 set_cap_per_period: int = 3,
                 period_hours: int = 24):
        self.quorum = quorum
        self.rate_limit = rate_limit_per_hour
        self.neighbor_required = neighbor_required
        self.clock = clock or LedgerClock()
        self.graph = graph or NeighborGraph()
        self.window_hours = window_hours
        self.pair_cap = pair_cap_per_period
        self.set_cap = set_cap_per_period
        self.period_hours = period_hours
        self._attestation_log: Dict[str, List[Tuple[int, str]]] = {}
        self._pair_log: Dict[Tuple[str, str], List[int]] = {}
        self._set_log: Dict[FrozenSet[str], List[int]] = {}
        self._nonce = 0

    @staticmethod
    def _sig(pod_id: str, claim_id: str) -> str:
        return hashlib.sha256(f"{pod_id}|{claim_id}".encode()).hexdigest()[:16]

    def _recent(self, times: List[int], span: int) -> int:
        now = self.clock.hour
        return sum(1 for t in times if now - t < span)

    def register(self, claimer: HouseholdPod, domain: str,
                 hours: float, peers: List[HouseholdPod],
                 hour: Optional[int] = None) -> Dict:
        """`hour` is accepted for v7 API compatibility and IGNORED."""
        now = self.clock.hour
        self._nonce += 1
        claim_id = hashlib.sha256(
            f"{claimer.node_id}|{domain}|{hours}|{now}|{self._nonce}".encode()
        ).hexdigest()[:12]
        accepted, rejected = [], []
        for peer in peers:
            if peer.node_id == claimer.node_id:
                rejected.append((peer.node_id, "self-attestation"))
                continue
            if self.neighbor_required and not self.graph.are_neighbors(
                    claimer.node_id, peer.node_id):
                rejected.append((peer.node_id, "not a registered mutual neighbor"))
                continue
            log = self._attestation_log.setdefault(peer.node_id, [])
            if self._recent([e[0] for e in log], self.window_hours) >= self.rate_limit:
                rejected.append((peer.node_id, "rate limit"))
                continue
            pair = (claimer.node_id, peer.node_id)
            if self._recent(self._pair_log.get(pair, []),
                            self.period_hours) >= self.pair_cap:
                rejected.append((peer.node_id, "pair cap"))
                continue
            accepted.append(peer.node_id)
        aset = frozenset(accepted)
        if accepted and self._recent(self._set_log.get(aset, []),
                                     self.period_hours) >= self.set_cap:
            rejected.extend((p, "attestor-set cap") for p in accepted)
            accepted = []
        for p in accepted:     # commit only what was accepted
            self._attestation_log.setdefault(p, []).append((now, claim_id))
            self._pair_log.setdefault((claimer.node_id, p), []).append(now)
        if accepted:
            self._set_log.setdefault(aset, []).append(now)
        return {
            "claim_id": claim_id,
            "hour": now,
            "accepted": accepted,
            "rejected": rejected,
            "quorum_met": len(accepted) >= self.quorum,
            "needed": self.quorum,
            "sigs": [self._sig(p, claim_id) for p in accepted],
        }

# ===========================================================================
# 3b. CaloriePlanner — goal-weight demand feeding the pod model
# ===========================================================================
# GENERAL GUIDANCE ONLY, NOT MEDICAL ADVICE. Mifflin-St Jeor 1990
# (Am J Clin Nutr 51:241-247): BMR = 10 W + 6.25 H - 5 A + 5 (male) / -161
# (female). Activity factors are the conventional multipliers. The
# 7,700 kcal/kg figure is a common rule of thumb that overstates long-run
# loss (metabolic adaptation); weeks-to-goal is a rough estimate.
ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55,
                    "active": 1.725, "very_active": 1.9}
KCAL_PER_KG = 7700.0
CALORIE_FLOOR = {"female": 1200.0, "male": 1500.0}   # general guidance


def mifflin_st_jeor(weight_kg: float, height_cm: float, age: float,
                    sex: str) -> float:
    s = 5.0 if sex == "male" else -161.0
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s


def plan_calories(height_cm: float, weight_kg: float, age: float, sex: str,
                  activity: str, goal_kg: float,
                  adjust_kcal: float = 500.0) -> Dict:
    if sex not in CALORIE_FLOOR:
        raise ValueError("sex must be 'female' or 'male' (BMR equation inputs)")
    goal_bmi = goal_kg / (height_cm / 100) ** 2
    if goal_bmi < 18.5:
        return {"status": "REFUSED",
                "reason": f"goal BMI {goal_bmi:.1f} < 18.5 (underweight)",
                "note": "general guidance, not medical advice"}
    bmr = mifflin_st_jeor(weight_kg, height_cm, age, sex)
    tdee = bmr * ACTIVITY_FACTORS[activity]
    # cap at ~0.5 kg/week: 0.5 * 7700 / 7 = 550 kcal/day
    step = min(max(300.0, min(adjust_kcal, 500.0)), 0.5 * KCAL_PER_KG / 7)
    diff = goal_kg - weight_kg
    if abs(diff) < 0.5:
        target, step = tdee, 0.0
    elif diff < 0:
        target = tdee - step
    else:
        target = tdee + step
    floor = CALORIE_FLOOR[sex]
    floored = target < floor
    target = max(target, floor)
    eff = abs(target - tdee)
    weeks = (abs(diff) * KCAL_PER_KG / (eff * 7)) if eff > 0 else (
        0.0 if abs(diff) < 0.5 else math.inf)
    return {"status": "OK", "bmr": round(bmr, 1), "tdee": round(tdee, 1),
            "target_kcal_day": round(target, 1), "floor_applied": floored,
            "goal_bmi": round(goal_bmi, 1),
            "weeks_to_goal_est": round(weeks, 1),
            "note": "general guidance, not medical advice"}


def pod_supply_for_plan(plan: Dict, params: Optional[PodParams] = None
                        ) -> Dict:
    """Feed one person's target into the pod model as the demand."""
    if plan.get("status") != "OK":
        return plan
    p = params or PodParams()
    p = PodParams(**{**p.__dict__, "persons": 1,
                     "kcal_per_person_day": plan["target_kcal_day"]})
    r = PodModel(p).report()
    return {"target_kcal_day": plan["target_kcal_day"],
            "pod_kcal_day": round(r["indoor_kcal_day"] + r["outdoor_kcal_day"], 1),
            "pod_share": r["calorie_share"]}


# ===========================================================================
# 4. TransparencyLog + Watcher
# ===========================================================================
class Watcher:
    def __init__(self, watcher_id: str):
        self.watcher_id = watcher_id
        self.comments: List[Dict] = []
        self.read_log: List[str] = []
        self.filed_proposals: List[str] = []

    def read(self, event_type: str, payload: Dict) -> None:
        self.read_log.append(event_type)

    def comment(self, target_id: str, text: str, day: int) -> Dict:
        c = {"watcher_id": self.watcher_id, "target": target_id,
             "text": text, "day": day}
        self.comments.append(c)
        return c


class TransparencyLog:
    """Public event log. v8 privacy mode (default on): identifying fields
    are replaced by per-subject pseudonyms (HMAC with a per-subject random
    key) and per-person hours are not published; hours are aggregated per
    (day, domain) and released only for cells with >= k contributors.
    Erasure deletes the subject key and rewrites that pseudonym
    (crypto-shredding)."""
    ID_KEYS = ("pod_id", "person_id")
    PRIVATE_KEYS = ("hours",)

    def __init__(self, privacy: bool = True, k_anon: int = 3):
        self.events: List[Dict] = []
        self.watchers: List[Watcher] = []
        self.privacy = privacy
        self.k_anon = k_anon
        self._subject_keys: Dict[str, bytes] = {}
        self._agg: Dict[Tuple[int, str], List] = {}

    def pseudonym(self, subject: str) -> str:
        key = self._subject_keys.setdefault(subject, secrets.token_bytes(16))
        return "ps:" + hmac.new(key, subject.encode(),
                                hashlib.sha256).hexdigest()[:12]

    def append(self, event_type: str, payload: Dict, day: int = 0) -> None:
        payload = dict(payload)
        if self.privacy:
            if "hours" in payload and "domain" in payload:
                cell = self._agg.setdefault((day, payload["domain"]), [0.0, set()])
                cell[0] += payload["hours"]
                cell[1].add(payload.get("pod_id", "?"))
            for k in self.PRIVATE_KEYS:
                payload.pop(k, None)
            for k in self.ID_KEYS:
                if k in payload and payload[k]:
                    payload[k] = self.pseudonym(str(payload[k]))
        event = {"seq": len(self.events), "type": event_type,
                 "day": day, "payload": payload}
        self.events.append(event)
        for w in self.watchers:
            w.read(event_type, payload)

    def aggregate_report(self) -> List[Dict]:
        out = []
        for (day, dom), (hrs, who) in sorted(self._agg.items()):
            if len(who) >= self.k_anon:
                out.append({"day": day, "domain": dom,
                            "hours": round(hrs, 1), "contributors": len(who)})
            else:
                out.append({"day": day, "domain": dom,
                            "hours": None, "contributors": f"<{self.k_anon}"})
        return out

    def erase_subject(self, subject: str) -> int:
        if subject not in self._subject_keys:
            return 0
        ps = self.pseudonym(subject)
        del self._subject_keys[subject]
        n = 0
        for e in self.events:
            for k in self.ID_KEYS:
                if e["payload"].get(k) == ps:
                    e["payload"][k] = "erased"
                    n += 1
        for cell in self._agg.values():
            if subject in cell[1]:
                cell[1].discard(subject)
                cell[1].add("erased:" + secrets.token_hex(4))
        return n

    def add_watcher(self, watcher: Watcher) -> None:
        self.watchers.append(watcher)

    def read_all(self, event_type: Optional[str] = None) -> List[Dict]:
        if event_type is None:
            return list(self.events)
        return [e for e in self.events if e["type"] == event_type]


# ===========================================================================
# 5. UNRecognition + EscalationLadder
# ===========================================================================
# LABEL: the "UN escalation ladder", "UN recognition" and the UNP codes are
# a FICTIONAL / PROPOSED mechanism for this model. No such UN process exists.
@dataclass
class UNProhibition:
    code: str
    description: str
    applies_to_domains: List[str]
    severity: str = "minor"


STANDARD_PROHIBITIONS = [
    UNProhibition("UNP-01", "Child labor", ["community_care"], "major"),
    UNProhibition("UNP-02", "Forced labor",
                  ["community_care", "permaculture", "field_study"], "major"),
    UNProhibition("UNP-03", "Environmental destruction",
                  ["permaculture", "field_study"], "minor"),
    UNProhibition("UNP-04", "Cultural erasure",
                  ["ritual_observation"], "minor"),
]


class UNEscalationLadder:
    """Severity comes from ONE source: the prohibition list. Appeals are
    filed, then decided by a quorum of independent registered reviewers;
    an approved appeal vacates a specific violation record, so later
    recomputes respect it."""
    STAGES = ["CLEAR", "WARNING", "PARTIAL_SUSP",
              "FULL_SUSPENSION", "DECERTIFICATION"]

    def __init__(self, prohibitions: Optional[List[UNProhibition]] = None,
                 review_quorum: int = 3):
        self.prohibitions = {p.code: p for p in
                             (prohibitions or STANDARD_PROHIBITIONS)}
        self.stage: Dict[str, int] = {}
        self.violation_log: Dict[str, List[Dict]] = {}
        self.reviewers: Dict[str, str] = {}      # reviewer_id -> home polity
        self.review_quorum = review_quorum
        self.appeals: List[Dict] = []

    def severity_of(self, code: str) -> str:
        return self.prohibitions[code].severity

    @property
    def MINOR_CODES(self) -> Set[str]:
        return {c for c, p in self.prohibitions.items() if p.severity == "minor"}

    @property
    def MAJOR_CODES(self) -> Set[str]:
        return {c for c, p in self.prohibitions.items() if p.severity == "major"}

    def record_violation(self, polity: str, code: str, day: int,
                         detail: str = "") -> Dict:
        if code not in self.prohibitions:
            raise ValueError(f"unknown prohibition {code}")
        self.violation_log.setdefault(polity, []).append(
            {"code": code, "day": day, "detail": detail, "vacated": False})
        self._recompute(polity)
        return {"polity": polity, "stage": self.stage[polity],
                "stage_name": self.STAGES[self.stage[polity]]}

    def _recompute(self, polity: str) -> None:
        v = [x for x in self.violation_log.get(polity, []) if not x["vacated"]]
        if not v:
            self.stage[polity] = 0
            return
        minor_c, major_c = self.MINOR_CODES, self.MAJOR_CODES
        last = v[-1]["day"]
        recent = [x for x in v if last - x["day"] <= 90]
        minors = [x for x in recent if x["code"] in minor_c]
        majors = [x for x in recent if x["code"] in major_c]
        life_majors = [x for x in v if x["code"] in major_c]
        stage = 0
        if len(minors) >= 1:
            stage = max(stage, 1)
        if len(minors) >= 3:
            stage = max(stage, 2)
        if len(majors) >= 1:
            stage = max(stage, 2)
        if len(majors) >= 2 or (len(majors) >= 1 and len(minors) >= 3):
            stage = max(stage, 3)
        if len(life_majors) >= 5:
            stage = max(stage, 4)
        if any(x["code"] == "UNP-02" and "forced_labor" in x.get("detail", "")
               for x in v):
            stage = 4
        self.stage[polity] = stage

    def current_stage(self, polity: str) -> Tuple[int, str]:
        s = self.stage.get(polity, 0)
        return s, self.STAGES[s]

    def register_reviewer(self, reviewer_id: str, home_polity: str) -> None:
        self.reviewers[reviewer_id] = home_polity

    def appeal(self, polity: str, justification: str, day: int,
               violation_index: Optional[int] = None) -> Dict:
        """File an appeal. Nothing changes until reviewers decide."""
        live = [i for i, x in enumerate(self.violation_log.get(polity, []))
                if not x["vacated"]]
        if not live or self.stage.get(polity, 0) == 0:
            return {"status": "FAILED", "reason": "no stage to appeal"}
        if not justification.strip():
            return {"status": "FAILED", "reason": "justification required"}
        idx = live[-1] if violation_index is None else violation_index
        if idx not in live:
            return {"status": "FAILED", "reason": "no such live violation"}
        a = {"appeal_id": len(self.appeals), "polity": polity, "index": idx,
             "justification": justification, "day": day,
             "votes": {}, "status": "PENDING"}
        self.appeals.append(a)
        return {"status": "PENDING", "appeal_id": a["appeal_id"],
                "needs": self.review_quorum}

    def review(self, appeal_id: int, reviewer_id: str, approve: bool) -> Dict:
        a = self.appeals[appeal_id]
        if a["status"] != "PENDING":
            return {"status": "FAILED", "reason": f"appeal {a['status']}"}
        home = self.reviewers.get(reviewer_id)
        if home is None:
            return {"status": "FAILED", "reason": "unregistered reviewer"}
        if home == a["polity"]:
            return {"status": "FAILED", "reason": "reviewer not independent"}
        a["votes"][reviewer_id] = approve
        yes = sum(a["votes"].values())
        no = len(a["votes"]) - yes
        if yes >= self.review_quorum:
            a["status"] = "GRANTED"
            old = self.stage.get(a["polity"], 0)
            self.violation_log[a["polity"]][a["index"]]["vacated"] = True
            self._recompute(a["polity"])
            return {"status": "GRANTED", "from_stage": old,
                    "to_stage": self.stage[a["polity"]]}
        if no >= self.review_quorum:
            a["status"] = "DENIED"
            return {"status": "DENIED"}
        return {"status": "PENDING", "yes": yes, "no": no}

    def can_cross_border(self, polity: str) -> bool:
        return self.stage.get(polity, 0) <= 1

    def can_register_pods(self, polity: str) -> bool:
        return self.stage.get(polity, 0) <= 2


class UNRecognition:
    STANDARD_PROHIBITIONS = STANDARD_PROHIBITIONS

    def __init__(self, signatories: List[str],
                 prohibitions: Optional[List[UNProhibition]] = None):
        self.signatories = set(signatories)
        self.prohibitions = prohibitions or list(STANDARD_PROHIBITIONS)
        self.recognized_polities: Set[str] = set()
        self.personhood_reports: List[Dict] = []
        self.ladder = UNEscalationLadder(self.prohibitions)

    def recognize(self, polity: str) -> Dict:
        stage, name = self.ladder.current_stage(polity)
        if stage >= 3:
            return {"status": "FAILED", "reason": f"polity is {name}"}
        self.recognized_polities.add(polity)
        return {"status": "RECOGNIZED", "polity": polity, "stage": name}

    def check_activity(self, domain: str, metadata: Dict,
                       polity: Optional[str] = None) -> Dict:
        for p in self.prohibitions:
            if domain in p.applies_to_domains:
                if metadata.get(f"violates_{p.code}", False):
                    if polity:
                        self.ladder.record_violation(
                            polity, p.code, metadata.get("day", 0),
                            metadata.get(f"detail_{p.code}", ""))
                    return {"status": "REJECT", "prohibition": p.code,
                            "severity": p.severity,
                            "reason": p.description}
        return {"status": "OK"}

    def report_personhood(self, person_id: str, polity: str,
                          day: int, log: Optional[TransparencyLog] = None
                          ) -> None:
        ps = (log.pseudonym(person_id) if log else
              "ps:" + hashlib.sha256(secrets.token_bytes(16) +
                                     person_id.encode()).hexdigest()[:12])
        self.personhood_reports.append({"person_ref": ps, "polity": polity,
                                        "day": day})

# ===========================================================================
# 6. AGIDataEngine
# ===========================================================================
@dataclass
class ProvenanceEvidence:
    """v7 added raw sensor_variance to the score, so noisier (or faked)
    sensors minted more. v8 scores provenance QUALITY: variance inside a
    plausible band scores 1.0; too flat (possible replay/synthetic) or too
    noisy scores lower. The band is an ASSUMPTION to be calibrated per
    sensor type."""
    sensor_variance: float
    quorum_attestations: int
    quorum_required: int = 3
    plausible_low: float = 0.05
    plausible_high: float = 0.60

    @property
    def quorum_score(self) -> float:
        return min(1.0, self.quorum_attestations / max(1, self.quorum_required))

    @property
    def quality(self) -> float:
        v, lo, hi = self.sensor_variance, self.plausible_low, self.plausible_high
        if v < 0:
            return 0.0
        if v < lo:
            return v / lo
        if v <= hi:
            return 1.0
        return max(0.0, 1.0 - (v - hi) / hi)

    @property
    def combined_entropy(self) -> float:
        """Name kept for API compatibility; now a provenance score."""
        return 0.6 * self.quality + 0.4 * self.quorum_score


@dataclass
class MammalianActivity:
    domain: str
    description: str
    hours_logged: float
    provenance: ProvenanceEvidence
    peer_consensus_rating: float
    pod_id: str = ""
    person_id: str = ""
    polity: str = ""
    metadata: Dict = field(default_factory=dict)


class AGIDataEngine:
    DEFAULT_DOMAIN_WEIGHTS = {
        "community_care":     1.50,
        "permaculture":       1.30,
        "ritual_observation": 1.40,
        "field_study":        1.25,
        "craft":              1.15,
        "synthetic_spam":     0.00,
    }

    def __init__(self, base_rate_per_hour: float = 12.0,
                 entropy_threshold: float = 0.75,
                 consensus_floor: float = 0.60,
                 un_recognition: Optional[UNRecognition] = None,
                 transparency_log: Optional[TransparencyLog] = None,
                 period_days: int = 30,
                 person_cap_per_period: float = 600.0,
                 global_cap_per_period: Optional[float] = None):
        self.base_rate = base_rate_per_hour
        self.entropy_threshold = entropy_threshold
        self.consensus_floor = consensus_floor
        self.domain_weights = dict(self.DEFAULT_DOMAIN_WEIGHTS)
        self.un = un_recognition
        self.log = transparency_log
        self.period_days = period_days
        self.person_cap = person_cap_per_period
        self.global_cap = global_cap_per_period
        self._minted: Dict[Tuple[int, str], float] = {}
        self._minted_global: Dict[int, float] = {}

    def minted_in_period(self, who: str, day: int) -> float:
        return self._minted.get((day // self.period_days, who), 0.0)

    def evaluate_and_mint(self, activity: MammalianActivity,
                          day: int = 0) -> Tuple[float, Optional[str]]:
        if self.un is not None:
            check = self.un.check_activity(activity.domain,
                                           activity.metadata,
                                           polity=activity.polity or None)
            if check["status"] == "REJECT":
                if self.log:
                    self.log.append("un_reject", {
                        "pod_id": activity.pod_id,
                        "domain": activity.domain,
                        "prohibition": check["prohibition"]}, day)
                return 0.0, (f"REJECTED [UN {check['prohibition']}]: "
                             f"{check['reason']}")

        entropy = activity.provenance.combined_entropy
        if entropy < self.entropy_threshold:
            return 0.0, (f"REJECTED [{activity.domain}]: provenance "
                         f"{entropy:.2f} < {self.entropy_threshold:.2f}")

        if activity.peer_consensus_rating < self.consensus_floor:
            return 0.0, (f"REJECTED [{activity.domain}]: consensus "
                         f"{activity.peer_consensus_rating:.2f} < "
                         f"{self.consensus_floor:.2f}")

        entropy_multiplier = (
            1.0 + (entropy - self.entropy_threshold) /
            (1.0 - self.entropy_threshold)
        )
        domain_weight = self.domain_weights.get(activity.domain, 1.0)
        minted = (activity.hours_logged * self.base_rate *
                  entropy_multiplier * domain_weight *
                  activity.peer_consensus_rating)
        # per-period caps
        period = day // self.period_days
        who = activity.person_id or activity.pod_id or "?"
        room = self.person_cap - self._minted.get((period, who), 0.0)
        if self.global_cap is not None:
            room = min(room, self.global_cap - self._minted_global.get(period, 0.0))
        capped = minted > room
        minted = max(0.0, min(minted, room))
        minted = round(minted, 2)
        if minted <= 0:
            return 0.0, f"REJECTED [{activity.domain}]: mint cap reached"
        self._minted[(period, who)] = self._minted.get((period, who), 0.0) + minted
        self._minted_global[period] = self._minted_global.get(period, 0.0) + minted
        if self.log:
            self.log.append("mint", {
                "pod_id": activity.pod_id, "domain": activity.domain,
                "hours": activity.hours_logged,
                "provenance": round(entropy, 3),
                "minted": minted, "capped": capped}, day)
        return minted, None


# ===========================================================================
# 7. MunicipalZone + MunicipalLedger (unchanged from v7)
# ===========================================================================
@dataclass
class MunicipalZone:
    zone_name: str
    base_tc_cost: float
    max_capacity: int
    current_occupancy: int
    ecological_fragility: float = 1.0

    def current_transit_toll(self) -> float:
        load_ratio = self.current_occupancy / max(1, self.max_capacity)
        congestion_penalty = (
            1.0 + max(0.0, load_ratio - 0.70) * 3.5 * self.ecological_fragility
        )
        return round(self.base_tc_cost * congestion_penalty, 2)

    def admit_traveler(self) -> bool:
        if self.current_occupancy < self.max_capacity:
            self.current_occupancy += 1
            return True
        return False


class MunicipalLedger:
    @staticmethod
    def process_trip(pod: HouseholdPod, zone: MunicipalZone,
                     polity_name: str = "",
                     un: Optional[UNRecognition] = None) -> Dict:
        if un is not None and polity_name:
            if not un.ladder.can_cross_border(polity_name):
                stage, name = un.ladder.current_stage(polity_name)
                return {"status": "DENIED",
                        "reason": f"polity {polity_name} at stage {stage} "
                                  f"({name}); cross-border travel suspended"}

        toll = zone.current_transit_toll()
        if pod.transit_credit_balance < toll:
            return {"status": "DENIED",
                    "reason": f"need {toll:.2f} TC, "
                              f"have {pod.transit_credit_balance:.2f}"}
        if zone.current_occupancy >= zone.max_capacity:
            return {"status": "DENIED",
                    "reason": f"zone {zone.zone_name} at capacity"}
        pod.transit_credit_balance -= toll
        pod.months_idle = 0
        zone.admit_traveler()
        return {
            "status": "APPROVED",
            "destination": zone.zone_name,
            "toll_paid": round(toll, 2),
            "remaining_balance": round(pod.transit_credit_balance, 2),
            "capacity_load": f"{zone.current_occupancy}/{zone.max_capacity}",
        }


# ===========================================================================
# 8. Polity (with exit) + Treasury
# ===========================================================================
class Treasury:
    """Spending needs approvals from >= min_approvals DISTINCT, currently
    registered (non-exited) members, and at least quorum_fraction of
    active members."""

    def __init__(self, registry: Optional[PersonhoodRegistry] = None,
                 min_approvals: int = 3, quorum_fraction: float = 0.0):
        self.balance = 0.0
        self.registry = registry
        self.min_approvals = min_approvals
        self.quorum_fraction = quorum_fraction
        self.ledger: List[Dict] = []

    def deposit(self, amount: float, source: str) -> None:
        self.balance += amount
        self.ledger.append({"in": amount, "source": source})

    def spend(self, amount: float, purpose: str,
              approvals: Set[str]) -> Dict:
        if self.registry is None:
            return {"status": "FAILED", "reason": "no member registry"}
        valid = {a for a in set(approvals) if self.registry.is_active_member(a)}
        active = sum(1 for p in self.registry.persons.values() if not p.exited)
        need = max(self.min_approvals, math.ceil(self.quorum_fraction * active))
        if len(valid) < need:
            return {"status": "FAILED",
                    "reason": f"need {need} distinct registered approvals, "
                              f"have {len(valid)}"}
        if amount <= 0 or amount > self.balance:
            return {"status": "FAILED", "reason": "invalid amount"}
        self.balance -= amount
        self.ledger.append({"out": amount, "purpose": purpose})
        return {"status": "OK", "balance": self.balance}


@dataclass
class Polity:
    name: str
    member_pods: Set[str] = field(default_factory=set)
    exit_settlements: List[Dict] = field(default_factory=list)

    def admit(self, pod: HouseholdPod) -> Dict:
        self.member_pods.add(pod.node_id)
        pod.polity = self.name
        return {"status": "ADMITTED", "polity": self.name,
                "members": len(self.member_pods)}

    def exit(self, pod: HouseholdPod,
             registry: Optional[PersonhoodRegistry] = None,
             log: Optional[TransparencyLog] = None,
             delete_personhood: bool = False, day: int = 0) -> Dict:
        """Leave the polity. Settlement rule: the full current balance
        (after any demurrage already applied) is paid out as a portable,
        non-expiring exit claim held by the person; nothing is confiscated,
        no exit fee. The pod is unbound; personhood is KEPT unless the
        person asks for deletion, in which case the registry entry is
        erased and the person's pseudonym is shredded in the log."""
        if pod.node_id not in self.member_pods:
            return {"status": "FAILED", "reason": "not a member"}
        self.member_pods.discard(pod.node_id)
        settled = round(pod.transit_credit_balance, 2)
        pod.transit_credit_balance = 0.0
        pod.polity = ""
        claim = {"claim_ref": secrets.token_hex(8),
                 "person_id": pod.person_id, "amount_tc": settled,
                 "day": day, "rule": "full balance, no fee"}
        self.exit_settlements.append(claim)
        if registry is not None and pod.person_id:
            registry.unbind_pod(pod.person_id)
            if delete_personhood:
                registry.erase(pod.person_id)
            elif pod.person_id in registry.persons:
                registry.persons[pod.person_id].exited = True
        erased = 0
        if log is not None:
            log.append("exit", {"pod_id": pod.node_id,
                                "settled": settled}, day)
            if delete_personhood:
                erased = log.erase_subject(pod.node_id)
                if pod.person_id:
                    erased += log.erase_subject(pod.person_id)
        if delete_personhood:
            claim["person_id"] = "erased"
        return {"status": "EXITED", "settled_tc": settled,
                "claim_ref": claim["claim_ref"],
                "personhood_kept": not delete_personhood,
                "log_entries_erased": erased}

# ===========================================================================
# 9. AuditorRegistry + ESGCertificate
# ===========================================================================
@dataclass
class Auditor:
    auditor_id: str
    name: str
    bond_tc: float
    issued_certificates: List[str] = field(default_factory=list)
    slash_count: int = 0
    removed: bool = False
    _key: bytes = field(default=b"", repr=False)   # auditor-held secret


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


class AuditorRegistry:
    """v8: certificates carry an HMAC-SHA256 tag made with the auditor's
    secret key; the registry holds a copy of the key to verify.
    LIMITATION: with HMAC the registry could also forge, and third parties
    cannot verify on their own; public-key signatures (e.g. Ed25519) would
    fix that but need a non-stdlib library.
    Slashing: removal only when (a) at least MIN_SAMPLE certificates exist
    and the Wilson 95% lower bound of the slash rate exceeds
    MAX_SLASH_RATE, or (b) slash_count reaches ABS_SLASH_LIMIT.
    Assignment: auditors are assigned at random by the registry, excluding
    declared conflicts; a sponsor cannot pick its own auditor."""
    MIN_BOND = 500.0
    MAX_SLASH_RATE = 0.05
    MIN_SAMPLE = 20
    ABS_SLASH_LIMIT = 3

    def __init__(self, seed: Optional[int] = None):
        self.auditors: Dict[str, Auditor] = {}
        self.contested: List[Dict] = []
        self._rng = random.Random(seed if seed is not None
                                  else secrets.randbits(64))
        self._conflicts: Set[Tuple[str, str]] = set()
        self._assignments: Dict[str, str] = {}

    def register(self, auditor_id: str, name: str, bond: float) -> Auditor:
        if bond < self.MIN_BOND:
            raise ValueError(f"bond {bond} below minimum {self.MIN_BOND}")
        a = Auditor(auditor_id, name, bond, _key=secrets.token_bytes(32))
        self.auditors[auditor_id] = a
        return a

    def declare_conflict(self, auditor_id: str, sponsor_id: str) -> None:
        self._conflicts.add((auditor_id, sponsor_id))

    @staticmethod
    def request_key(sponsor_id: str, pod_id: str, metric: str,
                    day: int) -> str:
        return f"{sponsor_id}|{pod_id}|{metric}|{day}"

    def assign(self, sponsor_id: str, pod_id: str, metric: str,
               day: int) -> Optional[Auditor]:
        key = self.request_key(sponsor_id, pod_id, metric, day)
        if key in self._assignments:
            return self.auditors[self._assignments[key]]
        pool = sorted(a for a in self.auditors
                      if self.is_valid(a) and (a, sponsor_id) not in self._conflicts)
        if not pool:
            return None
        aid = self._rng.choice(pool)
        self._assignments[key] = aid
        return self.auditors[aid]

    def slash(self, auditor_id: str, amount: float, reason: str) -> Dict:
        a = self.auditors.get(auditor_id)
        if a is None or a.removed:
            return {"status": "FAILED", "reason": "auditor unavailable"}
        slash = min(amount, a.bond_tc)
        a.bond_tc -= slash
        a.slash_count += 1
        rec = {"auditor_id": auditor_id, "amount": slash, "reason": reason}
        self.contested.append(rec)
        n = len(a.issued_certificates)
        lb = wilson_lower(a.slash_count, n) if n else 0.0
        if ((n >= self.MIN_SAMPLE and lb > self.MAX_SLASH_RATE) or
                a.slash_count >= self.ABS_SLASH_LIMIT):
            a.removed = True
        return {"status": "SLASHED", **rec, "bond_after": a.bond_tc,
                "wilson_lower": round(lb, 4), "removed": a.removed}

    def is_valid(self, auditor_id: str) -> bool:
        a = self.auditors.get(auditor_id)
        if a is None or a.removed:
            return False
        if a.bond_tc < self.MIN_BOND * 0.5:
            return False
        return True

    def sign(self, auditor: Auditor, payload: str) -> str:
        return hmac.new(auditor._key, payload.encode(),
                        hashlib.sha256).hexdigest()

    def verify(self, cert: "ESGCertificate") -> bool:
        a = self.auditors.get(cert.auditor_id)
        if a is None:
            return False
        expect = hmac.new(a._key, cert.payload().encode(),
                          hashlib.sha256).hexdigest()
        return hmac.compare_digest(expect, cert.auditor_signature)


@dataclass
class ESGCertificate:
    certificate_id: str
    sponsor_id: str
    pod_id: str
    metric: str
    amount: float
    issued_day: int
    auditor_id: str
    auditor_signature: str

    def payload(self) -> str:
        return (f"{self.certificate_id}|{self.sponsor_id}|{self.pod_id}|"
                f"{self.metric}|{self.amount:.4f}|{self.issued_day}|"
                f"{self.auditor_id}")

    @staticmethod
    def issue(sponsor_id: str, pod_id: str, metric: str, amount: float,
              day: int, auditor: Optional[Auditor],
              registry: AuditorRegistry) -> Optional["ESGCertificate"]:
        assigned = registry.assign(sponsor_id, pod_id, metric, day)
        if assigned is None:
            return None
        if auditor is not None and auditor.auditor_id != assigned.auditor_id:
            return None          # sponsor tried to choose its own auditor
        auditor = assigned
        if not registry.is_valid(auditor.auditor_id):
            return None
        cid = hashlib.sha256(
            f"{sponsor_id}|{pod_id}|{metric}|{amount:.4f}|{day}".encode()
        ).hexdigest()[:16]
        cert = ESGCertificate(cid, sponsor_id, pod_id, metric, amount,
                              day, auditor.auditor_id, "")
        cert.auditor_signature = registry.sign(auditor, cert.payload())
        auditor.issued_certificates.append(cid)
        return cert


# ===========================================================================
# 10. SponsorPool + CorporateSponsor
# ===========================================================================
@dataclass
class SponsorAllocation:
    sponsor_id: str
    amount_tc: float
    day_funded: int
    reason: str


class SponsorPool:
    def __init__(self, notice_period_days: int = 60,
                 cancellation_fee_pct: float = 0.10):
        self.notice_period_days = notice_period_days
        self.cancellation_fee_pct = cancellation_fee_pct
        self._allocations: Dict[str, List[SponsorAllocation]] = {}

    def contribute(self, pod: HouseholdPod, sponsor_id: str,
                   amount: float, day: int, reason: str) -> Dict:
        if amount <= 0:
            return {"status": "FAILED", "reason": "amount must be positive"}
        pod.credit(amount)
        pod.sponsors[sponsor_id] = pod.sponsors.get(sponsor_id, 0.0) + amount
        self._allocations.setdefault(pod.node_id, []).append(
            SponsorAllocation(sponsor_id, amount, day, reason))
        return {"status": "CONTRIBUTED", "pod": pod.node_id,
                "sponsor": sponsor_id, "amount": amount,
                "pod_total_sponsors": len(pod.sponsors)}

    def attribution(self, pod: HouseholdPod, total_amount: float
                    ) -> Dict[str, float]:
        total = pod.total_sponsor_share()
        if total <= 0:
            return {}
        return {sid: round(total_amount * (amt / total), 4)
                for sid, amt in pod.sponsors.items()}

    def largest_share(self, pod: HouseholdPod) -> Tuple[Optional[str], float]:
        if not pod.sponsors:
            return None, 0.0
        sid = max(pod.sponsors, key=lambda k: pod.sponsors[k])
        total = pod.total_sponsor_share()
        return sid, round(pod.sponsors[sid] / max(total, 1e-9), 4)


class CorporateSponsor:
    def __init__(self, sponsor_id: str, name: str,
                 annual_budget_tc: float, max_per_pod_tc: float,
                 focus_metrics: List[str]):
        self.sponsor_id = sponsor_id
        self.name = name
        self.annual_budget_tc = annual_budget_tc
        self.max_per_pod_tc = max_per_pod_tc
        self.focus_metrics = focus_metrics
        self.committed_tc = 0.0
        self.disbursed_tc = 0.0
        self.certificates: List[ESGCertificate] = []

    def remaining(self) -> float:
        return max(0.0, self.annual_budget_tc - self.disbursed_tc)

    def fund_pod(self, pool: SponsorPool, pod: HouseholdPod,
                 amount: float, reason: str, day: int) -> Dict:
        current = pod.sponsors.get(self.sponsor_id, 0.0)
        if current + amount > self.max_per_pod_tc:
            return {"status": "FAILED",
                    "reason": f"per-pod cap {self.max_per_pod_tc} exceeded"}
        if amount > self.remaining():
            return {"status": "FAILED", "reason": "annual budget exceeded"}
        r = pool.contribute(pod, self.sponsor_id, amount, day, reason)
        self.committed_tc += amount
        self.disbursed_tc += amount
        return {**r, "remaining_budget": round(self.remaining(), 2)}

    def issue_certificate(self, pod: HouseholdPod, metric: str,
                          amount: float, day: int,
                          auditor: Optional[Auditor] = None,
                          registry: Optional[AuditorRegistry] = None
                          ) -> Optional[ESGCertificate]:
        """v8: the registry assigns the auditor; passing a different one
        fails."""
        if registry is None:
            return None
        cert = ESGCertificate.issue(self.sponsor_id, pod.node_id, metric,
                                    amount, day, auditor, registry)
        if cert is not None:
            self.certificates.append(cert)
        return cert


# ===========================================================================
# 11. Federation
# ===========================================================================
@dataclass
class FederationTreaty:
    polity_a: str
    polity_b: str
    recognize_attestations: bool = True
    recognize_tc: bool = True
    recognize_personhood: bool = True
    cross_polity_fee_pct: float = 0.01
    active: bool = True
    signed_day: int = 0


class Federation:
    """v8: fees are deposited into fee_treasury (v7 silently burned them);
    personhood recognition never overwrites and requires the person to
    present the biometric to the target registry (per-registry peppers
    make hashes non-portable by design)."""

    def __init__(self, fee_treasury: Optional[Treasury] = None):
        self.treaties: Dict[Tuple[str, str], FederationTreaty] = {}
        self.fee_treasury = fee_treasury or Treasury()

    def sign(self, a: str, b: str, **kwargs) -> FederationTreaty:
        t = FederationTreaty(polity_a=a, polity_b=b, **kwargs)
        self.treaties[(a, b)] = t
        self.treaties[(b, a)] = t
        return t

    def get_treaty(self, a: str, b: str) -> Optional[FederationTreaty]:
        return self.treaties.get((a, b))

    def can_recognize_attestation(self, from_polity: str,
                                  to_polity: str) -> bool:
        if from_polity == to_polity:
            return True
        t = self.get_treaty(from_polity, to_polity)
        return bool(t and t.active and t.recognize_attestations)

    def cross_polity_transfer(self, src: HouseholdPod, dst: HouseholdPod,
                              amount: float, src_polity: Polity,
                              dst_polity: Polity) -> Dict:
        if src_polity.name == dst_polity.name:
            return {"status": "FAILED",
                    "reason": "use InterPodTransfer for same-polity"}
        t = self.get_treaty(src_polity.name, dst_polity.name)
        if t is None or not t.active or not t.recognize_tc:
            return {"status": "FAILED",
                    "reason": "no active treaty recognizing TC"}
        if amount <= 0 or src.transit_credit_balance < amount:
            return {"status": "FAILED", "reason": "insufficient balance"}
        base_fee = amount * 0.005
        treaty_fee = amount * t.cross_polity_fee_pct
        delivered = amount - base_fee - treaty_fee
        src.transit_credit_balance -= amount
        dst.credit(delivered)
        self.fee_treasury.deposit(base_fee + treaty_fee,
                                  f"fees {src_polity.name}->{dst_polity.name}")
        return {"status": "SETTLED", "amount": amount,
                "base_fee": round(base_fee, 4),
                "treaty_fee": round(treaty_fee, 4),
                "delivered": round(delivered, 4),
                "fees_to_treasury": round(base_fee + treaty_fee, 4),
                "from_polity": src_polity.name,
                "to_polity": dst_polity.name}

    def recognize_personhood(self, registry_a: PersonhoodRegistry,
                             registry_b: PersonhoodRegistry,
                             person_id: str,
                             biometric: Optional[str] = None) -> Dict:
        if person_id not in registry_a.persons:
            return {"status": "FAILED", "reason": "person not in registry_a"}
        if person_id in registry_b.persons:
            return {"status": "FAILED", "reason": "person_id exists in B"}
        att = registry_a.persons[person_id]
        if biometric is None or registry_a._hash(biometric) != att.biometric_hash:
            return {"status": "FAILED",
                    "reason": "person must present matching biometric"}
        return registry_b.import_attestation(att, biometric)

# ===========================================================================
# Demos
# ===========================================================================
POLITY = "River Watershed"


def _setup(seed: int = 42):
    """Honest bootstrap: a founding quorum of 3 registers together; every
    later person needs 2 DISTINCT existing sponsors (no ['p0','p0'])."""
    rng = random.Random(seed)
    registry = PersonhoodRegistry(vouches_required=2, genesis_quorum=3,
                                  pepper=b"demo-pepper-not-secret")
    registry.register_genesis([("p0", "bio0"), ("p1", "bio1"), ("p2", "bio2")])
    persons = ["p0", "p1", "p2"]
    for i in range(3, 6):
        sponsors = rng.sample(persons, 2)
        r = registry.register(f"p{i}", f"bio{i}", sponsors=sponsors,
                              in_person=True, day=0)
        if r["status"] == "REGISTERED":
            persons.append(f"p{i}")
    pods: List[HouseholdPod] = []
    for i in range(5):
        pid = f"p{i+1}"
        pod = HouseholdPod(f"POD-{i:03d}", transit_credit_balance=100.0,
                           person_id=pid)
        registry.bind_pod(pid, pod.node_id)
        pods.append(pod)
    return rng, registry, pods


def pod_report(out=print) -> Dict:
    m = PodModel()
    r = m.report()
    out("  Assumed pod: 2 persons @ 2,500 kcal, 10 m2 LED canopy (250 umol/m2/s, "
        "16 h, 2.7 umol/J), 6 kWp PV, 15 kWh battery")
    out(f"  indoor food:       {r['indoor_kcal_day']:.0f} kcal/day "
        f"= {r['calorie_share']:.1%} of {r['kcal_need_day']:.0f} kcal need "
        f"(light ceiling {r['light_ceiling_kcal_day']:.0f})")
    out(f"  grow electricity:  {r['grow_kwh_day']:.1f} kWh/day "
        f"({r['kwh_per_1000_kcal']:.0f} kWh per 1,000 kcal)")
    out(f"  load vs PV:        {r['load_kwh_day']:.1f} kWh/day vs "
        f"{r['pv_kwh_day_mean']:.1f} mean / {r['pv_kwh_day_worst_month']:.1f} Dec "
        f"-> needs {r['pv_kwp_needed_worst_month']:.1f} kWp for December")
    out(f"  water balance:     {r['water_balance_l_day']:+.0f} L/day")
    out(f"  capex:             ${r['capex_total_usd']:,} {r['capex_usd']}")
    out(f"  annualized cost:   ${r['annualized_cost_usd']:,}/yr "
        f"= ${r['annualized_cost_per_person_usd']:,}/person/yr")
    out(f"  energy_ok={r['energy_ok']} water_ok={r['water_ok']} "
        f"food_self_sufficient={r['food_self_sufficient']} "
        f"-> sustainable={r['sustainable']}")
    out(f"  FULL diet indoors, best case (2.6 kcal/mol): "
        f"{r['full_indoor_kwh_day_best_case']:.0f} kWh/day "
        f"= {r['full_indoor_pv_kwp_mean']:.0f} kWp PV (annual mean)")
    out(f"  FULL diet outdoors (potato, KR avg yield): "
        f"{r['full_outdoor_m2']:,} m2 for {r['persons']} persons")
    return r


def demo() -> None:
    print("=" * 88)
    print("SUBSTRATE v8 — FULL STACK DEMO   (Credit: humanity)")
    print("=" * 88)

    rng, registry, pods = _setup()
    log = TransparencyLog(privacy=True, k_anon=3)
    clock = LedgerClock()
    graph = NeighborGraph()
    for p in pods:
        graph.register_pod(p.node_id)
    for i, p in enumerate(pods):
        graph.link(p.node_id, pods[(i + 1) % len(pods)].node_id)
        p.polity = POLITY
    mesh = MeshConsensus(quorum=3, rate_limit_per_hour=4, clock=clock,
                         graph=graph)
    un = UNRecognition(signatories=["KR", "JP", "SG"])
    un.recognize(POLITY)
    agi = AGIDataEngine(un_recognition=un, transparency_log=log)

    print(f"\n[0] PERSONHOOD: genesis quorum 3, then 2 distinct sponsors each; "
          f"{len(registry.persons)} persons")
    r = registry.register("px", "biox", sponsors=["p0", "p0"], day=0)
    print(f"  duplicate vouch ['p0','p0']: {r['status']} ({r['reason']})")

    print("\n[1] MINTING (ledger clock, mutual neighbors, provenance quality)")
    for i, pod in enumerate(pods):
        clock.advance(1)
        peers = [q for q in pods if graph.are_neighbors(pod.node_id, q.node_id)]
        att = mesh.register(pod, "community_care", 6.0, peers)
        prov = ProvenanceEvidence(sensor_variance=0.30,
                                  quorum_attestations=len(att["accepted"]))
        act = MammalianActivity(
            domain="community_care", description="elder care",
            hours_logged=6.0, provenance=prov, peer_consensus_rating=0.90,
            pod_id=pod.node_id, person_id=pod.person_id, polity=POLITY)
        credits, err = agi.evaluate_and_mint(act, day=1)
        if err:
            print(f"  [-] {pod.node_id}: {err}")
        else:
            pod.credit(credits)
            print(f"  [+] {pod.node_id}: minted {credits:.2f} TC "
                  f"(provenance {prov.combined_entropy:.3f}, "
                  f"quorum {len(att['accepted'])}/{att['needed']})")
    noisy = ProvenanceEvidence(sensor_variance=0.92, quorum_attestations=2)
    _, err = agi.evaluate_and_mint(MammalianActivity(
        "community_care", "noisy sensor", 6.0, noisy, 0.9,
        pod_id=pods[0].node_id, person_id=pods[0].person_id), day=1)
    print(f"  [-] implausibly noisy sensor (v7 rewarded this): {err}")

    print("\n[2] DEMURRAGE (idle 4% + 10% on balance above 500 TC, everyone)")
    active, idle = pods[0], pods[1]
    active.credit(2000.0)
    idle.credit(2000.0)
    for m in range(1, 7):
        active.credit(400.0)       # active pod keeps earning
        r_a = active.step_month(is_active=True)
        r_i = idle.step_month(is_active=False)
        print(f"  month {m}: active={r_a['net_balance']:8.2f} | "
              f"idle={r_i['net_balance']:8.2f} (burned {r_i['decayed_burned']:.2f})")
    print(f"  active-pod bound at 400 TC/month inflow: "
          f"{active.balance_bound(400.0):.0f} TC")

    print("\n[3] MUNICIPAL ZONES (fragility-scaled tolls)")
    jeju = MunicipalZone("Jeju Volcanic Trail", base_tc_cost=30.0,
                         max_capacity=100, current_occupancy=85,
                         ecological_fragility=1.5)
    seorak = MunicipalZone("Seoraksan Rural Sanctuary", base_tc_cost=25.0,
                           max_capacity=200, current_occupancy=40,
                           ecological_fragility=0.8)
    print(f"  Jeju    (85% load, fragility 1.5): toll {jeju.current_transit_toll():.2f}")
    print(f"  Seorak  (20% load, fragility 0.8): toll {seorak.current_transit_toll():.2f}")

    print("\n[4] TRIP BOOKING")
    pod = pods[0]
    r = MunicipalLedger.process_trip(pod, jeju, pod.polity, un)
    print(f"  {pod.node_id} -> Jeju: {r['status']} | "
          f"{r.get('toll_paid', r.get('reason', ''))}")

    print("\n[5] ESCALATION LADDER (fictional/proposed mechanism, not a real UN process)")
    un.ladder.record_violation(POLITY, "UNP-04", 2)
    un.ladder.record_violation(POLITY, "UNP-04", 3)
    r = un.ladder.record_violation(POLITY, "UNP-01", 4)
    print(f"  after child-labor record (severity "
          f"{un.ladder.severity_of('UNP-01')}): stage {r['stage']} ({r['stage_name']})")
    a = un.ladder.appeal(POLITY, "record was a mislabeled school field trip", 6)
    print(f"  appeal filed: {a['status']} (needs {a['needs']} independent reviewers)")
    for rid, home in [("R1", "Other A"), ("R2", "Other B"), ("R3", POLITY), ("R4", "Other C")]:
        un.ladder.register_reviewer(rid, home)
    for rid in ["R1", "R3", "R2", "R4"]:
        v = un.ladder.review(a["appeal_id"], rid, True)
        extra = (f" ({v['reason']})" if "reason" in v else
                 f" (stage {v['from_stage']} -> {v['to_stage']})" if "to_stage" in v else "")
        print(f"   review {rid}: {v['status']}{extra}")
    un.ladder.record_violation(POLITY, "UNP-04", 7)
    print(f"  later minor violation; stage now {un.ladder.current_stage(POLITY)} "
          f"(appeal still respected)")
    un.ladder.record_violation(POLITY, "UNP-02", 8, detail="forced_labor incident")
    stage, name = un.ladder.current_stage(POLITY)
    print(f"  forced labor -> stage {stage} ({name})")
    r = MunicipalLedger.process_trip(pod, seorak, pod.polity, un)
    print(f"  {pod.node_id} -> Seorak: {r['status']} | {r.get('reason')}")

    print("\n[6] AUDITORS + SPONSORS (random assignment, HMAC certificates)")
    areg = AuditorRegistry(seed=7)
    a1 = areg.register("AUD-001", "GreenVerify", bond=2000.0)
    areg.register("AUD-002", "CareAudit", bond=1500.0)
    pool = SponsorPool()
    sponsor = CorporateSponsor("SP-001", "Korea Green Energy Co.",
                               annual_budget_tc=50_000.0, max_per_pod_tc=1_000.0,
                               focus_metrics=["carbon_kg", "care_hours"])
    for p in pods[:3]:
        r = sponsor.fund_pod(pool, p, 200.0, "Q1 ESG", day=1)
        print(f"  {sponsor.name} -> {p.node_id}: {r['status']}")
    cert = sponsor.issue_certificate(pods[0], "carbon_kg", 45.0, day=2, registry=areg)
    print(f"  cert {cert.certificate_id[:8]} assigned to {cert.auditor_id}; "
          f"verifies={areg.verify(cert)}")
    forged = ESGCertificate(**{**cert.__dict__, "amount": 4500.0})
    print(f"  tampered amount verifies={areg.verify(forged)}")
    r = areg.slash(cert.auditor_id, 300.0, "overturned")
    print(f"  slash: {r['status']} bond_after {r['bond_after']} "
          f"removed={r['removed']} (1 slash, {len(areg.auditors[cert.auditor_id].issued_certificates)} certs: "
          f"below minimum sample)")

    print("\n[7] MULTI-SPONSOR ATTRIBUTION")
    sponsor2 = CorporateSponsor("SP-002", "Frontier Alignment Lab",
                                annual_budget_tc=200_000.0, max_per_pod_tc=5_000.0,
                                focus_metrics=["care_hours"])
    sponsor2.fund_pod(pool, pods[0], 300.0, "Q1 alignment", day=1)
    print(f"  1000 TC to {pods[0].node_id}: {pool.attribution(pods[0], 1000.0)}")
    sid, share = pool.largest_share(pods[0])
    print(f"  largest sponsor: {sid} at {share:.0%}")

    print("\n[8] FEDERATION (fees to treasury)")
    home = Polity(POLITY, {p.node_id for p in pods})
    other = Polity("Island Watershed")
    other_pod = HouseholdPod("POD-ISL-01")
    other.admit(other_pod)
    fed = Federation()
    fed.sign(POLITY, "Island Watershed", recognize_tc=True, cross_polity_fee_pct=0.01)
    r = fed.cross_polity_transfer(pods[2], other_pod, 50.0, home, other)
    print(f"  transfer: {r['status']} | delivered {r.get('delivered')} | "
          f"fee treasury {fed.fee_treasury.balance:.2f}")

    print("\n[9] EXIT (settle balance, keep personhood unless deletion asked)")
    r = home.exit(pods[4], registry, log, delete_personhood=False, day=9)
    print(f"  {pods[4].node_id}: {r['status']} settled {r['settled_tc']} TC, "
          f"personhood kept={r['personhood_kept']}")
    r = home.exit(pods[3], registry, log, delete_personhood=True, day=9)
    print(f"  {pods[3].node_id}: {r['status']} settled {r['settled_tc']} TC, "
          f"erased {r['log_entries_erased']} log references")

    print("\n[10] TRANSPARENCY (pseudonymous, aggregated)")
    w = Watcher("W-001")
    log.add_watcher(w)
    log.append("test_event", {"x": 1}, day=10)
    sample = log.read_all("mint")[0]["payload"]
    print(f"  sample mint event: {sample}")
    print(f"  aggregate: {log.aggregate_report()}")
    print(f"  total transparency events: {len(log.events)}")

    print("\n[11] POD MODEL (sourced + labeled assumptions; see README_v8.md)")
    pod_report()

    print("\n[12] GOAL-WEIGHT CALORIE PLAN -> POD DEMAND (general guidance, not medical advice)")
    plan = plan_calories(175, 80, 40, "male", "moderate", goal_kg=72)
    print(f"  BMR {plan['bmr']} | TDEE {plan['tdee']} | target "
          f"{plan['target_kcal_day']} kcal/day | ~{plan['weeks_to_goal_est']} weeks")
    s = pod_supply_for_plan(plan)
    print(f"  one-person 10 m2 pod supplies {s['pod_kcal_day']} kcal = {s['pod_share']:.1%}")
    print(f"  underweight goal: {plan_calories(175, 80, 40, 'male', 'moderate', 55)['status']}")

    print("\n[13] HEALTH SNAPSHOT")
    print(f"  pods: {len(pods)} | registered persons: {len(registry.persons)}")
    print(f"  total TC in pods: {sum(p.transit_credit_balance for p in pods):.2f}")


def municipal_demo() -> None:
    print("=" * 88)
    print("MUNICIPAL DEMO — Fragility-scaled tolling")
    print("=" * 88)
    zones = [
        MunicipalZone("Robust City Park", 20.0, 500, 400, ecological_fragility=0.3),
        MunicipalZone("Balanced Forest Trail", 25.0, 200, 160, ecological_fragility=1.0),
        MunicipalZone("Fragile Alpine Ridge", 30.0, 50, 40, ecological_fragility=2.5),
    ]
    pod = HouseholdPod("POD-TEST", transit_credit_balance=1000.0)
    for z in zones:
        print(f"  {z.zone_name:<28} fragility {z.ecological_fragility:.1f} | "
              f"load {z.current_occupancy/z.max_capacity:.0%} | "
              f"toll {z.current_transit_toll():.2f} TC")
        r = MunicipalLedger.process_trip(pod, z)
        print(f"    -> {r['status']} | {r.get('toll_paid', 'N/A')}")


# ===========================================================================
# Built-in tests (v7 suite, updated where v8 intentionally changed behaviour;
# the full regression suite is test_substrate_v8.py)
# ===========================================================================
def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_unit_tests() -> None:
    print("Running substrate v8 built-in test suite...")

    # Max provenance. CHANGED: v7 used sensor_variance=1.0; v8 scores
    # plausibility, so an in-band variance gives quality 1.0.
    log = TransparencyLog()
    agi = AGIDataEngine(transparency_log=log)
    prov_max = ProvenanceEvidence(sensor_variance=0.3, quorum_attestations=3)
    _check(abs(prov_max.combined_entropy - 1.0) < 1e-9, "max provenance = 1.0")
    act = MammalianActivity(domain="community_care", description="test",
                            hours_logged=1.0, provenance=prov_max,
                            peer_consensus_rating=1.0, pod_id="P", polity="")
    credits, err = agi.evaluate_and_mint(act)
    _check(err is None, "max provenance accepted")
    _check(abs(credits - 36.0) < 1e-6, f"expected 36.0, got {credits}")

    # Demurrage. CHANGED: 1000 idle -> 960 (4% idle) then 10% of the 460
    # above the 500 threshold -> 914.
    pod = HouseholdPod("P", transit_credit_balance=1000.0)
    r = pod.step_month(is_active=False)
    _check(r["months_idle"] == 1, "idle counted")
    _check(abs(pod.transit_credit_balance - 914.0) < 0.01, "idle + holding decay")
    pod.credit(50.0)
    _check(pod.months_idle == 0, "credit resets idle")

    # Personhood uniqueness. CHANGED: genesis quorum instead of 1 bootstrap.
    reg = PersonhoodRegistry()
    _check(reg.register("p1", "same")["status"] == "FAILED", "no solo bootstrap")
    reg.register_genesis([("g1", "a"), ("g2", "b"), ("g3", "c")])
    r = reg.register("p1", "same", sponsors=["g1", "g2"])
    _check(r["status"] == "REGISTERED", "first")
    r = reg.register("p2", "same", sponsors=["g1", "g2"])
    _check(r["status"] == "FAILED", "sybil rejected")

    # Mesh neighbor rule. CHANGED: neighbors via the mutual NeighborGraph.
    pods = [HouseholdPod(f"P{i}") for i in range(5)]
    g = NeighborGraph()
    for p in pods:
        g.register_pod(p.node_id)
    for n in ("P1", "P2", "P3"):
        g.link("P0", n)
    mesh = MeshConsensus(quorum=3, rate_limit_per_hour=2, graph=g)
    r = mesh.register(pods[0], "community_care", 4.0, pods[1:5], hour=1)
    _check(r["quorum_met"], "quorum met")
    _check("P4" not in r["accepted"], "non-neighbor rejected")

    robust = MunicipalZone("R", 100.0, 100, 85, ecological_fragility=0.5)
    fragile = MunicipalZone("F", 100.0, 100, 85, ecological_fragility=2.0)
    _check(fragile.current_transit_toll() > robust.current_transit_toll(),
           "fragile costs more")

    un = UNRecognition(signatories=["KR"])
    un.ladder.record_violation("P", "UNP-02", 1, detail="forced_labor")
    stage, _ = un.ladder.current_stage("P")
    _check(stage >= 3, "forced labor -> stage >= 3")
    _check(not un.ladder.can_cross_border("P"), "cannot cross border")
    _check(not un.ladder.can_register_pods("P"), "decertified cannot register pods")

    pod2 = HouseholdPod("P2", transit_credit_balance=1000.0)
    zone = MunicipalZone("Z", 30.0, 100, 0, 1.0)
    r = MunicipalLedger.process_trip(pod2, zone, "P", un)
    _check(r["status"] == "DENIED", "trip denied by UN gate")

    areg = AuditorRegistry()
    try:
        areg.register("A1", "LowBond", bond=100.0)
        raise AssertionError("low bond should raise")
    except ValueError:
        pass

    a1 = areg.register("A1", "HighBond", bond=1000.0)
    _check(areg.is_valid("A1"), "valid auditor")
    for i in range(5):
        ESGCertificate.issue("SP", "POD", "carbon", 10.0, i, a1, areg)
    areg.slash("A1", 300.0, "disputed")
    areg.slash("A1", 300.0, "disputed")
    rate = a1.slash_count / max(1, len(a1.issued_certificates))
    _check(rate > areg.MAX_SLASH_RATE, f"slash rate {rate} > cap")
    # CHANGED: 2 slashes on 5 certs is below the minimum sample -> not removed
    _check(not a1.removed, "min-sample rule: not removed yet")

    pool = SponsorPool()
    p3 = HouseholdPod("P3")
    pool.contribute(p3, "A", 100.0, day=1, reason="esg")
    pool.contribute(p3, "B", 300.0, day=1, reason="data")
    attr = pool.attribution(p3, 1000.0)
    _check(abs(attr["A"] - 250.0) < 1e-6, "A gets 25%")
    _check(abs(attr["B"] - 750.0) < 1e-6, "B gets 75%")

    a = Polity("A"); b = Polity("B")
    pa = HouseholdPod("PA", transit_credit_balance=100)
    pb = HouseholdPod("PB")
    a.admit(pa); b.admit(pb)
    fed = Federation()
    r = fed.cross_polity_transfer(pa, pb, 50.0, a, b)
    _check(r["status"] == "FAILED", "no treaty = rejected")
    fed.sign("A", "B", cross_polity_fee_pct=0.01)
    r = fed.cross_polity_transfer(pa, pb, 50.0, a, b)
    _check(r["status"] == "SETTLED", "with treaty = settled")
    expected = 50 - 50 * 0.005 - 50 * 0.01
    _check(abs(pb.transit_credit_balance - expected) < 1e-6,
           f"delivered {pb.transit_credit_balance} vs {expected}")
    _check(abs(fed.fee_treasury.balance - 0.75) < 1e-9, "fees to treasury")

    # Federation personhood. CHANGED: biometric must be re-presented.
    reg_a = PersonhoodRegistry()
    reg_a.register_genesis([("person_x", "bio_x"), ("y", "by"), ("z", "bz")])
    reg_b = PersonhoodRegistry()
    _check(fed.recognize_personhood(reg_a, reg_b, "person_x")["status"] == "FAILED",
           "no biometric -> refused")
    r = fed.recognize_personhood(reg_a, reg_b, "person_x", biometric="bio_x")
    _check(r["status"] == "RECOGNIZED", "person recognized")
    _check("person_x" in reg_b.persons, "person in B registry")

    w = Watcher("W-1")
    log2 = TransparencyLog()
    log2.add_watcher(w)
    log2.append("evt", {"x": 1}, day=1)
    _check(len(w.read_log) == 1, "watcher sees event")
    _check("evt" in w.read_log, "correct event type")

    print("All checks passed.")


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--municipal-demo", action="store_true")
    p.add_argument("--pod-report", action="store_true")
    args = p.parse_args()

    if args.test:
        try:
            run_unit_tests()
            print("TEST SUITE: PASS")
            sys.exit(0)
        except AssertionError as e:
            print(f"TEST SUITE: FAIL — {e}")
            sys.exit(1)

    if args.municipal_demo:
        municipal_demo()
        return

    if args.pod_report:
        print(json.dumps(pod_report(out=lambda *_: None), indent=2))
        return

    demo()


if __name__ == "__main__":
    main()

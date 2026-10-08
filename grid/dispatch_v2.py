"""
REAL MODE v2: Wake-Aware Wind + Solar Dispatch with a Monte Carlo (probabilistic) forecast.
CC0 1.0. Python 3.9+, standard library only.

Fixes from the v1 review:
 1. Power curve P = Prated*(v^3 - vin^3)/(vr^3 - vin^3) between cut-in and rated speed.
 2. No fixed 0.25 MW floor. A turbine is curtailed only (a) to stay under the grid limit
    (local demand + export line capacity) or (b) when switching it off raises the farm total
    through wake recovery (brute-force search of on/off sets, fine for a few turbines).
    A switched-off rotor is feathered, not gone: it still leaves a weak wake (Ct = 0.05, ASSUMED).
 3. Partial wake overlap: deficit scaled by the share of the downstream rotor inside the wake.
 4. Several wakes combined with Katic et al. (1986): root of the sum of squared deficits.
 5. Solar: 25% commercial tandem module by default (32% kept only as LAB_TARGET), performance
    ratio 0.82, Kasten-Czeplak applied only to clear-sky GHI; measured GHI skips it.
 6. Honest names: WakeAwareDispatchEngine is a deterministic snapshot. Forecasting and
    uncertainty live in probabilistic_forecast() and run_monte_carlo().

CAMERA DATA: there is no real camera feed. Every camera input here is EXAMPLE/SIMULATED.
Error sizes (sigmas) are ASSUMPTIONS unless a source is listed next to them.
"""
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional, Sequence
import math
import random

SEED = 20261008
SIMULATED = "EXAMPLE/SIMULATED (no real camera feed)"
COMMERCIAL_TANDEM_EFF = 0.25  # Oxford PV: commercial tandem modules ~25% (pv magazine, 2026-06-18)
LAB_TARGET_TANDEM_EFF = 0.32  # LAB/TARGET only. LONGi certified tandem modules 29.4-31.4% (LONGi, 2026-07-14)
DEFAULT_PR = 0.82             # utility PR ~0.75-0.85; US federal fleet median 0.79 (US DOE 2022)
SOILING_IN_PR = 0.02          # soiling already inside PR; NREL fleet median annual soiling 2-3%
FEATHERED_CT = 0.05           # ASSUMED thrust coefficient of a feathered (switched-off) rotor
YAW_EXPONENT = 1.88           # power ~ cos(yaw)^p; literature p ~1.4-3, 1.88 = FLORIS default (NREL forum)
WAKE_SWITCH_MIN_GAIN_MW = 0.05  # ASSUMED: switch a turbine off only if the model gain beats this (model noise)


@dataclass
class TurbineUnit:
    id: str
    x: float  # m, along the reference wind direction (wind_dir_deg = 0 blows toward +x)
    y: float  # m, crosswind
    rotor_diameter: float = 120.0
    thrust_coeff: float = 0.8  # constant Ct while running (simplification)
    cut_in_speed: float = 3.0
    rated_speed: float = 11.5
    cut_out_speed: float = 25.0
    rated_capacity_mw: float = 3.5
    wake_decay_k: float = 0.04  # offshore/flat ~0.04-0.05
    feathered_ct: float = FEATHERED_CT


@dataclass
class SolarArray:
    id: str
    area_m2: float
    module_efficiency: float = COMMERCIAL_TANDEM_EFF
    performance_ratio: float = DEFAULT_PR  # heat, inverter, wiring, average soiling


@dataclass
class SiteConditions:
    wind_speed_ms: float
    wind_dir_deg: float = 0.0
    clear_sky_ghi: float = 0.0             # W/m2, cloud attenuation applied to this
    cloud_cover: float = 0.0               # 0..1 (= okta/8)
    measured_ghi: Optional[float] = None   # pyranometer value: already includes clouds
    soiling_loss: Optional[float] = None   # None = use the average inside the PR
    snow_cover: float = 0.0                # share of panel area covered
    yaw_error_deg: Dict[str, float] = field(default_factory=dict)
    iced: FrozenSet[str] = frozenset()
    local_demand_mw: Optional[float] = None
    export_capacity_mw: Optional[float] = None
    camera_sources: List[str] = field(default_factory=list)


# ---------- physics helpers ----------
def power_curve_mw(t: TurbineUnit, v: float) -> float:
    if v < t.cut_in_speed or v >= t.cut_out_speed:
        return 0.0
    if v >= t.rated_speed:
        return t.rated_capacity_mw
    vin3 = t.cut_in_speed ** 3
    return t.rated_capacity_mw * (v ** 3 - vin3) / (t.rated_speed ** 3 - vin3)


def yaw_factor(yaw_deg: float) -> float:
    return max(0.0, math.cos(math.radians(yaw_deg))) ** YAW_EXPONENT


def circle_overlap_fraction(r_wake: float, r_rotor: float, d: float) -> float:
    """Share of the rotor disc (r_rotor) inside the wake disc (r_wake); centres d apart."""
    if d >= r_wake + r_rotor:
        return 0.0
    if d <= abs(r_wake - r_rotor):
        return 1.0 if r_wake >= r_rotor else (r_wake / r_rotor) ** 2
    c1 = max(-1.0, min(1.0, (d * d + r_wake ** 2 - r_rotor ** 2) / (2 * d * r_wake)))
    c2 = max(-1.0, min(1.0, (d * d + r_rotor ** 2 - r_wake ** 2) / (2 * d * r_rotor)))
    k = (-d + r_wake + r_rotor) * (d + r_wake - r_rotor) * (d - r_wake + r_rotor) * (d + r_wake + r_rotor)
    lens = r_wake ** 2 * math.acos(c1) + r_rotor ** 2 * math.acos(c2) - 0.5 * math.sqrt(max(0.0, k))
    return lens / (math.pi * r_rotor ** 2)


def katic_combine(deficits: Sequence[float]) -> float:
    return math.sqrt(sum(d * d for d in deficits))


def wake_geometry_factor(up: TurbineUnit, down: TurbineUnit, wind_dir_deg: float) -> float:
    """(r0/rw)^2 * overlap share. Jensen deficit = (1 - sqrt(1 - Ct)) * this. 0 if not downstream."""
    th = math.radians(wind_dir_deg)
    ex, ey = down.x - up.x, down.y - up.y
    dx = ex * math.cos(th) + ey * math.sin(th)
    if dx <= 0:
        return 0.0
    dy = -ex * math.sin(th) + ey * math.cos(th)
    r0 = up.rotor_diameter / 2.0
    rw = r0 + up.wake_decay_k * dx
    return (r0 / rw) ** 2 * circle_overlap_fraction(rw, down.rotor_diameter / 2.0, abs(dy))


def kasten_czeplak(clear_sky_ghi: float, cloud_cover: float) -> float:
    """Kasten & Czeplak (1980): G(N)/G(0) = 1 - 0.75 (N/8)^3.4. Only valid on clear-sky GHI."""
    return clear_sky_ghi * (1.0 - 0.75 * min(1.0, max(0.0, cloud_cover)) ** 3.4)


def solar_mw(s: SolarArray, ghi: float, soiling: Optional[float] = None, snow: float = 0.0) -> float:
    soil_adj = 1.0 if soiling is None else (1.0 - soiling) / (1.0 - SOILING_IN_PR)
    return s.area_m2 * ghi * s.module_efficiency * s.performance_ratio * soil_adj * (1.0 - snow) / 1e6


# ---------- deterministic snapshot engine ----------
class WakeAwareDispatchEngine:
    """One snapshot: given site conditions, choose on/off turbines and grid curtailment. No forecasting."""

    def __init__(self, turbines: List[TurbineUnit], solar_units: List[SolarArray], max_bruteforce: int = 12,
                 min_gain_mw: float = WAKE_SWITCH_MIN_GAIN_MW):
        self.turbines = sorted(turbines, key=lambda t: t.x)
        self.solar_units = list(solar_units)
        self.max_bruteforce = max_bruteforce
        self.min_gain_mw = min_gain_mw
        self._cf_on = [1.0 - math.sqrt(1.0 - t.thrust_coeff) for t in self.turbines]
        self._cf_off = [1.0 - math.sqrt(1.0 - t.feathered_ct) for t in self.turbines]

    def wake_geometry(self, wind_dir_deg: float) -> List[List[float]]:
        ts = self.turbines
        return [[0.0 if i == j else wake_geometry_factor(ts[i], ts[j], wind_dir_deg)
                 for j in range(len(ts))] for i in range(len(ts))]

    def wind_state(self, v_amb, geom, on, yaw=None):
        """Katic-combined wakes. Returns (total MW, [(local wind m/s, MW) per turbine])."""
        out, total = [], 0.0
        for j, t in enumerate(self.turbines):
            s = 0.0
            for i in range(len(self.turbines)):
                g = geom[i][j]
                if g:
                    d = (self._cf_on[i] if i in on else self._cf_off[i]) * g
                    s += d * d
            v = max(0.0, v_amb * (1.0 - math.sqrt(s)))
            p = power_curve_mw(t, v) * yaw_factor((yaw or {}).get(t.id, 0.0)) if j in on else 0.0
            out.append((v, p))
            total += p
        return total, out

    def best_on_set(self, v_amb, geom, iced_idx=frozenset(), yaw=None):
        """Brute-force on/off search. Returns (on set, wind MW, best model gain over all-on).
        Turbines stay on unless switching some off gains more than min_gain_mw."""
        cand = [i for i in range(len(self.turbines)) if i not in iced_idx]
        full = best = frozenset(cand)
        full_total = best_total = self.wind_state(v_amb, geom, full, yaw)[0]
        if len(cand) > self.max_bruteforce or not any(any(r) for r in geom):
            return full, full_total, 0.0
        for mask in range((1 << len(cand)) - 1):
            on = frozenset(c for k, c in enumerate(cand) if mask >> k & 1)
            tot = self.wind_state(v_amb, geom, on, yaw)[0]
            if tot > best_total + 1e-9 or (abs(tot - best_total) <= 1e-9 and len(on) > len(best)):
                best, best_total = on, tot
        gain = best_total - full_total
        return (best, best_total, gain) if gain > self.min_gain_mw else (full, full_total, gain)

    def solar_total(self, c: SiteConditions):
        ghi = c.measured_ghi if c.measured_ghi is not None else kasten_czeplak(c.clear_sky_ghi, c.cloud_cover)
        return ghi, sum(solar_mw(s, ghi, c.soiling_loss, c.snow_cover) for s in self.solar_units)

    @staticmethod
    def grid_limit(c: SiteConditions) -> Optional[float]:
        parts = [p for p in (c.local_demand_mw, c.export_capacity_mw) if p is not None]
        return sum(parts) if parts else None

    def dispatch(self, c: SiteConditions, setpoint_mw: Optional[float] = None) -> Dict:
        geom = self.wake_geometry(c.wind_dir_deg)
        iced_idx = frozenset(i for i, t in enumerate(self.turbines) if t.id in c.iced)
        on, _, gain = self.best_on_set(c.wind_speed_ms, geom, iced_idx, c.yaw_error_deg)
        wind_total, per = self.wind_state(c.wind_speed_ms, geom, on, c.yaw_error_deg)
        ghi, solar_total = self.solar_total(c)
        available = wind_total + solar_total
        limits = [x for x in (self.grid_limit(c), setpoint_mw) if x is not None]
        limit = min(limits) if limits else None
        scale = 1.0 if limit is None or available <= limit else max(0.0, limit) / available
        status = {}
        for j, t in enumerate(self.turbines):
            v, p = per[j]
            if j in iced_idx:
                state = "STOPPED (ICING FLAG)"
            elif j not in on:
                state = "CURTAILED (WAKE RECOVERY)"
            elif v >= t.cut_out_speed:
                state = "STOPPED (CUT-OUT)"
            elif v < t.cut_in_speed:
                state = "IDLE (BELOW CUT-IN)"
            else:
                state = "ATTACHED" if scale == 1.0 else "ATTACHED (GRID-LIMITED)"
            status[t.id] = {"state": state, "effective_wind_ms": round(v, 3),
                            "available_mw": round(p, 4), "yield_mw": round(p * scale, 4)}
        return {
            "model": "REAL MODE dispatch v2 (deterministic snapshot)",
            "inputs": {"wind_ms": c.wind_speed_ms, "wind_dir_deg": c.wind_dir_deg,
                       "ghi_used_w_m2": round(ghi, 2),
                       "ghi_source": "measured" if c.measured_ghi is not None else "clear-sky x Kasten-Czeplak",
                       "cloud_cover": c.cloud_cover, "camera_sources": list(c.camera_sources)},
            "generation_mw": {"solar_available": round(solar_total, 4), "wind_available": round(wind_total, 4),
                              "total_available": round(available, 4), "grid_limit": limit,
                              "delivered": round(available * scale, 4),
                              "grid_curtailed": round(available * (1 - scale), 4)},
            "wake_search": {"best_model_gain_mw": round(gain, 4), "threshold_mw_ASSUMED": self.min_gain_mw},
            "wake_curtailed_turbines": [self.turbines[i].id for i in range(len(self.turbines))
                                        if i not in on and i not in iced_idx],
            "turbine_allocation": status,
        }


# ---------- camera layer: interfaces only, all data EXAMPLE/SIMULATED ----------
@dataclass
class SkyImagerNowcast:
    """All-sky imager cloud nowcast, 5-30 min ahead (SIMULATED)."""
    cloud_cover: float
    sigma: float
    lead_time_min: int = 15
    label: str = SIMULATED

    def __post_init__(self):
        if not 5 <= self.lead_time_min <= 30 or not 0.0 <= self.cloud_cover <= 1.0:
            raise ValueError("lead time must be 5-30 min and cloud cover 0..1")


@dataclass
class PanelCCTVSoiling:
    """CCTV estimate of panel dirt and snow (SIMULATED)."""
    soiling_loss: float
    snow_cover: float = 0.0
    sigma: float = 0.005
    label: str = SIMULATED


@dataclass
class TurbineCameraCheck:
    """Nacelle CCTV/lidar: yaw misalignment and icing flag (SIMULATED)."""
    turbine_id: str
    yaw_error_deg: float = 0.0
    icing: bool = False
    label: str = SIMULATED


@dataclass
class TrafficCountFeed:
    """Anonymous vehicle COUNTS from LPR/Flock-style cameras, used only to forecast EV charging demand.
    Privacy rules: (1) this type holds one integer count per bin of 15 min or more and nothing else;
    (2) counting happens on the camera, and plates, images and per-vehicle records are never sent to
    or stored by this system; (3) counts are used only for energy forecasting, never joined with other
    data or shared for enforcement; (4) a counting-only mode or plain traffic counters are preferred,
    because LPR products store plate reads themselves (Flock's stated default retention is 7 days)."""
    vehicles_per_hour: int
    bin_minutes: int = 15
    label: str = SIMULATED

    def __post_init__(self):
        if isinstance(self.vehicles_per_hour, bool) or not isinstance(self.vehicles_per_hour, int) \
                or self.vehicles_per_hour < 0:
            raise ValueError("counts only: a non-negative integer")
        if self.bin_minutes < 15:
            raise ValueError("bins shorter than 15 min are not allowed (privacy)")


def ev_charging_mw(vehicles_per_hour: float, ev_share=0.10, stop_share=0.15, kwh_per_session=40.0) -> float:
    """ASSUMED conversion: vehicles/h x EV share x share that stop to charge x kWh per session."""
    return max(0.0, vehicles_per_hour) * ev_share * stop_share * kwh_per_session / 1000.0


def apply_cameras(c: SiteConditions, sky=None, panel=None, checks=(), traffic=None,
                  base_demand_mw=None) -> SiteConditions:
    """Fold (SIMULATED) camera estimates into the site conditions."""
    upd, src = {}, list(c.camera_sources)
    if sky is not None:
        upd["cloud_cover"] = sky.cloud_cover
        src.append(f"sky imager ({sky.label})")
    if panel is not None:
        upd.update(soiling_loss=panel.soiling_loss, snow_cover=panel.snow_cover)
        src.append(f"panel CCTV ({panel.label})")
    if checks:
        upd["yaw_error_deg"] = {**c.yaw_error_deg, **{k.turbine_id: k.yaw_error_deg for k in checks}}
        upd["iced"] = frozenset(c.iced) | {k.turbine_id for k in checks if k.icing}
        src.append(f"turbine CCTV/lidar ({checks[0].label})")
    if traffic is not None:
        upd["local_demand_mw"] = (base_demand_mw or 0.0) + ev_charging_mw(traffic.vehicles_per_hour)
        src.append(f"traffic counts only ({traffic.label})")
    return replace(c, camera_sources=src, **upd)


# ---------- statistics ----------
def percentile(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def describe(values: Sequence[float]) -> Dict[str, float]:
    """pct10/pct50/pct90 are plain percentiles. (In energy-yield jargon 'P90' = our pct10.)"""
    return {"mean": round(sum(values) / len(values), 4), "pct10": round(percentile(values, 0.1), 4),
            "pct50": round(percentile(values, 0.5), 4), "pct90": round(percentile(values, 0.9), 4)}


def rmse(errors: Sequence[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def probabilistic_forecast(engine, point: SiteConditions, wind_sigma=1.2, dir_sigma=8.0,
                           cloud_sigma=0.20, n=5000, seed=SEED) -> Dict:
    """Spread a point forecast into a distribution of available MW. Sigmas are ASSUMED;
    pass a SkyImagerNowcast.sigma as cloud_sigma when an imager is available."""
    rng = random.Random(seed)
    tot, sol, wnd = [], [], []
    for _ in range(n):
        c = replace(point, wind_speed_ms=max(0.0, rng.gauss(point.wind_speed_ms, wind_sigma)),
                    wind_dir_deg=rng.gauss(point.wind_dir_deg, dir_sigma),
                    cloud_cover=min(1.0, max(0.0, rng.gauss(point.cloud_cover, cloud_sigma))),
                    local_demand_mw=None, export_capacity_mw=None)
        g = engine.dispatch(c)["generation_mw"]
        tot.append(g["total_available"]); sol.append(g["solar_available"]); wnd.append(g["wind_available"])
    return {"n": n, "seed": seed, "sigmas_ASSUMED": {"wind_ms": wind_sigma, "dir_deg": dir_sigma,
            "cloud": cloud_sigma}, "total_mw": describe(tot), "solar_mw": describe(sol), "wind_mw": describe(wnd)}


# ---------- Monte Carlo: without vs with (SIMULATED) cameras ----------
TRUTH = {  # ALL ASSUMED: what the "real" hour looks like, same draws for both scenarios
    "weibull_c_ms": 9.0, "weibull_k": 2.0, "dir_sigma_deg": 15.0, "cloud_beta": (0.8, 1.6),
    "soiling_beta": (2.0, 98.0), "yaw_sigma_deg": 6.0, "icing_prob": 0.01, "clear_sky_ghi": 850.0,
    "base_demand_mw": (6.0, 1.0), "vehicles_per_hour": (2000.0, 500.0), "export_capacity_mw": 8.0,
    "interval_h": 1.0, "margin_z": 1.2816}
SCENARIOS = {  # forecast error sizes, ALL ASSUMED. Cameras do not measure wind speed, so wind error is equal.
    "without_cameras": {"wind_ms": 1.2, "dir_deg": 8.0, "cloud": 0.20, "soiling": None, "yaw_deg": None,
                        "icing_detect": 0.0, "vph": None, "base_mw": 0.4},
    "with_cameras_SIMULATED": {"wind_ms": 1.2, "dir_deg": 8.0, "cloud": 0.16, "soiling": 0.005,
                               "yaw_deg": 1.5, "icing_detect": 0.95, "vph": 100.0, "base_mw": 0.4}}


def demo_farm():
    """The original v1 layout: 4 x 3.5 MW in a row plus a 2.5 ha tandem array."""
    return ([TurbineUnit("WTG_01", 0, 0), TurbineUnit("WTG_02", 350, 10),
             TurbineUnit("WTG_03", 700, 20), TurbineUnit("WTG_04", 1200, 300)],
            [SolarArray("PV_TANDEM_ALPHA", 25000)])


def _draw_truth(r, ids):
    T = TRUTH
    return {"v": r.weibullvariate(T["weibull_c_ms"], T["weibull_k"]), "dir": r.gauss(0.0, T["dir_sigma_deg"]),
            "cloud": r.betavariate(*T["cloud_beta"]), "soil": r.betavariate(*T["soiling_beta"]),
            "yaw": {i: r.gauss(0.0, T["yaw_sigma_deg"]) for i in ids},
            "iced": frozenset(k for k in range(len(ids)) if r.random() < T["icing_prob"]),
            "base": max(0.0, r.gauss(*T["base_demand_mw"])), "vph": max(0.0, r.gauss(*T["vehicles_per_hour"]))}


def run_monte_carlo(n=100_000, seed=SEED) -> Dict:
    eng = WakeAwareDispatchEngine(*demo_farm())
    ids = [t.id for t in eng.turbines]
    T, rt = TRUTH, random.Random(seed)
    rfs = {name: random.Random(seed + k + 1) for k, name in enumerate(SCENARIOS)}
    keys = ("delivered", "solar", "wind", "curtailed_mwh", "e_deliv", "e_avail", "e_solar", "e_wind", "e_demand")
    acc = {name: {k: [] for k in keys} for name in SCENARIOS}
    viol = {name: 0 for name in SCENARIOS}
    wake_cut = {name: 0 for name in SCENARIOS}
    vph_mu, vph_sd = T["vehicles_per_hour"]
    for _ in range(n):
        tr = _draw_truth(rt, ids)
        g_true = eng.wake_geometry(tr["dir"])
        sol_true = sum(solar_mw(s, kasten_czeplak(T["clear_sky_ghi"], tr["cloud"]), tr["soil"])
                       for s in eng.solar_units)
        dem_true = tr["base"] + ev_charging_mw(tr["vph"])
        lim_true = dem_true + T["export_capacity_mw"]
        for name, sc in SCENARIOS.items():
            r = rfs[name]
            v_fc = max(0.0, tr["v"] + r.gauss(0.0, sc["wind_ms"]))
            g_fc = eng.wake_geometry(tr["dir"] + r.gauss(0.0, sc["dir_deg"]))
            cloud_fc = min(1.0, max(0.0, tr["cloud"] + r.gauss(0.0, sc["cloud"])))
            soil_fc = None if sc["soiling"] is None else min(1.0, max(0.0, tr["soil"] + r.gauss(0, sc["soiling"])))
            yaw_fc = {} if sc["yaw_deg"] is None else {i: y + r.gauss(0.0, sc["yaw_deg"]) for i, y in tr["yaw"].items()}
            iced_fc = frozenset(k for k in tr["iced"] if r.random() < sc["icing_detect"])
            vph_fc, vph_sig = (vph_mu, vph_sd) if sc["vph"] is None else (tr["vph"] + r.gauss(0, sc["vph"]), sc["vph"])
            dem_fc = max(0.0, tr["base"] + r.gauss(0.0, sc["base_mw"])) + ev_charging_mw(vph_fc)
            margin = T["margin_z"] * math.hypot(sc["base_mw"], ev_charging_mw(vph_sig))
            setpoint = dem_fc + T["export_capacity_mw"] - margin
            on, wind_fc, _ = eng.best_on_set(v_fc, g_fc, iced_fc, yaw_fc)
            sol_fc = sum(solar_mw(s, kasten_czeplak(T["clear_sky_ghi"], cloud_fc), soil_fc) for s in eng.solar_units)
            avail_fc = wind_fc + sol_fc
            wind_true = eng.wind_state(tr["v"], g_true, on - tr["iced"], tr["yaw"])[0]
            avail_true = wind_true + sol_true
            delivered = min(avail_true, setpoint, lim_true)
            viol[name] += min(avail_true, setpoint) > lim_true + 1e-9
            wake_cut[name] += len(on) < len(ids) - len(iced_fc)
            a = acc[name]
            a["delivered"].append(delivered); a["solar"].append(sol_true); a["wind"].append(wind_true)
            a["curtailed_mwh"].append((avail_true - delivered) * T["interval_h"])
            a["e_deliv"].append(min(avail_fc, setpoint) - delivered); a["e_avail"].append(avail_fc - avail_true)
            a["e_solar"].append(sol_fc - sol_true); a["e_wind"].append(wind_fc - wind_true)
            a["e_demand"].append(dem_fc - dem_true)
    out = {"label": "Monte Carlo on SIMULATED camera data; all distributions and sigmas ASSUMED",
           "n": n, "seed": seed, "truth_assumptions": TRUTH, "scenario_sigmas_assumed": SCENARIOS, "scenarios": {}}
    for name, a in acc.items():
        out["scenarios"][name] = {
            "delivered_mw": describe(a["delivered"]), "solar_mw": describe(a["solar"]),
            "wind_mw": describe(a["wind"]), "grid_curtailed_mwh_per_interval": describe(a["curtailed_mwh"]),
            "rmse_mw": {k[2:]: round(rmse(a[k]), 4) for k in keys if k.startswith("e_")},
            "bias_mw": {k[2:]: round(sum(a[k]) / n, 4) for k in keys if k.startswith("e_")},
            "grid_limit_breach_share": round(viol[name] / n, 5), "wake_curtailment_share": round(wake_cut[name] / n, 5)}
    base, cam = (out["scenarios"][k]["rmse_mw"] for k in SCENARIOS)
    out["rmse_reduction_pct_SIMULATED"] = {k: round(100 * (1 - cam[k] / base[k]), 2) if base[k] else None for k in base}
    return out


if __name__ == "__main__":
    import json
    import sys
    eng = WakeAwareDispatchEngine(*demo_farm())
    demo = SiteConditions(wind_speed_ms=4.8, clear_sky_ghi=850.0, cloud_cover=0.15)
    print(json.dumps(eng.dispatch(demo), indent=2))
    sky = SkyImagerNowcast(cloud_cover=0.15, sigma=0.16, lead_time_min=15)  # SIMULATED
    print(json.dumps(probabilistic_forecast(eng, apply_cameras(demo, sky=sky), cloud_sigma=sky.sigma, n=2000), indent=2))
    print(json.dumps(run_monte_carlo(int(sys.argv[1]) if len(sys.argv) > 1 else 20_000), indent=2))

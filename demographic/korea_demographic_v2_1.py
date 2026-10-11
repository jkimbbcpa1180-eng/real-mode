#!/usr/bin/env python3
"""Real Mode Korea demographic scenario simulator, v2.1 (Python 3.9+, stdlib).

v2.1 CHANGES (copy of v2.0; original file untouched):
 * Demo recalibrated toward 2025 Korea (see CALIBRATION in demo_config):
   pop 51.7M, 0-14 10.1%, 65+ 20.7%, TFR 0.80, births ~254k, deaths ~363k,
   life expectancy F 86.6 / M 80.8, sex ratio at birth 105.8. Still APPROXIMATE.
 * Six policy scenarios, each low/central/high. EFFECT SIZES ARE GUESSES,
   loosely anchored to published evidence (POLICY_EVIDENCE). Not forecasts.
 * Validation: scenario paths must have exactly the coefficient keys (typos no
   longer silently become zero effect); immigration bounded; ramp integer.
 * Median age is interpolated within the median single-year group.

This patches the pasted Korea Demographic + Surveillance Simulator. It is a
standalone demographic extension, not a modification of the attached Real Mode
probability engine. No TP, calibrated probabilities, or extinction forecasts
are manufactured. Default data and policy effects are SYNTHETIC demonstrations.

Run:
  python korea_demographic_realmode.py --test
  python korea_demographic_realmode.py --years 100 --scenario baseline
  python korea_demographic_realmode.py --scenario pro_natal --json
  python korea_demographic_realmode.py --print-template > inputs.json
  python korea_demographic_realmode.py --input inputs.json --years 30 --json

INPUT SCHEMA: --print-template emits the complete documented-by-example schema.
Replace the synthetic population, female/male annual death probabilities, ASFR,
flows, and provenance with observed age-sex-region data. All vectors cover single
years of age 0..max_age inclusive; max_age is an OPEN terminal age group. ASFR
units are births per female person-year, NOT births per 1,000 women. Supply zero
outside reproductive ages. Seoul means Seoul CITY, not the capital region.
All input series share start_year; provenance should state source, observation
date, geography, units and retrieval date. This program cannot authenticate it.

EQUATIONS:
 F_rt = Fmax * sigmoid(logit(F_r0/Fmax) + sum(beta_k * delta_X_rkt))
 f_rat = F_rt * f_ra0 / sum_a(f_ra0)   [fixed fertility-age pattern]
 S_ras = (1-q_ras)^mortality_multiplier
 E_raF = N_raF * (1 + S_raF)/2         [trapezoidal annual exposure]
 B_rt = sum_a(E_raF * f_rat)
 P_next = P + B - D + immigration - emigration

Annual timing: record initial population; births use start-of-year cohorts with
approximate within-year mortality exposure; survivors age one year (terminal
survivors remain); newborns receive half-year survival; domestic migration,
emigration, then immigration occur at year end. Domestic flows are simultaneous
and preserve age/sex. Births/deaths/flows on each output row describe the interval
ending at that row's year, with an explicit period_start. Initial flows are null.
National TFR is the sum of age-specific national births/exposures, never weighted
by total regional population. It is null if a fertility age has no exposure.

Scenario coefficients are signed changes in log odds of TFR/Fmax per unit of
driver change. Positive beta raises fertility. Driver values are standardized
scenario units, not automatically measured percentages. Illustrative policy paths
ramp over 20 years then stop, rather than compounding for a century. The demo
surveillance beta is zero: evidence is not established. Changing it is sensitivity
analysis only. No artificial deprivation floor, marriage-pool doom feedback,
immigration minimum, or population cutoff. Pro-natal changes fertility drivers;
immigration_support changes immigration separately. Policy success is not assumed.

LIMITS: annual deterministic expected counts (fractional people); no endogenous
economy, causal estimation, fertility timing shifts, migration selection, or
uncertainty intervals. The mortality multiplier and supplied life tables remain
constant. A scenario is NOT a backtested Korean forecast. Population stability
does not ensure wellbeing or a sustainable age structure.

Reference pointers (not sources for the synthetic defaults):
 OECD, Korea's Unborn Future (2025):
 https://www.oecd.org/en/publications/korea-s-unborn-future_005ce8f7-en.html
 Statistics Korea population projections, 2022-2072:
 https://www.kostat.go.kr/board.es?act=view&bid=207&list_no=428476&mid=a10301020100

PATCH NOTES: baseline-anchored fertility can rise or fall; sex-specific cohorts;
hazard-scaled survival; open terminal age; correct initial year/horizon; age-
preserving migration; age-exposure national TFR; source-labelled inputs; finite
input validation; annual accounting assertions; isolated policy scenarios.
The old births expressions (TFR/2)/years*N and TFR/years*(N/2) were equivalent.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

VERSION = "2.1"
SEXES = ("female", "male")
BASE_SCENARIOS = ("baseline", "pro_natal", "surveillance_heavy", "immigration_support")
# GUESSED full-effect (after the 20-year ramp) relative change in TFR, by
# variant (low, central, high), and the drivers each policy moves.
POLICIES = {
    "homes_for_families": (("housing",), (0.05, 0.12, 0.25)),
    "work_hours": (("work_conflict",), (0.03, 0.08, 0.15)),
    "end_hagwon": (("child_cost",), (0.03, 0.10, 0.20)),
    "equal_pay_subcontractors": (("job_insecurity",), (0.02, 0.06, 0.12)),
    "free_care": (("support", "child_cost"), (0.05, 0.12, 0.25)),
}
VARIANTS = ("low", "central", "high")
POLICY_SCENARIOS = tuple(f"{p}_{v}" for p in list(POLICIES) + ["all_combined"] for v in VARIANTS)
SCENARIOS = BASE_SCENARIOS + POLICY_SCENARIOS
POLICY_EVIDENCE = {
    "status": "GUESSES. Ranges are judgement calls informed by, not estimated from, these sources.",
    "overall": "OECD (2025) Korea's Unborn Future: its reform scenario ASSUMES TFR 0.72 (2023) -> 1.1 by 2040. "
               "all_combined central lands near/above that path; high is optimistic.",
    "homes_for_families": "OECD 2025 cites Kang (2022): doubling of house prices 2013-19 cut marriage likelihood 4-5.7%; "
                          "Fluchtmann et al. (2023, OECD): housing spending lowers fertility across OECD.",
    "work_hours": "OECD 2025: 68->52h cap (2018-21) had no visible fertility effect yet (masked by COVID). "
                  "Paid-leave expansions abroad: e.g. Lalive & Zweimueller (2009, Austria) a few % more births.",
    "end_hagwon": "OECD 2025: ~80% of schoolchildren in private tutoring, ~10% of disposable income. "
                  "No causal TFR estimate for abolishing it; range is a guess.",
    "equal_pay_subcontractors": "OECD 2025: labour-market dualism drives tutoring race and job insecurity. "
                                "No direct TFR estimate; smallest guessed effect.",
    "free_care": "Rindfuss et al. (2010, Norway): childcare expansion raised completed fertility substantially; "
                 "Raute (2019, Germany Elterngeld) and Milligan (2005, Quebec) found +10-25% in some groups. "
                 "OECD 2025: Korea's 2013 free-childcare reform had 'underwhelming' fertility effects.",
    "all_combined": "Sum of the five in log-odds (assumes no overlap; free_care and end_hagwon both move child_cost, "
                    "so this may OVERSTATE the combined effect).",
}


def number(x, label, low=None, high=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(label + " must be a finite number")
    if low is not None and x < low or high is not None and x > high:
        raise ValueError(label + " outside permitted range")
    return float(x)


def integer(x, label, low=0):
    number(x, label, low)
    if int(x) != x:
        raise ValueError(label + " must be an integer")
    return int(x)


def vector(v, n, label, high=None):
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"{label} must have {n} age entries")
    for x in v:
        number(x, label, 0, high)


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def fertility(base, ceiling, eta):
    number(base, "baseline TFR", 0)
    number(ceiling, "fertility ceiling", 0)
    number(eta, "fertility predictor")
    if not 0 < base < ceiling:
        raise ValueError("Require 0 < baseline TFR < fertility ceiling")
    return ceiling * sigmoid(math.log(base / (ceiling - base)) + eta)


def survival(q, multiplier):
    number(q, "annual death probability", 0, 1)
    number(multiplier, "mortality multiplier", 0)
    return (1 - q) ** multiplier if multiplier else 1.0


def validate(c):
    integer(c["start_year"], "start_year", 1)
    n = integer(c["max_age"], "max_age", 1) + 1
    number(c["female_birth_fraction"], "female birth fraction", 0, 1)
    number(c["mortality_multiplier"], "mortality multiplier", 0)
    number(c["fertility_ceiling"], "fertility ceiling", 0)
    if not isinstance(c["provenance"], dict) or not c["provenance"]:
        raise ValueError("Nonempty provenance object required")
    if not isinstance(c["regions"], list) or not c["regions"]:
        raise ValueError("At least one region required")
    names = [r["name"] for r in c["regions"]]
    if any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
        raise ValueError("Region names must be unique nonempty strings")
    for r in c["regions"]:
        for field in ("population", "death_probability", "immigration", "emigration_rate"):
            for sex in SEXES:
                vector(r[field][sex], n, field, 1 if field in
                       ("death_probability", "emigration_rate") else None)
        vector(r["asfr"], n, "ASFR")
        fertility(sum(r["asfr"]), c["fertility_ceiling"], 0)
        if not isinstance(c["coefficients"], dict) or not c["coefficients"]:
            raise ValueError("coefficients must be a nonempty object")
        if set(r["annual_driver_change"]) != set(SCENARIOS):
            raise ValueError("annual_driver_change must list exactly the known scenarios")
        for scenario in SCENARIOS:
            if set(r["annual_driver_change"][scenario]) != set(c["coefficients"]):
                raise ValueError(f"driver keys for {scenario} must match coefficient keys")
        for key, b in c["coefficients"].items():
            number(b, "coefficient " + key)
            for scenario in SCENARIOS:
                number(r["annual_driver_change"][scenario][key], "driver " + key)
    if set(c["immigration_factors"]) != set(SCENARIOS):
        raise ValueError("immigration_factors must list exactly the known scenarios")
    initial = sum(sum(r["population"][s]) for r in c["regions"] for s in SEXES)
    for scenario in SCENARIOS:
        f = number(c["immigration_factors"][scenario], "immigration factor", 0, 10)
        arrivals = f * sum(sum(r["immigration"][s]) for r in c["regions"] for s in SEXES)
        if initial and arrivals > 0.05 * initial:
            raise ValueError("annual immigration above 5% of initial population; check units")
    integer(c["policy_ramp_years"], "policy ramp years", 0)
    totals = {}
    for flow in c["domestic_migration"]:
        origin, dest = flow["from"], flow["to"]
        if origin not in names or dest not in names or origin == dest:
            raise ValueError("Invalid domestic migration endpoints")
        for sex in SEXES:
            vector(flow[sex], n, "domestic migration rates", 1)
            for a, rate in enumerate(flow[sex]):
                key = (origin, sex, a)
                totals[key] = totals.get(key, 0) + rate
                if totals[key] > 1:
                    raise ValueError("Total outward domestic migration rate exceeds one")


def population(state):
    return sum(sum(v) for r in state.values() for v in r.values())


def rates_for(c, elapsed, scenario):
    rates, tfrs = {}, {}
    for r in c["regions"]:
        eta = min(elapsed, c["policy_ramp_years"]) * sum(
            b * r["annual_driver_change"][scenario][k]
            for k, b in c["coefficients"].items())
        base = sum(r["asfr"])
        f = fertility(base, c["fertility_ceiling"], eta)
        tfrs[r["name"]] = f
        rates[r["name"]] = [x * f / base for x in r["asfr"]]
    return rates, tfrs


def exposures(c, state):
    return {r["name"]: [x * (1 + survival(q, c["mortality_multiplier"])) / 2
            for x, q in zip(state[r["name"]]["female"], r["death_probability"]["female"])]
            for r in c["regions"]}


def national_tfr(exposure, rates):
    total = 0.0
    for age in range(len(next(iter(rates.values())))):
        den = sum(e[age] for e in exposure.values())
        if den:
            total += sum(exposure[r][age] * rates[r][age] for r in rates) / den
        elif any(v[age] > 0 for v in rates.values()):
            return None
    return total


def migrate(state, flows):
    """All rates applied to the same pre-migration populations."""
    out = copy.deepcopy(state)
    for f in flows:
        for sex in SEXES:
            for a, rate in enumerate(f[sex]):
                amount = state[f["from"]][sex][a] * rate
                out[f["from"]][sex][a] -= amount
                out[f["to"]][sex][a] += amount
    return out


def advance(c, state, rates, scenario):
    n = c["max_age"] + 1
    exposure = exposures(c, state)
    births_by_region = {name: sum(e * f for e, f in zip(exposure[name], rates[name]))
                        for name in state}
    nxt = {}
    deaths = immigration = emigration = 0.0
    for r in c["regions"]:
        name = r["name"]
        nxt[name] = {}
        for sex in SEXES:
            out = [0.0] * n
            sv = [survival(q, c["mortality_multiplier"]) for q in r["death_probability"][sex]]
            for a, count in enumerate(state[name][sex]):
                out[min(a + 1, n - 1)] += count * sv[a]
                deaths += count * (1 - sv[a])
            share = c["female_birth_fraction"] if sex == "female" else 1 - c["female_birth_fraction"]
            newborns = births_by_region[name] * share
            out[0] = newborns * math.sqrt(sv[0])
            deaths += newborns - out[0]
            nxt[name][sex] = out
    nxt = migrate(nxt, c["domestic_migration"])
    for r in c["regions"]:
        for sex in SEXES:
            for a in range(n):
                depart = nxt[r["name"]][sex][a] * r["emigration_rate"][sex][a]
                arrive = r["immigration"][sex][a] * c["immigration_factors"][scenario]
                nxt[r["name"]][sex][a] += arrive - depart
                immigration += arrive
                emigration += depart
    births = sum(births_by_region.values())
    expected = population(state) + births - deaths + immigration - emigration
    residual = population(nxt) - expected
    if not math.isclose(population(nxt), expected, rel_tol=1e-11, abs_tol=1e-6):
        raise ArithmeticError("Population accounting failed")
    if any(not math.isfinite(x) or x < -1e-8 for r in nxt.values() for v in r.values() for x in v):
        raise ArithmeticError("Invalid resulting cohort")
    return nxt, dict(births=births, deaths=deaths, immigration=immigration,
                     emigration=emigration, births_by_region=births_by_region,
                     national_tfr=national_tfr(exposure, rates), accounting_residual=residual)


def snapshot(state):
    ages = [sum(r[s][a] for r in state.values() for s in SEXES)
            for a in range(len(next(iter(state.values()))["female"]))]
    total = sum(ages)
    median = None
    cumulative = 0.0
    for a, count in enumerate(ages):
        cumulative += count
        if total > 0 and cumulative >= total / 2:
            median = a + (total / 2 - (cumulative - count)) / count  # within [a, a+1)
            break
    return dict(total_pop=total, regional_pop={k: sum(map(sum, r.values())) for k, r in state.items()},
                median_age=median, share_65_plus=sum(ages[65:]) / total if total else None)


def simulate(c, years=100, scenario="baseline"):
    validate(c)
    integer(years, "years")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    state = {r["name"]: copy.deepcopy(r["population"]) for r in c["regions"]}
    rows = [dict(year=c["start_year"], period_start=None, **snapshot(state),
                 flows=None, regional_tfr_used=None)]
    for t in range(years):
        rates, tfrs = rates_for(c, t, scenario)
        state, flows = advance(c, state, rates, scenario)
        rows.append(dict(year=c["start_year"] + t + 1, period_start=c["start_year"] + t,
                         **snapshot(state), flows=flows, regional_tfr_used=tfrs))
    digest = hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(version=VERSION, classification="UNCALIBRATED CONDITIONAL SCENARIO",
                scenario=scenario, input_sha256=digest, inputs=c, rows=rows,
                calibrated=False, truth_probability=None,
                warning="Synthetic defaults are not Korean measurements. User data remain unverified.")


# CALIBRATION SOURCES (Statistics Korea / Ministry of Data and Statistics):
#  * Preliminary Birth and Death Statistics 2025 (25 Feb 2026): births 254.5k,
#    deaths 363.4k, TFR 0.80, sex ratio at birth 105.8, mean age at birth 33.8.
#    https://mods.go.kr/board.es?act=view&bid=11773&list_no=444910&mid=a20108010000
#  * Register-based census, 1 Nov 2025: 51.82M; 0-14 5.225M (10.1%),
#    15-64 35.87M (69.2%), 65+ 10.72M (20.7%).
#  * Social Indicators 2025: life expectancy 83.7 (2024), sex gap 5.8 years;
#    2072 medium projection 36.22M.
# 5-year age totals below are APPROXIMATE (rounded, constrained to the broad
# group totals above); replace with KOSIS single-year tables for real use.
AGE_GROUPS_M = (1.25, 1.75, 2.25, 2.30, 2.75, 3.35, 3.55, 3.20, 3.75, 3.95,
                4.30, 4.15, 4.35, 3.70, 2.65, 1.90, 1.40, 0.75, 0.32)  # 0-4 ... 85-89, 90+
LIFE_EXPECTANCY = {"female": 86.6, "male": 80.8}


def gompertz_table(e0, n=101, q0=0.0024, g=0.105):
    """q(a) = 1-exp(-(A + B e^{g a})) for a>=1, infant q0; B solved for target e0."""
    def table(B):
        return [q0] + [1 - math.exp(-(0.00008 + B * math.exp(g * a))) for a in range(1, n - 1)] + [1.0]

    def e0_of(q):
        alive, e = 1.0, 0.0
        for a, x in enumerate(q):
            e += alive * ((1 - x / 2) if a < n - 1 else 1 / max(-math.log(1 - min(q[a - 1], .999)), 1e-9))
            alive *= 1 - x
        return e
    lo, hi = 1e-9, 1e-3
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if e0_of(table(mid)) > e0 else (lo, mid)
    q = table(math.sqrt(lo * hi))
    q[-1] = q[-2]  # open terminal group keeps the last single-year hazard
    return q


def demo_config():
    """Approximate 2025-Korea demo (v2.1). Not a forecast; see CALIBRATION SOURCES."""
    n = 101
    total = [0.0] * n
    for g, millions in enumerate(AGE_GROUPS_M):
        if g < 18:
            for a in range(5 * g, 5 * g + 5):
                total[a] = millions * 1e6 / 5
        else:  # 90+: geometric decay over 90..100 (100 is open)
            w = [0.75 ** k for k in range(11)]
            for k in range(11):
                total[90 + k] = millions * 1e6 * w[k] / sum(w)
    scale = 51_700_000 / sum(total)
    total = [x * scale for x in total]

    def female_share(a):
        return 0.48 if a < 50 else 0.50 if a < 65 else 0.53 if a < 75 else 0.58 if a < 85 else 0.70
    # 2025 ASFR per 1,000 by 5-year group 15-19..45-49 (25-29 21.3, 30-34 73.2, 35-39 52.0 published;
    # the outer groups are approximate). Used as a fixed age pattern.
    asfr5 = (0.6, 4.0, 21.3, 73.2, 52.0, 9.0, 0.3)
    pattern = [asfr5[(a - 15) // 5] if 15 <= a <= 49 else 0 for a in range(n)]
    pattern = [x / sum(pattern) for x in pattern]
    coefficients = dict(housing=-0.3, child_cost=-0.2, work_conflict=-0.3,
                        job_insecurity=-0.2, support=0.3, surveillance=0.0)
    ramp = 20
    mortality = {s: gompertz_table(LIFE_EXPECTANCY[s]) for s in SEXES}
    regions = []
    # Seoul city TFR ~0.58-0.6 vs national 0.80 (Statistics Korea 2024/25); rest set so national ~0.80.
    for name, share, tfr in (("Seoul city", 0.181, 0.60), ("Rest of Korea", 0.819, 0.845)):
        paths = {s: {k: 0.0 for k in coefficients} for s in SCENARIOS}
        paths["pro_natal"].update(housing=-0.02, child_cost=-0.02,
                                  work_conflict=-0.02, job_insecurity=-0.01, support=0.03)
        paths["surveillance_heavy"]["surveillance"] = 0.03
        for i, v in enumerate(VARIANTS):
            combined = {k: 0.0 for k in coefficients}
            for policy, (drivers, effects) in POLICIES.items():
                ceiling = 4.0
                target = min(tfr * (1 + effects[i]), ceiling * 0.99)
                d_eta = (math.log(target / (ceiling - target)) - math.log(tfr / (ceiling - tfr)))
                for k in drivers:  # split the log-odds change across the policy's drivers
                    step = d_eta / len(drivers) / (ramp * coefficients[k])
                    paths[f"{policy}_{v}"][k] += step
                    combined[k] += step
            paths[f"all_combined_{v}"].update(combined)
        pop = {"female": [x * share * female_share(a) for a, x in enumerate(total)],
               "male": [x * share * (1 - female_share(a)) for a, x in enumerate(total)]}
        # Net international migration GUESS: +80k/yr at ages 20-39 (2025 implied ~+120k:
        # population +11k with natural change -110k; long-run official assumptions are lower).
        arrivals = {s: [80_000 * share * 0.5 / 20 if 20 <= a < 40 else 0.0
                        for a in range(n)] for s in SEXES}
        regions.append(dict(name=name, population=pop, death_probability=mortality,
                            asfr=[tfr * w for w in pattern], immigration=arrivals,
                            emigration_rate={s: [0.0] * n for s in SEXES}, annual_driver_change=paths))
    # Seoul city loses residents domestically (mostly to Gyeonggi); small two-way flows, net outflow.
    flows = [{"from": "Rest of Korea", "to": "Seoul city",
              **{s: [0.006 if 19 <= a < 30 else 0.0 for a in range(n)] for s in SEXES}},
             {"from": "Seoul city", "to": "Rest of Korea",
              **{s: [0.012 if 30 <= a < 60 else 0.006 if a < 19 or a >= 60 else 0.0
                     for a in range(n)] for s in SEXES}}]
    return dict(start_year=2026, max_age=100, female_birth_fraction=100 / 205.8,
                mortality_multiplier=1.0, fertility_ceiling=4.0, policy_ramp_years=ramp,
                coefficients=coefficients, regions=regions, domestic_migration=flows,
                immigration_factors={s: 2.0 if s == "immigration_support" else 1.0 for s in SCENARIOS},
                provenance={"status": "APPROXIMATE_2025_KOREA_DEMO", "observation_date": "2025 (prelim.)",
                            "description": "Broad totals from Statistics Korea 2025; single-year detail, mortality "
                                           "curve, migration and all policy effects are approximations/GUESSES.",
                            "policy_evidence": POLICY_EVIDENCE,
                            "driver_units": "standardized scenario units; not measured percentages",
                            "fertility_ceiling": "4.0 is an assumed numerical bound, not a target",
                            "surveillance": "zero coefficient pending evidence; nonzero values are sensitivity assumptions"})


class Tests(unittest.TestCase):
    def test_baseline_and_direction(self):
        self.assertAlmostEqual(fertility(.72, 4, 0), .72)
        self.assertGreater(fertility(.72, 4, .5), .72)
        self.assertLess(fertility(.72, 4, -.5), .72)

    def test_survival_hazard(self):
        self.assertAlmostEqual(survival(.2, 2), .64)
        self.assertEqual(survival(1, 1), 0)
        self.assertEqual(survival(1, 0), 1)

    def test_horizon_initial_and_accounting(self):
        c = demo_config()
        for s in SCENARIOS:
            r = simulate(c, 100, s)
            self.assertEqual(len(r["rows"]), 101)
            self.assertEqual(r["rows"][-1]["year"], 2126)
            self.assertAlmostEqual(r["rows"][0]["total_pop"], 51_700_000, places=4)
            self.assertIsNone(r["rows"][0]["flows"])
            self.assertTrue(all(abs(x["flows"]["accounting_residual"]) < 1e-5 for x in r["rows"][1:]))

    def test_scenario_isolation(self):
        c = demo_config()
        b, s, p = [simulate(c, 25, x) for x in ("baseline", "surveillance_heavy", "pro_natal")]
        self.assertEqual(b["rows"], s["rows"])
        self.assertGreater(p["rows"][-1]["flows"]["births"], b["rows"][-1]["flows"]["births"])
        self.assertEqual(p["rows"][-1]["flows"]["immigration"], b["rows"][-1]["flows"]["immigration"])

    def test_migration_preserves_age_sex(self):
        state = {"A": {s: [10., 20., 30.] for s in SEXES}, "B": {s: [5., 6., 7.] for s in SEXES}}
        flows = [{"from": "A", "to": "B", **{s: [.1, .2, .3] for s in SEXES}}]
        out = migrate(state, flows)
        for s in SEXES:
            for a in range(3):
                self.assertEqual(out["A"][s][a] + out["B"][s][a], state["A"][s][a] + state["B"][s][a])

    def test_terminal_and_births(self):
        c = demo_config()
        c["domestic_migration"] = []
        c["female_birth_fraction"] = 0.5
        for r in c["regions"]:
            for s in SEXES:
                for k in ("population", "death_probability", "immigration", "emigration_rate"):
                    r[k][s] = [0.] * 101
        c["regions"][0]["population"]["female"][30] = 100
        c["regions"][0]["population"]["male"][100] = 7
        state = {r["name"]: r["population"] for r in c["regions"]}
        rates = {r["name"]: [0.] * 101 for r in c["regions"]}
        rates["Seoul city"][30] = .1
        out, f = advance(c, state, rates, "baseline")
        self.assertEqual(f["births"], 10)
        self.assertEqual(out["Seoul city"]["male"][100], 7)
        self.assertEqual(out["Seoul city"]["female"][0], 5)

    def test_age_specific_national_tfr(self):
        self.assertAlmostEqual(national_tfr({"A": [90, 10], "B": [10, 90]},
                                           {"A": [.1, .2], "B": [.3, .4]}), .5)
        self.assertIsNone(national_tfr({"A": [0]}, {"A": [.1]}))

    def test_invalid_inputs(self):
        for bad in (float("nan"), float("inf"), -1):
            c = demo_config()
            c["regions"][0]["population"]["female"][0] = bad
            with self.assertRaises(ValueError):
                simulate(c, 1)
        c = demo_config()
        c["domestic_migration"] *= 201
        with self.assertRaises(ValueError):
            validate(c)

    def test_v21_driver_key_typo_rejected(self):
        c = demo_config()
        paths = c["regions"][0]["annual_driver_change"]["homes_for_families_central"]
        paths["housng"] = paths.pop("housing")
        with self.assertRaises(ValueError):
            validate(c)
        c = demo_config()
        del c["regions"][1]["annual_driver_change"]["work_hours_low"]
        with self.assertRaises(ValueError):
            validate(c)

    def test_v21_immigration_bounded(self):
        c = demo_config()
        c["regions"][0]["immigration"]["male"][25] = 1e9
        with self.assertRaises(ValueError):
            validate(c)
        c = demo_config()
        c["immigration_factors"]["baseline"] = 50
        with self.assertRaises(ValueError):
            validate(c)
        c = demo_config()
        c["policy_ramp_years"] = 2.5
        with self.assertRaises(ValueError):
            validate(c)

    def test_v21_median_interpolated(self):
        self.assertAlmostEqual(snapshot({"A": {"female": [1., 1.], "male": [0., 0.]}})["median_age"], 1.0)
        self.assertAlmostEqual(snapshot({"A": {"female": [3., 1.], "male": [0., 0.]}})["median_age"], 2 / 3)

    def test_v21_calibration_bands(self):
        r = simulate(demo_config(), 1)["rows"]
        f = r[1]["flows"]
        self.assertTrue(0.78 < f["national_tfr"] < 0.82)
        self.assertTrue(240_000 < f["births"] < 275_000)
        self.assertTrue(340_000 < f["deaths"] < 395_000)
        self.assertTrue(0.195 < r[0]["share_65_plus"] < 0.215)

    def test_v21_policy_ordering(self):
        c = demo_config()
        tfr = {s: rates_for(c, 30, s)[1]["Rest of Korea"] for s in SCENARIOS}
        base = tfr["baseline"]
        for p in list(POLICIES) + ["all_combined"]:
            self.assertLess(base, tfr[p + "_low"])
            self.assertLess(tfr[p + "_low"], tfr[p + "_central"])
            self.assertLess(tfr[p + "_central"], tfr[p + "_high"])
        for p, (_, effects) in POLICIES.items():
            self.assertAlmostEqual(tfr[p + "_central"] / base, 1 + effects[1], places=6)
            self.assertGreater(tfr["all_combined_central"], tfr[p + "_central"])
        self.assertEqual(rates_for(c, 0, "all_combined_high")[1], rates_for(c, 0, "baseline")[1])

    def test_zero_population(self):
        c = demo_config()
        for r in c["regions"]:
            for s in SEXES:
                r["population"][s] = [0.] * 101
                r["immigration"][s] = [0.] * 101
        rows = simulate(c, 2)["rows"]
        self.assertTrue(all(x["total_pop"] == 0 and x["median_age"] is None for x in rows))
        self.assertEqual(len(simulate(c, 0)["rows"]), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=100)
    ap.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    ap.add_argument("--input", type=Path, help="JSON configuration; otherwise synthetic demonstration")
    ap.add_argument("--json", action="store_true", help="Full annual results plus input provenance")
    ap.add_argument("--print-template", action="store_true")
    ap.add_argument("--report-step", type=int, default=10)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if result.wasSuccessful() else 1
    if args.print_template:
        print(json.dumps(demo_config(), indent=2, allow_nan=False))
        return 0
    try:
        integer(args.report_step, "report step", 1)
        c = json.loads(args.input.read_text(encoding="utf-8")) if args.input else demo_config()
        r = simulate(c, args.years, args.scenario)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        ap.error(str(exc))
    if args.json:
        print(json.dumps(r, indent=2, allow_nan=False))
    else:
        print("Real Mode demographic simulator v" + VERSION)
        print(r["classification"] + " | " + args.scenario)
        print("SYNTHETIC DEMO INPUTS" if not args.input else "USER-SUPPLIED INPUTS: not independently verified")
        print("Year       Population   Births/year   Deaths/year  Period TFR")
        for i, row in enumerate(r["rows"]):
            if i % args.report_step and i != len(r["rows"]) - 1:
                continue
            f = row["flows"]
            tail = "     --            --          --" if f is None else (
                f"{f['births']:12,.0f} {f['deaths']:13,.0f} " +
                (f"{f['national_tfr']:11.3f}" if f["national_tfr"] is not None else "         NA"))
            print(f"{row['year']:<6} {row['total_pop']:14,.0f} {tail}")
        print("Flows describe the preceding year. Terminal age is an open group.")
        print("Policy coefficients are uncalibrated; no truth probability assigned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

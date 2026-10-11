#!/usr/bin/env python3
"""Real Mode Korea demographic scenario simulator, v2.0 (Python 3.9+, stdlib).

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

VERSION = "2.0"
SEXES = ("female", "male")
SCENARIOS = ("baseline", "pro_natal", "surveillance_heavy", "immigration_support")


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
        for key, b in c["coefficients"].items():
            number(b, "coefficient " + key)
            for scenario in SCENARIOS:
                number(r["annual_driver_change"][scenario][key], "driver " + key)
    for scenario in SCENARIOS:
        number(c["immigration_factors"][scenario], "immigration factor", 0)
    number(c["policy_ramp_years"], "policy ramp years", 0)
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
            median = a
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


def demo_config():
    """Deliberately synthetic template: no claims about actual 2026 age shares."""
    n = 101
    shape = [math.exp(-((a - 48) / 28) ** 2) for a in range(n)]
    shape = [x / sum(shape) for x in shape]
    pattern = [math.exp(-0.5 * ((a - 33) / 5) ** 2) if 15 <= a <= 49 else 0 for a in range(n)]
    pattern = [x / sum(pattern) for x in pattern]
    coefficients = dict(housing=-0.3, child_cost=-0.2, work_conflict=-0.3,
                        job_insecurity=-0.2, support=0.3, surveillance=0.0)
    regions = []
    for name, share in (("Seoul city", 0.18), ("Rest of Korea", 0.82)):
        paths = {s: {k: 0.0 for k in coefficients} for s in SCENARIOS}
        paths["pro_natal"].update(housing=-0.02, child_cost=-0.02,
                                  work_conflict=-0.02, job_insecurity=-0.01, support=0.03)
        paths["surveillance_heavy"]["surveillance"] = 0.03
        pop = {s: [51_300_000 * share * 0.5 * w for w in shape] for s in SEXES}
        mortality = {s: [1 - math.exp(-(0.00015 + 0.00002 * math.exp(a / 9)))
                         for a in range(n)] for s in SEXES}
        arrivals = {s: [60_000 * share * 0.5 / 20 if 20 <= a < 40 else 0.0
                        for a in range(n)] for s in SEXES}
        regions.append(dict(name=name, population=pop, death_probability=mortality,
                            asfr=[0.72 * w for w in pattern], immigration=arrivals,
                            emigration_rate={s: [0.0] * n for s in SEXES}, annual_driver_change=paths))
    return dict(start_year=2026, max_age=100, female_birth_fraction=0.5,
                mortality_multiplier=1.0, fertility_ceiling=4.0, policy_ramp_years=20,
                coefficients=coefficients, regions=regions,
                domestic_migration=[{"from": "Rest of Korea", "to": "Seoul city",
                                     **{s: [0.005 if 22 <= a < 35 else 0.0 for a in range(n)] for s in SEXES}}],
                immigration_factors={s: 2.0 if s == "immigration_support" else 1.0 for s in SCENARIOS},
                provenance={"status": "SYNTHETIC_DEMO", "observation_date": None,
                            "description": "All demographic values, coefficients, and policy paths are illustrative. Replace them before empirical use.",
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
            self.assertAlmostEqual(r["rows"][0]["total_pop"], 51_300_000, places=5)
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

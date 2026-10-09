# SPDX-License-Identifier: CC0-1.0
"""Tests for mountain_updraft_v4_2 (python3 -m unittest). License: CC0 1.0 Universal."""
import dataclasses
import inspect
import math
import os
import re
import unittest

import mountain_updraft_v4_1 as V41
import mountain_updraft_v4_2 as M

S = M.DEMO_SOUNDINGS
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mountain_updraft_v4_2.py")


def solve(cfg=None, soundings=S, neutral=False, irr=780.0, v_guess=10.0):
    cfg = cfg or M.FacilityConfig()
    return M.AtmosphericThermodynamics.solve_draft(cfg, soundings[0], soundings[-1], irr, soundings,
                                                   neutral=neutral, v_guess=v_guess)


def neutral_sounding():
    """Dry-adiabatic temperatures, dewpoints giving the same specific humidity."""
    s0 = S[0]
    out = [s0]
    for z, p in ((300.0, 984.0), (685.0, 940.0)):
        t = s0.temp_c - M.DRY_ADIABATIC_LAPSE_K_M * (z - s0.altitude_m)
        q = M._specific_humidity(s0.dewpoint_c, s0.pressure_hpa)
        e = q * p / (0.622 + 0.378 * q)
        td = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
        out.append(M.SoundingLayer(p, z, t, td, s0.wind_speed_mps))
    return out


class TestDraft(unittest.TestCase):
    def test_energy_limit(self):
        cfg = M.FacilityConfig()
        for irr in (200.0, 780.0, 1100.0):
            dT, dp, v, m, Q = solve(cfg, irr=irr)
            p_el = M.compute_turbine_power(cfg, dp, Q)[1]
            q_th_mw = m * M.CP_AIR * dT / 1e6
            self.assertLessEqual(p_el, M.chimney_efficiency(cfg, S[0]) * q_th_mw * 1.0001)
            self.assertAlmostEqual(M.chimney_efficiency(cfg, S[0]), 0.0221, places=3)

    def test_power_invariant_to_velocity(self):
        """P ~ eta x g H Q_th / (cp T): flow losses change v, not the power (neutral air)."""
        ps = []
        for f, k_exit, k_esp in ((0.0, 1.0, 0.0), (0.010, 1.0, 0.5), (0.03, 2.0, 2.0)):
            cfg = M.FacilityConfig(friction_factor=f, k_exit=k_exit, k_esp=k_esp)
            dT, dp, v, m, Q = solve(cfg, neutral=True)
            ps.append((v, M.compute_turbine_power(cfg, dp, Q)[1]))
        self.assertGreater(ps[0][0], ps[2][0] * 1.3)                   # velocity changes a lot
        # power changes only at second order: buoyancy ~ dT / (T + dT), and a slower flow is hotter
        self.assertLess(abs(ps[0][1] - ps[1][1]) / ps[0][1], 0.03)
        self.assertLess(abs(ps[0][1] - ps[2][1]) / ps[0][1], 0.08)
        for vg in (0.5, 10.0, 80.0):                                   # nor does the initial guess
            dT, dp, v, m, Q = solve(v_guess=vg)
            self.assertAlmostEqual(M.compute_turbine_power(M.FacilityConfig(), dp, Q)[1], 1.872132, places=5)

    def test_flow_lower_than_v41(self):
        r41 = V41.run_system_cycle(config=V41.FacilityConfig(),
                                   soundings=[V41.SoundingLayer(*dataclasses.astuple(s)) for s in S], **M.DEMO_INPUTS)
        q41 = r41["thermodynamics_and_updraft"]["volumetric_flow_m3_s"]
        q42 = solve()[4]
        self.assertLess(q42, q41)
        q_exit_only = solve(M.FacilityConfig(friction_factor=0.0, k_esp=0.0), neutral=True)[4]
        self.assertTrue(7000.0 < q_exit_only < 7800.0)                 # reviewer's ~7,640 check

    def test_stable_sounding_lowers_draft(self):
        dT, dp_s, v_s, m, q_s = solve()
        _, dp_n, v_n, _, q_n = solve(neutral=True)
        self.assertLess(q_s, q_n)
        warm_crest = [*S[:-1], dataclasses.replace(S[-1], temp_c=S[-1].temp_c + 3.0)]
        self.assertLess(solve(soundings=warm_crest)[4], q_s)

    def test_neutral_sounding_matches_v41_buoyancy(self):
        cfg = M.FacilityConfig()
        dT = 20.0
        head_snd = M.AtmosphericThermodynamics.buoyant_head(cfg, neutral_sounding(), S[0], dT)
        head_neu = M.AtmosphericThermodynamics.buoyant_head(cfg, None, S[0], dT)
        t = S[0].temp_c + 273.15
        rho = S[0].pressure_hpa * 100 / (M.R_SPECIFIC_AIR * t)
        head_v41 = rho * M.G_ACCEL * (cfg.crest_elevation_m - cfg.base_elevation_m) * dT / t
        self.assertLess(abs(head_snd - head_neu) / head_neu, 0.01)
        self.assertLess(abs(head_neu - head_v41) / head_v41, 0.10)   # density falls ~7 % over 650 m


class TestESP(unittest.TestCase):
    def test_monotonic_in_area_and_voltage(self):
        effs = [M.ElectrostaticPrecipitatorEngine.calculate_filtration(M.FacilityConfig(esp_plate_area_m2=a), 5000.0, 50.0)["efficiency"]
                for a in (2000.0, 9200.0, 50000.0)]
        self.assertTrue(effs[0] < effs[1] < effs[2])
        effs = [M.ElectrostaticPrecipitatorEngine.calculate_filtration(M.FacilityConfig(), 5000.0, kv)["efficiency"]
                for kv in (20.0, 50.0, 80.0)]
        self.assertTrue(effs[0] < effs[1] < effs[2])

    def test_combined_charge_and_derating(self):
        e = M.ElectrostaticPrecipitatorEngine
        for r in (0.3e-6, 1.25e-6, 5e-6):
            self.assertGreater(e._particle_charge(r, 5e5), 0.0)
        out = e.calculate_filtration(M.FacilityConfig(), 5000.0, 50.0)
        self.assertAlmostEqual(out["migration_velocity_effective_mps"], 0.25 * out["migration_velocity_theory_mps"])
        self.assertLess(out["efficiency"], out["efficiency_theory"])
        self.assertGreater(out["corona_power_mw"], 0.0)


class TestGrid(unittest.TestCase):
    def test_bess_soc_limits(self):
        cfg = M.FacilityConfig()
        bess = M.BatteryStorage(capacity_mwh=0.5, power_mw=2.0, soc=0.5)
        g = M.GridIntertieController(bess, dispatch_interval_s=300.0)
        for k in range(40):
            f = 59.5 if k < 20 else 60.5                       # long under- then over-frequency
            r = g.modulate_grid_export(cfg, 1.0, f, 44.0, now_s=300.0 * k)
            self.assertGreaterEqual(bess.soc, bess.soc_min - 1e-12)
            self.assertLessEqual(bess.soc, bess.soc_max + 1e-12)
            self.assertLessEqual(r["net_intertie_export_mw"],
                                 1.0 + r["synthetic_inertia_discharged_mw"] + max(0.0, r["bess_flow_mw"]) + 1e-9)
        self.assertAlmostEqual(bess.soc, bess.soc_max, places=6)
        empty = M.GridIntertieController(M.BatteryStorage(soc=0.10))
        r = empty.modulate_grid_export(cfg, 1.0, 59.9, 0.0, now_s=0.0)
        self.assertAlmostEqual(r["net_intertie_export_mw"], 1.0)     # nothing to discharge: export = generation
        self.assertTrue(r["bess_limited"])

    def test_deterministic_with_injected_clock(self):
        cfg = M.FacilityConfig()

        def run():
            g = M.GridIntertieController()
            return [g.modulate_grid_export(cfg, 1.0, 59.78, 44.0, now_s=t)["synthetic_inertia_discharged_mw"]
                    for t in (0.0, 10.0, 31.0, 40.0)]
        a, b = run(), run()
        self.assertEqual(a, b)
        self.assertGreater(a[0], 0.0)
        self.assertEqual(a[1], 0.0)          # inside the 30 s cooldown
        self.assertGreater(a[2], 0.0)        # after the cooldown
        self.assertNotIn("time.time", inspect.getsource(M))

    def test_droop_sign(self):
        cfg = M.FacilityConfig()
        lo = M.GridIntertieController().modulate_grid_export(cfg, 1.0, 59.95, 0.0)["pfr_droop_command_mw"]
        hi = M.GridIntertieController().modulate_grid_export(cfg, 1.0, 60.05, 0.0)["pfr_droop_command_mw"]
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 0.0)
        self.assertAlmostEqual(lo, (1 / 0.04) * (0.05 / 60.0) * 5.0, places=3)


class TestReport(unittest.TestCase):
    def test_no_hardcoded_probability_and_computed_status(self):
        src = open(SRC).read()
        self.assertNotIn("system_truth_" + "probability", src)
        self.assertIsNone(re.search(r"probability\"?\s*:\s*[0-9.]", src))
        r = M.run_system_cycle(config=M.FacilityConfig(), soundings=S, **M.DEMO_INPUTS)
        self.assertIn(r["execution_status"], ("NOMINAL", "DEGRADED", "FAULT"))
        self.assertEqual(r["execution_status"], "DEGRADED")       # ESP SCA far below the industrial range
        night = M.run_system_cycle(config=M.FacilityConfig(), soundings=S, irradiance_w_m2=0.0,
                                   esp_voltage_kv=50.0, grid_freq_hz=60.0, rotor_rpm=0.0)
        self.assertEqual(night["execution_status"], "FAULT")

    def test_centrifugal_root_stress(self):
        b = M.TurbineBladeSpecs()
        st = M.TurbomachineryStressEngine.evaluate_structural_margin(10.0, b, 44.0)
        w = 44.0 * 2 * math.pi / 60
        exp = 0.5 * b.blade_density_kg_m3 * w ** 2 * (b.rotor_radius_m ** 2 - (b.rotor_radius_m - b.blade_length_m) ** 2) / 1e6
        self.assertAlmostEqual(st["centrifugal_stress_mpa"], round(exp, 3))

    def test_privacy_and_header(self):
        for name in ("mountain_updraft_v4_2.py", "test_mountain_updraft_v4_2.py", "make_demo_v4_2.py"):
            text = open(os.path.join(os.path.dirname(SRC), name)).read()
            self.assertIn("SPDX-License-Identifier: CC0-1.0", text[:200])
            self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text))
            for bad in ("/" + "home/", "/" + "workspace", "/" + "Users/"):
                self.assertNotIn(bad, text)
            # optional private terms come from the environment, never stored here
            for term in filter(None, (t.strip().lower() for t in os.environ.get("UPDRAFT_PRIVACY_TERMS", "").split(","))):
                self.assertNotIn(term, text.lower(), name)


if __name__ == "__main__":
    unittest.main()

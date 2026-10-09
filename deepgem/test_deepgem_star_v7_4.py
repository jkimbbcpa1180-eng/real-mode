# SPDX-License-Identifier: CC0-1.0
"""Tests for deepgem_star_v7_4.py. Run: python3 -m unittest test_deepgem_star_v7_4

Reference values are hard-coded with their source. Each module has checks
against published/tabulated data, plus regression checks for the v7.3.1 bugs.
"""
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "deepgem_star_v7_4.py")
_spec = importlib.util.spec_from_file_location("dg74", SRC)
dg = importlib.util.module_from_spec(_spec)
sys.modules["dg74"] = dg
_spec.loader.exec_module(dg)

# --- Reference data ---------------------------------------------------------
# NIST-JANAF H(T)-H(298.15), kJ/mol (tables C-002 graphite, C-067 CH4, H-050 H2)
JANAF_GRAPHITE = {298.15: 0.0, 1000.0: 11.795, 1200.0: 16.240, 1300.0: 18.539}
JANAF_CH4_1273 = 59.145      # interpolated 1200->1300 from C-067
JANAF_H2_1273 = 29.080       # interpolated 1200->1300 from H-050
JANAF_GRAPHITE_1273 = 17.922 # interpolated 1200->1300 from C-002
JANAF_DH_RXN_1273 = 91.809   # -dfH(CH4) at 1273.15 K, interpolated, C-067
JANAF_LHV_CH4 = 50.010       # MJ/kg from JANAF dfH (CO2, H2O(g), CH4)
JANAF_LHV_H2 = 119.961       # MJ/kg from JANAF dfH H2O(g)
# ENSDF Ni-63 (J. Chen, NDS 196, 2024)
ENSDF_NI63_HALF_LIFE_Y = 100.8
ENSDF_NI63_Q_KEV = 66.977
ENSDF_NI63_MEAN_KEV = 17.439
MEASURED_MU_NI_CM = 13184.0  # Schweitzer 1952 via Belghachi arXiv:1903.09098
PROTOTYPE_VOC_V = 1.02       # Bormashov et al. 2018, Diamond Relat. Mater.
PROTOTYPE_ISC_A = 1.27e-6
BORMASHOV_CELL_EFF = (0.05, 0.06)

_DEFAULT_RUN = None


def default_run():
    global _DEFAULT_RUN
    if _DEFAULT_RUN is None:
        _DEFAULT_RUN = dg.run_hybrid_system_simulation()
    return _DEFAULT_RUN


# ===========================================================================
class TestModule1Pyrolysis(unittest.TestCase):

    def test_graphite_h298_is_zero(self):
        self.assertEqual(dg.graphite_h_minus_h298_kj_mol(298.15), 0.0)

    def test_graphite_matches_janaf_table_points(self):
        for t, h in JANAF_GRAPHITE.items():
            self.assertAlmostEqual(dg.graphite_h_minus_h298_kj_mol(t), h, places=3)

    def test_graphite_at_reactor_temp_vs_janaf(self):
        self.assertAlmostEqual(dg.graphite_h_minus_h298_kj_mol(1273.15),
                               JANAF_GRAPHITE_1273, delta=0.01)

    def test_graphite_regression_not_bogus_set(self):
        # v7.3.1's set, used as a 298->1273 K delta, gave 13.52 (JANAF 17.92).
        self.assertGreater(dg.graphite_h_minus_h298_kj_mol(1273.15), 15.0)

    def test_ch4_shomate_vs_janaf(self):
        self.assertAlmostEqual(dg._ch4_sensible_kj_mol(1273.15, 298.15),
                               JANAF_CH4_1273, delta=0.1)

    def test_h2_uses_high_range_set_above_1000k(self):
        self.assertAlmostEqual(dg._h2_sensible_kj_mol(1273.15, 298.15),
                               JANAF_H2_1273, delta=0.05)
        # below 1000 K the low set is used: H2 at 1000 K JANAF 20.680
        self.assertAlmostEqual(dg._h2_sensible_kj_mol(1000.0, 298.15),
                               20.68, delta=0.05)

    def test_reaction_enthalpy_at_t_vs_janaf(self):
        r = dg.run_pyrolysis_stage()
        mol = 1000.0 / dg.M_CH4
        self.assertAlmostEqual(r.dh_reaction_at_t_mj * 1000.0 / mol,
                               JANAF_DH_RXN_1273, delta=0.3)
        self.assertAlmostEqual(r.dh_reaction_298k_mj * 1000.0 / mol, 74.873, places=2)

    def test_first_law_balance_uses_products(self):
        r = dg.run_pyrolysis_stage()
        expected = r.dh_reaction_298k_mj + r.sensible_products_mj - r.heat_recuperated_mj
        self.assertAlmostEqual(r.net_process_heat_mj, expected, places=3)
        # products-based heat must exceed the old feed-based figure
        old = r.dh_reaction_298k_mj + r.sensible_preheat_feed_mj - r.heat_recuperated_mj
        self.assertGreater(r.net_process_heat_mj, old)

    def test_electrical_efficiency_matches_review_reference(self):
        # review_v6: first-law + JANAF at SOFC 0.65, HX 0.85 -> 30.60%
        r = dg.run_pyrolysis_stage(sofc_eff=0.65)
        self.assertAlmostEqual(r.electrical_efficiency_pct, 30.60, delta=0.02)

    def test_default_sofc_is_labeled_realistic(self):
        r = dg.run_pyrolysis_stage()
        self.assertEqual(r.sofc_eff_assumed, 0.60)
        self.assertAlmostEqual(r.electrical_efficiency_pct, 28.24, delta=0.02)

    def test_lhv_constants_vs_janaf(self):
        self.assertAlmostEqual(dg.LHV_CH4, JANAF_LHV_CH4, delta=0.01)
        self.assertAlmostEqual(dg.LHV_H2, JANAF_LHV_H2, delta=0.01)

    def test_thermal_eff_below_theoretical_ceiling(self):
        # 2*LHV_H2(mol)/LHV_CH4(mol) = 60.28% ceiling (no H2 burned)
        self.assertLess(dg.run_pyrolysis_stage().thermal_efficiency_pct, 60.28)

    def test_sofc_sweep_monotonic(self):
        effs = [e for _, e in dg.sofc_sweep()]
        self.assertEqual(effs, sorted(effs))

    def test_conversion_scales_h2(self):
        full = dg.run_pyrolysis_stage(conversion=1.0).total_h2_produced_kg
        part = dg.run_pyrolysis_stage(conversion=0.982).total_h2_produced_kg
        self.assertAlmostEqual(part / full, 0.982, places=4)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            dg.run_pyrolysis_stage(methane_kg=0)
        with self.assertRaises(ValueError):
            dg.run_pyrolysis_stage(sofc_eff=1.5)
        with self.assertRaises(ValueError):
            dg.run_pyrolysis_stage(conversion=0.0)


# ===========================================================================
class TestModule2Betavoltaic(unittest.TestCase):

    def test_thickness_uses_nickel_density(self):
        b = dg.run_betavoltaic_stage()
        self.assertAlmostEqual(b.source_thickness_um,
                               25e-3 / (8.908 * 5.0) * 1e4, places=3)
        self.assertAlmostEqual(b.source_thickness_um, 5.61, delta=0.01)

    def test_density_cancels_in_self_absorption(self):
        a = dg.run_betavoltaic_stage(source_density_g_cm3=8.908)
        b = dg.run_betavoltaic_stage(source_density_g_cm3=3.515)
        self.assertAlmostEqual(a.self_absorption_factor, b.self_absorption_factor, places=9)
        self.assertNotAlmostEqual(a.source_thickness_um, b.source_thickness_um)

    def test_measured_ni_attenuation(self):
        self.assertAlmostEqual(dg.BETA_MU_RHO_CM2_PER_G["Ni-63"],
                               MEASURED_MU_NI_CM / dg.NICKEL_DENSITY_G_CM3, delta=1.0)

    def test_self_absorption_regression(self):
        # measured mu -> 0.135; v7.3.1's 25 cm^2/g gave 0.940 (~7x inflated)
        b = dg.run_betavoltaic_stage()
        self.assertAlmostEqual(b.self_absorption_factor, 0.1351, delta=0.001)

    def test_specific_activity_pure_and_purity(self):
        pure = dg.run_betavoltaic_stage(isotope_purity=1.0)
        # NRC 10 CFR 71 App A: Ni-63 specific activity 2.1 TBq/g = ~56.8 Ci/g
        self.assertAlmostEqual(pure.specific_activity_ci_g, 56.1, delta=0.5)
        d = dg.run_betavoltaic_stage()  # default purity 0.18 ~ 10 Ci/g
        self.assertAlmostEqual(d.specific_activity_ci_g, 10.1, delta=0.2)

    def test_activity_linear_in_purity(self):
        a = dg.run_betavoltaic_stage(isotope_purity=0.2).activity_bq
        b = dg.run_betavoltaic_stage(isotope_purity=0.4).activity_bq
        self.assertAlmostEqual(b / a, 2.0, places=9)

    def test_half_space_emission_geometry(self):
        one = dg.run_betavoltaic_stage(emission_faces=1).escaping_power_uw_per_cm2
        two = dg.run_betavoltaic_stage(emission_faces=2).escaping_power_uw_per_cm2
        self.assertAlmostEqual(two / one, 2.0, places=6)

    def test_pair_energy_is_measured(self):
        self.assertAlmostEqual(dg.DIAMOND_EHP_ENERGY_EV, 13.1, places=6)
        self.assertNotAlmostEqual(dg.DIAMOND_EHP_ENERGY_EV, 2.8 * 5.47 + 0.5, places=2)

    def test_i0_matches_prototype_voc(self):
        voc = dg.diode_voc(PROTOTYPE_ISC_A, 0.25, 2.7e-20)
        self.assertAlmostEqual(voc, PROTOTYPE_VOC_V, delta=0.02)

    def test_saturation_cross_check_within_factor_two(self):
        model = dg.ni63_saturated_escaping_uw_cm2(1.0)
        ratio = model / dg.PUBLISHED_NI63_SATURATION_UW_CM2
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)
        # documented as a conservative under-estimate (spectral hardening not modeled)
        self.assertLess(ratio, 1.0)

    def test_cell_efficiency_near_published(self):
        eff = dg.run_betavoltaic_stage().cell_efficiency_pct / 100.0
        self.assertGreater(eff, 0.03)
        self.assertLess(eff, 0.08)

    def test_carbon_loop_is_noop(self):
        a = dg.run_betavoltaic_stage(carbon_loop_fraction=0.0, carbon_available_kg=0.75)
        b = dg.run_betavoltaic_stage(carbon_loop_fraction=1.0, carbon_available_kg=0.75)
        self.assertEqual(a.electrical_power_uw, b.electrical_power_uw)
        self.assertIn("no-op", b.carbon_loop_note)

    def test_ni63_constants_vs_ensdf(self):
        s = dg.ISOTOPE_DATABASE["Ni-63"]
        self.assertLess(abs(s["half_life_yr"] / ENSDF_NI63_HALF_LIFE_Y - 1), 0.005)
        self.assertLess(abs(s["q_val_ev"] / 1e3 / ENSDF_NI63_Q_KEV - 1), 0.001)
        self.assertLess(abs(s["avg_beta_ev"] / 1e3 / ENSDF_NI63_MEAN_KEV - 1), 0.002)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            dg.run_betavoltaic_stage(isotope_purity=0.0)
        with self.assertRaises(ValueError):
            dg.run_betavoltaic_stage(emission_faces=3)
        with self.assertRaises(ValueError):
            dg.run_betavoltaic_stage(isotope="U-235")


# ===========================================================================
class TestModule3PowerBuffer(unittest.TestCase):

    def test_rearm_must_exceed_low(self):
        with self.assertRaises(ValueError):
            dg.DeepGemPowerState(v_low=2.5, v_rearm=2.0)

    def test_schmitt_disables_and_rearms_only_at_v_rearm(self):
        ps = dg.DeepGemPowerState(v_cap=3.3, quiescent_leakage_w=0.0)
        ps.v_cap = 1.9
        ps._update_schmitt()
        self.assertFalse(ps.tactical_inference_enabled)
        ps.v_cap = 2.4   # between v_low and v_rearm: must stay OFF (hysteresis)
        ps._update_schmitt()
        self.assertFalse(ps.tactical_inference_enabled)
        ps.v_cap = 2.7   # above v_rearm: re-arm
        ps._update_schmitt()
        self.assertTrue(ps.tactical_inference_enabled)
        self.assertEqual(ps.safety_transitions, 2)

    def test_refused_burst_trips_safe_mode(self):
        ps = dg.DeepGemPowerState(v_cap=2.05, quiescent_leakage_w=0.0)
        ok, _ = ps.attempt_burst_execution(1.1, 0.040)
        self.assertFalse(ok)
        self.assertFalse(ps.tactical_inference_enabled)
        self.assertEqual(ps.safety_transitions, 1)
        # stays off while below v_rearm even if a tiny burst would fit
        ok2, _ = ps.attempt_burst_execution(0.001, 0.0001)
        self.assertFalse(ok2)

    def test_hysteresis_engages_in_simulation_for_large_burst(self):
        r = dg.run_hybrid_system_simulation(inference_power_w=1.1,
                                            inference_duration_s=0.040,
                                            held_out_n=100)
        self.assertGreater(r["energy_harvesting_totals"]["safety_transitions"], 0)
        self.assertGreater(r["star_drone_tactical_module"]["safety_halts"], 0)

    def test_right_sized_burst_no_brownouts(self):
        s = default_run()["star_drone_tactical_module"]
        self.assertEqual(s["brownout_events"], 0)
        self.assertAlmostEqual(s["inference_burst_uj"], 6.0, places=6)

    def test_dcdc_efficiency_reduces_stored_energy(self):
        a = dg.DeepGemPowerState(v_cap=2.2, dcdc_efficiency=1.0, quiescent_leakage_w=0.0)
        b = dg.DeepGemPowerState(v_cap=2.2, dcdc_efficiency=0.5, quiescent_leakage_w=0.0)
        a.recharge_step(1.0, irradiance_w_m2=0.0, wind_speed_m_s=5.0, is_perched=True)
        b.recharge_step(1.0, irradiance_w_m2=0.0, wind_speed_m_s=5.0, is_perched=True)
        self.assertGreater(a.v_cap, b.v_cap)

    def test_clipping_when_full(self):
        ps = dg.DeepGemPowerState(v_cap=3.3)
        out = ps.recharge_step(360.0, irradiance_w_m2=850.0)
        self.assertGreater(out["clipped_j"], 0.0)
        self.assertAlmostEqual(ps.v_cap, 3.3, places=6)

    def test_burst_feasibility_storage(self):
        ps = dg.DeepGemPowerState()
        ok, why = ps.burst_feasible(2.0, 0.040)
        self.assertFalse(ok)
        self.assertIn("usable", why)
        self.assertTrue(ps.burst_feasible(0.012, 0.0005)[0])

    def test_coin_cell_preset_peak_current(self):
        esr, leak, ipk = dg.STORAGE_PRESETS["coin_47mF"]
        ps = dg.DeepGemPowerState(esr_ohms=esr, quiescent_leakage_w=leak,
                                  peak_current_a=ipk)
        self.assertTrue(ps.burst_feasible(0.012, 0.0005)[0])   # ~4.5 mA < 14 mA
        ok, why = ps.burst_feasible(1.1, 0.040)                  # ~0.42 A
        self.assertFalse(ok)
        self.assertIn("peak", why)

    def test_energy_available_formula(self):
        ps = dg.DeepGemPowerState(v_cap=3.3)
        self.assertAlmostEqual(ps.energy_available_j(),
                               0.5 * 0.05 * (3.3 ** 2 - 2.0 ** 2), places=12)

    def test_burst_sweep_monotone_in_feasibility(self):
        rows = dg.burst_sweep(bursts=((0.012, 0.0005), (2.0, 0.040)), n_steps=480)
        self.assertTrue(rows[0]["feasible"])
        self.assertFalse(rows[1]["feasible"])
        self.assertEqual(rows[0]["brownout_rate"], 0.0)


# ===========================================================================
class TestModule4Classifier(unittest.TestCase):

    def test_label_ignores_energy_reserve(self):
        f = [0.3, 0.7, 0.2, 0.0, 0.9, 0.4]
        g = list(f)
        g[dg.ENERGY_RESERVE_INDEX] = 1.0
        self.assertEqual(dg.label_from_features(f), dg.label_from_features(g))
        for row in dg.LABEL_WEIGHTS:
            self.assertEqual(row[dg.ENERGY_RESERVE_INDEX], 0.0)

    def test_label_rule_uses_all_options(self):
        rng = random.Random(0)
        seen = {dg.label_from_features([rng.random() for _ in range(6)])
                for _ in range(5000)}
        self.assertGreaterEqual(len(seen), 5)

    def test_bayes_ceiling_formula(self):
        self.assertAlmostEqual(dg.bayes_ceiling_accuracy(), 0.85 + 0.15 / 7, places=12)

    def test_reflex_threshold_relative_to_chance(self):
        self.assertAlmostEqual(dg.reflex_threshold(), 1.5 / 7, places=12)
        self.assertGreater(dg.reflex_threshold(), 1.0 / dg.N_OPTIONS)

    def test_fit_temperature_recovers_known_t(self):
        rng = random.Random(3)
        true_t = 2.0
        logits, labels = [], []
        for _ in range(4000):
            lg = [rng.gauss(0, 3) for _ in range(7)]
            p = dg._logits_softmax(lg, true_t)
            u, c, y = rng.random(), 0.0, 6
            for i, pi in enumerate(p):
                c += pi
                if u <= c:
                    y = i
                    break
            logits.append(lg)
            labels.append(y)
        self.assertAlmostEqual(dg.fit_temperature(logits, labels), true_t, delta=0.15)

    def test_ece_small_for_calibrated_data(self):
        rng = random.Random(5)
        probs, labels = [], []
        for _ in range(6000):
            lg = [rng.gauss(0, 2) for _ in range(7)]
            p = dg._logits_softmax(lg, 1.0)
            u, c, y = rng.random(), 0.0, 6
            for i, pi in enumerate(p):
                c += pi
                if u <= c:
                    y = i
                    break
            probs.append(p)
            labels.append(y)
        self.assertLess(dg.expected_calibration_error(probs, labels), 0.03)

    def test_held_out_above_chance_below_ceiling(self):
        ho = default_run()["held_out_evaluation"]
        self.assertGreater(ho["drone_accuracy"], 0.5)
        self.assertLess(ho["drone_accuracy"], ho["bayes_ceiling"])
        self.assertGreater(ho["server_accuracy"], 1.0 / dg.N_OPTIONS + 0.2)

    def test_temperature_scaling_reduces_ece(self):
        ho = default_run()["held_out_evaluation"]
        self.assertLess(ho["drone_ece_fitted"], ho["drone_ece_T1"])
        self.assertLess(ho["server_ece_fitted"], ho["server_ece_T1"])

    def test_fitted_temperature_not_at_grid_edge(self):
        ho = default_run()["held_out_evaluation"]
        for k in ("drone_fitted_temperature", "server_fitted_temperature"):
            self.assertGreater(ho[k], 0.21)
            self.assertLess(ho[k], 4.99)

    def test_old_hash_label_is_floor_of_fraction(self):
        # documents why v7.3.1 labels were unlearnable: int(7S)%7 == floor(7*frac(S))
        rng = random.Random(9)
        for _ in range(5000):
            s = sum((i + 1) * rng.random() for i in range(6))
            self.assertEqual(int(s * 7) % 7, int(7 * (s - math.floor(s))))

    def test_brier_uniform_reference(self):
        t = dg.STARCalibrationTracker(dg.TACTICAL_OPTIONS)
        for y in range(7):
            t.add([1.0 / 7] * 7, y)
        self.assertAlmostEqual(t.compute_brier(), 6.0 / 7.0, places=12)

    def test_train_step_reduces_loss_on_repeat(self):
        clf = dg.STALinearClassifier.initialize(7, 6, dg.TACTICAL_OPTIONS, seed=0)
        f = [0.2, 0.8, 0.5, 1.0, 0.6, 0.1]
        first = clf.train_step(f, 3)
        for _ in range(50):
            last = clf.train_step(f, 3)
        self.assertLess(last, first)


# ===========================================================================
class TestIntegrationAndHonesty(unittest.TestCase):

    def test_deterministic(self):
        a = dg.run_hybrid_system_simulation(n_steps=240, held_out_n=200)
        b = dg.run_hybrid_system_simulation(n_steps=240, held_out_n=200)
        self.assertEqual(json.dumps(a, default=str), json.dumps(b, default=str))

    def test_cli_json_runs(self):
        out = subprocess.run([sys.executable, SRC, "--json", "--steps", "240"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        self.assertEqual(data["version"], "7.4")

    def test_cli_keeps_v731_flags(self):
        out = subprocess.run([sys.executable, SRC, "--methane-kg", "2", "--isotope",
                              "Ni-63", "--isotope-mg", "25", "--carbon-loop", "0.5",
                              "--steps", "240", "--seed", "7", "--json"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_solar_unchanged_from_v731(self):
        # solar is deterministic (no RNG) and the skin is unchanged from v7.3.1
        h = default_run()["energy_harvesting_totals"]
        self.assertAlmostEqual(h["cumulative_solar_joules"], 4488065.83, places=2)

    def test_honest_docstring(self):
        doc = dg.__doc__
        self.assertNotIn("FINAL", doc)
        self.assertNotIn("production release", doc.lower())
        self.assertIn("NOT production", doc)
        self.assertIn("flight", doc.lower())
        self.assertIn("license", doc.lower())
        self.assertIn("10 CFR 30.71", doc)

    def test_cc0_header_and_no_personal_info(self):
        import os
        import re
        with open(SRC) as fh:
            text = fh.read()
        self.assertIn("CC0", text[:300])
        self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text))  # no emails
        self.assertNotIn("/home/", text)
        # Optional private terms are supplied via the environment, never stored here.
        terms = os.environ.get("DEEPGEM_PRIVACY_TERMS", "")
        for term in filter(None, (t.strip().lower() for t in terms.split(","))):
            self.assertNotIn(term, text.lower())

    def test_tactical_dt_used_in_beta_night_check(self):
        s = default_run()["star_drone_tactical_module"]
        # 6 uJ every 30 s = 0.2 uW average brain draw
        self.assertAlmostEqual(s["brain_avg_power_uw_at_tactical_dt"], 0.2, places=6)
        # honest result: beta alone (~0.07 uW usable) can't cover brain + leakage
        self.assertFalse(s["beta_alone_sustains_brain"])
        self.assertGreater(s["beta_shortfall_factor"], 1.0)


if __name__ == "__main__":
    unittest.main()

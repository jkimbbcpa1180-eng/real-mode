# SPDX-License-Identifier: CC0-1.0
"""Tests for gnss_budget_v2 (python3 -m unittest). License: CC0 1.0 Universal."""
import contextlib
import io
import math
import os
import re
import unittest

import gnss_budget_v2 as B

HERE = os.path.dirname(os.path.abspath(__file__))
SC = {s.name: s for s in B.SCENARIOS}
WEATHER = (B.QUIET, B.ACTIVE, B.STORM)


def h(sw, name, env="suburban"):
    return B.compute_budget(sw, SC[name], environment=env)["position_gnss_only"]["horizontal_m"]


class TestBudget(unittest.TestCase):
    def test_original_crash_fixed(self):
        b = B.compute_budget(B.QUIET, SC["L1 only (bare)"], outage_s=2.0)     # imu default
        self.assertGreater(b["position_fused"]["horizontal_m"], 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            B.main()

    def test_sbas_never_worse_than_l1(self):
        for sw in WEATHER:
            for env in ("open", "suburban", "urban", "urban_canyon"):
                self.assertLess(h(sw, "L1 + SBAS (MSAS)", env), h(sw, "L1 only (bare)", env))
                self.assertLessEqual(h(sw, "L1 + L5 + SBAS", env), h(sw, "L1 + L5 dual-freq", env))
            terms = B.compute_budget(sw, SC["L1 + SBAS (MSAS)"])["terms_pr_m"]
            self.assertNotIn("sbas_geo", terms)                               # no stacked GEO term

    def test_l5_removes_most_iono(self):
        for sw in WEATHER:
            l1 = B.compute_budget(sw, SC["L1 only (bare)"])["terms_pr_m"]["iono"]
            l5 = B.compute_budget(sw, SC["L1 + L5 dual-freq"])["terms_pr_m"]["iono"]
            self.assertLess(l5, 0.15 * l1)

    def test_ppp_better_than_sbas(self):
        for sw in WEATHER:
            ppp = h(sw, "L1 + L5 + PPP (converged)")
            self.assertLess(ppp, h(sw, "L1 + SBAS (MSAS)"))
            self.assertLess(ppp, h(sw, "L1 + L5 + SBAS"))
            self.assertLess(ppp, h(sw, "L1 + L5 + PPP (not converged)"))

    def test_storm_ge_quiet(self):
        for name in SC:
            self.assertGreaterEqual(h(B.STORM, name), h(B.QUIET, name))
            self.assertGreaterEqual(h(B.ACTIVE, name), h(B.QUIET, name))

    def test_split_adds_up(self):
        for sw in WEATHER:
            for name, sc in SC.items():
                comps = B.pseudorange_components(sw, sc.cfg)
                sp = B.split_slow_fast(comps)
                total, _ = B.pseudorange_sigma_m(sw, sc.cfg)
                self.assertAlmostEqual(sp["slow_m"] ** 2 + sp["fast_m"] ** 2, total ** 2, places=9)
                g = B.drone_gps_model(sw, sc.cfg)
                k = sc.cfg.hdop / math.sqrt(2)
                self.assertAlmostEqual(g["bias_h_m"] ** 2 + g["white_h_m"] ** 2, (total * k) ** 2, places=3)
                self.assertTrue(min(c.tau_s for c in comps if c.slow_m > 0) <= sp["tau_s"]
                                <= max(c.tau_s for c in comps))

    def test_cep_definitions(self):
        p = B.position_error_from_pseudorange(1.0, B.GNSSConfig(hdop=1.0, vdop=1.0), {})
        self.assertAlmostEqual(p.horizontal_per_axis_m, 1 / math.sqrt(2), places=3)
        self.assertAlmostEqual(p.horizontal_cep50_m, 0.8326, places=3)        # 1.1774 / sqrt(2)
        self.assertAlmostEqual(p.vertical_cep50_m, 0.674, places=2)

    def test_receiver_noise_depends_on_cn0(self):
        E = B.ErrorSources
        self.assertGreater(E.receiver_noise_m(35.0), 2.5 * E.receiver_noise_m(45.0))
        self.assertTrue(0.2 < E.receiver_noise_m(45.0) < 0.6)


class TestIMU(unittest.TestCase):
    def test_noise_t15_bias_t2_no_rate(self):
        noise = B.IMUConfig(accel_bias_mps2=0.0, gyro_bias_dps=0.0, gyro_noise_dps_per_sqrt_hz=0.0)
        bias = B.IMUConfig(accel_noise_mps2_per_sqrt_hz=0.0, gyro_bias_dps=0.0, gyro_noise_dps_per_sqrt_hz=0.0)
        r_n = B.imu_position_drift_m(noise, 8.0) / B.imu_position_drift_m(noise, 2.0)
        r_b = B.imu_position_drift_m(bias, 8.0) / B.imu_position_drift_m(bias, 2.0)
        self.assertAlmostEqual(r_n, 4.0 ** 1.5, places=9)
        self.assertAlmostEqual(r_b, 4.0 ** 2, places=9)
        self.assertAlmostEqual(B.imu_position_drift_m(noise, 3.0), 0.05 * 3.0 ** 1.5 / math.sqrt(3), places=12)
        fast = B.IMUConfig(sample_rate_hz=1000.0)
        self.assertEqual(B.imu_position_drift_m(fast, 5.0), B.imu_position_drift_m(B.IMUConfig(), 5.0))
        self.assertGreater(B.imu_position_drift_m(B.IMUConfig(), 5.0),
                           B.imu_position_drift_m(B.IMUConfig(), 5.0, include_tilt=False))


class TestHygiene(unittest.TestCase):
    def test_privacy_and_header(self):
        for name in sorted(os.listdir(HERE)):
            if not name.endswith((".py", ".md")) or name == "gnss_imu_budget.py":
                continue
            text = open(os.path.join(HERE, name), encoding="utf-8").read()
            if name.endswith(".py"):
                self.assertIn("SPDX-License-Identifier: CC0-1.0", text[:200], name)
            self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text), name)
            for bad in ("/" + "home/", "/" + "workspace", "/" + "Users/"):
                self.assertNotIn(bad, text, name)
            terms = os.environ.get("GNSS_PRIVACY_TERMS", "")
            for term in filter(None, (t.strip().lower() for t in terms.split(","))):
                self.assertNotIn(term, text.lower(), name)


if __name__ == "__main__":
    unittest.main()

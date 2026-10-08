"""Tests for REAL MODE dispatch v2. Run: python3 -m unittest test_dispatch_v2 -v"""
import math
import unittest

import dispatch_v2 as d


def farm(**kw):
    return d.WakeAwareDispatchEngine(*d.demo_farm(), **kw)


class PhysicsTests(unittest.TestCase):
    def test_power_curve_cubic_in_v(self):
        t = d.TurbineUnit("T", 0, 0)
        self.assertEqual(d.power_curve_mw(t, 2.9), 0.0)
        self.assertAlmostEqual(d.power_curve_mw(t, 3.0), 0.0)
        self.assertEqual(d.power_curve_mw(t, 11.5), 3.5)
        self.assertEqual(d.power_curve_mw(t, 25.0), 0.0)
        self.assertAlmostEqual(d.power_curve_mw(t, 4.8), 3.5 * (4.8 ** 3 - 27) / (11.5 ** 3 - 27))
        self.assertAlmostEqual(d.power_curve_mw(t, 9.0), 1.6447, places=4)

    def test_overlap_fraction(self):
        self.assertEqual(d.circle_overlap_fraction(70, 60, 200), 0.0)
        self.assertEqual(d.circle_overlap_fraction(80, 60, 10), 1.0)
        lens = (2 * math.acos(0.5) - 0.5 * math.sqrt(3)) / math.pi  # equal radii, d = r
        self.assertAlmostEqual(d.circle_overlap_fraction(60, 60, 60), lens, places=9)

    def test_partial_overlap_scales_deficit(self):
        up = d.TurbineUnit("U", 0, 0)
        full = d.wake_geometry_factor(up, d.TurbineUnit("A", 350, 0), 0.0)
        part = d.wake_geometry_factor(up, d.TurbineUnit("B", 350, 80), 0.0)
        self.assertGreater(full, part)
        self.assertGreater(part, 0.0)

    def test_katic_sum_of_squares(self):
        self.assertAlmostEqual(d.katic_combine([0.3, 0.4]), 0.5)

    def test_wind_direction_rotation_removes_row_wakes(self):
        e = farm()
        r = e.dispatch(d.SiteConditions(9.0, wind_dir_deg=90.0))
        for tid in ("WTG_01", "WTG_02", "WTG_03"):
            self.assertAlmostEqual(r["turbine_allocation"][tid]["effective_wind_ms"], 9.0)

    def test_yaw_factor(self):
        self.assertAlmostEqual(d.yaw_factor(10.0), math.cos(math.radians(10)) ** d.YAW_EXPONENT)


class DispatchTests(unittest.TestCase):
    def test_no_fixed_floor_at_demo_point(self):
        r = farm().dispatch(d.SiteConditions(4.8, clear_sky_ghi=850.0, cloud_cover=0.15))
        self.assertGreater(r["generation_mw"]["wind_available"], 0.39)
        self.assertEqual(r["turbine_allocation"]["WTG_01"]["state"], "ATTACHED")
        self.assertEqual(r["wake_curtailed_turbines"], [])

    def test_grid_limit_curtails_to_limit(self):
        c = d.SiteConditions(12.0, clear_sky_ghi=850.0, local_demand_mw=4.0, export_capacity_mw=6.0)
        g = farm().dispatch(c)["generation_mw"]
        self.assertAlmostEqual(g["delivered"], 10.0, places=3)
        self.assertAlmostEqual(g["grid_curtailed"], g["total_available"] - 10.0, places=3)

    def test_wake_search_switches_off_when_it_pays(self):
        small = d.TurbineUnit("SMALL", 0, 0, rated_capacity_mw=0.3)
        big = d.TurbineUnit("BIG", 250, 0, rated_capacity_mw=8.0, rated_speed=13.0)
        e = d.WakeAwareDispatchEngine([small, big], [])
        r = e.dispatch(d.SiteConditions(10.0))
        self.assertEqual(r["wake_curtailed_turbines"], ["SMALL"])
        all_on = e.wind_state(10.0, e.wake_geometry(0.0), frozenset({0, 1}))[0]
        self.assertGreater(r["generation_mw"]["wind_available"], all_on + d.WAKE_SWITCH_MIN_GAIN_MW)

    def test_wake_search_respects_threshold(self):
        r = farm(min_gain_mw=0.0).dispatch(d.SiteConditions(9.0))
        self.assertEqual(r["wake_curtailed_turbines"], ["WTG_02"])  # tiny model gain only
        self.assertEqual(farm().dispatch(d.SiteConditions(9.0))["wake_curtailed_turbines"], [])

    def test_icing_flag_stops_turbine(self):
        chk = d.TurbineCameraCheck("WTG_04", icing=True)
        c = d.apply_cameras(d.SiteConditions(9.0), checks=[chk])
        r = farm().dispatch(c)
        self.assertEqual(r["turbine_allocation"]["WTG_04"]["state"], "STOPPED (ICING FLAG)")
        self.assertEqual(r["turbine_allocation"]["WTG_04"]["yield_mw"], 0.0)


class SolarTests(unittest.TestCase):
    def test_defaults_are_commercial(self):
        s = d.SolarArray("PV", 25000)
        self.assertEqual(s.module_efficiency, 0.25)
        self.assertEqual(s.performance_ratio, 0.82)
        self.assertAlmostEqual(d.solar_mw(s, 1000.0), 25000 * 1000 * 0.25 * 0.82 / 1e6)

    def test_measured_ghi_skips_attenuation(self):
        e = farm()
        a = e.dispatch(d.SiteConditions(0.0, clear_sky_ghi=850.0, cloud_cover=0.9, measured_ghi=500.0))
        self.assertEqual(a["inputs"]["ghi_used_w_m2"], 500.0)
        self.assertAlmostEqual(d.kasten_czeplak(800.0, 1.0), 200.0)

    def test_camera_soiling_replaces_pr_average(self):
        s = d.SolarArray("PV", 1000)
        self.assertAlmostEqual(d.solar_mw(s, 1000.0, soiling=d.SOILING_IN_PR), d.solar_mw(s, 1000.0))
        self.assertLess(d.solar_mw(s, 1000.0, soiling=0.10), d.solar_mw(s, 1000.0))


class CameraAndPrivacyTests(unittest.TestCase):
    def test_simulated_labels_flow_to_output(self):
        c = d.apply_cameras(d.SiteConditions(9.0), sky=d.SkyImagerNowcast(0.2, 0.16),
                            traffic=d.TrafficCountFeed(1800), base_demand_mw=6.0)
        srcs = farm().dispatch(c)["inputs"]["camera_sources"]
        self.assertEqual(len(srcs), 2)
        self.assertTrue(all("SIMULATED" in s for s in srcs))

    def test_traffic_feed_is_counts_only(self):
        with self.assertRaises(ValueError):
            d.TrafficCountFeed("12GA3456")
        with self.assertRaises(ValueError):
            d.TrafficCountFeed(-1)
        with self.assertRaises(ValueError):
            d.TrafficCountFeed(100, bin_minutes=1)
        self.assertEqual(set(d.TrafficCountFeed.__dataclass_fields__), {"vehicles_per_hour", "bin_minutes", "label"})

    def test_sky_imager_lead_time_window(self):
        with self.assertRaises(ValueError):
            d.SkyImagerNowcast(0.3, 0.1, lead_time_min=60)


class MonteCarloTests(unittest.TestCase):
    def test_reproducible_and_cameras_cut_solar_error(self):
        a, b = d.run_monte_carlo(1500), d.run_monte_carlo(1500)
        self.assertEqual(a, b)
        s = a["scenarios"]
        self.assertLess(s["with_cameras_SIMULATED"]["rmse_mw"]["solar"], s["without_cameras"]["rmse_mw"]["solar"])
        self.assertLess(s["with_cameras_SIMULATED"]["rmse_mw"]["demand"], s["without_cameras"]["rmse_mw"]["demand"])
        self.assertIn("SIMULATED", a["label"])

    def test_probabilistic_forecast_ordered(self):
        f = d.probabilistic_forecast(farm(), d.SiteConditions(9.0, clear_sky_ghi=850.0, cloud_cover=0.15), n=300)
        t = f["total_mw"]
        self.assertLessEqual(t["pct10"], t["pct50"])
        self.assertLessEqual(t["pct50"], t["pct90"])


if __name__ == "__main__":
    unittest.main()

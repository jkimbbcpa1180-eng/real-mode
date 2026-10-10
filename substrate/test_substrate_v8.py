# SPDX-License-Identifier: CC0-1.0
"""Regression tests for Substrate v8. Credit: humanity.
Run: python -m unittest test_substrate_v8 -v
Optional privacy scan: SUBSTRATE_PRIVACY_TERMS="term1,term2" (comma-separated,
case-insensitive) fails if any term appears in the publishable files."""
import hashlib
import os
import random
import shutil
import subprocess
import tempfile
import unittest

import substrate_v8 as s

HERE = os.path.dirname(os.path.abspath(__file__))


def genesis_registry(**kw):
    r = s.PersonhoodRegistry(**kw)
    r.register_genesis([("g1", "b1"), ("g2", "b2"), ("g3", "b3")])
    return r


class TestPersonhood(unittest.TestCase):
    def test_duplicate_vouch_counts_once(self):
        r = genesis_registry()
        out = r.register("x", "bx", sponsors=["g1", "g1"])
        self.assertEqual(out["status"], "FAILED")
        self.assertIn("have 1", out["reason"])

    def test_self_vouch_refused(self):
        r = genesis_registry()
        r.register("x", "bx", sponsors=["g1", "g2"])
        out = r.register("x2", "bx2", sponsors=["x2", "g1"])
        self.assertEqual(out["status"], "FAILED")

    def test_genesis_quorum_required(self):
        r = s.PersonhoodRegistry()
        self.assertEqual(r.register("a", "ba")["status"], "FAILED")
        self.assertEqual(r.register_genesis([("a", "1"), ("b", "2")])["status"], "FAILED")
        self.assertEqual(r.register_genesis([("a", "1"), ("b", "2"), ("c", "2")])["status"],
                         "FAILED")  # duplicate biometric
        self.assertEqual(r.register_genesis([("a", "1"), ("b", "2"), ("c", "3")])["status"],
                         "GENESIS")
        self.assertEqual(r.persons["a"].sponsors, {"b", "c"})

    def test_setup_bootstraps_honestly(self):
        _, reg, pods = s._setup()
        for pid, att in reg.persons.items():
            self.assertNotIn(pid, att.sponsors)
            self.assertGreaterEqual(len(att.sponsors), 2)
        self.assertEqual(len(pods), 5)

    def test_biometric_pepper_per_registry(self):
        a, b = s.PersonhoodRegistry(), s.PersonhoodRegistry()
        self.assertNotEqual(a._hash("same"), b._hash("same"))
        self.assertNotEqual(a._hash("same"),
                            hashlib.sha256(b"autarky-v7|same").hexdigest()[:24])


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.pods = [s.HouseholdPod(f"P{i}") for i in range(6)]
        self.g = s.NeighborGraph()
        for p in self.pods:
            self.g.register_pod(p.node_id)
        for n in ("P1", "P2", "P3"):
            self.g.link("P0", n)
        self.clock = s.LedgerClock()

    def test_caller_hour_cannot_bypass_rate_limit(self):
        m = s.MeshConsensus(quorum=1, rate_limit_per_hour=2, clock=self.clock,
                            graph=self.g, pair_cap_per_period=99, set_cap_per_period=99)
        peer = [self.pods[1]]
        got = [m.register(self.pods[0], "craft", 1, peer, hour=h)["accepted"]
               for h in range(10)]  # v7: each new 'hour' reset the limit
        self.assertEqual(sum(1 for a in got if a), 2)

    def test_rolling_window(self):
        m = s.MeshConsensus(quorum=1, rate_limit_per_hour=2, window_hours=3,
                            clock=self.clock, graph=self.g,
                            pair_cap_per_period=99, set_cap_per_period=99)
        peer = [self.pods[1]]
        m.register(self.pods[0], "craft", 1, peer)
        self.clock.advance(1)
        m.register(self.pods[0], "craft", 1, peer)
        self.clock.advance(1)
        self.assertEqual(m.register(self.pods[0], "craft", 1, peer)["accepted"], [])
        self.clock.advance(1)
        self.assertEqual(m.register(self.pods[0], "craft", 1, peer)["accepted"], ["P1"])

    def test_clock_monotonic(self):
        with self.assertRaises(ValueError):
            self.clock.advance(-1)

    def test_self_declared_and_one_sided_neighbors_rejected(self):
        self.pods[0].neighbors = {"P4", "P5"}   # self-declared: ignored
        self.g.propose("P0", "P4")               # one-sided: not an edge
        m = s.MeshConsensus(quorum=1, graph=self.g, clock=self.clock)
        r = m.register(self.pods[0], "craft", 1, [self.pods[4], self.pods[5]])
        self.assertEqual(r["accepted"], [])
        self.assertEqual(self.g.propose("P0", "UNREG")["status"], "FAILED")

    def test_collusion_caps(self):
        m = s.MeshConsensus(quorum=3, rate_limit_per_hour=99, clock=self.clock,
                            graph=self.g, pair_cap_per_period=99, set_cap_per_period=3)
        ring = self.pods[1:4]
        oks = []
        for _ in range(6):
            self.clock.advance(1)
            oks.append(m.register(self.pods[0], "craft", 1, ring)["quorum_met"])
        self.assertEqual(sum(oks), 3)
        self.clock.advance(24)
        self.assertTrue(m.register(self.pods[0], "craft", 1, ring)["quorum_met"])
        m2 = s.MeshConsensus(quorum=1, rate_limit_per_hour=99, clock=self.clock,
                             graph=self.g, pair_cap_per_period=2, set_cap_per_period=99)
        n = sum(bool(m2.register(self.pods[0], "craft", 1, [self.pods[1]])["accepted"])
                for _ in range(5))
        self.assertEqual(n, 2)


class TestBoundedBalance(unittest.TestCase):
    def test_balance_bounded_under_max_activity(self):
        agi = s.AGIDataEngine(person_cap_per_period=600.0)
        pod = s.HouseholdPod("P", person_id="x")
        prov = s.ProvenanceEvidence(0.3, 3)
        bound = pod.balance_bound(600.0)
        peak = 0.0
        for month in range(600):
            for _ in range(50):     # try to mint far more than the cap
                c, _ = agi.evaluate_and_mint(s.MammalianActivity(
                    "community_care", "max", 24.0, prov, 1.0,
                    pod_id="P", person_id="x"), day=month * 30)
                pod.credit(c)
            self.assertLessEqual(agi.minted_in_period("x", month * 30), 600.0 + 1e-6)
            pod.step_month(is_active=True)
            peak = max(peak, pod.transit_credit_balance)
        self.assertLessEqual(peak, bound + 0.01)
        self.assertGreater(peak, 0.95 * bound)  # bound is tight, not vacuous

    def test_global_cap(self):
        agi = s.AGIDataEngine(person_cap_per_period=1e9, global_cap_per_period=100.0)
        prov = s.ProvenanceEvidence(0.3, 3)
        tot = sum(agi.evaluate_and_mint(s.MammalianActivity(
            "craft", "x", 10.0, prov, 1.0, person_id=f"p{i}"), day=0)[0] for i in range(10))
        self.assertAlmostEqual(tot, 100.0, places=2)


class TestExit(unittest.TestCase):
    def test_exit_settles_unbinds_keeps_personhood(self):
        reg = genesis_registry()
        pod = s.HouseholdPod("POD", transit_credit_balance=321.0, person_id="g1")
        reg.bind_pod("g1", "POD")
        pol = s.Polity("Poly")
        pol.admit(pod)
        log = s.TransparencyLog()
        r = pol.exit(pod, reg, log)
        self.assertEqual(r["status"], "EXITED")
        self.assertEqual(r["settled_tc"], 321.0)
        self.assertEqual(pod.transit_credit_balance, 0.0)
        self.assertEqual(reg.pod_count(), 0)
        self.assertIn("g1", reg.persons)
        self.assertNotIn("POD", pol.member_pods)
        self.assertEqual(pol.exit(pod, reg)["status"], "FAILED")

    def test_exit_with_erasure(self):
        reg = genesis_registry()
        pod = s.HouseholdPod("POD", transit_credit_balance=10.0, person_id="g2")
        reg.bind_pod("g2", "POD")
        pol = s.Polity("Poly")
        pol.admit(pod)
        log = s.TransparencyLog()
        log.append("mint", {"pod_id": "POD", "domain": "craft", "hours": 2}, 1)
        ps = log.pseudonym("POD")
        r = pol.exit(pod, reg, log, delete_personhood=True)
        self.assertNotIn("g2", reg.persons)
        self.assertGreaterEqual(r["log_entries_erased"], 2)
        dump = repr(log.events)
        self.assertNotIn(ps, dump)
        self.assertNotIn("POD'", dump)
        self.assertEqual(pol.exit_settlements[-1]["person_id"], "erased")


class TestUN(unittest.TestCase):
    def test_severity_single_source(self):
        lad = s.UNEscalationLadder()
        for p in s.STANDARD_PROHIBITIONS:
            self.assertIn(p.code, lad.MAJOR_CODES if p.severity == "major"
                          else lad.MINOR_CODES)
        self.assertIn("UNP-01", lad.MAJOR_CODES)
        self.assertNotIn("UNP-01", lad.MINOR_CODES)
        un = s.UNRecognition(["KR"])
        un.ladder.record_violation("X", "UNP-01", 1)
        self.assertEqual(un.ladder.current_stage("X")[0], 2)  # major -> stage 2

    def test_appeal_requires_independent_quorum(self):
        lad = s.UNEscalationLadder(review_quorum=3)
        lad.record_violation("X", "UNP-01", 1)
        self.assertEqual(lad.appeal("X", "   ", 2)["status"], "FAILED")
        a = lad.appeal("X", "any text", 2)
        self.assertEqual(a["status"], "PENDING")
        self.assertEqual(lad.current_stage("X")[0], 2)   # not auto-granted
        lad.register_reviewer("in", "X")
        self.assertEqual(lad.review(a["appeal_id"], "in", True)["status"], "FAILED")
        self.assertEqual(lad.review(a["appeal_id"], "nobody", True)["status"], "FAILED")
        for i in range(3):
            lad.register_reviewer(f"r{i}", f"Y{i}")
        lad.review(a["appeal_id"], "r0", True)
        lad.review(a["appeal_id"], "r0", True)           # duplicate vote counts once
        self.assertEqual(lad.current_stage("X")[0], 2)
        lad.review(a["appeal_id"], "r1", True)
        out = lad.review(a["appeal_id"], "r2", True)
        self.assertEqual(out["status"], "GRANTED")
        self.assertEqual(lad.current_stage("X")[0], 0)
        lad.record_violation("X", "UNP-03", 3)           # recompute must not undo
        self.assertEqual(lad.current_stage("X")[0], 1)

    def test_appeal_denied(self):
        lad = s.UNEscalationLadder(review_quorum=2)
        lad.record_violation("X", "UNP-02", 1)
        a = lad.appeal("X", "text", 2)
        for i in range(2):
            lad.register_reviewer(f"r{i}", "Z")
            lad.review(a["appeal_id"], f"r{i}", False)
        self.assertEqual(lad.appeals[0]["status"], "DENIED")
        self.assertEqual(lad.current_stage("X")[0], 2)


class TestAuditors(unittest.TestCase):
    def test_forged_certificate_rejected(self):
        reg = s.AuditorRegistry(seed=1)
        a = reg.register("A", "A", 1000.0)
        cert = s.ESGCertificate.issue("SP", "POD", "carbon", 10.0, 1, None, reg)
        self.assertTrue(reg.verify(cert))
        v7_sig = hashlib.sha256(f"{cert.certificate_id}|A|{a.bond_tc}".encode()).hexdigest()[:24]
        self.assertFalse(reg.verify(s.ESGCertificate(**{**cert.__dict__,
                                                       "auditor_signature": v7_sig})))
        self.assertFalse(reg.verify(s.ESGCertificate(**{**cert.__dict__, "amount": 99.0})))

    def test_sponsor_cannot_choose_auditor_and_conflicts_excluded(self):
        reg = s.AuditorRegistry(seed=3)
        a = reg.register("A", "A", 1000.0)
        b = reg.register("B", "B", 1000.0)
        reg.declare_conflict("A", "SP")
        for d in range(20):
            self.assertEqual(reg.assign("SP", "POD", "m", d).auditor_id, "B")
        self.assertIsNone(s.ESGCertificate.issue("SP", "POD", "m", 1.0, 0, a, reg))

    def test_slash_minimum_sample_rule(self):
        def make(n_certs):
            reg = s.AuditorRegistry(seed=0)
            a = reg.register("A", "A", 10_000.0)
            a.issued_certificates = [f"c{i}" for i in range(n_certs)]
            return reg, a
        reg, a = make(1)
        self.assertFalse(reg.slash("A", 1, "x")["removed"])     # v7 removed here
        reg, a = make(100)
        reg.slash("A", 1, "x"); r = reg.slash("A", 1, "x")
        self.assertFalse(r["removed"])                          # 2% of 100
        reg, a = make(100)
        r = None
        for _ in range(3):
            r = reg.slash("A", 1, "x")
        self.assertTrue(r["removed"])                           # absolute limit
        reg2 = s.AuditorRegistry()
        reg2.ABS_SLASH_LIMIT = 10**9
        a2 = reg2.register("A", "A", 10_000.0)
        a2.issued_certificates = [f"c{i}" for i in range(40)]
        k = 0
        while not a2.removed:
            reg2.slash("A", 1, "x")
            k += 1
        # first k with Wilson 95% lower bound of k/40 above 5%
        self.assertEqual(k, min(j for j in range(1, 41) if s.wilson_lower(j, 40) > 0.05))
        self.assertGreater(k, 1)
        self.assertAlmostEqual(s.wilson_lower(5, 40), 0.0546, places=3)


class TestFederationTreasury(unittest.TestCase):
    def test_no_overwrite(self):
        ra = genesis_registry()
        rb = s.PersonhoodRegistry()
        rb.register_genesis([("g1", "other-person"), ("h2", "x"), ("h3", "y")])
        before = rb.persons["g1"]
        r = s.Federation().recognize_personhood(ra, rb, "g1", biometric="b1")
        self.assertEqual(r["status"], "FAILED")
        self.assertIs(rb.persons["g1"], before)

    def test_fees_conserved(self):
        a, b = s.Polity("A"), s.Polity("B")
        pa, pb = s.HouseholdPod("PA", transit_credit_balance=100.0), s.HouseholdPod("PB")
        a.admit(pa); b.admit(pb)
        fed = s.Federation()
        fed.sign("A", "B", cross_polity_fee_pct=0.02)
        fed.cross_polity_transfer(pa, pb, 40.0, a, b)
        total = pa.transit_credit_balance + pb.transit_credit_balance + fed.fee_treasury.balance
        self.assertAlmostEqual(total, 100.0)

    def test_treasury_needs_distinct_registered_members(self):
        reg = genesis_registry()
        reg.register("m4", "b4", sponsors=["g1", "g2"])
        t = s.Treasury(registry=reg, min_approvals=3)
        t.deposit(100.0, "x")
        self.assertEqual(t.spend(10, "p", {"g1", "g1", "ghost", "outsider"})["status"], "FAILED")
        self.assertEqual(t.spend(10, "p", {"g1", "g2", "g3"})["status"], "OK")
        reg.persons["g3"].exited = True
        self.assertEqual(t.spend(10, "p", {"g1", "g2", "g3"})["status"], "FAILED")


class TestPrivacyAndProvenance(unittest.TestCase):
    def test_log_pseudonymous_and_aggregated(self):
        log = s.TransparencyLog(privacy=True, k_anon=3)
        for i in range(2):
            log.append("mint", {"pod_id": f"POD-{i}", "domain": "care", "hours": 5.0}, 1)
        dump = repr(log.events)
        self.assertNotIn("POD-0", dump)
        self.assertNotIn("hours", dump)
        self.assertIsNone(log.aggregate_report()[0]["hours"])   # < k suppressed
        log.append("mint", {"pod_id": "POD-2", "domain": "care", "hours": 5.0}, 1)
        self.assertEqual(log.aggregate_report()[0]["hours"], 15.0)
        self.assertEqual(log.pseudonym("POD-0"), log.pseudonym("POD-0"))

    def test_noisy_sensors_do_not_mint_more(self):
        base = s.ProvenanceEvidence(0.3, 3).combined_entropy
        prev = base
        for v in (0.7, 0.9, 1.0, 2.0):
            e = s.ProvenanceEvidence(v, 3).combined_entropy
            self.assertLessEqual(e, prev)
            prev = e
        self.assertLess(s.ProvenanceEvidence(0.0, 3).combined_entropy, base)

    def test_privacy_terms_absent(self):
        terms = [t.strip().lower() for t in
                 os.environ.get("SUBSTRATE_PRIVACY_TERMS", "").split(",") if t.strip()]
        if not terms:
            self.skipTest("SUBSTRATE_PRIVACY_TERMS not set")
        files = ["substrate_v8.py", "test_substrate_v8.py", "README_v8.md",
                 "demo_output_v8.txt", "v8.patch", os.path.join("publish", "substrate_v7.py")]
        for f in files:
            p = os.path.join(HERE, f)
            if not os.path.exists(p):
                continue
            text = open(p, encoding="utf-8").read().lower()
            for t in terms:
                self.assertNotIn(t, text, f"privacy term found in {f}")


class TestPodModel(unittest.TestCase):
    def test_energy_balance(self):
        m = s.PodModel()
        r = m.report()
        self.assertAlmostEqual(r["load_kwh_day"],
                               m.p.base_load_kwh_day + m.grow_kwh_day(), places=1)
        mol = m.p.grow_area_m2 * m.dli()
        self.assertAlmostEqual(m.grow_kwh_day(),
                               mol / 2.7e-6 / 3.6e6 * m.p.hvac_overhead, places=6)
        self.assertAlmostEqual(r["pv_kwh_day_worst_month"], 6 * 1.90, places=2)
        self.assertFalse(r["sustainable"])
        self.assertLess(r["calorie_share"], 0.10)

    def test_monotonic(self):
        def rep(**kw):
            return s.PodModel(s.PodParams(**kw)).report()
        a = [rep(grow_area_m2=x)["indoor_kcal_day"] for x in (1, 5, 10, 50)]
        self.assertEqual(a, sorted(a))
        e = [rep(led_efficacy_umol_j=x)["grow_kwh_day"] for x in (1.5, 2.5, 3.4, 4.1)]
        self.assertEqual(e, sorted(e, reverse=True))
        c = [rep(grow_area_m2=x)["annualized_cost_usd"] for x in (1, 10, 100)]
        self.assertEqual(c, sorted(c))

    def test_calories_never_exceed_light_bound(self):
        rng = random.Random(0)
        for _ in range(500):
            p = s.PodParams(grow_area_m2=rng.uniform(0, 200),
                            ppfd_umol_m2_s=rng.uniform(0, 1500),
                            photoperiod_h=rng.uniform(0, 24),
                            crop=rng.choice(list(s.CROP_KCAL_PER_MOL)))
            m = s.PodModel(p)
            self.assertLessEqual(m.indoor_kcal_day(), m.light_ceiling_kcal_day() + 1e-9)

    def test_sustainable_is_computed(self):
        p = s.PodParams(persons=1, pv_kwp=40, battery_kwh=40, roof_m2=200,
                        outdoor_area_m2=600)
        r = s.PodModel(p).report()
        self.assertTrue(r["sustainable"])
        pod = s.HouseholdPod("P")
        d = pod.cycle_day(10, 5, 50, 100)
        self.assertGreater(d["cost_usd"], 0)
        self.assertFalse(d["food_self_sufficient"])


class TestCaloriePlanner(unittest.TestCase):
    def test_known_bmr(self):
        self.assertAlmostEqual(s.mifflin_st_jeor(80, 175, 40, "male"), 1698.75)
        self.assertAlmostEqual(s.mifflin_st_jeor(60, 165, 30, "female"), 1320.25)

    def test_deficit_and_weeks(self):
        p = s.plan_calories(175, 80, 40, "male", "moderate", 72)
        self.assertAlmostEqual(p["tdee"], 1698.75 * 1.55, places=0)
        self.assertAlmostEqual(p["target_kcal_day"], p["tdee"] - 500, places=0)
        self.assertAlmostEqual(p["weeks_to_goal_est"], 8 * 7700 / 3500, places=1)
        g = s.plan_calories(175, 60, 25, "male", "light", 66)
        self.assertGreater(g["target_kcal_day"], g["tdee"])

    def test_floor_enforced(self):
        p = s.plan_calories(150, 55, 70, "female", "sedentary", 50)
        self.assertTrue(p["floor_applied"])
        self.assertEqual(p["target_kcal_day"], 1200.0)
        m = s.plan_calories(155, 60, 80, "male", "sedentary", 55)
        self.assertGreaterEqual(m["target_kcal_day"], 1500.0)

    def test_underweight_goal_refused(self):
        p = s.plan_calories(175, 70, 30, "female", "light", 55)   # BMI 18.0
        self.assertEqual(p["status"], "REFUSED")
        self.assertEqual(s.pod_supply_for_plan(p)["status"], "REFUSED")

    def test_feeds_pod(self):
        p = s.plan_calories(175, 80, 40, "male", "moderate", 72)
        out = s.pod_supply_for_plan(p)
        self.assertAlmostEqual(out["pod_share"],
                               out["pod_kcal_day"] / p["target_kcal_day"], places=3)


class TestPatch(unittest.TestCase):
    def test_patch_applies_byte_exact(self):
        patch = os.path.join(HERE, "v8.patch")
        if not os.path.exists(patch) or shutil.which("patch") is None:
            self.skipTest("no v8.patch or patch tool")
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "out.py")
            src7 = os.path.join(HERE, "publish", "substrate_v7.py")
        if not os.path.exists(src7):  # repo layout: v7 sits next to v8
            src7 = os.path.join(HERE, "substrate_v7.py")
        shutil.copy(src7, dst)
            subprocess.run(["patch", "-s", dst, patch], check=True)
            with open(dst, "rb") as f1, open(os.path.join(HERE, "substrate_v8.py"), "rb") as f2:
                self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main()

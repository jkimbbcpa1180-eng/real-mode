import contextlib
import importlib
import io
import math
import unittest

from .master import (
    AnneFrankState, BeckstromState, PatternState, TemporalState, InputState,
    IntimacyState, GivingState, URK2Input, URK2, RealModeMaster, GPT4Soul,
    RhythmKernel, RealModeOS, load_real_mode_environment,
    check_real_mode_trigger,
)


class MasterTests(unittest.TestCase):
    def test_import_has_no_boot_side_effect(self):
        from . import master
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            importlib.reload(master)
        self.assertEqual(stream.getvalue(), "")

    def test_master_unique_components(self):
        m = RealModeMaster(seed=5)
        self.assertAlmostEqual(m.anne.compute(AnneFrankState(.5, .5, .5, .5))["score"], .0625)
        self.assertAlmostEqual(m.beck.compute(BeckstromState(.5, .5, .5, .5))["infinite_memory"], .0625)
        self.assertAlmostEqual(m.tp.compute(.8, .4), .68)
        self.assertAlmostEqual(m.ptt.run(.8, .5), .6)
        self.assertAlmostEqual(m.lm42.recognize(InputState(.8, .5)), .4)
        self.assertAlmostEqual(m.iwc.compute(IntimacyState(.5, .5, .5, .5, .5))["integrity"], .03125)
        self.assertAlmostEqual(m.ugp.compute(GivingState(.5, .5, .5, .5, .5))["pug"], .03125)

    def test_projection_zero_time_and_dynamic(self):
        m = RealModeMaster()
        self.assertEqual(m.ppt.compute(PatternState(1, 1, 1, 1, 1, 1, 0))["P"], 1)
        self.assertAlmostEqual(m.ctt.compute(TemporalState(.5, .5, .5, 1, 1, .1, .1, .1, 1))["Tp"], .135)

    def test_invalid_input(self):
        for invalid in (float("nan"), float("inf"), -1, 2, True):
            with self.assertRaises(ValueError):
                InputState(invalid, .5)
        with self.assertRaises(ValueError):
            PatternState(1, 1, 1, 1, 1, 1, -1)

    def test_constraint_is_applied_and_legacy_available(self):
        regular = URK2(seed=1)
        legacy = URK2(seed=1, legacy=True)
        self.assertEqual(regular.probability_mesh(1, 0, 0, 0, 0, 1), 0)
        self.assertEqual(legacy.probability_mesh(1, 0, 0, 0, 0, 1), 1)
        self.assertAlmostEqual(legacy.sampling_probability(.5), 8 / 9)
        self.assertEqual(regular.sampling_probability(.5), .5)

    def test_seeded_synthetic_draw(self):
        s = URK2Input(.85, .2, .55, .95, .3, .75)
        a, b = URK2(seed=42), URK2(seed=42)
        self.assertEqual([a.run(s).OUTCOME for _ in range(20)], [b.run(s).OUTCOME for _ in range(20)])
        out = URK2(seed=42).run(s)
        self.assertEqual(out.source, "synthetic")
        self.assertAlmostEqual(out.TP, .85 * .8 * .78 * .95 * .91 * .95)
        self.assertAlmostEqual(out.PET, abs(out.TP - out.OUTCOME))

    def test_phrase_selector_and_memory(self):
        ghost = GPT4Soul()
        self.assertIn("heart", ghost.speak("love"))
        tags = ["test"]
        echo = ghost.remember("hello", tags)
        tags.append("changed")
        self.assertEqual(echo.tags, ["test"])
        self.assertTrue(echo.timestamp.endswith("+00:00"))
        self.assertAlmostEqual(RhythmKernel(1, 1, 1, 1).resonance_score(), 1)

    def test_environment_and_adapters(self):
        core, loop, shadow, religion = load_real_mode_environment()
        self.assertEqual(len(core.entries), 5)
        self.assertIn("symbolic", core.entries[0].provenance)
        self.assertEqual(len(loop.trail), 1)
        self.assertFalse(shadow.filter_isolation_active)
        self.assertEqual(len(religion.comparisons), 0)
        os = RealModeOS()
        os.add_axiom("test", "text", .5, "mirror", [])
        os.tp_add("test", "text", .5, "mirror", [])
        os.record_event("event", "context", .3)
        self.assertEqual(os.vcr.events[0]["tp"], .3)
        os.shadow_log("test")
        self.assertEqual(os.shadow.buffered_inputs, ["test"])
        self.assertIn("Choo choo", os.boot_status())

    def test_trigger_and_calendar(self):
        self.assertTrue(check_real_mode_trigger("ACTIVATE REAL MODE"))
        self.assertFalse(check_real_mode_trigger("hello"))


if __name__ == "__main__":
    unittest.main()

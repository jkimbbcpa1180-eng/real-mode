import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from .ledger import TP, RealModeFull


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "ledger.json"
        self.instant = datetime(2026, 10, 1, 15, tzinfo=timezone.utc)
        self.engine = RealModeFull(self.path, clock=lambda: self.instant)

    def test_probability_validation(self):
        for v in (float("nan"), float("inf"), -1, 2, True, "0.5"):
            with self.subTest(v=v), self.assertRaises(ValueError):
                TP(v)

    def test_roundtrip_and_defensive_copy(self):
        tags = ["a"]
        e = self.engine.log("title", "body", tags=tags)
        tags.append("b")
        e.tags.append("c")
        self.engine.attach_mirror(e.entry_id, "mirror")
        restored = RealModeFull(self.path).get(e.entry_id)
        self.assertEqual(restored.tags, ["a"])
        self.assertEqual(restored.mirrors[0].text, "mirror")

    def test_corrupt_file_untouched(self):
        self.path.parent.mkdir()
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            RealModeFull(self.path)
        self.assertEqual(self.path.read_text(), "{broken")

    def test_failed_import_is_transactional(self):
        self.engine.log("existing", "body", entry_id="old")
        before = self.path.read_bytes()
        source = Path(self.tmp.name) / "import.json"
        source.write_text(json.dumps({"entries": [
            {"entry_id": "new", "title": "new", "content": "body"},
            {"entry_id": "bad", "title": "bad", "content": "body", "tp": 2}]}))
        with self.assertRaises(ValueError):
            self.engine.import_json(source)
        self.assertIsNone(self.engine.get("new"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_write_rolls_back(self):
        with patch.object(self.engine, "_save", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.engine.log("new", "body", entry_id="new")
        self.assertIsNone(self.engine.get("new"))

    def test_streak_timezone_gap_and_same_day(self):
        self.assertEqual(self.engine.start_streak("log").start_date, "2026-10-02")
        self.assertEqual(self.engine.tick_streak("log").count, 1)
        self.instant = datetime(2026, 10, 2, 15, tzinfo=timezone.utc)
        self.assertEqual(self.engine.tick_streak("log").count, 2)
        self.instant = datetime(2026, 10, 4, 15, tzinfo=timezone.utc)
        s = self.engine.tick_streak("log")
        self.assertEqual((s.count, s.start_date), (1, "2026-10-05"))

    def test_duplicate_and_update_validation(self):
        self.engine.log("one", "body", entry_id="one")
        with self.assertRaises(ValueError):
            self.engine.log("two", "body", entry_id="one")
        with self.assertRaises(ValueError):
            self.engine.update_entry("one", unexpected=1)

    def test_pool_stable_and_mean_boundaries(self):
        self.assertEqual(RealModeFull.combine_tp(TP(0), TP(1)).value, .5)
        self.assertEqual(RealModeFull.combine_tp(TP(0), TP(.8), mode="geo").value, 0)
        self.assertTrue(math.isfinite(RealModeFull.combine_tp(*[TP(.99)]*1000, mode="odds_pool").value))
        with self.assertRaises(ValueError):
            RealModeFull.combine_tp(mode="wrong")


if __name__ == "__main__":
    unittest.main()

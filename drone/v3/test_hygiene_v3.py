# SPDX-License-Identifier: CC0-1.0
"""v3 hygiene: SPDX/CC0 headers, no personal names, v2 files unchanged. CC0 1.0."""
import hashlib
import pathlib
import re
import unittest

HERE = pathlib.Path(__file__).resolve().parent
V2 = HERE.parent / "v2" if (HERE.parent / "v2").is_dir() else HERE.parent   # repo vs workspace layout

# sha256 of the v2 files when v3 work started (v3 must not modify them)
V2_SHA256 = {
    "drone_core_v2.py": "c0da1c9e5f3d26c5ff01784532e18c7608b27f64fe37fb899d5956fc2279e36b",
    "drone_uplink_v2.py": "309bf5fc60eff19bc0213c601d9bd1ad78c187ab29b62ac9d3b122d598c081d7",
    "drone_recon_v2.py": "d28e780dabe85d97ba9755a7645b195c7a1885ebfb3298ac32d9c99521679598",
    "drone_slam_v2.py": "aa27cc182bff1d38fec5fff9d10bca0af340ecff08738fe51480e882b465b215",
    "mc_recon_v2.py": "70251fb860f835a03615c062ddee074922edff4db3faee01c940a2e042adf314",
    "mc_slam_v2.py": "a31989a1e03513f8e6c97b953171abc87429dfd1355b934d12003421a72c1ecd",
    "make_demo_v2.py": "5b5c239ce3a3a97d3a4429cf9974ad393ca30240d3144a7dd2e544366d351e86",
    "make_parts_v2.py": "6e36ec074869bb95de44018dd162abe35fa03ac306d22665c5c4a7e1ada71193",
    "README.md": "0b998e6771ac02e6d679a2aa2c5579d5dc03145573619e7ee709ce95cc55e30e",
    "test_drone_core_v2.py": "1d7c48fd326f9600cd78977f9ad3b04814a1db9999c9043a9832da26ca920188",
    "test_drone_slam_v2.py": "794ef45438ac24ae090086aab181d1fefe09deb44c98dced72bb0acd07e0e052",
}


class TestHygieneV3(unittest.TestCase):
    def test_v2_files_unchanged(self):
        for name, h in V2_SHA256.items():
            self.assertEqual(hashlib.sha256((V2 / name).read_bytes()).hexdigest(), h, name)

    def test_spdx_and_no_personal_names(self):
        words = ["jo" + "hn", r"\b" + "ki" + "m" + r"\b", "an" + "san", "jki" + "mbb"]
        pat = re.compile("|".join(words), re.I)
        files = sorted(list(HERE.glob("*.py")) + list(HERE.glob("*.md")) + list(HERE.glob("*.txt")))
        self.assertGreater(len(files), 3)
        for f in files:
            txt = f.read_text()
            self.assertIsNone(pat.search(txt), f.name)
            if f.suffix == ".py":
                self.assertTrue(txt.startswith("# SPDX-License-Identifier: CC0-1.0"), f.name)


if __name__ == "__main__":
    unittest.main()

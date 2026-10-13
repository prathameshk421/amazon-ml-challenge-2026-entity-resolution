"""Romanizer tests (plan §9): goldens with Indic input, ASCII output, TIPPI, ZWNJ."""

import json
import unittest
from pathlib import Path

from ber.translit import classify, romanize

GOLDENS = json.loads((Path(__file__).resolve().parents[4] / "plan/reference/goldens.json")
                     .read_text(encoding="utf-8"))


class TestTranslit(unittest.TestCase):
    def test_indic_goldens_ascii(self):
        indic = [g for g in GOLDENS if g["had_indic"]]
        self.assertGreater(len(indic), 50)
        for g in indic:
            out, had = romanize(g["in"])
            self.assertTrue(had, g["in"])
            self.assertTrue(all(ord(c) < 128 or not 0x0900 <= ord(c) <= 0x0D7F for c in out), g["in"])

    def test_non_indic_untouched(self):
        self.assertEqual(romanize("Café 12"), ("Café 12", False))

    def test_tippi(self):
        self.assertEqual(classify("ੰ"), ("anusvara", ""))
        self.assertEqual(romanize("ਅਨੰਦ"), ("anand", True))

    def test_zwnj_not_indic(self):
        self.assertEqual(classify("‌")[0], "other")


if __name__ == "__main__":
    unittest.main()

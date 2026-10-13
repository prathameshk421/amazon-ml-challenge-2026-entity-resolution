"""Normalization tests (plan §9): 83 goldens, ASCII Indic output, idempotence, placeholders, ZWNJ,
S.A.S., st/ste classes, digit split, step 11b."""

import json
import random
import string
import unittest
from pathlib import Path

from ber.textnorm import normalize_field, skeleton

GOLDENS = json.loads((Path(__file__).resolve().parents[4] / "plan/reference/goldens.json")
                     .read_text(encoding="utf-8"))


class TestGoldens(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(GOLDENS), 83)

    def test_all_goldens(self):
        for g in GOLDENS:
            with self.subTest(inp=g["in"], kind=g["kind"]):
                out, had = normalize_field(g["in"], g["kind"])
                self.assertEqual(out, g["out"])
                self.assertEqual(had, g["had_indic"])
                if "skeletons" in g:
                    self.assertEqual([skeleton(t) for t in out.split()], g["skeletons"])

    def test_indic_outputs_ascii(self):
        for g in GOLDENS:
            if g["had_indic"]:
                self.assertTrue(normalize_field(g["in"], g["kind"])[0].isascii(), g["in"])


class TestRules(unittest.TestCase):
    def test_idempotent_ascii(self):
        # addresses only: the reference name path is not idempotent ("C O" -> "co" -> "company").
        rng = random.Random(3)
        alphabet = string.ascii_letters + string.digits + string.punctuation + " "
        for _ in range(1000):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            once = normalize_field(s, "addr")[0]
            self.assertEqual(normalize_field(once, "addr")[0], once, s)

    def test_placeholders(self):
        self.assertEqual(normalize_field("A<NULL>B", "addr")[0], "a b")
        self.assertEqual(normalize_field("<CITY_NAME> Homes", "name")[0], "cityname homes")

    def test_zwnj_deleted(self):
        self.assertEqual(normalize_field("ab‌cd", "name")[0], "abcd")

    def test_sas(self):
        self.assertEqual(normalize_field("BORDEAUX CLUB S.A.S.", "name")[0], "bordeaux club sas")
        self.assertEqual(normalize_field("S.A.S.", "addr")[0], "s a s")

    def test_st_ste_classes(self):
        self.assertEqual(normalize_field("12 Main St, St Louis", "addr")[0], "12 main st st louis")
        self.assertEqual(normalize_field("Suite 5, 100 Saint Street", "addr")[0], "ste 5 100 st st")
        self.assertEqual(normalize_field("Sainte Marie STE 4", "addr")[0], "ste marie ste 4")

    def test_digit_split(self):
        self.assertEqual(normalize_field("1945 214TH LN", "addr")[0], "1945 214 th lane")
        self.assertEqual(normalize_field("A12B", "name")[0], "a12b")

    def test_11b(self):
        m = {"r": "rue", "st": "street", "rd": "xx", "av": "avenue"}
        self.assertEqual(normalize_field("12 R. X Av St Rd", "addr", abbrev_map=m)[0],
                         "12 rue x avenue st road")
        self.assertEqual(normalize_field("12 R. X", "addr")[0], "12 r x")
        self.assertEqual(normalize_field("R Av", "name", abbrev_map=m)[0], "r av")


if __name__ == "__main__":
    unittest.main()

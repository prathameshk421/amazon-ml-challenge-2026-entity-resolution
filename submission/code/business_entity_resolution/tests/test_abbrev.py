"""Abbreviation-miner tests (plan §9): planted pairs recovered, low-frequency target rejected,
chain resolution, no cycles."""

import unittest

from ber.abbrev_mine import mean_jaccard, mine, resolve_chains

CFG = {"min_frac": 0.001, "short_max_len": 4, "cos_min": 0.95, "target_min_frac": 0.01}


def corpus():
    docs = []
    for short, long_ in (("rte", "route"), ("rd", "road")):
        docs += [["12", short, "x"]] * 100 + [["12", long_, "x"]] * 100
    docs += [["12", "pk", "x"]] * 50 + [["7", "pk", "y"]] * 50
    docs += [["12", "prk", "x"]] * 100 + [["7", "prk", "y"]] * 100
    docs += [["7", "park", "x"]] * 45 + [["7", "park", "y"]] * 55
    docs += [["5", "q", "z"]] * 30 + [["5", "quai", "z"]] * 3
    return docs


class TestMiner(unittest.TestCase):
    def test_planted(self):
        maps, detail = mine(corpus(), CFG)
        self.assertEqual(maps["rte"], "route")
        self.assertNotIn("rd", maps)            # already mapped by step 11 (ADDR_ABBREV)
        self.assertEqual(detail["rd"][0], "road")
        self.assertNotIn("q", maps)             # df(quai) = 3 < target_min_frac * N

    def test_target_floor_is_the_reason(self):
        maps, _ = mine(corpus(), dict(CFG, target_min_frac=0.0))
        self.assertEqual(maps["q"], "quai")

    def test_chain(self):
        maps, detail = mine(corpus(), CFG)
        self.assertEqual(detail["pk"][0], "prk")
        self.assertEqual(detail["prk"][0], "park")
        self.assertEqual(maps["pk"], "park")
        self.assertEqual(maps["prk"], "park")

    def test_resolve_chains(self):
        self.assertEqual(resolve_chains({"bd": "blvd", "blvd": "boulevard"}),
                         {"bd": "boulevard", "blvd": "boulevard"})
        self.assertEqual(resolve_chains({"a": "b", "b": "c", "c": "d", "d": "e"}),
                         {"a": "d", "b": "e", "c": "e", "d": "e"})

    def test_no_cycles(self):
        self.assertEqual(resolve_chains({"a": "ab", "ab": "a", "c": "a", "x": "xy"}), {"x": "xy"})


class TestE0Jaccard(unittest.TestCase):
    def test_skips_pairs_with_an_empty_side(self):
        pairs = [({"a", "b"}, {"b", "c"}), (set(), {"a"}), ({"a"}, set()), (set(), set()), ({"x"}, {"x"})]
        mean, skipped = mean_jaccard(pairs)
        self.assertAlmostEqual(mean, (1 / 3 + 1) / 2)
        self.assertEqual(skipped, 3)


if __name__ == "__main__":
    unittest.main()

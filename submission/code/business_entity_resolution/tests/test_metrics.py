"""§5.9 metric functions (plan §9)."""

import unittest

import numpy as np

from ber.evaluate import query_metrics, size_stats, summarize, summarize_segments


class TestQueryMetrics(unittest.TestCase):
    def test_singleton_ceiling_is_one(self):
        r, c = query_metrics([0], [0])
        self.assertTrue(np.isnan(r[0]))
        self.assertEqual(c[0], 1.0)

    def test_zero_recall_gives_zero(self):
        self.assertEqual(query_metrics([3], [0])[1][0], 0.0)

    def test_full_recall_gives_one(self):
        self.assertAlmostEqual(query_metrics([4], [4])[1][0], 1.0)

    def test_formula(self):
        r, c = query_metrics([2], [1])
        self.assertEqual(r[0], 0.5)
        self.assertAlmostEqual(c[0], 1.25 * 0.5 / 0.75)

    def test_readme_f05_formula(self):
        # README example: P = 2/3, R = 1 -> F0.5 = 0.714; the ceiling is the same formula with P = 1.
        p, r = 2 / 3, 1.0
        self.assertAlmostEqual(1.25 * p * r / (0.25 * p + r), 0.714, places=3)
        self.assertAlmostEqual(query_metrics([2], [2])[1][0], 1.25 * 1 * r / (0.25 + r))

    def test_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            query_metrics([1], [2])


class TestSummaries(unittest.TestCase):
    n_true = np.array([0, 2, 3, 1, 4])
    n_hit = np.array([0, 2, 1, 0, 4])
    sizes = np.array([0, 10, 20, 30, 40])

    def test_summarize(self):
        s = summarize(self.n_true, self.n_hit, self.sizes, n_pool=100)
        c = [1.0, 1.0, 1.25 * (1 / 3) / (0.25 + 1 / 3), 0.0, 1.0]
        self.assertAlmostEqual(s["f05_ceiling"], np.mean(c))
        self.assertEqual(s["entity_recall"], 3 / 4)
        self.assertEqual(s["full_recall"], 2 / 4)
        self.assertEqual(s["pair_recall"], 7 / 10)
        self.assertEqual((s["n_queries"], s["n_singletons"], s["n_true_pairs"]), (5, 1, 10))
        self.assertEqual(s["sizes"]["mean"], 20.0)
        self.assertEqual(s["sizes"]["p50"], 20.0)
        self.assertEqual(s["sizes"]["max"], 40)
        self.assertAlmostEqual(s["sizes"]["reduction_ratio"], 0.8)

    def test_all_singletons(self):
        s = summarize([0, 0], [0, 0], [0, 0], n_pool=10)
        self.assertEqual(s["f05_ceiling"], 1.0)
        self.assertIsNone(s["entity_recall"])
        self.assertIsNone(s["pair_recall"])

    def test_segments(self):
        masks = {"a": [True, True, False, False, False], "b": [False, False, True, True, True],
                 "none": [False] * 5}
        seg = summarize_segments(self.n_true, self.n_hit, self.sizes, 100, masks)
        self.assertEqual(seg["a"]["f05_ceiling"], 1.0)
        self.assertEqual(seg["b"]["pair_recall"], 5 / 8)
        self.assertEqual(seg["b"]["sizes"]["mean"], 30.0)
        self.assertEqual(seg["none"]["n_queries"], 0)
        self.assertIsNone(seg["none"]["f05_ceiling"])

    def test_size_stats_empty(self):
        self.assertIsNone(size_stats([], 10)["mean"])


if __name__ == "__main__":
    unittest.main()

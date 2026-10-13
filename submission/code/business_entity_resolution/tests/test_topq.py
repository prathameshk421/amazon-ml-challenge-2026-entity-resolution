"""top_q tie rule (plan §3.3, §9)."""

import unittest

import numpy as np

from ber.retrieve import top_q


def _ref(score, q):
    return sorted(range(len(score)), key=lambda i: (-score[i], i))[:q]


class TestTopQ(unittest.TestCase):
    def test_ties_by_index(self):
        self.assertEqual(top_q(np.array([1.0, 3.0, 3.0, 2.0, 3.0]), 2).tolist(), [1, 2])

    def test_q_at_least_n(self):
        s = np.array([0.5, 2.0, 0.5, 1.0])
        self.assertEqual(top_q(s, 4).tolist(), [1, 3, 0, 2])
        self.assertEqual(top_q(s, 10).tolist(), [1, 3, 0, 2])

    def test_all_equal(self):
        self.assertEqual(top_q(np.zeros(6), 3).tolist(), [0, 1, 2])

    def test_partition_boundary_ties(self):
        s = np.array([1.0, 4.0, 5.0, 4.0, 4.0])
        self.assertEqual(top_q(s, 2).tolist(), [2, 1])
        self.assertEqual(top_q(s, 3).tolist(), [2, 1, 3])
        self.assertEqual(top_q(s, 4).tolist(), [2, 1, 3, 4])

    def test_zero_and_empty(self):
        self.assertEqual(top_q(np.array([1.0, 2.0]), 0).tolist(), [])
        self.assertEqual(top_q(np.zeros(0), 5).tolist(), [])

    def test_random_against_sort(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 60))
            s = rng.integers(0, 5, n).astype(np.float64)
            q = int(rng.integers(1, 70))
            self.assertEqual(top_q(s, q).tolist(), _ref(s.tolist(), q))


if __name__ == "__main__":
    unittest.main()

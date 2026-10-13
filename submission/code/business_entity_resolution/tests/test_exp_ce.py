"""Cross-encoder feature group (exp.py `ce:<name>`): join on (s1_idx, pool_idx), list rank/gap, NaN, missing file."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import exp


class TestCeGroup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = mock.patch("ber.io_utils.paths", return_value={"work_root": self.root})
        self.patch.start()
        self.s1 = np.array([0, 0, 0, 5, 7], dtype=np.int32)
        self.pool = np.array([10, 11, 12, 10, 3], dtype=np.uint32)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def write(self, s1, pool, p):
        d = self.root / "ce" / "m"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"s1_idx": pa.array(s1, pa.int32()), "pool_idx": pa.array(pool, pa.uint32()),
                                 "p_ce": pa.array(p, pa.float32())}), d / "MATCH_VALID.parquet")

    def test_columns_and_aligned(self):
        self.write(self.s1, self.pool, [0.2, 0.9, 0.5, 0.4, 0.1])
        cols, _ = exp.group_spec("ce:m")
        self.assertEqual(cols, ["p_m", "p_m_rank", "p_m_gap"])
        G = exp.load_group("MATCH_VALID", "ce:m", None, self.s1, self.pool)
        np.testing.assert_allclose(G[:, 0], [0.2, 0.9, 0.5, 0.4, 0.1], rtol=1e-6)
        np.testing.assert_array_equal(G[:, 1], [3, 1, 2, 1, 1])
        np.testing.assert_allclose(G[:, 2], [0.7, 0.0, 0.4, 0.0, 0.0], atol=1e-6)

    def test_shuffled_and_missing_rows_join_by_key(self):
        # score file in another order and without (5, 10): that row must be NaN, others joined by key
        self.write([7, 0, 0, 0], [3, 12, 11, 10], [0.1, 0.5, 0.9, 0.2])
        G = exp.load_group("MATCH_VALID", "ce:m", None, self.s1, self.pool)
        np.testing.assert_allclose(G[[0, 1, 2, 4], 0], [0.2, 0.9, 0.5, 0.1], rtol=1e-6)
        self.assertTrue(np.isnan(G[3]).all())

    def test_missing_file_raises(self):
        with self.assertRaises(RuntimeError):
            exp.load_group("TEST", "ce:m", None, self.s1, self.pool)


if __name__ == "__main__":
    unittest.main()

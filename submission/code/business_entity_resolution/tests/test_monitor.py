"""Tests for ber.monitor (plan §5.10 M9)."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io_utils, monitor


class TestSummary(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(monitor._summary([]), {"n": 0, "mean": None, "p50": None, "p90": None})

    def test_drops_nan(self):
        s = monitor._summary([1.0, np.nan, 3.0])
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["mean"], 2.0)


class TestPerCountryStats(unittest.TestCase):
    def test_shares(self):
        country = np.array(["US", "US", "France", "France"])
        size = np.array([5, 0, 0, 0])
        max_p = np.array([0.9, np.nan, 0.1, 0.05])
        rank_p = {r: np.full(4, 0.5) for r in monitor.RANKS}
        n_dropped = np.array([0, 0, 1, 0])
        out = monitor.per_country_stats(country, size, max_p, rank_p, n_dropped)
        self.assertEqual(out["US"]["share_empty"], 0.5)
        self.assertEqual(out["France"]["share_empty"], 1.0)
        self.assertEqual(out["France"]["share_hit_postings_limit"], 0.5)
        self.assertEqual(out["US"]["n_queries"], 2)


class TestAlarmsFor(unittest.TestCase):
    def test_low_max_p_triggers_alarm(self):
        per_country = {
            "US": {"max_p": {"mean": 0.9}, "share_empty": 0.01},
            "India": {"max_p": {"mean": 0.8}, "share_empty": 0.02},
            "France": {"max_p": {"mean": 0.1}, "share_empty": 0.02},
        }
        ref_max_p, ref_empty, alarms = monitor.alarms_for(per_country, ["US", "India"])
        self.assertAlmostEqual(ref_max_p, 0.85)
        self.assertEqual(len(alarms), 1)
        self.assertIn("France", alarms[0])
        self.assertIn("max-p", alarms[0])

    def test_high_empty_share_triggers_alarm(self):
        per_country = {
            "US": {"max_p": {"mean": 0.9}, "share_empty": 0.01},
            "India": {"max_p": {"mean": 0.9}, "share_empty": 0.01},
            "France": {"max_p": {"mean": 0.85}, "share_empty": 0.1},
        }
        _, ref_empty, alarms = monitor.alarms_for(per_country, ["US", "India"])
        self.assertAlmostEqual(ref_empty, 0.01)
        self.assertEqual(len(alarms), 1)
        self.assertIn("empty-list", alarms[0])

    def test_no_alarm_when_within_bounds(self):
        per_country = {"US": {"max_p": {"mean": 0.9}, "share_empty": 0.01},
                       "India": {"max_p": {"mean": 0.85}, "share_empty": 0.015}}
        _, _, alarms = monitor.alarms_for(per_country, ["US", "India"])
        self.assertEqual(alarms, [])

    def test_no_train_covered_countries_no_alarms(self):
        per_country = {"France": {"max_p": {"mean": 0.1}, "share_empty": 0.5}}
        ref_max_p, ref_empty, alarms = monitor.alarms_for(per_country, [])
        self.assertIsNone(ref_max_p)
        self.assertIsNone(ref_empty)
        self.assertEqual(alarms, [])


class TestRunMonitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        out = Path("work/candidates/TEST")
        out.mkdir(parents=True)
        n_ranks = {f"p_rank{r}": pa.array([0.9, 0.05], pa.float32()) for r in monitor.RANKS}
        pq.write_table(pa.table({
            "s1_idx": pa.array([0, 1], pa.int32()), "country_raw": ["US", "France"],
            "n_union": pa.array([100, 50], pa.int32()), "n_dropped": pa.array([0, 0], pa.int32()),
            "size": pa.array([10, 0], pa.int32()), "max_p": pa.array([0.9, 0.05], pa.float32()),
            **n_ranks}), out / "query_stats.parquet")

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_missing_stats_raises(self):
        (Path("work/candidates/TEST") / "query_stats.parquet").unlink()
        with self.assertRaises(RuntimeError):
            monitor.run_monitor(None)

    def test_writes_metrics_with_alarm(self):
        monitor.run_monitor(None)
        doc = io_utils.read_json(io_utils.paths()["metrics_root"] / "monitor_test.json")
        self.assertIn("France", doc["per_country"])
        self.assertTrue(any("France" in a for a in doc["alarms"]))


if __name__ == "__main__":
    unittest.main()

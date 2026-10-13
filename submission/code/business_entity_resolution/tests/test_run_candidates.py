"""Tests for ber.run_candidates (plan §7 `candidates`): the staleness guard shared with rerank-eval,
end-to-end candidate generation for a train-derived set and for TEST (incl. query_stats.parquet, which
monitor.py depends on), and --limit truncation. Small synthetic fixtures only; never touches work/ from
a real run."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import index_build, io_utils, rerank, retrieve, run_candidates
from tests.test_rerank import _write_pool_store


def _cfg():
    cfg = copy.deepcopy(io_utils.load_config())
    cfg["max_workers"] = 2
    cfg["build_df_max_frac"] = 0.9
    cfg["families"] = ["EN", "NT", "AT"]
    cfg["cap_frac"] = {F: 0.9 for F in cfg["cap_frac"]}
    cfg["edge_tokens"]["min_frac"] = 0.5
    return cfg


def _build_split(split, cfg, s1_entity_ids, s1_names, s1_addrs, s1_countries,
                 pool_names, pool_addrs, pool_countries, pool_indic, pool_src, pool_num):
    w = Path("work")
    (w / "records").mkdir(parents=True, exist_ok=True)
    (w / "norm").mkdir(parents=True, exist_ok=True)
    n_s1, n_pool = len(s1_entity_ids), len(pool_names)
    pq.write_table(pa.table({
        "s1_idx": pa.array(range(n_s1), pa.int32()), "entity_id": s1_entity_ids,
        "country_raw": s1_countries}), w / f"records/{split}_s1.parquet")
    pq.write_table(pa.table({
        "pool_idx": pa.array(range(n_pool), pa.uint32()), "name_raw": pool_names,
        "addr_raw": pool_addrs}), w / f"records/{split}_pool.parquet")
    pq.write_table(pa.table({
        "s1_idx": pa.array(range(n_s1), pa.int32()), "name_norm": s1_names, "addr_norm": s1_addrs}),
        w / f"norm/{split}_s1.parquet")
    np.save(w / f"norm/{split}_s1_chain.npy", np.ones(n_s1, dtype=np.int32))
    pq.write_table(pa.table({
        "pool_idx": pa.array(range(n_pool), pa.uint32()), "name_norm": pool_names,
        "addr_norm": pool_addrs, "name_indic": pool_indic, "addr_indic": [False] * n_pool}),
        w / f"norm/{split}_pool.parquet")
    _write_pool_store(w / f"norm/{split}_pool_store", pool_names, pool_addrs, pool_countries,
                      pool_indic, pool_src, pool_num)
    index_build.build_split_index(w / f"norm/{split}_pool.parquet", w / f"index/{split}",
                                  index_build.split_families(split, cfg), cfg, 2)


def _write_stage1_model(cfg):
    r = retrieve.Retriever("train", cfg)
    model = {"features": r.feature_names, "families": r.families, "coef": [0.0] * len(r.feature_names),
             "intercept": 0.0, "mean": [0.0] * len(r.feature_names), "std": [1.0] * len(r.feature_names),
             "config_hash": io_utils.config_hash(cfg), "stage1_config": retrieve.stage1_config(cfg)}
    io_utils.atomic_write_json(io_utils.work_path("stage1", "model_v1.json"), model)
    return model


def _write_rerank_model(cfg, model, n_features):
    feat_names = [f"f{i}" for i in range(n_features)]
    X = np.random.default_rng(0).random((20, n_features)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 1},
                        lgb.Dataset(X, label=y, feature_name=feat_names), num_boost_round=2)
    booster.save_model(str(io_utils.work_path("rerank", "model_v1.txt")))
    meta = {"features": feat_names, "stage1_config_hash": io_utils.config_hash(retrieve.stage1_config(cfg)),
           "stage1_model_hash": io_utils.config_hash(model)}
    io_utils.atomic_write_json(io_utils.work_path("rerank", "model_v1_meta.json"), meta)
    return feat_names


class TestStage2ModelGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.cfg = copy.deepcopy(io_utils.load_config())
        self.cfg["max_workers"] = 2
        self._blocking = io_utils.BLOCKING_JSON
        io_utils.BLOCKING_JSON = Path("blocking.json").resolve()
        io_utils.atomic_write_json(io_utils.BLOCKING_JSON, self.cfg)
        self.model = {"features": retrieve.feature_names(retrieve.kept_families(self.cfg)),
                     "families": retrieve.kept_families(self.cfg), "coef": [0.0], "intercept": 0.0,
                     "mean": [0.0], "std": [1.0], "config_hash": io_utils.config_hash(self.cfg),
                     "stage1_config": retrieve.stage1_config(self.cfg)}
        io_utils.atomic_write_json(io_utils.work_path("stage1", "model_v1.json"), self.model)

    def tearDown(self):
        io_utils.BLOCKING_JSON = self._blocking
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_stale_config_hash_raises(self):
        meta = {"features": [], "stage1_config_hash": "not-the-real-hash",
                "stage1_model_hash": io_utils.config_hash(self.model)}
        io_utils.atomic_write_json(io_utils.work_path("rerank", "model_v1_meta.json"), meta)
        with self.assertRaises(RuntimeError) as ctx:
            run_candidates._stage2_model(self.cfg)
        self.assertIn("stale", str(ctx.exception))

    def test_no_stage1_model_raises(self):
        os.remove(io_utils.work_path("stage1", "model_v1.json"))
        with self.assertRaises(FileNotFoundError) as ctx:
            run_candidates._stage2_model(self.cfg)
        self.assertIn("stage1-train", str(ctx.exception))


class TestRunCandidatesEndToEnd(unittest.TestCase):
    """Tiny real train + test indices/models; run_candidates for a train-derived set and for TEST."""

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        cls._tmp = tempfile.TemporaryDirectory()
        os.chdir(cls._tmp.name)
        cls.cfg = _cfg()
        cls._blocking = io_utils.BLOCKING_JSON
        io_utils.BLOCKING_JSON = Path("blocking.json").resolve()
        io_utils.atomic_write_json(io_utils.BLOCKING_JSON, cls.cfg)

        pool_names = ["happy nails", "happy nailz", "zz noise", "kumar traders", "other biz"]
        pool_addrs = ["1 oak st springfield", "1 oak st springfield", "9 elm st", "", "5 pine ave"]
        pool_countries = ["US", "US", "US", "India", "India"]
        pool_indic = [False, False, False, True, False]
        pool_src = [2, 2, 3, 2, 3]
        pool_num = [1, 2, 3, 4, 5]
        for split, s1_names in (("train", ["happy nail"]), ("test", ["happy nail", "zz"])):
            n_s1 = len(s1_names)
            _build_split(split, cls.cfg, [f"S1-{i}" for i in range(n_s1)], s1_names,
                        ["1 oak st"] * n_s1, ["US"] * n_s1, pool_names, pool_addrs, pool_countries,
                        pool_indic, pool_src, pool_num)
        np.save(Path("work/records/train_s1_nmatch.npy"), np.array([1], dtype=np.int16))
        pq.write_table(pa.table({"s1_idx": pa.array([0], pa.int32()),
                                 "pool_idx": pa.array([0], pa.uint32())}),
                      Path("work/records/train_gt.parquet"))
        io_utils.atomic_write_text(
            io_utils.work_path("splits", "query_sets.json"),
            json.dumps({"sets": {"MATCH_VALID": [0], "HOLDOUT_TEST": [0]}}))

        cls.model = _write_stage1_model(cls.cfg)
        r = retrieve.Retriever("train", cls.cfg, cls.model)
        cls.feat_names = _write_rerank_model(cls.cfg, cls.model, len(rerank.feature_names(r)))

    @classmethod
    def tearDownClass(cls):
        io_utils.BLOCKING_JSON = cls._blocking
        os.chdir(cls._cwd)
        cls._tmp.cleanup()

    def test_train_derived_set(self):
        run_candidates.run_candidates(SimpleNamespace(set="MATCH_VALID", limit=None))
        out_dir = io_utils.work_path("candidates", "MATCH_VALID")
        parts = sorted(out_dir.glob("part-*.parquet"))
        self.assertTrue(parts)
        self.assertTrue((out_dir / "part-00000._SUCCESS").exists())
        self.assertFalse((out_dir / "query_stats.parquet").exists())  # only TEST collects this
        doc = io_utils.read_json(io_utils.paths()["metrics_root"] / "candidates_MATCH_VALID.json")
        self.assertEqual(doc["n_queries"], 1)
        self.assertEqual(doc["split"], "train")

    def test_val_s1_alias_maps_to_holdout_test_dir(self):
        run_candidates.run_candidates(SimpleNamespace(set="val-S1", limit=None))
        self.assertTrue(io_utils.work_path("candidates", "HOLDOUT_TEST", "part-00000.parquet").exists())

    def test_test_set_writes_query_stats(self):
        run_candidates.run_candidates(SimpleNamespace(set="TEST", limit=None))
        out_dir = io_utils.work_path("candidates", "TEST")
        self.assertTrue((out_dir / "query_stats.parquet").exists())
        stats = pq.read_table(out_dir / "query_stats.parquet").to_pydict()
        self.assertEqual(sorted(stats["s1_idx"]), [0, 1])
        doc = io_utils.read_json(io_utils.paths()["metrics_root"] / "candidates_TEST.json")
        self.assertEqual(doc["n_queries"], 2)

    def test_limit_truncates_queries(self):
        run_candidates.run_candidates(SimpleNamespace(set="TEST", limit=1))
        doc = io_utils.read_json(io_utils.paths()["metrics_root"] / "candidates_TEST.json")
        self.assertEqual(doc["n_queries"], 1)
        self.assertTrue(doc["limited"])

    def test_unknown_set_raises(self):
        with self.assertRaises(ValueError):
            run_candidates.run_candidates(SimpleNamespace(set="NOT_A_SET", limit=None))


if __name__ == "__main__":
    unittest.main()

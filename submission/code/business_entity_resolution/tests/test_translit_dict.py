"""Learned-dictionary and normalize-command tests (plan §9, §5.3, §5.2 steps 11b/13): planted entry
recovered, count/ratio thresholds, JW floor, ties, usable-token gate, per-country abbreviation map,
dictionary only on had_indic fields, and a tiny end-to-end normalize -> learn-dict -> normalize run."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io_utils
from ber.textnorm import normalize_records, run_normalize, s1_chain
from ber.translit_dict import (first_indic_script, gate_decision, learn, name_tokens, pool_df,
                               run_learn_dict, shares, usable_flags)

HIT = (["kumar"], ["kumaar", "traders"])    # JW(kumar, kumaar) = 0.967
MISS = (["kumar"], ["zzzz"])                 # JW = 0 -> counts in c(r) only


class TestLearner(unittest.TestCase):
    def test_planted_recovered(self):
        self.assertEqual(learn([HIT] * 5), {"kumar": "kumaar"})

    def test_count_threshold(self):
        self.assertEqual(learn([HIT] * 4), {})

    def test_ratio_threshold(self):
        self.assertEqual(learn([HIT] * 5 + [MISS]), {"kumar": "kumaar"})    # 5/6 = 0.83
        self.assertEqual(learn([HIT] * 5 + [MISS] * 2), {})                  # 5/7 = 0.71

    def test_low_jw_not_counted(self):
        self.assertEqual(learn([(["abc"], ["xyz"])] * 10), {})

    def test_tie_first_s1_token(self):
        self.assertEqual(learn([(["ab"], ["ac", "ad"])] * 5), {"ab": "ac"})
        self.assertEqual(learn([(["ab"], ["ad", "ac"])] * 5), {"ab": "ad"})

    def test_verbatim_and_short_tokens_skipped(self):
        self.assertEqual(learn([(["kumar", "x"], ["kumar", "xy"])] * 6), {})

    def test_distinct_per_pair(self):
        self.assertEqual(learn([(["kumar", "kumar"], ["kumaar"])] * 3), {})


class TestGate(unittest.TestCase):
    def test_usable_and_decision(self):
        df = {"rare": 1, "common": 100}
        pairs = [({"rare", "x"}, {"rare"}), ({"common"}, {"common"}), ({"a"}, {"b"})]
        flags = usable_flags(pairs, df, cap=10)
        self.assertEqual(flags.tolist(), [True, False, False])
        before = shares(flags, ["DEVANAGARI", "BENGALI", "BENGALI"])
        self.assertAlmostEqual(before["overall"], 1 / 3)
        self.assertEqual(before["per_script"]["BENGALI"], {"n": 2, "share": 0.0})
        up = shares([True, False, True], ["DEVANAGARI", "BENGALI", "BENGALI"])
        down = shares([False, True, True], ["DEVANAGARI", "BENGALI", "BENGALI"])
        self.assertTrue(gate_decision(before, up))
        self.assertFalse(gate_decision(before, down))   # overall up, DEVANAGARI down
        self.assertFalse(gate_decision(before, before))

    def test_pool_df_dict_only_on_indic(self):
        before, after = pool_df(["kumar shop", "kumar", "kumaar"], [True, False, False],
                                {"kumar": "kumaar"}, {"kumar", "kumaar"})
        self.assertEqual(before, {"kumar": 2, "kumaar": 1})
        self.assertEqual(after, {"kumar": 1, "kumaar": 2})

    def test_helpers(self):
        self.assertEqual(name_tokens("a kumar kumar b", {"kumar": "kumaar"}), {"kumaar"})
        self.assertEqual(first_indic_script("Om কলকাতা"), "BENGALI")


class TestNormalizeRecords(unittest.TestCase):
    def test_abbrev_by_country_and_dict_on_indic_only(self):
        nn, an, ni, ai = normalize_records(
            ["कुमार", "kumar"], ["5 rte de paris", "5 rte de paris"], ["France", "US"],
            {"name": {"kumar": "kumaar"}, "addr": {}}, {"France": {"rte": "route"}})
        self.assertEqual(nn, ["kumaar", "kumar"])
        self.assertEqual(an, ["5 route de paris", "5 rte de paris"])
        self.assertEqual((ni, ai), ([True, False], [False, False]))

    def test_chain(self):
        self.assertEqual(s1_chain(pa.chunked_array([["a", "b"], ["a"]])).tolist(), [2, 1, 2])


def _write(path, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


class TestEndToEnd(unittest.TestCase):
    """normalize --no-dict -> learn-dict -> normalize on a tiny synthetic train split."""

    N_FILL, N_IND = 6000, 12   # cap = floor(0.002 * 6013) = 12 >= df(kumaar) after the dictionary

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        cfg = io_utils.load_config()
        cfg["max_workers"] = 2
        cfg["abbrev_mining"]["enabled"] = True
        # Pinned regardless of what E3 has tuned cap_frac.NT to in the real blocking.json: the fixture
        # size (N_FILL/N_IND) is calibrated against this exact value (see the class docstring above).
        cfg["cap_frac"]["NT"] = 0.002
        self.patch = mock.patch.object(io_utils, "load_config", return_value=cfg)
        self.patch.start()
        w = Path("work")
        n_s1 = self.N_IND + 1
        _write(w / "records/train_s1.parquet", {
            "s1_idx": pa.array(range(n_s1), pa.int32()), "entity_id": [f"S1-{i}" for i in range(n_s1)],
            "name_raw": ["Kumaar"] * self.N_IND + ["Other Co"],
            "addr_raw": ["1 Main Rd"] * n_s1, "country_raw": ["India"] * n_s1})
        names = [f"filler{i}" for i in range(self.N_FILL)] + ["कुमार"] * self.N_IND + ["kumar"]
        n_pool = len(names)
        addrs = ["5 rte de paris"] * n_pool
        countries = ["France"] * self.N_FILL + ["India"] * (n_pool - self.N_FILL)
        _write(w / "records/train_pool.parquet", {
            "pool_idx": pa.array(range(n_pool), pa.uint32()),
            "src": pa.array([2] * n_pool, pa.int8()), "num": pa.array(range(n_pool), pa.uint32()),
            "name_raw": names, "addr_raw": addrs, "country_raw": countries})
        _write(w / "records/train_gt.parquet", {
            "s1_idx": pa.array(range(self.N_IND), pa.int32()),
            "pool_idx": pa.array(range(self.N_FILL, self.N_FILL + self.N_IND), pa.uint32())})
        (w / "splits").mkdir()
        (w / "splits/query_sets.json").write_text(json.dumps(
            {"sets": {"DICT_SRC": list(range(8)), "DEV_A": list(range(8, 12))}}))
        (w / "norm").mkdir()
        (w / "norm/abbrev_train.json").write_text(json.dumps({"maps": {"France": {"rte": "route"}}}))

    def tearDown(self):
        self.patch.stop()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_pipeline(self):
        norm = Path("work/norm")
        run_normalize(SimpleNamespace(split="train", no_dict=True, no_abbrev=False))
        meta = json.loads((norm / "train_meta.json").read_text())
        self.assertEqual((meta["dict"], meta["abbrev"], meta["n_pool"]), (False, True, self.N_FILL + 13))
        pool = pq.read_table(norm / "train_pool.parquet").to_pydict()
        self.assertEqual(pool["name_norm"][self.N_FILL], "kumar")
        self.assertEqual(pool["addr_norm"][0], "5 route de paris")           # France map
        self.assertEqual(pool["addr_norm"][self.N_FILL], "5 rte de paris")   # India: no map
        store = norm / "train_pool_store"
        for f in ("name", "addr"):
            off = np.load(store / f"{f}.off.npy")
            data = (store / f"{f}.bin").read_bytes()
            got = [data[off[i]:off[i + 1]].decode() for i in range(off.size - 1)]
            self.assertEqual(got, pool[f"{f}_norm"])
        codes = np.load(store / "country.npy")
        labels = json.loads((store / "countries.json").read_text())
        self.assertEqual(codes.dtype, np.int16)
        self.assertEqual([labels[c] for c in codes[[0, -1]]], ["France", "India"])
        self.assertEqual(np.load(store / "indic.npy").tolist(), pool["name_indic"])
        self.assertEqual(np.load(norm / "train_s1_chain.npy").tolist(), [self.N_IND] * self.N_IND + [1])

        run_learn_dict(SimpleNamespace())
        doc = json.loads(Path("work/translit/dict_v1.json").read_text())
        self.assertEqual(doc["name"], {"kumar": "kumaar"})
        self.assertTrue(doc["meta"]["gate_pass"])
        m = json.loads(Path("artifacts/metrics/translit_dict.json").read_text())
        self.assertEqual((m["before"]["overall"], m["after"]["overall"]), (0.0, 1.0))
        self.assertEqual(list(m["after"]["per_script"]), ["DEVANAGARI"])

        run_normalize(SimpleNamespace(split="train", no_dict=False, no_abbrev=False))
        pool = pq.read_table(norm / "train_pool.parquet").to_pydict()
        self.assertEqual(pool["name_norm"][self.N_FILL], "kumaar")   # Indic record
        self.assertEqual(pool["name_norm"][-1], "kumar")             # Latin record untouched
        self.assertTrue(json.loads((norm / "train_meta.json").read_text())["dict"])
        with self.assertRaises(RuntimeError):
            run_learn_dict(SimpleNamespace())   # requires a no-dict normalization


if __name__ == "__main__":
    unittest.main()

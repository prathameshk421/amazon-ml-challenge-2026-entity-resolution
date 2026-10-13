"""Index build / store / sanity tests on a tiny synthetic norm pool (plan §9 integration, §5.6)."""

import copy
import math
import os
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ber import index_build, io_utils
from ber.hashing import key64
from ber.index_store import IndexStore
from ber.keys import edge_token_set, record_keys

N = 240
FAMILIES = index_build.split_families("train", io_utils.load_config())


def _synthetic_pool(rng):
    heads = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "happynails",
             "international"]
    names, addrs = [], []
    for i in range(N):
        toks = rng.sample(heads, rng.randint(1, 3))
        if rng.random() < 0.6:
            toks.append("llc")  # frequent edge token: in G and over the test build limit in NT
        names.append("" if i % 50 == 7 else " ".join(toks))
        at = [str(rng.randint(1, 30)), rng.choice(["oak", "elm", "pine", "maple"]), "street"]
        if rng.random() < 0.3:
            at.append("springfield")
        addrs.append(" ".join(at))
    return pd.DataFrame({"pool_idx": np.arange(N, dtype=np.uint32), "name_norm": names,
                         "addr_norm": addrs, "name_indic": False, "addr_indic": False})


def _brute_force(df, cfg, families):
    """{F: ({key64: sorted pool_idx list} kept, sorted dropped keys)} with the same rules."""
    n = len(df)
    et = cfg["edge_tokens"]
    G = set(edge_token_set(df.name_norm, n, et["min_frac"], et["min_edge_share"]))
    at_df = Counter(t for a in df.addr_norm for t in set(a.split()))
    at_cap = math.floor(cfg["cap_frac"]["AT"] * n)
    limit = math.floor(cfg["build_df_max_frac"] * n)
    out = {}
    for F in families:
        post = defaultdict(set)
        for i, (nm, ad) in enumerate(zip(df.name_norm, df.addr_norm)):
            for k in record_keys(nm, ad, (F,), G, at_df, at_cap, "pool")[F]:
                post[key64(F, k)].add(i)
        kept = {k: sorted(v) for k, v in post.items() if len(v) <= limit}
        out[F] = (kept, sorted(k for k, v in post.items() if len(v) > limit))
    return out, at_df, at_cap


class TestIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        cls._tmp = tempfile.TemporaryDirectory()
        os.chdir(cls._tmp.name)
        cfg = copy.deepcopy(io_utils.load_config())
        cfg["build_df_max_frac"] = 0.3
        cfg["cap_frac"]["AT"] = 0.2
        cfg["edge_tokens"]["min_frac"] = 0.05
        cls.cfg = cfg
        cls.df = _synthetic_pool(random.Random(5))
        cls.norm = Path("work/norm/train_pool.parquet").resolve()
        cls.norm.parent.mkdir(parents=True)
        cls.df.to_parquet(cls.norm, index=False, row_group_size=50)
        cls.out = Path("work/index/train").resolve()
        cls._chunk, index_build.CHUNK = index_build.CHUNK, 37  # several chunks over several row groups
        cls.meta = index_build.build_split_index(cls.norm, cls.out, FAMILIES, cfg, 2)
        cls.expected, cls.at_df, cls.at_cap = _brute_force(cls.df, cfg, FAMILIES)

    @classmethod
    def tearDownClass(cls):
        index_build.CHUNK = cls._chunk
        os.chdir(cls._cwd)
        cls._tmp.cleanup()

    def _load(self, F):
        return [np.load(self.out / f"{F}.{x}.npy") for x in index_build.FILES]

    def test_postings_match_brute_force(self):
        for F in FAMILIES:
            k, ip, p, d = self._load(F)
            kept, dropped = self.expected[F]
            self.assertEqual(k.tolist(), sorted(kept), F)
            got = {int(key): p[ip[j]:ip[j + 1]].tolist() for j, key in enumerate(k)}
            self.assertEqual(got, kept, F)
            self.assertEqual(d.tolist(), dropped, F)
            self.assertIsNone(index_build.check_family(self.out, F, N), F)
        self.assertEqual(sorted(os.listdir(self.out)),
                         sorted([f"{F}.{x}.npy" for F in FAMILIES for x in index_build.FILES] + ["meta.json"]))

    def test_dropped_sentinel(self):
        store = IndexStore("train", ["NT", "XK"])
        llc = key64("NT", "llc")
        self.assertIn(llc, self.expected["NT"][1])
        present = next(iter(self.expected["NT"][0]))
        df = store.df("NT", np.array([llc, present, key64("NT", "zzzz")], dtype=np.uint64))
        self.assertEqual(df.tolist(), [N + 1, len(self.expected["NT"][0][present]), 0])
        posts = store.postings("NT", np.array([llc, present], dtype=np.uint64))
        self.assertEqual([x.tolist() for x in posts], [[], self.expected["NT"][0][present]])
        self.assertEqual(self.meta["families"]["NT"]["n_dropped"], len(self.expected["NT"][1]))
        self.assertIn("llc", store.G)

    def test_at_df_lookup(self):
        store = IndexStore("train", ["XK"])
        limit = math.floor(self.cfg["build_df_max_frac"] * N)
        self.assertEqual(store.at_cap, self.at_cap)
        self.assertGreater(self.at_df["street"], limit)
        self.assertEqual(store.at_df_lookup.get("street", 0), N)
        self.assertEqual(store.at_df_lookup.get("nowhere", 0), 0)
        for t, c in self.at_df.items():
            if c <= limit:
                self.assertEqual(store.at_df_lookup.get(t, 0), c, t)
        self.assertTrue(any(self.at_cap < c <= limit for c in self.at_df.values()))

    def test_sanity(self):
        res = index_build.index_sanity(self.norm, IndexStore("train", ("EN", "NT")))
        self.assertEqual(res["share"], 1.0)
        self.assertGreater(res["per_family"]["NT"]["exempt"], 0)
        self.assertEqual(res["n_records"], N)

    def test_rerun_skips(self):
        before = {F: os.stat(self.out / f"{F}.post.npy") for F in FAMILIES}
        meta = index_build.build_split_index(self.norm, self.out, FAMILIES, self.cfg, 2)
        for F in FAMILIES:
            st = os.stat(self.out / f"{F}.post.npy")
            self.assertEqual((st.st_ino, st.st_mtime_ns), (before[F].st_ino, before[F].st_mtime_ns))
            self.assertEqual(meta["families"][F], self.meta["families"][F])


if __name__ == "__main__":
    unittest.main()

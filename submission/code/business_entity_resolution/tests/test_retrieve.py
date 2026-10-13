"""Stage-1 retrieval (plan §5.7) on a tiny synthetic index: S[F]/weights/cov vs brute force, caps,
postings limit, quotas + chain rule, evaluation mode == direct selection, the spawn driver, stage-1
sampling, and an end-to-end run of stage1-train and s1-eval E1-E6 on a patched blocking.json."""

import copy
import json
import math
import os
import random
import tempfile
import unittest
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ber import evaluate, index_build, io_utils, retrieve, stage1_model
from ber.hashing import key64
from ber.keys import edge_token_set, record_keys

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "happy", "nails",
         "international", "kilo", "lima", "mike"]
STREETS = ["oak", "elm", "pine", "maple", "cedar"]


def _typo(tok, rng):
    i = rng.randrange(len(tok))
    return tok[:i] + tok[i + 1:] if len(tok) > 3 else tok


def _make_data(rng, n_s1=60, n_noise=150):
    """S1 entities with 0-3 perturbed pool copies each, plus noise pool records."""
    s1, pool, gt = [], [], []
    for e in range(n_s1):
        if e < 12:
            name = "happy nails"  # a chain of 12 identical S1 names
        else:
            name = " ".join(rng.sample(VOCAB, rng.randint(1, 3)) + (["llc"] if rng.random() < 0.5 else []))
        addr = f"{rng.randint(1, 40)} {rng.choice(STREETS)} street" + (" springfield" if rng.random() < .3 else "")
        country = rng.choice(["US", "India"])
        s1.append((name, addr, country))
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            toks = name.split()
            if rng.random() < 0.5:
                toks = [_typo(t, rng) for t in toks]
            pool.append([" ".join(toks), addr if rng.random() < 0.8 else "", country, e])
    for _ in range(n_noise):
        name = " ".join(rng.sample(VOCAB, rng.randint(1, 3)))
        pool.append([name, f"{rng.randint(1, 40)} {rng.choice(STREETS)} street", rng.choice(["US", "India"]), -1])
    rng.shuffle(pool)
    for i, p in enumerate(pool):
        if p[3] >= 0:
            gt.append((p[3], i))
    return s1, pool, sorted(gt)


def _write_work(cfg, rng):
    """All work/ inputs of stage 1 under the current directory; returns (s1, pool, gt)."""
    s1, pool, gt = _make_data(rng)
    w = Path("work")
    for d in ("records", "norm/train_pool_store", "splits", "index/train"):
        (w / d).mkdir(parents=True, exist_ok=True)
    n_pool = len(pool)
    indic = np.array([i % 17 == 3 for i in range(n_pool)])
    raw_names = [("नाम " if indic[i] else "é" if i % 13 == 5 else "") + p[0].upper() for i, p in enumerate(pool)]
    pd.DataFrame({"pool_idx": np.arange(n_pool, dtype=np.uint32), "name_norm": [p[0] for p in pool],
                  "addr_norm": [p[1] for p in pool], "name_indic": indic, "addr_indic": False}
                 ).to_parquet(w / "norm/train_pool.parquet", index=False, row_group_size=64)
    pd.DataFrame({"pool_idx": np.arange(n_pool, dtype=np.uint32), "src": np.int8(2),
                  "num": np.arange(n_pool, dtype=np.uint32), "name_raw": raw_names,
                  "addr_raw": [p[1].upper() for p in pool], "country_raw": [p[2] for p in pool]}
                 ).to_parquet(w / "records/train_pool.parquet", index=False, row_group_size=64)
    np.save(w / "norm/train_pool_store/indic.npy", indic)
    pd.DataFrame({"s1_idx": np.arange(len(s1), dtype=np.int32), "name_norm": [s[0] for s in s1],
                  "addr_norm": [s[1] for s in s1], "name_indic": False, "addr_indic": False}
                 ).to_parquet(w / "norm/train_s1.parquet", index=False)
    pd.DataFrame({"s1_idx": np.arange(len(s1), dtype=np.int32), "entity_id": [f"S1-{i}" for i in range(len(s1))],
                  "name_raw": [s[0] for s in s1], "addr_raw": [s[1] for s in s1],
                  "country_raw": [s[2] for s in s1]}).to_parquet(w / "records/train_s1.parquet", index=False)
    pd.DataFrame({"s1_idx": np.array([g[0] for g in gt], dtype=np.int32),
                  "pool_idx": np.array([g[1] for g in gt], dtype=np.uint32)}
                 ).to_parquet(w / "records/train_gt.parquet", index=False)
    np.save(w / "records/train_s1_nmatch.npy",
            np.bincount([g[0] for g in gt], minlength=len(s1)).astype(np.int16))
    cnt = Counter(s[0] for s in s1)
    np.save(w / "norm/train_s1_chain.npy", np.array([cnt[s[0]] for s in s1], dtype=np.int32))
    perm = np.random.default_rng(3).permutation(len(s1))
    cuts = {"DEV_A": perm[:24], "DEV_B": perm[24:36], "S1R_TRAIN": perm[36:52], "HOLDOUT_TEST": perm[52:]}
    sets = {k: sorted(int(x) for x in v) for k, v in cuts.items()}
    (w / "splits/query_sets.json").write_text(json.dumps({"sets": sets, "sizes": {k: len(v) for k, v in sets.items()}}))
    index_build.build_split_index(w / "norm/train_pool.parquet", w / "index/train",
                                  index_build.split_families("train", cfg), cfg, 2)
    return s1, pool, gt


def _test_cfg():
    cfg = copy.deepcopy(io_utils.load_config())
    cfg["max_workers"] = 2
    cfg["build_df_max_frac"] = 0.3
    cfg["cap_frac"] = {F: 0.1 for F in cfg["cap_frac"]}
    cfg["cap_frac"]["NG"] = 0.05
    cfg["edge_tokens"]["min_frac"] = 0.05
    cfg["quota"] = {"LEARN": 6, "NAME": 3, "ADDR": 3, "PAIR": 2, "XK": 2, "CAT": 1}
    cfg["chain_rule"] = {"enabled": True, "min_chain": 3, "mult": 2, "channels": ["ADDR", "PAIR", "XK"]}
    cfg["stage1_model"] = {"neg_top": 10, "neg_rand": 10, "C": 1.0, "max_iter": 500}
    return cfg


class _WorkDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        cls._tmp = tempfile.TemporaryDirectory()
        os.chdir(cls._tmp.name)
        cls.cfg = _test_cfg()
        cls.s1, cls.pool, cls.gt = _write_work(cls.cfg, random.Random(11))
        cls.n = len(cls.pool)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls._tmp.cleanup()


class TestRetrieve(_WorkDir):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cfg = cls.cfg
        et = cfg["edge_tokens"]
        G = set(edge_token_set([p[0] for p in cls.pool], cls.n, et["min_frac"], et["min_edge_share"]))
        at_df = Counter(t for p in cls.pool for t in set(p[1].split()))
        at_cap = math.floor(cfg["cap_frac"]["AT"] * cls.n)
        cls.post = {F: defaultdict(set) for F in retrieve.FAMILY_ORDER}
        for i, p in enumerate(cls.pool):
            for F, ks in record_keys(p[0], p[1], retrieve.FAMILY_ORDER, G, at_df, at_cap, "pool").items():
                for k in ks:
                    cls.post[F][key64(F, k)].add(i)
        cls.queries = [(s[0], s[1]) for s in cls.s1[:25]] + [("zulu alpha", "7 oak street"), ("", ""),
                                                            ("happy nails llc", "")]

    def _brute(self, cfg, name, addr):
        """(S {F: {pool_idx: score}}, Q {F: q}, d_en, n_dropped) straight from the definitions."""
        r = retrieve.Retriever("train", cfg)
        fams = retrieve.kept_families(cfg)
        kd = record_keys(name, addr, fams, r.store.G, r.store.at_df_lookup, r.store.at_cap, "query")
        rows = []
        for j, F in enumerate(fams):
            for k in kd[F]:
                h = key64(F, k)
                rows.append([F, j, h, len(self.post[F].get(h, ()))])
        d_en = next((d for F, _, _, d in rows if F == "EN"), 0)
        if d_en > math.floor(cfg["build_df_max_frac"] * self.n):
            d_en = self.n + 1  # IndexStore sentinel for keys over the build limit
        used = [x for x in rows if x[3] <= math.floor(cfg["cap_frac"][x[0]] * self.n)]
        used.sort(key=lambda x: (-x[3], x[1], x[2]))
        n_drop = 0
        while sum(x[3] for x in used[n_drop:]) > cfg["max_postings_per_query"]:
            n_drop += 1
        S, Q = {F: defaultdict(float) for F in fams}, {F: 0.0 for F in fams}
        for F, _, h, d in used[n_drop:]:
            w = 1.0 if F == "EN" else math.log((self.n + 1) / (d + 1))
            Q[F] += w
            for i in self.post[F].get(h, ()):
                S[F][i] += w
        return S, Q, d_en, n_drop

    def _check_union(self, cfg):
        r = retrieve.Retriever("train", cfg)
        fams = retrieve.kept_families(cfg)
        drops = 0
        for name, addr in self.queries:
            un = r.union(name, addr)
            S, Q, d_en, n_drop = self._brute(cfg, name, addr)
            drops += n_drop
            items = sorted({i for F in fams for i in S[F]})
            self.assertEqual(un.u.tolist(), items)
            self.assertEqual((un.d_en, un.n_dropped), (d_en, n_drop))
            for j, F in enumerate(fams):
                self.assertAlmostEqual(un.Q[j], Q[F])
                np.testing.assert_allclose(un.S[j], [S[F].get(i, 0.0) for i in items])
            X = r.features(un)
            self.assertEqual(X.shape, (len(items), len(r.feature_names)))
            for c, F in enumerate(f for f in retrieve.COV_FAMILIES if f in fams):
                col = r.feature_names.index(f"cov_{F}")
                want = [S[F].get(i, 0.0) / Q[F] if Q[F] > 0 else 0.0 for i in items]
                np.testing.assert_allclose(X[:, col], want)
            np.testing.assert_allclose(X[:, -len(fams):], np.log1p(X[:, :len(fams)]))
        return drops

    def test_union_matches_brute_force(self):
        self.assertEqual(self._check_union(self.cfg), 0)

    def test_caps_leave_indexed_keys_unused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["cap_frac"]["NT"] = 0.02
        r = retrieve.Retriever("train", cfg)
        un = r.union("happy nails llc", "")
        self.assertGreater(len(self.post["NT"][key64("NT", "happy")]), math.floor(0.02 * self.n))
        self.assertEqual(un.Q[r.families.index("NT")], 0.0)
        self._check_union(cfg)

    def test_postings_limit(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["max_postings_per_query"] = 40
        self.assertGreater(self._check_union(cfg), 0)

    def test_family_subset(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["families"] = ["EN", "NT", "NG", "PH", "AT"]
        self._check_union(cfg)
        r = retrieve.Retriever("train", cfg)
        self.assertEqual(r.quotas(1)["PAIR"], 0)
        self.assertIsNone(r.scores(r.union("alpha", ""), r.features(r.union("alpha", "")))["XK"])

    def _brute_select(self, r, un, chain):
        X = r.features(un)
        cols = r.scores(un, X)
        q = r.quotas(chain)
        sel = set()
        for C in retrieve.CHANNELS:
            if cols[C] is None:
                continue
            s, e = cols[C]
            items = [i for i in range(un.u.size) if e is None or e[i]]
            sel |= set(sorted(items, key=lambda i: (-s[i], i))[:q[C]])
        learn = cols["LEARN"][0]
        return [int(un.u[i]) for i in sorted(sel, key=lambda i: (-learn[i], i))]

    def _fake_model(self, r):
        rng = np.random.default_rng(5)
        k = len(r.feature_names)
        return {"features": r.feature_names, "coef": rng.normal(size=k).tolist(), "intercept": 0.3,
                "mean": rng.normal(size=k).tolist(), "std": rng.uniform(0.5, 2, size=k).tolist()}

    def test_quota_selection_and_chain_rule(self):
        r0 = retrieve.Retriever("train", self.cfg)
        for model in (None, self._fake_model(r0)):
            r = retrieve.Retriever("train", self.cfg, model)
            self.assertEqual(r.quotas(2)["ADDR"], 3)
            self.assertEqual(r.quotas(3)["ADDR"], 6)
            self.assertEqual(r.quotas(3)["NAME"], 3)
            for name, addr in self.queries:
                for chain in (1, 5):
                    out = r.retrieve(name, addr, chain)
                    un = r.union(name, addr)
                    self.assertEqual(out["pool_idx"].tolist(), self._brute_select(r, un, chain))
                    self.assertEqual(out["n_union"], un.u.size)
                    self.assertEqual(out["X"].shape, (out["pool_idx"].size, len(r.feature_names)))
                    self.assertEqual(out["ranks"].shape, (out["pool_idx"].size, 3))
                    self.assertTrue(np.all((out["ranks"] >= 1) & (out["ranks"] <= retrieve.RANK_CAP)))
                    if out["pool_idx"].size:
                        self.assertEqual(int(out["ranks"][0, 0]), 1)

    def test_learn_score_is_standardized_dot(self):
        r0 = retrieve.Retriever("train", self.cfg)
        m = self._fake_model(r0)
        r = retrieve.Retriever("train", self.cfg, m)
        un = r.union(*self.queries[3])
        X = r.features(un)
        want = ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["coef"]) + m["intercept"]
        np.testing.assert_allclose(r.scores(un, X)["LEARN"][0], want)
        hb = copy.deepcopy(self.cfg)
        hb["channels"]["LEARN"] = "hand_both"
        rh = retrieve.Retriever("train", hb, m)
        np.testing.assert_allclose(rh.scores(un, X)["LEARN"][0], rh.hand_score(un))

    def test_evaluation_mode_equals_direct_selection(self):
        quotas = [self.cfg["quota"], {"LEARN": 2, "NAME": 0, "ADDR": 5, "PAIR": 0, "XK": 1, "CAT": 0},
                  {"LEARN": 10, "NAME": 1, "ADDR": 1, "PAIR": 1, "XK": 0, "CAT": 2}]
        chains = [(3, 2), (0, 1), (10, 3)]
        grid = retrieve.make_grid(["a", "b", "c"], quotas, self.cfg["chain_rule"]["channels"], chains)
        direct = []
        for q, (mc, m) in zip(quotas, chains):
            cfg = copy.deepcopy(self.cfg)
            cfg["quota"] = dict(q)
            cfg["chain_rule"].update(enabled=m > 1, min_chain=mc, mult=m)
            direct.append(retrieve.Retriever("train", cfg))
        r = direct[0]
        for name, addr in self.queries:
            un = r.union(name, addr)
            truth = np.concatenate([un.u[::2], np.array([self.n + 5], dtype=np.uint32)])
            for chain in (1, 4, 12):
                ev = r.evaluate(name, addr, chain, truth, grid)
                self.assertEqual(ev["reached"].tolist(), [True] * (truth.size - 1) + [False])
                for j, d in enumerate(direct):
                    sel = d.retrieve(name, addr, chain)["pool_idx"]
                    self.assertEqual(int(ev["sizes"][j]), sel.size)
                    self.assertEqual(ev["sel_true"][:, j].tolist(), np.isin(truth, sel).tolist())
                bits = [sum(1 << retrieve.FAMILY_ORDER.index(F) for F in r.families
                            if int(t) in self.post[F] and False) for t in truth]
                self.assertEqual(len(bits), truth.size)
                for t, fm in zip(truth[:-1], ev["famask"][:-1]):
                    j = int(np.searchsorted(un.u, t))
                    want = sum(1 << retrieve.FAMILY_ORDER.index(F) for k, F in enumerate(r.families)
                               if un.S[k, j] > 0)
                    self.assertEqual(int(fm), want)

    def test_query_runner_matches_direct(self):
        idx = np.arange(0, 40, 3)
        with retrieve.QueryRunner("train", self.cfg) as qr:
            got = [x for chunk in qr.map(self.cfg, None, idx, retrieve.retrieve_chunk, chunk=4) for x in chunk]
        r = retrieve.Retriever("train", self.cfg)
        chain = np.load("work/norm/train_s1_chain.npy")
        for i, g in zip(idx, got):
            want = r.retrieve(self.s1[i][0], self.s1[i][1], int(chain[i]))
            self.assertEqual(g["pool_idx"].tolist(), want["pool_idx"].tolist())
            np.testing.assert_array_equal(g["X"], want["X"])


class TestStage1Sampling(unittest.TestCase):
    def test_sample_rows(self):
        y = np.zeros(50, dtype=bool)
        y[[3, 40]] = True
        hand = np.arange(50, dtype=float)
        rows, w = stage1_model.sample_rows(y, hand, 17, 5, 10)
        top = set(range(45, 50))
        self.assertTrue(top | {3, 40} <= set(rows.tolist()))
        rest = [i for i in rows.tolist() if i not in top and i not in (3, 40)]
        self.assertEqual(len(rest), 10)
        self.assertTrue(np.allclose(w[np.isin(rows, rest)], 43 / 10))
        self.assertTrue(np.all(w[~np.isin(rows, rest)] == 1.0))
        rows2, _ = stage1_model.sample_rows(y, hand, 17, 5, 10)
        self.assertEqual(rows.tolist(), rows2.tolist())
        rows3, w3 = stage1_model.sample_rows(y, hand, 17, 5, 100)
        self.assertEqual(rows3.tolist(), list(range(50)))
        self.assertTrue(np.all(w3 == 1.0))


class TestExperiments(_WorkDir):
    """stage1-train and s1-eval E1-E6 + ad-hoc end to end on the synthetic data, with blocking.json
    redirected to a temp copy."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._blocking = io_utils.BLOCKING_JSON
        io_utils.BLOCKING_JSON = Path("blocking.json").resolve()
        io_utils.atomic_write_json(io_utils.BLOCKING_JSON, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        io_utils.BLOCKING_JSON = cls._blocking
        super().tearDownClass()

    def _eval(self, exp, set_name="DEV_A", **kw):
        args = Namespace(set=set_name, exp=exp, families=None, cap=[], quota_grid=False, scorer="hand")
        vars(args).update(kw)
        evaluate.run_s1_eval(args)
        return io_utils.read_json(Path("artifacts/metrics") / f"s1_{exp}.json")

    def test_pipeline(self):
        m = self._eval("adhoc1", families=["EN", "NT", "AT"], cap=["NT=0.05"])
        self.assertEqual(m["setup"]["caps"], {"NT": 0.05})
        cfgs = m["runs"]["adhoc1"]["configs"]
        self.assertEqual(len(cfgs), 1 + len(evaluate.ADHOC_TOPK))
        self.assertIn("name_unseen", m["runs"]["adhoc1"]["primary_segments"])
        self.assertIn("pool_indic", m["runs"]["adhoc1"]["primary_segments"])
        self.assertEqual(io_utils.load_config(), self.cfg)  # ad-hoc never changes the config

        e1 = self._eval("E1")
        self.assertTrue(e1["decision"]["passed"])
        self.assertIn("reachability", e1["runs"]["E1"])
        self._eval("E2")
        cfg = io_utils.load_config()
        self.assertIn("families", cfg["tuned_by"])
        self.assertTrue(set(evaluate.E1_FAMILIES) - {"PH"} <= set(cfg["families"]))
        self._eval("E3")
        cfg = io_utils.load_config()
        self.assertTrue(any(k.startswith("cap_frac.") for k in cfg["tuned_by"]))
        with self.assertRaises(RuntimeError):
            self._eval("E3b")  # no model trained with the current config yet
        stage1_model.run_stage1_train(Namespace())
        model = io_utils.read_json("work/stage1/model_v1.json")
        self.assertEqual(model["features"], retrieve.feature_names(retrieve.kept_families(cfg)))
        self._eval("E3b")
        e4 = self._eval("E4")
        self.assertEqual(len(e4["runs"]["E4"]["configs"]), 486)
        cfg = io_utils.load_config()
        self.assertEqual(cfg["quota"], e4["decision"]["chosen"]["quota"])
        means = [p["mean_size"] for p in e4["decision"]["pareto"]]
        self.assertEqual(means, sorted(means))
        self._eval("E5")
        self.assertIn("chain_rule.enabled", io_utils.load_config()["tuned_by"])
        with self.assertRaises(ValueError):
            self._eval("E6")  # E6 runs on DEV_B only
        try:
            e6 = self._eval("E6", set_name="DEV_B")
            self.assertIn("dev_b_f05", e6["decision"])
        except RuntimeError as e:
            self.assertIn("BLOCKER", str(e))
        with self.assertRaises(ValueError):
            self._eval("adhoc2", set_name="HOLDOUT_TEST")


if __name__ == "__main__":
    unittest.main()

"""Stage-3 matcher (plan/MATCHER_PLAN.md): metric, decision rules, features, output file checks."""

import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ber import match
from ber.match import (FEATURES, LIST_FEATURES, PAIR_FEATURES, Scorer, calibrate, check_matching_file,
                       decide_expected_f, entity_f05, expected_f05_table, list_features, one_owner,
                       pair_features, pair_keys, write_matching_file)

REPO = Path(__file__).resolve().parents[4]
OFFICIAL = REPO / "student_resource" / "utils" / "validate_submission.py"
G = frozenset({"llc", "pvt", "ltd"})


def f05_bruteforce(pred, truth):
    return entity_f05([len(pred)], [len(truth)], [len(pred & truth)])[0]


class TestMetric(unittest.TestCase):
    def test_readme_example(self):
        # README: predict 3, truth 2, both true ones found -> 0.714
        self.assertAlmostEqual(entity_f05([3], [2], [2])[0], 0.7142857, places=6)

    def test_singleton_rule(self):
        np.testing.assert_array_equal(entity_f05([0, 1], [0, 0], [0, 0]), [1.0, 0.0])

    def test_matched_entity_empty_prediction_is_zero(self):
        self.assertEqual(entity_f05([0], [3], [0])[0], 0.0)

    def test_scorer_counts_queries_without_candidates(self):
        queries = np.array([1, 5, 9], dtype=np.int32)       # 9 has no candidate rows
        s1 = np.array([1, 1, 5], dtype=np.int32)
        y = np.array([True, False, False])
        sc = Scorer(queries, s1, y, np.array([1, 0, 2]))
        keep = np.array([True, False, False])
        # S1 1: perfect (1.0); S1 5: singleton, empty (1.0); S1 9: matched, nothing predicted (0.0)
        self.assertAlmostEqual(sc.f05(keep), 2 / 3)
        rep, _ = sc.report(keep)
        self.assertEqual(rep["loss_breakdown"]["blocking_miss_only"]["n_entities"], 1)

    def test_scorer_rejects_rows_outside_query_set(self):
        with self.assertRaises(RuntimeError):
            Scorer(np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32), np.array([True]),
                   np.array([1, 1]))


class TestExpectedF(unittest.TestCase):
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for n in range(1, 7):
            P = -np.sort(-rng.random((4, n)), axis=1)
            E = expected_f05_table(P)
            for g in range(4):
                for k in range(n + 1):
                    exp = 0.0
                    for labels in itertools.product((0, 1), repeat=n):
                        pr = np.prod([P[g, j] if l else 1 - P[g, j] for j, l in enumerate(labels)])
                        truth = {j for j, l in enumerate(labels) if l}
                        exp += pr * f05_bruteforce(set(range(k)), truth)
                    self.assertAlmostEqual(E[g, k], exp, places=10)

    def test_decide_extremes(self):
        s1 = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        p = np.array([0.01, 0.02, 0.01, 0.99, 0.98])
        np.testing.assert_array_equal(decide_expected_f(s1, p), [False, False, False, True, True])

    def test_decide_keeps_prefix_by_p(self):
        s1 = np.zeros(4, dtype=np.int32)
        p = np.array([0.1, 0.95, 0.9, 0.05])
        keep = decide_expected_f(s1, p)
        self.assertTrue(keep[1] and keep[2])
        self.assertFalse(keep[0] or keep[3])


class TestPairedGate(unittest.TestCase):
    def test_no_change_not_adopted(self):
        f = np.ones(1000)
        cmp = match.paired_delta(f, f)
        self.assertEqual(cmp["delta"], 0.0)
        self.assertFalse(match.adopt(cmp, 0.0))

    def test_consistent_gain_adopted_noisy_gain_not(self):
        a = np.zeros(10_000)
        b = a.copy()
        b[:200] = 1.0  # 200 entities better, none worse
        self.assertTrue(match.adopt(match.paired_delta(b, a), 0.0))
        self.assertFalse(match.adopt(match.paired_delta(b, a), 0.05))  # below min_gain
        c = a.copy()
        c[:3], c[3:5] = 1.0, -1.0  # +1 entity net out of 10,000: CI spans zero
        self.assertFalse(match.adopt(match.paired_delta(c, a), 0.0))


class TestOneOwner(unittest.TestCase):
    def test_keeps_highest_owner_only(self):
        pool = np.array([7, 7, 8, 7], dtype=np.uint32)
        p = np.array([0.9, 0.6, 0.5, 0.9], dtype=np.float32)
        np.testing.assert_allclose(one_owner(pool, p), [0.9, 0.0, 0.5, 0.9])  # ties both kept


class TestFeatures(unittest.TestCase):
    def rec(self, name, addr, country="US", **kw):
        return dict({"name": name, "addr": addr, "country": country, "chain": 1, "src": 2,
                     "indic": False}, **kw)

    def feats(self, q, p):
        # PAIR_FEATURES = [p_rerank, rank, ...pair_features() output..., n_tfidf_cos, a_tfidf_cos]; the
        # tfidf cosines are filled in _feature_chunk() from a whole chunk, not by pair_features() itself.
        vals = pair_features(q, p, {}, {}, G)
        self.assertEqual(len(vals), len(PAIR_FEATURES) - 4)
        return dict(zip(PAIR_FEATURES[2:-2], vals))

    def test_identical(self):
        f = self.feats(self.rec("happy nails llc", "12 main st 10001"),
                       self.rec("happy nails llc", "12 main st 10001"))
        self.assertEqual(f["n_eq"], 1.0)
        self.assertEqual(f["a_longnum"], 2.0)
        self.assertEqual(f["a_house"], 1.0)
        self.assertEqual(f["country3"], 1.0)

    def test_branch_conflict(self):
        f = self.feats(self.rec("subway", "12 main st 10001"), self.rec("subway", "480 oak ave 10002"))
        self.assertEqual(f["a_longnum"], 1.0)
        self.assertEqual(f["a_house"], 0.0)

    def test_missing_address_is_nan_not_zero(self):
        f = self.feats(self.rec("acme", ""), self.rec("acme", ""))
        for k in ("a_tset", "a_tsort", "a_idf_jacc", "a_num_jacc", "a_longnum", "a_house"):
            self.assertTrue(np.isnan(f[k]), k)
        self.assertEqual(f["a_empty"], 3.0)

    def test_core_nospace_acronym(self):
        self.assertEqual(self.feats(self.rec("happy nails llc", ""), self.rec("nails happy", ""))["n_eq_core"], 1.0)
        self.assertEqual(self.feats(self.rec("happy nails", ""), self.rec("happynails", ""))["n_eq_nospace"], 1.0)
        self.assertEqual(self.feats(self.rec("international business machines", ""),
                                    self.rec("ibm", ""))["n_acronym"], 1.0)

    def test_country_open_set(self):
        f = self.feats(self.rec("a b", "", "France"), self.rec("a b", "", "US"))
        self.assertEqual(f["country3"], 0.0)
        f = self.feats(self.rec("a b", "", "France"), self.rec("a b", "", ""))
        self.assertEqual(f["country3"], 2.0)

    def test_digits_state_four_way(self):
        f = self.feats(self.rec("x", "412 main st"), self.rec("x", "412 main st"))
        self.assertEqual(f["a_digits_state"], 3.0)
        f = self.feats(self.rec("x", "412 main st"), self.rec("x", "412 480 main st"))
        self.assertEqual(f["a_digits_state"], 2.0)
        f = self.feats(self.rec("x", "412 main st"), self.rec("x", "480 main st"))
        self.assertEqual(f["a_digits_state"], 1.0)
        f = self.feats(self.rec("x", "412 main st"), self.rec("x", "main st"))
        self.assertEqual(f["a_digits_state"], 0.0)
        f = self.feats(self.rec("x", "main st"), self.rec("x", "main st"))
        self.assertTrue(np.isnan(f["a_digits_state"]))
        # leading-zero rescue: raw sets differ, stripped sets equal -> match, not conflict
        f = self.feats(self.rec("x", "412 brunswick dr"), self.rec("x", "0412 brunswick dr"))
        self.assertEqual(f["a_digits_state"], 3.0)

    def test_zero_padded_numbers_agree(self):
        # pool addresses write 0412 / door no 0381; the same number must not count as a conflict
        f = self.feats(self.rec("uptown pizza", "412 brunswick drive troy"),
                       self.rec("uptown pizza", "0412 brunswick drive troy"))
        self.assertEqual(f["a_house"], 1.0)
        self.assertEqual(f["a_longnum"], 2.0)
        self.assertEqual(f["a_num_jacc"], 1.0)
        self.assertEqual(f["a_numstreet"], 1.0)
        self.assertEqual(f["a_num_minlev"], 0.0)

    def test_address_shape_features(self):
        f = self.feats(self.rec("x", "19010 meridian place bothell"), self.rec("x", "9010 meridian place bothell"))
        self.assertEqual(f["a_num_minlev"], 1.0)
        self.assertEqual(f["a_numstreet"], 0.0)
        f = self.feats(self.rec("x", "mg road"), self.rec("x", "mgroad"))
        self.assertEqual(f["a_char3_jacc"], 1.0)
        f = self.feats(self.rec("x", "main st"), self.rec("x", "oak ave"))
        for k in ("a_num_minlev", "a_numstreet"):
            self.assertTrue(np.isnan(f[k]), k)

    def test_shared_maxidf_and_counts(self):
        idf = {"acme": 5.0, "llc": 0.5, "store": 1.0}
        vals = pair_features(self.rec("acme store llc", ""), self.rec("acme llc", ""), idf, {}, G)
        f = dict(zip(PAIR_FEATURES[2:], vals))
        self.assertEqual(f["n_shared_maxidf"], 5.0)
        self.assertTrue(np.isnan(f["a_shared_maxidf"]))
        self.assertTrue(np.isnan(f["q_name_cnt"]))  # not supplied -> NaN, never 0
        vals = pair_features(self.rec("a", "", name_cnt=1.5), self.rec("a", "", addr_cnt=2.0), {}, {}, G)
        f = dict(zip(PAIR_FEATURES[2:], vals))
        self.assertEqual(f["q_name_cnt"], 1.5)
        self.assertEqual(f["p_addr_cnt"], 2.0)

    def test_count_keys(self):
        self.assertEqual(match._count_key("name", "happy nails llc", G), match._count_key("name", "nails happy", G))
        self.assertEqual(match._count_key("addr", "508 raemar drive co", G),
                         match._count_key("addr", "co 508 raemar drive", G))
        self.assertNotEqual(match._count_key("addr", "508 raemar drive", G),
                            match._count_key("addr", "510 raemar drive", G))

    def test_list_features(self):
        X = np.full((3, len(FEATURES)), np.nan, dtype=np.float32)
        c = match._COL
        X[:, c["p_rerank"]] = [0.9, 0.5, 0.1]
        X[:, c["n_tset"]] = [1.0, 0.95, 0.2]
        X[:, c["a_tset"]] = [np.nan, 0.5, 0.7]
        X[:, c["n_idf_jacc"]] = [0.8, np.nan, 0.1]
        X[:, c["src"]] = [2, 2, 3]
        X[:, c["a_longnum"]] = [2, 1, 2]
        list_features(X)
        np.testing.assert_allclose(X[:, c["l_p_gap_next"]], [0.4, 0.4, 0.1], atol=1e-6)
        np.testing.assert_allclose(X[:, c["l_p_gap_max"]], [0.0, 0.4, 0.8], atol=1e-6)
        np.testing.assert_array_equal(X[:, c["l_rank_ntset"]], [1, 2, 3])
        np.testing.assert_array_equal(X[:, c["l_rank_atset"]], [3, 2, 1])
        np.testing.assert_allclose(X[:, c["l_ntset_max_other"]], [0.95, 1.0, 1.0])
        np.testing.assert_array_equal(X[:, c["l_n_name_close"]], [2, 2, 2])
        np.testing.assert_array_equal(X[:, c["l_n_longnum_shared"]], [2, 2, 2])
        self.assertTrue(np.isnan(X[1, c["l_nidf_gap_max"]]))
        np.testing.assert_array_equal(X[:, c["l_n_addr_close"]], [0, 0, 0])
        for f in LIST_FEATURES:
            self.assertFalse(np.isnan(X[0, c[f]]), f)

    def test_monge_elkan(self):
        # identical name -> both directions are self-similarity 1.0
        f = self.feats(self.rec("happy nails llc", ""), self.rec("happy nails llc", ""))
        self.assertAlmostEqual(f["n_me_mean"], 1.0, places=6)
        self.assertAlmostEqual(f["n_me_min"], 1.0, places=6)
        # disjoint tokens -> low similarity both ways
        f = self.feats(self.rec("acme corp", ""), self.rec("zzqq wwvv", ""))
        self.assertLess(f["n_me_mean"], 0.5)
        self.assertLess(f["n_me_min"], 0.5)
        # no tokens on one side -> NaN, not 0
        f = self.feats(self.rec("", ""), self.rec("acme corp", ""))
        self.assertTrue(np.isnan(f["n_me_mean"]))
        self.assertTrue(np.isnan(f["n_me_min"]))

    def test_phonetic_jaccard(self):
        # smith/smyth share both a metaphone and a soundex code
        f = self.feats(self.rec("smith enterprises", ""), self.rec("smyth enterprises", ""))
        self.assertEqual(f["n_metaphone_jacc"], 1.0)
        self.assertEqual(f["n_soundex_jacc"], 1.0)
        # digit-only "name" has no eligible token -> NaN
        f = self.feats(self.rec("123 456", ""), self.rec("acme corp", ""))
        self.assertTrue(np.isnan(f["n_metaphone_jacc"]))
        self.assertTrue(np.isnan(f["n_soundex_jacc"]))
        # disjoint words -> jaccard < 1
        f = self.feats(self.rec("acme corp", ""), self.rec("zzqq wwvv", ""))
        self.assertLess(f["n_metaphone_jacc"], 1.0)

    def test_tfidf_pair_cosine(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 12,
                               alternate_sign=False, norm=None)
        idf = np.ones(2 ** 12, dtype=np.float32)  # uniform weights: isolates the cosine mechanism
        q_texts = ["acme corp", "zz totally unrelated"]
        p_texts = ["acme corp", "acme corp", "nothing in common here"]
        bounds = [(0, 2), (2, 3)]
        cos = match.tfidf_pair_cosine(hv, idf, q_texts, p_texts, bounds)
        self.assertAlmostEqual(cos[0], 1.0, places=5)  # identical strings -> cosine 1.0
        self.assertAlmostEqual(cos[1], 1.0, places=5)
        self.assertLess(cos[2], 0.3)  # near-disjoint char n-grams -> low cosine

        # address variant: NaN wherever either side's address is empty, not 0.0
        a_cos = match.tfidf_addr_cosine(hv, idf, ["12 main st", ""], ["12 main st", "5 oak ave", "9 elm rd"],
                                        [(0, 1), (1, 3)])
        self.assertAlmostEqual(a_cos[0], 1.0, places=5)
        self.assertTrue(np.isnan(a_cos[1]))
        self.assertTrue(np.isnan(a_cos[2]))

    def test_calibrate_clips(self):
        np.testing.assert_allclose(calibrate(np.array([-1.0, 0.5, 2.0]), {"x": [0.0, 1.0], "y": [0.1, 0.9]}),
                                   [0.1, 0.5, 0.9], atol=1e-6)


class TestMatchingFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.s1_ids = ["S1-5", "S1-12", "S1-7"]
        # pool_idx -> (src, num)
        self.src = np.array([2, 3, 2, 3], dtype=np.uint8)
        self.num = np.array([7, 7, 40, 9], dtype=np.uint32)
        # candidates: S1 row 0 -> pools 0,1 ; row 2 -> pools 2,3 ; row 1 -> none
        self.cand_s1 = np.array([0, 0, 2, 2], dtype=np.int32)
        self.cand_pool = np.array([0, 1, 2, 3], dtype=np.uint32)
        self.cand_keys = np.sort(pair_keys(self.cand_s1, self.src[self.cand_pool], self.num[self.cand_pool]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_passes(self):
        path = self.dir / "matching_results.tsv"
        write_matching_file(path, self.s1_ids, np.array([2, 0, 0]), np.array([3, 1, 0]), self.src, self.num)
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text, "source1_entity_id\tmatched_entity_ids\nS1-5\tS3-7,S2-7\nS1-12\t\nS1-7\tS3-9\n")
        self.assertEqual(check_matching_file(path, self.s1_ids, self.cand_keys), [])

    def test_non_candidate_detected(self):
        path = self.dir / "m.tsv"
        path.write_text("source1_entity_id\tmatched_entity_ids\nS1-5\tS2-40\nS1-12\t\nS1-7\t\n",
                        encoding="utf-8", newline="\n")
        self.assertTrue(any("not candidates" in i for i in check_matching_file(path, self.s1_ids, self.cand_keys)))

    def test_order_and_duplicates_detected(self):
        path = self.dir / "m.tsv"
        path.write_text("source1_entity_id\tmatched_entity_ids\nS1-12\t\nS1-5\t\nS1-7\t\n",
                        encoding="utf-8", newline="\n")
        self.assertTrue(check_matching_file(path, self.s1_ids, self.cand_keys))
        path.write_text("source1_entity_id\tmatched_entity_ids\nS1-5\tS2-7,S2-7\nS1-12\t\nS1-7\t\n",
                        encoding="utf-8", newline="\n")
        self.assertTrue(any("duplicate" in i for i in check_matching_file(path, self.s1_ids, self.cand_keys)))

    @unittest.skipUnless(OFFICIAL.exists(), "official validator not found")
    def test_official_validator_passes(self):
        test_dir = self.dir / "test"
        test_dir.mkdir()
        for k, ids in ((1, self.s1_ids), (2, ["S2-7", "S2-40"]), (3, ["S3-7", "S3-9"])):
            rows = "".join(f"{i}\tname\taddr\tUS\n" for i in ids)
            (test_dir / f"test_source{k}.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + rows, encoding="utf-8", newline="\n")
        path = self.dir / "matching_results.tsv"
        write_matching_file(path, self.s1_ids, np.array([2, 0, 0]), np.array([3, 1, 0]), self.src, self.num)
        r = subprocess.run([sys.executable, str(OFFICIAL), "--matching", str(path), "--test-dir", str(test_dir)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()

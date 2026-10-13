"""Key-family tests (plan §9 keys)."""

import json
import multiprocessing
import unittest
from pathlib import Path

from ber.hashing import format_id, key64, keys64, pack_ids
from ber.keys import FAMILIES, addr_keys, edge_token_set, name_keys, record_keys, xk_keys

KEYS_CHECK = Path(__file__).resolve().parents[4] / "plan" / "reference" / "keys_check.json"
NAMES = ["phaedra zeng pegasus corporation", "hu desert alto llc", "happy nails llc", "happynails com",
         "ram marketing private limited", "a b c", "the alpha beta gamma delta epsilon zeta eta theta iota",
         "x", ""]
ADDR = "12 main street oak unknown main"
HASH_KEYS = ["ram marketing private limited", "happy nails", "é ü 字", ""]


class TestKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(KEYS_CHECK, encoding="utf-8") as f:
            cls.check = json.load(f)
        cls.G = set(cls.check["G"])

    def test_reference_examples(self):
        for ex in self.check["examples"].values():
            q = name_keys(ex["norm"], self.G, "query")
            self.assertEqual({f: q[f] for f in ex["query"]}, ex["query"])
            self.assertEqual(name_keys(ex["norm"], self.G, "pool")["NC"], ex["pool_NC"])

    def test_query_pool_identity(self):
        at_df = {"main": 5, "street": 6}
        fams = list(FAMILIES) + ["SK"]
        for nm in NAMES:
            q = record_keys(nm, ADDR, fams, self.G, at_df, 5, "query")
            p = record_keys(nm, ADDR, fams, self.G, at_df, 5, "pool")
            self.assertEqual(set(q), set(fams))
            for f in fams:
                if f == "NC":
                    self.assertTrue(set(q[f]) <= set(p[f]), nm)
                else:
                    self.assertEqual(q[f], p[f], (nm, f))

    def test_requested_families_only(self):
        out = record_keys("happy nails llc", ADDR, ["NT", "AT"], self.G, {}, 5, "query")
        self.assertEqual(set(out), {"NT", "AT"})
        self.assertNotIn("SK", record_keys("happy nails", ADDR, FAMILIES, self.G, {}, 5, "query"))

    def test_np_count(self):
        for n in range(0, 9):
            nm = " ".join(f"tok{i}" for i in range(n))
            self.assertEqual(len(name_keys(nm, set(), "query")["NP"]), n * (n - 1) // 2)
        nm = " ".join(f"tok{i}" for i in range(12))
        self.assertEqual(len(name_keys(nm, set(), "query")["NP"]), 28)

    def test_xk_at_cap(self):
        at_df = {"main": 5, "street": 6, "oak": 1}
        keys = xk_keys("happy nails", ADDR, at_df, 5)
        toks = {k.split("#", 1)[1] for k in keys}
        self.assertEqual(toks, {"12", "main", "oak", "unknown"})
        self.assertNotIn("street", toks)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(xk_keys("happy nails", ADDR, at_df, 6), xk_keys("happy nails", ADDR, {}, 5))

    def test_addr_keys(self):
        a = addr_keys("12 main street main")
        self.assertEqual(a["AT"], ["12", "main", "street"])
        self.assertEqual(a["AB"], ["12 main", "main street", "street main"])

    def test_key64_golden(self):
        for fk, v in self.check["key64_golden"].items():
            fam, key = fk.split("|", 1)
            self.assertEqual(key64(fam, key), v)
            self.assertEqual(int(keys64(fam, [key])[0]), v)

    def test_keys64_matches_key64(self):
        arr = keys64("NT", HASH_KEYS)
        self.assertEqual(arr.dtype.name, "uint64")
        self.assertEqual([int(x) for x in arr], [key64("NT", k) for k in HASH_KEYS])
        self.assertNotEqual(key64("NT", "abc"), key64("EN", "abc"))

    def test_hash_across_spawned_processes(self):
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(2, maxtasksperchild=1) as pool:
            res = [pool.apply_async(keys64, ("EN", HASH_KEYS)) for _ in range(2)]
            a, b = (r.get(timeout=120) for r in res)
        local = keys64("EN", HASH_KEYS)
        self.assertEqual(a.tolist(), local.tolist())
        self.assertEqual(b.tolist(), local.tolist())

    def test_ids(self):
        self.assertEqual(int(pack_ids(3, 7)), (3 << 32) | 7)
        self.assertEqual(pack_ids([2, 3], [0, 4294967295]).tolist(), [2 << 32, (3 << 32) | 4294967295])
        self.assertEqual(format_id(2, 17), "S2-17")

    def test_edge_token_set(self):
        names = ["acme llc", "beta llc", "a alpha b c", "d alpha e f", "",
                 "co p", "co q", "co r", "co s", "g co h i",
                 "zz t", "zz u", "zz v", "j zz k l"]
        # thr = 0.2 * 10 = 2: llc (2, share 1) in; alpha (2, share 0) out; co (5, share 0.8) in;
        # zz (4, share 0.75) out; every other token occurs once.
        self.assertEqual(edge_token_set(names, 10, 0.2, 0.80), ["co", "llc"])
        self.assertEqual(edge_token_set(names, 10, 0.21, 0.80), ["co"])
        self.assertEqual(edge_token_set(names, 10, 0.2, 0.75), ["co", "llc", "zz"])


if __name__ == "__main__":
    unittest.main()

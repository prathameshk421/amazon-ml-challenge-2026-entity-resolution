"""Critique job I: arithmetic derived from verified counts (b_gt_facts.json, data_facts.json)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "plan", "critique", "out")
b = json.load(open(os.path.join(OUT, "b_gt_facts.json")))
tr_s1, tr_pool, te_s1, te_pool = 2206821, b["n_pool"], 1732544, 4887273 + 5082316
rate = b["n_links"] / tr_s1
est_links = te_s1 * rate
R = {"pool_per_s1_train": tr_pool / tr_s1, "pool_per_s1_test": te_pool / te_s1,
     "links_per_s1_train": rate, "train_orphan_share": b["orphan_pool_records"] / tr_pool,
     "ESTIMATE_test_links_if_same_rate": est_links, "ESTIMATE_test_orphan_share_if_same_rate": 1 - est_links / te_pool,
     "test_france_s1_share": 259452 / te_s1}
json.dump(R, open(os.path.join(OUT, "i_derived.json"), "w"), indent=1); print(R)

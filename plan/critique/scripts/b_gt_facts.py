"""Critique job B: facts over ALL 7,638,365 train links (streamed GT, numpy joins), EN (exact normalized
name) bucket sizes, chain structure, name-level overlap (train split / test S1), F0.5 ceilings.
Inputs: plan/critique/out/norm_*.parquet (job A), train_ground_truth.tsv, old fit_sample_K150 candidates.
Output: plan/critique/out/b_gt_facts.json
"""
import json
import os
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "plan", "critique", "out")
GT = os.path.join(ROOT, "student_resource", "dataset", "train", "train_ground_truth.tsv")
R = {}


def load(part, cols):
    return pq.read_table(os.path.join(OUT, f"norm_{part}.parquet"), columns=cols)


def f05(p, r):
    return 0.0 if p == 0 or r == 0 else 1.25 * p * r / (0.25 * p + r)


# ---------------- pool ----------------
cols = ["src", "num", "name_raw", "name_norm", "had_indic", "country", "addr_state"]
t2, t3 = load("train_pool_s2", cols), load("train_pool_s3", cols)
src = np.concatenate([t2["src"].to_numpy(), t3["src"].to_numpy()]).astype(np.int64)
num = np.concatenate([t2["num"].to_numpy(), t3["num"].to_numpy()]).astype(np.int64)
pkey = (src << 32) | num
p_country = np.concatenate([t2["country"].to_numpy(zero_copy_only=False), t3["country"].to_numpy(zero_copy_only=False)])
p_addr = np.concatenate([t2["addr_state"].to_numpy(), t3["addr_state"].to_numpy()])
p_indic = np.concatenate([t2["had_indic"].to_numpy(zero_copy_only=False), t3["had_indic"].to_numpy(zero_copy_only=False)])
p_name = t2["name_norm"].to_pylist() + t3["name_norm"].to_pylist()
p_name_raw_empty = np.array([len(x) == 0 for x in t2["name_raw"].to_pylist() + t3["name_raw"].to_pylist()])
del t2, t3
n_pool = len(pkey)
R["n_pool"] = int(n_pool)
R["pool_addr_state_counts"] = {"nonempty": int((p_addr == 0).sum()), "empty": int((p_addr == 1).sum()),
                               "null_or_punct_only": int((p_addr == 2).sum())}
R["pool_name_raw_empty"] = int(p_name_raw_empty.sum())
R["pool_name_norm_empty"] = int(sum(1 for x in p_name if not x))
order = np.argsort(pkey)
pkey_sorted = pkey[order]

# ---------------- S1 ----------------
s1 = load("train_s1", ["num", "name_norm", "country"])
s1_num = s1["num"].to_numpy().astype(np.int64)
s1_name = s1["name_norm"].to_pylist()
s1_country = s1["country"].to_numpy(zero_copy_only=False)
n_s1 = len(s1_num)
s1_order = np.argsort(s1_num)
s1_sorted = s1_num[s1_order]

# ---------------- GT stream ----------------
link_s1, link_pool = [], []
n_rows = 0
nmatch = np.zeros(n_s1, np.int16)
with open(GT, encoding="utf-8", newline="") as f:
    next(f)
    for line in f:
        a, b = line.rstrip("\n").split("\t")
        n_rows += 1
        i = s1_order[np.searchsorted(s1_sorted, int(a[3:]))]
        if b:
            ids = b.split(",")
            nmatch[i] = len(ids)
            for x in ids:
                link_s1.append(i)
                link_pool.append((int(x[1]) << 32) | int(x[3:]))
link_s1 = np.array(link_s1, np.int64)
link_pool = np.array(link_pool, np.int64)
pos = np.searchsorted(pkey_sorted, link_pool)
assert (pkey_sorted[np.minimum(pos, n_pool - 1)] == link_pool).all(), "missing pool id"
link_pidx = order[pos]
n_links = len(link_pidx)
R["gt_rows"] = n_rows
R["n_links"] = int(n_links)
R["singletons"] = int((nmatch == 0).sum())
R["max_matches"] = int(nmatch.max())
R["unique_pool_in_links"] = int(len(np.unique(link_pidx)))
R["one_owner_holds"] = R["unique_pool_in_links"] == n_links
R["orphan_pool_records"] = int(n_pool - R["unique_pool_in_links"])

# country agreement over all links
lc_s1 = s1_country[link_s1]
lc_p = p_country[link_pidx]
ct = Counter(zip(lc_s1.tolist(), lc_p.tolist()))
R["country_crosstab_all_links"] = {f"{a}->{b}": v for (a, b), v in ct.items()}
R["cross_country_links"] = int(sum(v for (a, b), v in ct.items() if a != b))

# empty names / addresses in true pairs
R["true_pairs_pool_name_raw_empty"] = int(p_name_raw_empty[link_pidx].sum())
la = p_addr[link_pidx]
R["true_pairs_pool_addr"] = {"empty": int((la == 1).sum()), "null_or_punct_only": int((la == 2).sum()),
                             "share_empty_or_null": float(((la > 0).mean()))}
ents_nonsingle = np.nonzero(nmatch > 0)[0]
bad_addr_per_ent = np.bincount(link_s1, weights=(la > 0).astype(np.float64), minlength=n_s1)
indic_per_ent = np.bincount(link_s1, weights=p_indic[link_pidx].astype(np.float64), minlength=n_s1)
both_bad = np.bincount(link_s1, weights=((la > 0) & p_indic[link_pidx]).astype(np.float64), minlength=n_s1)
R["entities_all_matches_addr_empty_or_null"] = int((bad_addr_per_ent[ents_nonsingle] == nmatch[ents_nonsingle]).sum())
R["true_pairs_indic_name_and_empty_addr"] = int(((la > 0) & p_indic[link_pidx]).sum())
R["entities_all_matches_indic_name"] = int((indic_per_ent[ents_nonsingle] == nmatch[ents_nonsingle]).sum())
R["true_pairs_indic_name"] = int(p_indic[link_pidx].sum())

# ---------------- EN buckets & chains ----------------
pool_en_df = Counter(p_name)
s1_name_cnt = Counter(s1_name)
chain = np.array([s1_name_cnt[x] for x in s1_name], np.int64)
den = np.array([pool_en_df.get(x, 0) if x else 0 for x in s1_name], np.int64)
same_en = np.array([p_name[p] == s1_name[s] for s, p in zip(link_s1.tolist(), link_pidx.tolist())])
R["true_pairs_exact_name_equal"] = float(same_en.mean())
lchain = chain[link_s1]
lden = den[link_s1]


def seg(mask):
    return {"n_pairs": int(mask.sum()), "share_pairs": float(mask.mean()),
            "exact_name_equal": float(same_en[mask].mean()) if mask.any() else None,
            "d_EN_p50": float(np.median(lden[mask])) if mask.any() else None,
            "d_EN_p90": float(np.percentile(lden[mask], 90)) if mask.any() else None,
            "share_dEN_gt_300": float((lden[mask] > 300).mean()) if mask.any() else None,
            "share_dEN_gt_1000": float((lden[mask] > 1000).mean()) if mask.any() else None}


R["segments_by_chain"] = {"1": seg(lchain == 1), "2-9": seg((lchain >= 2) & (lchain <= 9)), ">=10": seg(lchain >= 10)}
R["s1_entities_in_chain_ge2"] = float((chain >= 2).mean())
R["s1_entities_in_chain_ge10"] = float((chain >= 10).mean())
cap = int(0.002 * n_pool)
R["EN_cap_0.002"] = cap
R["share_true_pairs_query_dEN_gt_cap"] = float((lden > cap).mean())
R["top_EN_df_pool"] = [[k, v] for k, v in pool_en_df.most_common(15)]
# true matches whose name equals the query name, and the query's EN bucket exceeds the BOTH quota:
R["share_true_pairs_sameEN_and_dEN_gt_300"] = float((same_en & (lden > 300)).mean())

# ---------------- name-level overlap: random 80/20 of train S1 ----------------
rng = np.random.default_rng(7)
hold = rng.random(n_s1) < 0.2
fit_names = Counter(x for x, h in zip(s1_name, hold) if not h)
seen = np.array([fit_names.get(x, 0) > 0 for x in s1_name])
R["heldout20_share_name_seen_in_fit80"] = float(seen[hold].mean())
R["heldout20_share_name_seen_by_country"] = {c: float(seen[hold & (s1_country == c)].mean()) for c in np.unique(s1_country)}
# pairs-weighted
R["heldout20_true_pairs_share_query_name_seen"] = float(seen[link_s1][hold[link_s1]].mean())

# ---------------- test S1 structure + overlap with train S1 names ----------------
ts = load("test_s1", ["name_norm", "country"])
t_name = ts["name_norm"].to_pylist()
t_country = ts["country"].to_numpy(zero_copy_only=False)
t_cnt = Counter(t_name)
t_chain = np.array([t_cnt[x] for x in t_name])
train_names = set(s1_name)
t_seen = np.array([x in train_names for x in t_name])
R["test_s1"] = {"n": len(t_name), "countries": dict(Counter(t_country.tolist())),
                "chain_ge2": float((t_chain >= 2).mean()), "chain_ge10": float((t_chain >= 10).mean()),
                "share_name_seen_in_train_s1": float(t_seen.mean()),
                "by_country": {c: {"n": int((t_country == c).sum()),
                                   "chain_ge2": float((t_chain[t_country == c] >= 2).mean()),
                                   "chain_ge10": float((t_chain[t_country == c] >= 10).mean()),
                                   "share_name_seen_in_train_s1": float(t_seen[t_country == c].mean())}
                               for c in np.unique(t_country)}}
R["train_s1_by_country"] = {c: {"chain_ge2": float((chain[s1_country == c] >= 2).mean()),
                                "chain_ge10": float((chain[s1_country == c] >= 10).mean()),
                                "heldout_seen": float(seen[hold & (s1_country == c)].mean())}
                            for c in np.unique(s1_country)}
R["test_top_chains"] = [[k, v] for k, v in t_cnt.most_common(10)]

# ---------------- metric: F0.5 ceilings from old fit_sample_K150 candidates ----------------
cf = pq.read_table(os.path.join(ROOT, "artifacts", "candidates", "fit_sample_K150.parquet")).to_pydict()
psrc = np.load(os.path.join(ROOT, "artifacts", "index", "pool.src.npy"))
pnum = np.load(os.path.join(ROOT, "artifacts", "index", "pool.num.npy"))
old_key = ((np.where(psrc == 0, 2, 3).astype(np.int64) if psrc.max() <= 1 else psrc.astype(np.int64)) << 32) | pnum.astype(np.int64)
# truth per entity (as pool keys)
s1_by_num = {int(n): i for i, n in enumerate(s1_num.tolist())}
lk = pkey[link_pidx]
starts = np.argsort(link_s1, kind="stable")
ls_sorted = link_s1[starts]
f_vals, pr_num, pr_den, ent_hit, ent_full, nonsing, sing = [], 0, 0, 0, 0, 0, 0
lost_one = []
for sid, cand in zip(cf["s1_id"], cf["cand"]):
    i = s1_by_num[int(sid[3:])]
    lo, hi = np.searchsorted(ls_sorted, i), np.searchsorted(ls_sorted, i, side="right")
    truth = set(lk[starts[lo:hi]].tolist())
    if not truth:
        sing += 1
        f_vals.append(1.0)  # ceiling: matcher predicts empty
        continue
    nonsing += 1
    cset = set(old_key[np.array(cand, np.int64)].tolist()) if len(cand) else set()
    h = len(truth & cset)
    pr_num += h
    pr_den += len(truth)
    ent_hit += h > 0
    ent_full += h == len(truth)
    r = h / len(truth)
    f_vals.append(f05(1.0, r) if h else 0.0)
R["old_K150_fit_sample"] = {"n_entities": len(f_vals), "singletons": sing,
                            "pair_recall": pr_num / pr_den, "entity_recall": ent_hit / nonsing,
                            "full_recall": ent_full / nonsing,
                            "F05_ceiling_perfect_precision_incl_singletons": float(np.mean(f_vals)),
                            "F05_ceiling_nonsingletons": float(np.mean([v for v in f_vals][:]))}
# analytic: value of one missed link per entity size
R["F05_ceiling_if_miss_one_of_k"] = {str(k): f05(1.0, (k - 1) / k) for k in range(1, 12)}
R["match_count_hist"] = {str(k): int(v) for k, v in zip(*np.unique(nmatch, return_counts=True))}

json.dump(R, open(os.path.join(OUT, "b_gt_facts.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str)[:6000])

"""Unsupervised per-country address-abbreviation mining (plan §5.2.2) and its gate E0 (plan §8).

Algorithm = plan/checks/abbrev_miner.py at a single cosine threshold, plus the target-frequency floor,
chain resolution and the step-11 drop.
"""

import random
import re
from collections import Counter, defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber import io_utils
from ber.textnorm import ADDR_ABBREV, normalize_field

_ALPHA = re.compile(r"^[a-z]+$")
CHAIN_DEPTH = 3


def is_sub(s, l):
    """True if s is a subsequence of l."""
    it = iter(l)
    return all(c in it for c in s)


def resolve_chains(pairs):
    """Follow s -> l -> ... up to CHAIN_DEPTH hops; drop every s whose chain runs into a cycle."""
    out = {}
    for s in pairs:
        path, t = [s], pairs[s]
        while t in pairs and t not in path and len(path) < CHAIN_DEPTH:
            path.append(t)
            t = pairs[t]
        if t not in path:
            out[s] = t
    return out


def mine(docs, acfg):
    """docs: token lists (steps 1-11, no 11b). -> ({short: long}, {short: [l*, cos, df_s, df_l, margin]})."""
    N = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    thr = acfg["min_frac"] * N
    S = [t for t, c in df.items() if c >= thr and _ALPHA.match(t) and 1 <= len(t) <= acfg["short_max_len"]]
    L = [t for t, c in df.items() if c >= thr and _ALPHA.match(t) and len(t) >= 2]
    cand = defaultdict(list)
    for s in S:
        for l in L:
            if l != s and l[0] == s[0] and len(l) > len(s) and is_sub(s, l):
                cand[s].append(l)
    need = set(cand) | {l for v in cand.values() for l in v}
    vocab = {t: i for i, t in enumerate(sorted({t for t, c in df.items() if c >= thr} | {"<num>", "<s>", "</s>"}))}
    V = len(vocab)
    ctx = {t: np.zeros(2 * V) for t in need}
    for d in docs:
        m = ["<num>" if t.isdigit() else t for t in d]
        for i, t in enumerate(d):
            if t in ctx:
                lt = m[i - 1] if i > 0 else "<s>"
                rt = m[i + 1] if i + 1 < len(m) else "</s>"
                if lt in vocab:
                    ctx[t][vocab[lt]] += 1
                if rt in vocab:
                    ctx[t][V + vocab[rt]] += 1
    for t in ctx:
        n = np.linalg.norm(ctx[t])
        if n:
            ctx[t] /= n
    kept, detail = {}, {}
    for s, ls in cand.items():
        sc = sorted(((float(ctx[s] @ ctx[l]), l) for l in ls), reverse=True)
        cos, l = sc[0]
        if cos >= acfg["cos_min"] and df[l] >= 0.1 * df[s] and df[l] >= acfg["target_min_frac"] * N:
            second = sc[1][0] if len(sc) > 1 else 0.0
            kept[s] = l
            detail[s] = [l, round(cos, 3), df[s], df[l], round(cos - second, 3)]
    maps = {s: l for s, l in resolve_chains(kept).items() if s not in ADDR_ABBREV}
    return dict(sorted(maps.items())), dict(sorted(detail.items()))


def run_mine_abbrev(args):
    """mine-abbrev --split: per country label with >= min_country_records pool records."""
    cfg = io_utils.load_config()
    acfg = cfg["abbrev_mining"]
    log = io_utils.get_logger("mine-abbrev")
    src = io_utils.work_path("records", f"{args.split}_pool.parquet")
    tbl = pq.read_table(src, columns=["addr_raw", "country_raw"])
    counts = {r["values"]: r["counts"] for r in pc.value_counts(tbl["country_raw"]).to_pylist()}
    maps, stats = {}, {"n_pool": tbl.num_rows, "countries": {}}
    for country in sorted(counts):
        if counts[country] < acfg["min_country_records"]:
            continue
        rows = tbl.filter(pc.equal(tbl["country_raw"], country))["addr_raw"].to_pylist()
        random.Random(acfg["seed"]).shuffle(rows)
        docs = [normalize_field(a, "addr")[0].split() for a in rows[:acfg["sample"]]]
        del rows
        maps[country], detail = mine(docs, acfg)
        stats["countries"][country] = {"n_pool_records": counts[country], "n_sample": len(docs),
                                       "n_mined": len(maps[country]), "pairs": detail}
        log.info("%s: %d records, sample %d, mined %d", country, counts[country], len(docs),
                 len(maps[country]))
    stats["skipped_countries"] = {c: n for c, n in sorted(counts.items())
                                  if n < acfg["min_country_records"]}
    stats["peak_rss_mb"] = round(io_utils.peak_rss_mb(), 1)
    out = io_utils.work_path("norm", f"abbrev_{args.split}.json")
    io_utils.atomic_write_json(out, {"maps": maps, "stats": stats})
    io_utils.write_metrics(f"abbrev_{args.split}", {"maps": maps, "stats": stats}, 0, 0, cfg)
    if args.split == "test":
        io_utils.log_test_usage("mined per-country address-abbreviation map (unsupervised)",
                                "plan §5.2.2 / R4", str(src))
    log.info("wrote %s", out)


def mean_jaccard(set_pairs):
    """Mean token Jaccard over (a, b) set pairs, skipping pairs where either set is empty (E0 decision,
    as plan/checks/norm_v21_check.py). -> (mean, n_skipped)."""
    jac = [len(a & b) / len(a | b) for a, b in set_pairs if a and b]
    return (float(np.mean(jac)) if jac else float("nan")), len(set_pairs) - len(jac)


def _take(path, cols, idx, key):
    """Rows `idx` (row position == `key` column value per the data contract) of a records parquet."""
    tbl = pq.read_table(path, columns=[key] + cols).take(idx)
    assert np.array_equal(tbl[key].to_numpy(), idx)
    return {c: tbl[c].to_pylist() for c in cols}


def run_abbrev_e0(args):
    """E0 (plan §8): adopt step 11b iff true-pair address Jaccard rises >= 0.005 and more than negatives'."""
    from ber.splits import load_query_sets

    cfg = io_utils.load_config()
    log = io_utils.get_logger("abbrev-e0")
    maps = io_utils.read_json(io_utils.work_path("norm", "abbrev_train.json"))["maps"]
    dev = np.sort(load_query_sets()["DEV_A"]).astype(np.int32)
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet")).to_pandas()
    gt = gt[np.isin(gt["s1_idx"].to_numpy(), dev)]
    pools = gt.groupby("s1_idx")["pool_idx"].apply(list).to_dict()
    s1 = _take(io_utils.work_path("records", "train_s1.parquet"), ["addr_raw", "country_raw"], dev, "s1_idx")
    pidx = np.unique(gt["pool_idx"].to_numpy()).astype(np.uint32)
    pool = _take(io_utils.work_path("records", "train_pool.parquet"), ["addr_raw", "country_raw"], pidx,
                 "pool_idx")

    def toks(addrs, countries):
        plain = [set(normalize_field(a, "addr")[0].split()) for a in addrs]
        mined = [set(normalize_field(a, "addr", abbrev_map=maps.get(c))[0].split())
                 for a, c in zip(addrs, countries)]
        return plain, mined

    q_plain, q_mined = toks(s1["addr_raw"], s1["country_raw"])
    p_plain, p_mined = toks(pool["addr_raw"], pool["country_raw"])
    qpos = {int(q): i for i, q in enumerate(dev)}
    ppos = {int(p): i for i, p in enumerate(pidx)}

    by_country = defaultdict(list)
    for q, c in zip(dev, s1["country_raw"]):
        by_country[c].append(int(q))
    rng = np.random.default_rng(11)
    pos_pairs, neg_pairs = [], []
    for q, c in zip(dev, s1["country_raw"]):
        q = int(q)
        pos_pairs += [(q, p) for p in pools.get(q, [])]
        others = np.array([o for o in by_country[c] if o != q], dtype=np.int32)
        for o in rng.choice(others, size=min(5, len(others)), replace=False):
            neg_pairs += [(q, p) for p in pools.get(int(o), [])]

    def mean_jac(pairs, qs, ps):
        return mean_jaccard([(qs[qpos[q]], ps[ppos[p]]) for q, p in pairs])

    res = {}
    for tag, pairs in (("true", pos_pairs), ("neg", neg_pairs)):
        without, skip_without = mean_jac(pairs, q_plain, p_plain)
        with_, skip_with = mean_jac(pairs, q_mined, p_mined)
        res[tag] = {"n_pairs": len(pairs), "jaccard_without_11b": without, "jaccard_with_11b": with_,
                    "n_skipped_empty_without_11b": skip_without, "n_skipped_empty_with_11b": skip_with,
                    "rise": with_ - without}
    adopt = res["true"]["rise"] >= 0.005 and res["true"]["rise"] > res["neg"]["rise"]
    payload = {"set": "DEV_A", "true_pairs": res["true"], "random_negatives": res["neg"],
               "adopt_11b": bool(adopt), "rule": "true rise >= 0.005 and true rise > negative rise",
               "empty_rule": "pairs with an empty address token set on either side are skipped"}
    path = io_utils.write_metrics("e0_abbrev", payload, len(dev), len(pos_pairs), cfg)
    io_utils.apply_tuned({"abbrev_mining.enabled": bool(adopt)}, "E0", path)
    log.info("E0: true rise %.4f, negative rise %.4f, adopt=%s", res["true"]["rise"], res["neg"]["rise"],
             adopt)

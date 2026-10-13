"""Stage-3 experiment runner (plan/EXPERIMENT_ROUTINE.md): MATCH_TRAIN 5-fold OOF with optional extra feature
groups and sample weights, paired against a parent run, WITHOUT touching the live model in work/match/models/.

  exp-train --exp exps/M005.json [--workers W]
  exp-predict --run M005            TEST -> work/outputs/<run>/matching_results.tsv (inputs only)

exps/<id>.json: {"id", "parent" (run dir under work/match/runs), "hypothesis", "groups": [...],
"weights": null | {"kind": ..., ...}, "drop_features": [...], "lgb": {overrides}, "seeds": [42]}

Every run writes work/match/runs/<id>/models/ (model_fold*.txt + matcher.json with "extra_groups"), which
`match-predict`/`exp-predict` can score on TEST, plus OOF parquet, per-entity scores and
artifacts/metrics/exp_<id>.json (paired delta vs parent overall, per segment and on the parent's
uncertain-band stratum).

Feature groups compute from inputs only (feature table, candidate lists, p_rerank); labels are joined only
for fitting and scoring MATCH_TRAIN, as in match-train.
"""
import json
import math
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io_utils, match

TRAIN_UNIVERSE = ("MATCH_TRAIN", "MATCH_VALID", "DEV_REST_180K", "DEV_REST_REST", "DEV_B", "HOLDOUT_TEST")
BAND = (0.1, 0.9)


# ---------------------------------------------------------------- feature groups

def _col(X, name):
    return X[:, match._COL[name]].astype(np.float64)


def _list_stats(s1, v):
    """Per-row list mean, std (ddof 0) and size of v over each S1's rows, NaN-aware."""
    df = pd.DataFrame({"s": s1, "v": v})
    g = df.groupby("s", sort=False)["v"]
    return g.transform("mean").to_numpy(), g.transform("std", ddof=0).to_numpy(), g.transform("size").to_numpy()


def _z(s1, v):
    mean, std, n = _list_stats(s1, v)
    z = (v - mean) / np.where(std > 1e-9, std, np.nan)
    return np.where(n >= 2, z, np.nan)


def group_dinv(set_name, X, s1, pool):
    """Density-invariant list features: position of the pair relative to its own list, as z-scores and
    shares instead of absolute counts (l_n / l_n_name_close grow with pool density; test is denser)."""
    prr, ntset, atset, nidf = (_col(X, c) for c in ("p_rerank", "n_tset", "a_tset", "n_idf_jacc"))
    rank = _col(X, "rank")
    df = pd.DataFrame({"s": s1, "p": prr})
    g = df.groupby("s", sort=False)["p"]
    n = g.transform("size").to_numpy().astype(np.float64)
    med = g.transform("median").to_numpy()
    share05 = (df.assign(b=(prr >= 0.5).astype(np.float64)).groupby("s", sort=False)["b"]
               .transform("mean").to_numpy())
    # share of the OTHER candidates whose p_rerank is within 0.1 of this one (near-ties)
    close = np.zeros(prr.size)
    _, start, stop = match.group_bounds(s1)
    for a, b in zip(start.tolist(), stop.tolist()):
        if b - a > 1:
            p = prr[a:b]
            close[a:b] = ((np.abs(p[:, None] - p[None, :]) < 0.1).sum(1) - 1) / (b - a - 1)
    close[n < 2] = np.nan
    out = np.column_stack([_z(s1, prr), np.where(n >= 2, (rank - 1) / np.maximum(n - 1, 1), np.nan),
                           share05, prr - med, close, _z(s1, ntset), _z(s1, atset), _z(s1, nidf)])
    return out.astype(np.float32)


def _universe(set_name):
    """(s1_idx, pool_idx, p_rerank) of every candidate list on the set's side, with match.json's cand_filter
    (the same rows the feature files hold). Train side: all train-split S1 with candidates, including
    validation-S1 inputs (no labels). Test side: all test S1."""
    sets = ("TEST",) if set_name == "TEST" else TRAIN_UNIVERSE
    cf = match.load_match_config().get("cand_filter")
    parts = []
    for s in sets:
        d = io_utils.paths()["work_root"] / "candidates" / s
        for p in sorted(d.glob("part-*.parquet")):
            t = pq.read_table(p, columns=["s1_idx", "pool_idx", "rank", "p_rerank"])
            rank, prr = t.column("rank").to_numpy(), t.column("p_rerank").to_numpy()
            keep = (rank <= cf["K_min"]) | (prr >= cf["p_min"]) if cf else np.ones(rank.size, bool)
            parts.append(pd.DataFrame({"s1": t.column("s1_idx").to_numpy()[keep].astype(np.int64),
                                       "pool": t.column("pool_idx").to_numpy()[keep].astype(np.int64),
                                       "prr": prr[keep].astype(np.float64)}))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["s1", "pool"])


def group_comp(set_name, X, s1, pool):
    """Pool-side contention (reverse retrieval): how pool record p is contested by other S1 lists.
    c_margin = own p_rerank - best p_rerank of p in any other S1 list (NaN when p is in one list only)."""
    u = _universe(set_name).sort_values(["pool", "prr"], ascending=[True, False], kind="stable")
    u = u.reset_index(drop=True)
    pl = u["pool"].to_numpy()
    start = np.flatnonzero(np.r_[True, pl[1:] != pl[:-1]])
    cnt = np.diff(np.r_[start, pl.size])
    grp = np.repeat(np.arange(start.size), cnt)
    pr = u["prr"].to_numpy()
    rank_in_pool = np.arange(pl.size) - start[grp] + 1
    best, n_own = pr[start][grp], cnt[grp]
    second = np.where(cnt > 1, pr[np.minimum(start + 1, pl.size - 1)], np.nan)[grp]
    best_other = np.where(rank_in_pool == 1, second, best)
    n_ge05 = np.bincount(grp, weights=pr >= 0.5)[grp] - (pr >= 0.5)
    u["c_log_n_own"] = np.log(n_own)
    u["c_rank_q"] = rank_in_pool.astype(np.float64)
    u["c_margin"] = np.where(n_own > 1, pr - best_other, np.nan)
    u["c_n_other_ge05"] = n_ge05
    u["c_best_other"] = np.where(n_own > 1, best_other, np.nan)
    key = s1.astype(np.int64) << 32 | pool.astype(np.int64)
    ukey = u["s1"].to_numpy() << 32 | u["pool"].to_numpy()
    order = np.argsort(ukey)
    pos = order[np.searchsorted(ukey[order], key)]
    if not np.array_equal(ukey[pos], key):
        raise RuntimeError(f"{set_name}: feature rows missing from the candidate universe")
    cols = ["c_log_n_own", "c_rank_q", "c_margin", "c_n_other_ge05", "c_best_other"]
    out = u[cols].to_numpy()[pos]
    # per-S1: share of its list it does not win, and share it loses by >= 0.2
    df = pd.DataFrame({"s": s1, "nb": (out[:, 1] > 1).astype(np.float64),
                       "lose": (np.nan_to_num(out[:, 2], nan=1.0) <= -0.2).astype(np.float64)})
    g = df.groupby("s", sort=False)
    out = np.column_stack([out, g["nb"].transform("mean").to_numpy(), g["lose"].transform("mean").to_numpy()])
    return out.astype(np.float32)


_SIB_NAME, _SIB_ADDR = 90.0, 80.0


def _sib_chunk(task):
    """Candidate-to-candidate evidence for a chunk of S1 lists (worker)."""
    from rapidfuzz import fuzz, process
    split, pool, prr, ntset, atset, bounds = task
    from ber.rerank import PoolTextReader
    with PoolTextReader(split) as ptr:
        names, addrs, src = ptr.name(pool), ptr.addr(pool), np.asarray(ptr.src(pool))
    out = np.full((pool.size, 8), np.nan, dtype=np.float32)
    for a, b in bounds:
        n = b - a
        if n < 2:
            out[a:b, 0:2] = 0.0
            continue
        nm = process.cdist(names[a:b], names[a:b], scorer=fuzz.token_set_ratio, workers=1)
        ad = process.cdist(addrs[a:b], addrs[a:b], scorer=fuzz.token_set_ratio, workers=1).astype(np.float64)
        empty = np.array([not s for s in addrs[a:b]])
        ad[empty, :] = np.nan
        ad[:, empty] = np.nan
        np.fill_diagonal(nm, -1)
        np.fill_diagonal(ad, np.nan)
        dup = (nm >= _SIB_NAME) & ((ad >= _SIB_ADDR) | np.isnan(ad))
        np.fill_diagonal(dup, False)
        other_src = src[a:b][None, :] != src[a:b][:, None]
        p, nt, at = prr[a:b], ntset[a:b], atset[a:b]
        out[a:b, 0] = dup.sum(1)
        out[a:b, 1] = (dup & other_src).sum(1)
        has = dup.any(1)
        pm = np.where(dup, p[None, :], -np.inf).max(1)
        out[a:b, 2] = np.where(has, pm, np.nan)
        ntm = np.where(dup, np.nan_to_num(nt, nan=-1)[None, :], -np.inf).max(1)
        out[a:b, 3] = np.where(has, ntm, np.nan)
        atm = np.where(dup, np.nan_to_num(at, nan=-1)[None, :], -np.inf).max(1)
        out[a:b, 4] = np.where(has, atm, np.nan)
        out[a:b, 5] = nm.max(1)
        with np.errstate(all="ignore"):
            am = np.nanmax(np.where(np.isnan(ad), -np.inf, ad), 1)
        out[a:b, 6] = np.where(np.isfinite(am), am, np.nan)
        out[a:b, 7] = (1.0 + dup.sum(1)) / n
    return out


def group_sib(set_name, X, s1, pool):
    """Copy-cluster support and cross-source triangulation, from candidate-to-candidate text similarity only
    (no model score is stacked; X3's stacked pass-1 p broke singletons). A near-duplicate = another candidate
    of the same S1 with name token-set >= 90 and address token-set >= 80 (or an empty address)."""
    from concurrent.futures import ProcessPoolExecutor
    split = "test" if set_name == "TEST" else "train"
    prr, ntset, atset = _col(X, "p_rerank"), _col(X, "n_tset"), _col(X, "a_tset")
    _, start, stop = match.group_bounds(s1)
    tasks, step = [], 20000
    for lo in range(0, start.size, step):
        hi = min(lo + step, start.size)
        a0, b0 = int(start[lo]), int(stop[hi - 1])
        bounds = list(zip((start[lo:hi] - a0).tolist(), (stop[lo:hi] - a0).tolist()))
        tasks.append((split, pool[a0:b0], prr[a0:b0], ntset[a0:b0], atset[a0:b0], bounds))
    with ProcessPoolExecutor(max_workers=io_utils.n_workers(io_utils.load_config())) as ex:
        return np.vstack(list(ex.map(_sib_chunk, tasks)))


_UNIT_WORDS = {"unit", "apartment", "apt", "suite", "ste", "flat", "room", "rm", "floor", "fl", "no"}


def _num_state(a, b):
    """0 both missing, 1 one missing, 2 equal, 3 differ."""
    if not a and not b:
        return 0
    if not a or not b:
        return 1
    return 2 if a == b else 3


def _addr_nums(text):
    toks = text.split()
    nums = [t for t in toks if t.isdigit()]
    unit = frozenset(toks[i + 1].lstrip("0") for i in range(len(toks) - 1)
                     if toks[i] in _UNIT_WORDS and toks[i + 1].isdigit())
    post = frozenset(t for t in nums if len(t) >= 5)
    return nums, unit, post


def _num_rows(q_addr, p_addr):
    qn, qu, qp = _addr_nums(q_addr)
    pn, pu, pp = _addr_nums(p_addr)
    out = [np.nan] * 8
    if qn and pn:
        a, b = qn[0].lstrip("0") or "0", pn[0].lstrip("0") or "0"
        if a == b:
            out[0] = 2
        elif len(a) != len(b) and (a.endswith(b) or a.startswith(b) or b.endswith(a) or b.startswith(a)):
            out[0] = 3  # digit dropped / added: 106 vs 06, 313 vs 31
        else:
            out[0] = 4
        out[1] = math.log1p(abs(int(a[:15]) - int(b[:15])))
        qs, ps = {t.lstrip("0") for t in qn}, {t.lstrip("0") for t in pn}
        out[2] = float(any(x != y and len(x) != len(y) and (x.endswith(y) or x.startswith(y) or y.endswith(x)
                                                             or y.startswith(x)) for x in qs for y in ps))
        shared = [t.lstrip("0") for t in qn if t.lstrip("0") in ps]
        pshared = [t.lstrip("0") for t in pn if t.lstrip("0") in qs]
        out[7] = float(shared == pshared) if len(shared) >= 2 else np.nan
    else:
        out[0] = 0 if not qn and not pn else 1
    out[3] = _num_state(qp, pp)
    out[4] = _num_state(qu, pu)
    out[5], out[6] = len(qn), len(pn)
    return out


def _num_chunk(task):
    split, q_addrs, pool = task
    from ber.rerank import PoolTextReader
    with PoolTextReader(split) as ptr:
        p_addrs = ptr.addr(pool)
    return np.array([_num_rows(q, p) for q, p in zip(q_addrs, p_addrs)], dtype=np.float32).reshape(-1, 8)


def group_num(set_name, X, s1, pool):
    """Structured address-number evidence (item 7), country-agnostic: first number equal / digit dropped /
    different and its log distance, any digit-drop pair, postal-like (>= 5 digits) and unit-number agreement,
    number counts, and whether shared numbers appear in the same order (word-order transposition check)."""
    from concurrent.futures import ProcessPoolExecutor
    split = "test" if set_name == "TEST" else "train"
    norm = pd.read_parquet(io_utils.work_path("norm", f"{split}_s1.parquet"), columns=["s1_idx", "addr_norm"])
    if not np.array_equal(norm["s1_idx"].to_numpy(), np.arange(len(norm))):
        raise RuntimeError(f"norm/{split}_s1.parquet: s1_idx is not 0..n-1")
    q_addr = norm["addr_norm"].to_numpy()[s1]
    del norm
    step = 100_000
    tasks = [(split, [a or "" for a in q_addr[i:i + step]], pool[i:i + step]) for i in range(0, s1.size, step)]
    with ProcessPoolExecutor(max_workers=io_utils.n_workers(io_utils.load_config())) as ex:
        return np.vstack(list(ex.map(_num_chunk, tasks)))


_NFREQ_CACHE = {}
_LEGAL_RE = r"\b(pvt|private|ltd|limited|llc|inc|co|corp|corporation|company|the)\b"


def _name_key(names):
    s = pd.Series(names).fillna("").astype(str).str.lower()
    return s.str.replace(_LEGAL_RE, "", regex=True).str.replace(r"[^a-z0-9]", "", regex=True).to_numpy()


def _nfreq_tables(split):
    """(S1 name keys, pool name keys, S1 key counts, pool key counts) of one split's own tables (label-free)."""
    if split not in _NFREQ_CACHE:
        s1k = _name_key(pq.read_table(io_utils.work_path("records", f"{split}_s1.parquet"),
                                      columns=["name_raw"]).column("name_raw").to_pylist())
        pk = _name_key(pq.read_table(io_utils.work_path("records", f"{split}_pool.parquet"),
                                     columns=["name_raw"]).column("name_raw").to_pylist())
        _NFREQ_CACHE[split] = (s1k, pk, pd.Series(s1k).value_counts(), pd.Series(pk).value_counts())
    return _NFREQ_CACHE[split]


def group_nfreq(set_name, X, s1, pool):
    """Whole-name frequency (legal suffixes and punctuation stripped) in the split's own S1 and pool tables:
    how many S1 / pool records carry the pool record's name, and how many S1 carry the S1's name. A name-only
    pool record whose name is unique among S1 is far more often a true link (ce_headroom probe)."""
    s1k, pk, c1, cp = _nfreq_tables("test" if set_name == "TEST" else "train")
    pkey, skey = pk[pool], s1k[s1]
    cnt = lambda c, k: np.log1p(pd.Series(k).map(c).fillna(0).to_numpy(dtype=np.float64))
    return np.column_stack([cnt(c1, pkey), cnt(cp, pkey), cnt(c1, skey)])


REV_COLS = ["rv_in20", "rv_rank", "rv_cos", "rv_top", "rv_gap", "rv_margin", "rv_nclose"]


def group_rev(set_name, X, s1, pool):
    """Reverse-name retrieval for name-only pool candidates (plan/checks/rev_prep.py): where this S1 ranks among all
    same-country S1 names nearest to the pool record's name. Read from work/retrieval/rev_<SET>.parquet."""
    t = pq.read_table(io_utils.paths()["work_root"] / "retrieval" / f"rev_{set_name}.parquet")
    ts1, tpool = t.column("s1_idx").to_numpy(), t.column("pool_idx").to_numpy()
    F = np.column_stack([t.column(c).to_numpy() for c in REV_COLS])
    if ts1.size == s1.size and np.array_equal(ts1, s1) and np.array_equal(tpool, pool):
        return F
    # a row subset (e.g. REST_FOLD in train_final_comp.py): align by (s1_idx, pool_idx); every row must be found
    key = s1.astype(np.int64) << 32 | pool.astype(np.int64)
    tkey = ts1.astype(np.int64) << 32 | tpool.astype(np.int64)
    order = np.argsort(tkey)
    pos = np.searchsorted(tkey, key, sorter=order)
    pos = np.minimum(pos, tkey.size - 1)
    if not np.array_equal(tkey[order[pos]], key):
        raise RuntimeError(f"rev_{set_name}.parquet does not cover the feature rows")
    return F[order[pos]]


GRP_COLS = ["g_n_acc", "g_name_max", "g_name_rmax", "g_tok_cov", "g_addr_max", "g_same_src"]


def group_grp(set_name, X, s1, pool):
    """Group-profile features vs the S1's first-pass accepted pool records (plan/checks/grp_feats.py)."""
    return _aligned("grp", set_name, GRP_COLS, s1, pool)


def _aligned(prefix, set_name, cols, s1, pool):
    """Columns of work/retrieval/<prefix>_<SET>.parquet aligned to the feature rows by (s1_idx, pool_idx)."""
    t = pq.read_table(io_utils.paths()["work_root"] / "retrieval" / f"{prefix}_{set_name}.parquet")
    ts1, tpool = t.column("s1_idx").to_numpy(), t.column("pool_idx").to_numpy()
    F = np.column_stack([t.column(c).to_numpy() for c in cols])
    if ts1.size == s1.size and np.array_equal(ts1, s1) and np.array_equal(tpool, pool):
        return F
    key = s1.astype(np.int64) << 32 | pool.astype(np.int64)
    tkey = ts1.astype(np.int64) << 32 | tpool.astype(np.int64)
    order = np.argsort(tkey)
    pos = np.minimum(np.searchsorted(tkey, key, sorter=order), tkey.size - 1)
    if not np.array_equal(tkey[order[pos]], key):
        raise RuntimeError(f"{prefix}_{set_name}.parquet does not cover the feature rows")
    return F[order[pos]]


def group_grp_full(set_name, X, s1, pool):
    """grp built from full-scale first-pass OOF decisions (plan/checks/firstpass_oof.py); TEST from S9."""
    return _aligned("grp_full", set_name, GRP_COLS, s1, pool)


GROUPS = {
    "grp": (GRP_COLS, group_grp),
    "grp_full": (GRP_COLS, group_grp_full),
    "rev": (REV_COLS, group_rev),
    "nfreq": (["nf_pool_name_in_s1", "nf_pool_name_in_pool", "nf_s1_name_in_s1"], group_nfreq),
    "num":(["nm_first_state", "nm_first_logdiff", "nm_any_digit_drop", "nm_post_state", "nm_unit_state",
             "nm_q_n", "nm_p_n", "nm_shared_same_order"], group_num),
    "sib": (["sb_n_dup", "sb_n_dup_other_src", "sb_dup_max_prr", "sb_dup_max_ntset", "sb_dup_max_atset",
             "sb_best_name_sim", "sb_best_addr_sim", "sb_cluster_share"], group_sib),
    "dinv": (["d_prr_z", "d_rank_frac", "d_share_prr05", "d_prr_minus_med", "d_share_prr_close",
              "d_ntset_z", "d_atset_z", "d_nidf_z"], group_dinv),
    "comp": (["c_log_n_own", "c_rank_q", "c_margin", "c_n_other_ge05", "c_best_other",
              "c_q_share_not_best", "c_q_share_lose02"], group_comp),
}


def group_ce(ce_name, set_name, s1, pool):
    """Cross-encoder score from work/ce/<ce_name>/<SET>.parquet (ce/inference.py), joined on (s1_idx, pool_idx):
    p, its rank inside the S1 list (1 = best) and the gap to the list max. Rows without a score stay NaN; a
    missing file raises (never filled silently)."""
    p = _ce_p(ce_name, set_name, s1, pool)
    g = pd.DataFrame({"s": s1, "p": p}).groupby("s", sort=False)["p"]
    rank = g.rank(ascending=False, method="min").to_numpy()
    gap = g.transform("max").to_numpy() - p
    return np.column_stack([p, rank, gap])


def _ce_p(ce_name, set_name, s1, pool):
    path = io_utils.paths()["work_root"] / "ce" / ce_name / f"{set_name}.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} missing: score {set_name} with ce/inference.py first")
    t = pq.read_table(path, columns=["s1_idx", "pool_idx", "p_ce"])
    cs1, cpool = t.column("s1_idx").to_numpy(), t.column("pool_idx").to_numpy()
    pc = t.column("p_ce").to_numpy().astype(np.float64)
    if cs1.size == s1.size and np.array_equal(cs1, s1) and np.array_equal(cpool, pool):
        p = pc
    else:
        key = s1.astype(np.int64) << 32 | pool.astype(np.int64)
        ckey = cs1.astype(np.int64) << 32 | cpool.astype(np.int64)
        order = np.argsort(ckey)
        pos = np.minimum(np.searchsorted(ckey, key, sorter=order), max(ckey.size - 1, 0))
        hit = ckey[order[pos]] == key
        p = np.where(hit, pc[order[pos]], np.nan)
    return p


# CE-derived feature families (plan/checks/cefe_ablation.py): everything a tree cannot build from p_ce alone in
# one split — position and competition inside the S1 list, list shape, disagreement with stage 2, and products
# with single evidence columns. Each family repeats ce_p so it can be tested alone on top of the 68 base columns.
_SAME_NAME = 0.95  # n_tset at or above this = "same-name" candidate
CEX_FAMILIES = {
    "p": ["ce_p"],
    "raw": ["ce_p", "ce_logit", "ce_entropy", "ce_conf"],
    "rank": ["ce_p", "ce_rank", "ce_rank_pct", "ce_top1", "ce_top2", "ce_top3", "ce_gap_top1", "ce_gap_next",
             "ce_gap_prev", "ce_t1_t2", "ce_t2_t3"],
    "list": ["ce_p", "ce_l_mean", "ce_l_std", "ce_l_max", "ce_l_min", "ce_z", "ce_pctile", "ce_softmax",
             "ce_ratio_max", "ce_n_ge05", "ce_n_ge07", "ce_n_ge08", "ce_n_ge09", "ce_frac_above", "ce_l_entropy",
             "ce_softmax_hhi", "ce_t1_t2", "ce_t1_t3", "ce_samename_n", "ce_samename_rank"],
    "dis": ["ce_p", "ce_minus_prr", "ce_abs_prr", "ce_rank_minus_s2", "ce_top1_is_s2top1", "ce_s2_top1_agree",
            "ce_margin_minus_s2", "ce_minus_s2top1"],
    "int": ["ce_p", "ce_x_ntset", "ce_x_atset", "ce_x_cmatch", "ce_x_house", "ce_x_numjacc", "ce_x_prr",
            "ce_x_s2rank", "ce_x_ln", "ce_x_aempty", "ce_x_chain", "ce_x_src"],
}
CEX_FAMILIES["all"] = list(dict.fromkeys(c for f in ("raw", "rank", "list", "dis", "int") for c in CEX_FAMILIES[f]))
_CEX_CACHE = {}


def cex_table(ce_name, set_name, X, s1, pool):
    """All CE-derived columns for the feature rows (dict name -> float64 array); inputs only (p_ce, base X)."""
    key = (ce_name, set_name, int(s1.size))
    if key in _CEX_CACHE:
        return _CEX_CACHE[key]
    p = _ce_p(ce_name, set_name, s1, pool)
    if np.isnan(p).any():
        raise RuntimeError(f"{ce_name}/{set_name}: {int(np.isnan(p).sum())} rows without p_ce")
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(pc / (1 - pc))
    c = {"ce_p": p, "ce_logit": logit,
         "ce_entropy": -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)), "ce_conf": np.abs(p - 0.5) * 2}

    # sorted view: S1 ascending, p descending; position 0 = CE top1 of the list
    order = np.lexsort((-p, s1))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    ss, ps = s1[order], p[order]
    start = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(start) - 1
    gs = np.flatnonzero(start)
    gn = np.diff(np.r_[gs, ss.size])
    n = gn[gid].astype(np.float64)
    pos = np.arange(ss.size) - gs[gid]

    def at(k):  # k-th best p of the row's list (NaN if the list is shorter)
        idx = gs + k
        v = np.where(k < gn, ps[np.minimum(idx, ss.size - 1)], np.nan)
        return v[gid][inv]
    t1, t2, t3 = at(0), at(1), at(2)
    nxt = np.where(pos < gn[gid] - 1, ps[np.minimum(np.arange(ss.size) + 1, ss.size - 1)], np.nan)[inv]
    prv = np.where(pos > 0, ps[np.maximum(np.arange(ss.size) - 1, 0)], np.nan)[inv]
    n = n[inv]

    df = pd.DataFrame({"s": s1, "p": p, "e": np.exp(logit)})
    g = df.groupby("s", sort=False)
    rank = g["p"].rank(ascending=False, method="min").to_numpy()
    rank_max = g["p"].rank(ascending=False, method="max").to_numpy()
    mean, std = g["p"].transform("mean").to_numpy(), g["p"].transform("std", ddof=0).to_numpy()
    soft = df["e"].to_numpy() / g["e"].transform("sum").to_numpy()
    df["soft2"] = soft ** 2
    q = p / np.maximum(g["p"].transform("sum").to_numpy(), 1e-12)
    df["qlq"] = -q * np.log(np.maximum(q, 1e-12))
    g = df.groupby("s", sort=False)
    multi = n >= 2
    c.update({
        "ce_rank": rank, "ce_rank_pct": np.where(multi, (rank - 1) / np.maximum(n - 1, 1), np.nan),
        "ce_top1": (rank <= 1).astype(np.float64), "ce_top2": (rank <= 2).astype(np.float64),
        "ce_top3": (rank <= 3).astype(np.float64),
        "ce_gap_top1": t1 - p, "ce_gap_next": p - nxt, "ce_gap_prev": prv - p,
        "ce_t1_t2": t1 - t2, "ce_t2_t3": t2 - t3, "ce_t1_t3": t1 - t3,
        "ce_l_mean": mean, "ce_l_std": std, "ce_l_max": t1, "ce_l_min": g["p"].transform("min").to_numpy(),
        "ce_z": np.where(multi & (std > 1e-9), (p - mean) / np.where(std > 1e-9, std, 1), np.nan),
        "ce_pctile": np.where(multi, (n - rank_max) / np.maximum(n - 1, 1), np.nan),
        "ce_softmax": soft, "ce_ratio_max": p / np.maximum(t1, 1e-12),
        "ce_frac_above": (rank - 1) / n,
        "ce_l_entropy": np.where(multi, g["qlq"].transform("sum").to_numpy() / np.log(np.maximum(n, 2)), np.nan),
        "ce_softmax_hhi": g["soft2"].transform("sum").to_numpy(),
    })
    for thr, name in ((0.5, "05"), (0.7, "07"), (0.8, "08"), (0.9, "09")):
        df["b"] = (p >= thr).astype(np.float64)
        c[f"ce_n_ge{name}"] = df.groupby("s", sort=False)["b"].transform("sum").to_numpy()

    # rank among same-name candidates (n_tset >= 0.95) of the list; NaN when the row itself is not same-name
    same = _col(X, "n_tset") >= _SAME_NAME
    df["sp"] = np.where(same, p, np.nan)
    gsn = df.groupby("s", sort=False)["sp"]
    c["ce_samename_n"] = gsn.transform("count").to_numpy().astype(np.float64)
    c["ce_samename_rank"] = np.where(same, gsn.rank(ascending=False, method="min").to_numpy(), np.nan)

    # disagreement with stage 2 (p_rerank, its list rank)
    prr, s2rank = _col(X, "p_rerank"), _col(X, "rank")
    s2_top = s2rank <= 1
    df["ce_at_s2top"] = np.where(s2_top, p, np.nan)
    df["prr"] = prr
    g = df.groupby("s", sort=False)
    ce_at_s2top = g["ce_at_s2top"].transform("max").to_numpy()
    prr_rank = g["prr"].rank(ascending=False, method="first").to_numpy()
    df["prr2"] = np.where(prr_rank == 2, prr, np.nan)
    s2_margin = g["prr"].transform("max").to_numpy() - df.groupby("s", sort=False)["prr2"].transform("max").to_numpy()
    c.update({
        "ce_minus_prr": p - prr, "ce_abs_prr": np.abs(p - prr), "ce_rank_minus_s2": rank - s2rank,
        "ce_top1_is_s2top1": ((rank <= 1) & s2_top).astype(np.float64),
        "ce_s2_top1_agree": (np.abs(ce_at_s2top - t1) < 1e-12).astype(np.float64),
        "ce_margin_minus_s2": (t1 - t2) - s2_margin,
        "ce_minus_s2top1": p - ce_at_s2top,
    })

    # products with one evidence column each; binary/categorical columns as indicators, not codes
    col = lambda f: np.nan_to_num(_col(X, f), nan=-1.0)
    c.update({
        "ce_x_ntset": p * _col(X, "n_tset"), "ce_x_atset": p * _col(X, "a_tset"),
        "ce_x_cmatch": p * (col("country3") == 1), "ce_x_house": p * _col(X, "a_house"),
        "ce_x_numjacc": p * _col(X, "a_num_jacc"), "ce_x_prr": p * prr, "ce_x_s2rank": p / np.maximum(s2rank, 1),
        "ce_x_ln": p * _col(X, "l_n"), "ce_x_aempty": p * (col("a_empty") == 1), "ce_x_chain": p * _col(X, "log_chain"),
        "ce_x_src": p * (col("src") == 3),
    })
    _CEX_CACHE[key] = c
    return c


def group_spec(name):
    """(columns, fn) of a feature group; 'ce:<name>' is a cross-encoder score group and 'cex_<family>:<name>' a
    CE-derived family (neither cached: the score file is the cache, and a re-scored file must not be shadowed
    by an old copy)."""
    if name.startswith("ce:"):
        ce = name[3:]
        return ([f"p_{ce}", f"p_{ce}_rank", f"p_{ce}_gap"],
                lambda set_name, X, s1, pool: group_ce(ce, set_name, s1, pool))
    if name.startswith("cex_"):
        fam, ce = name[4:].split(":")
        cols = CEX_FAMILIES[fam]
        return (list(cols), lambda set_name, X, s1, pool:
                np.column_stack([cex_table(ce, set_name, X, s1, pool)[k] for k in cols]))
    return GROUPS[name]


def group_path(set_name, name):
    return io_utils.work_path("exp", "groups", set_name, f"{name}.parquet")


def load_group(set_name, name, X, s1, pool, lg=None):
    """Group columns for the feature rows (cached per set; the cache stores s1/pool to verify row order)."""
    cols, fn = group_spec(name)
    if name.startswith(("ce:", "cex_")) or name in ("rev", "grp", "grp_full"):
        return fn(set_name, X, s1, pool).astype(np.float32)
    path = group_path(set_name, name)
    if path.exists():
        t = pq.read_table(path)
        if (t.num_rows == s1.size and np.array_equal(t.column("s1_idx").to_numpy(), s1)
                and np.array_equal(t.column("pool_idx").to_numpy(), pool) and t.column_names[2:] == cols):
            return np.column_stack([t.column(c).to_numpy() for c in cols]).astype(np.float32)
    t0 = time.time()
    G = fn(set_name, X, s1, pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab = {"s1_idx": pa.array(s1), "pool_idx": pa.array(pool)}
    tab.update({c: pa.array(G[:, i]) for i, c in enumerate(cols)})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.table(tab), tmp, compression="zstd")
    tmp.replace(path)
    if lg:
        lg.info("group %s on %s: %d rows in %.0fs", name, set_name, s1.size, time.time() - t0)
    return G


def full_matrix(set_name, groups, fingerprint, lg=None):
    """(X with base FEATURES + group columns, names, s1, pool, meta)."""
    X, s1, pool, meta = match.load_features(set_name, fingerprint)
    names = list(match.FEATURES)
    blocks = [X]
    for g in groups:
        blocks.append(load_group(set_name, g, X, s1, pool, lg))
        names += group_spec(g)[0]
    return (np.hstack(blocks) if len(blocks) > 1 else X), names, s1, pool, meta


# ---------------------------------------------------------------- sample weights

def sample_weights(spec, Xf, names, s1):
    """Label-free weights (features only). Normalised to mean 1."""
    if not spec:
        return None
    col = {n: i for i, n in enumerate(names)}
    kind = spec["kind"]
    if kind == "macro":
        # each S1 contributes equally to the loss, as each S1 does to the macro F0.5
        n = pd.Series(s1).groupby(s1).transform("size").to_numpy().astype(np.float64)
        w = 1.0 / n ** spec.get("power", 1.0)
    elif kind == "hard":
        # upweight pairs that look ambiguous before any model sees them (token-set scores are 0-1): strong
        # name with weak or empty address, or a near-tie with the next p_rerank while p_rerank is mid-range.
        # On MATCH_TRAIN this marks 18.5% of rows and 58% of the reference run's uncertain-band rows.
        nt, at = Xf[:, col["n_tset"]], Xf[:, col["a_tset"]]
        gap, prr = np.abs(Xf[:, col["l_p_gap_next"]]), Xf[:, col["p_rerank"]]
        hard = ((nt >= spec.get("name_min", 0.9)) & ~(at >= spec.get("addr_max", 0.6))) | \
               ((gap < spec.get("gap", 0.1)) & (prr >= 0.05) & (prr <= 0.95))
        w = np.where(hard, spec.get("alpha", 2.0), 1.0)
    else:
        raise ValueError(f"unknown weight kind {kind!r}")
    return (w / w.mean()).astype(np.float64)


# ---------------------------------------------------------------- train

# Specialist models (items 9 / 11): each sees one evidence family only; their OOF scores and the name-vs-address
# disagreement are stacked into the main model.
SPECIALISTS = {
    "name": lambda f: f.startswith("n_") or f in ("q_ntok", "p_ntok", "q_name_cnt", "p_name_cnt"),
    "addr": lambda f: f.startswith("a_") or f in ("q_addr_cnt", "p_addr_cnt"),
    "ctx": lambda f: f in ("p_rerank", "rank", "src", "country3", "pool_indic", "log_chain") or f.startswith("l_"),
}


# Fellegi-Sunter m/u evidence (item 12): agreement pattern = binned name token-set x address token-set x
# address-number Jaccard; weight = log P(pattern | match) / P(pattern | non-match), Laplace-smoothed, fitted on
# the fold's fit rows only (test time: on all MATCH_TRAIN rows).
_MU_BINS = {"n_tset": [0.5, 0.7, 0.85, 0.95], "a_tset": [0.5, 0.7, 0.85, 0.95], "a_num_jacc": [1e-9, 1.0 - 1e-9]}


def mu_pattern(cols):
    code = np.zeros(next(iter(cols.values())).size, dtype=np.int64)
    for f, edges in _MU_BINS.items():
        v = cols[f]
        b = np.where(np.isnan(v), len(edges) + 1, np.digitize(v, edges))
        code = code * (len(edges) + 2) + b
    return code


def mu_table(code, y):
    n_cells = int(np.prod([len(e) + 2 for e in _MU_BINS.values()]))
    c1 = np.bincount(code[y], minlength=n_cells).astype(np.float64)
    c0 = np.bincount(code[~y], minlength=n_cells).astype(np.float64)
    return (np.log((c1 + 1) / (c1.sum() + n_cells)) - np.log((c0 + 1) / (c0.sum() + n_cells))).tolist()


def _folds(y, s1, mcfg):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    gkf = GroupKFold(n_splits=mcfg["n_folds"], shuffle=True, random_state=mcfg["fold_seed"])
    out = []
    for k, (tr, te) in enumerate(gkf.split(np.zeros(y.size), y, s1)):
        gss = GroupShuffleSplit(n_splits=1, test_size=mcfg["early_stop_frac"], random_state=mcfg["fold_seed"] + k)
        fit_i, es_i = next(gss.split(tr, groups=s1[tr]))
        out.append((k, tr, te, tr[fit_i], tr[es_i]))
    return out


def _train_one(params, n_round, esr, X, y, w, feats, fit_i, es_i):
    import lightgbm as lgb
    dfit = lgb.Dataset(X[fit_i], label=y[fit_i], weight=None if w is None else w[fit_i], feature_name=feats,
                       free_raw_data=False)
    # early stopping on UNWEIGHTED logloss: the stopping rule should not depend on the weighting
    des = lgb.Dataset(X[es_i], label=y[es_i], reference=dfit, free_raw_data=False)
    return lgb.train(params, dfit, num_boost_round=n_round, valid_sets=[des], valid_names=["early_stop"],
                     callbacks=[lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(period=0)])


def _model_name(prefix, k, j, n_seeds, part):
    s = f"{prefix}_fold{k}"
    if n_seeds > 1:
        s += f"_s{j}"
    if part is not None:
        s += f"_part{int(part)}"
    return s + ".txt"


def run_exp_train(args):
    from sklearn.isotonic import IsotonicRegression

    t0 = time.time()
    spec = json.loads(open(args.exp).read())
    run = spec["id"]
    lg = io_utils.get_logger(f"exp-train_{run}")
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    groups = spec.get("groups", [])
    Xf, names, s1, pool, meta = full_matrix("MATCH_TRAIN", groups, fp, lg)
    if meta["limited"]:
        raise RuntimeError("MATCH_TRAIN features built with --limit")
    queries = match.set_queries("MATCH_TRAIN")
    rows = None
    if spec.get("smoke_s1"):  # code check on the first N S1 only; never a result
        cut = np.unique(s1)[int(spec["smoke_s1"])]
        rows = np.flatnonzero(s1 < cut)
        Xf, s1, pool, queries = Xf[rows], s1[rows], pool[rows], queries[queries < cut]
    drop = set(spec.get("drop_features", [])) | set(mcfg.get("drop_features", []))
    feats = [n for n in names if n not in drop]
    ci = {n: i for i, n in enumerate(names)}
    X = Xf[:, [ci[f] for f in feats]]
    y = match.pair_labels(s1, pool)
    w = sample_weights(spec.get("weights"), Xf, names, s1)
    params = dict(mcfg["lgb"])
    params.update(spec.get("lgb", {}))
    params["num_threads"] = args.workers or io_utils.n_workers(cfg)
    n_round = int(spec.get("num_boost_round", mcfg["num_boost_round"]))
    esr = mcfg["early_stopping_rounds"]
    seeds = spec.get("seeds", [params.get("seed", 42)])
    folds_idx = _folds(y, s1, mcfg)
    rd = match.runs_dir(run)
    (rd / "models").mkdir(parents=True, exist_ok=True)
    folds = []

    def log_fold(tag, k, seed, b, cap=None):
        cap = cap or n_round
        truncated = b.best_iteration >= cap - esr
        folds.append({"model": tag, "fold": k, "seed": seed, "best_iteration": int(b.best_iteration),
                      "cap": cap, "truncated": bool(truncated),
                      "early_stop_logloss": float(b.best_score["early_stop"]["binary_logloss"])})
        lg.info("%s fold %d seed %d: best_iteration %d/%d%s (%.0fs)", tag, k, seed, b.best_iteration, cap,
                " TRUNCATED" if truncated else "", time.time() - t0)

    # ---- stacked specialists (OOF over the same folds; test time = fold average)
    stack = {}
    if spec.get("specialists"):
        cols_sp, S = {}, []
        sp_round = int(spec.get("spec_num_boost_round", n_round))  # weaker feature sets need more rounds (M011)
        for sname in spec["specialists"]:
            cols = [f for f in feats if SPECIALISTS[sname](f)]
            cols_sp[sname] = cols
            Xs = X[:, [feats.index(c) for c in cols]]
            o = np.zeros(y.size)
            p = dict(params, seed=42)
            for k, tr, te, fit_i, es_i in folds_idx:
                b = _train_one(p, sp_round, esr, Xs, y, None, cols, fit_i, es_i)
                o[te] = b.predict(Xs[te], num_iteration=b.best_iteration)
                b.save_model(str(rd / "models" / f"spec_{sname}_fold{k}.txt"), num_iteration=b.best_iteration)
                log_fold(f"spec_{sname}", k, 42, b, cap=sp_round)
            S.append(o)
        S = np.column_stack(S)
        sp_names = [f"sp_{s}" for s in spec["specialists"]]
        if "name" in spec["specialists"] and "addr" in spec["specialists"]:
            S = np.column_stack([S, S[:, sp_names.index("sp_name")] - S[:, sp_names.index("sp_addr")]])
            sp_names.append("sp_name_minus_addr")
        X = np.hstack([X, S.astype(np.float32)])
        feats = feats + sp_names
        stack = {"specialists": spec["specialists"], "cols": cols_sp, "names": sp_names}

    # ---- residual model on the parent's uncertain band (item 10)
    residual = None
    train_mask = np.ones(y.size, bool)
    if spec.get("residual_of"):
        par = spec["residual_of"]
        t = pq.read_table(match.runs_dir(par) / "oof.parquet").to_pandas()
        if rows is not None:
            t = t.iloc[rows].reset_index(drop=True)
        if not (np.array_equal(t["s1_idx"].to_numpy(), s1) and np.array_equal(t["pool_idx"].to_numpy(), pool)):
            raise RuntimeError(f"{par} OOF rows differ from MATCH_TRAIN feature rows")
        par_raw, par_cal = t["p_raw"].to_numpy().astype(np.float64), t["p_cal"].to_numpy().astype(np.float64)
        train_mask = (par_cal >= BAND[0]) & (par_cal < BAND[1])
        X = np.hstack([X, par_raw[:, None].astype(np.float32)])
        feats = feats + ["par_p"]
        residual = {"parent": par, "band": list(BAND)}

    mu_code = mu_pattern({f: Xf[:, ci[f]] for f in _MU_BINS}) if spec.get("mu") else None
    mu_full = None
    if mu_code is not None:
        mu_full = mu_table(mu_code, y)
        X = np.hstack([X, np.zeros((y.size, 1), np.float32)])
        feats = feats + ["mu_llr"]

    # ---- main model(s): optional partition by one feature's value (item 13)
    split_by = spec.get("split_by")
    part_vals = [None]
    if split_by:
        part_col = Xf[:, ci[split_by]]
        part_vals = sorted(float(v) for v in np.unique(part_col[~np.isnan(part_col)]))
        if np.isnan(part_col).any():
            raise RuntimeError(f"split_by {split_by} has NaN values")
    oof = np.zeros(y.size)
    gain = np.zeros(len(feats))
    lg.info("exp %s: %d rows (%d trainable), %d features, groups %s, weights %s, seeds %s, specialists %s, "
            "residual_of %s, split_by %s, cap %d", run, y.size, int(train_mask.sum()), len(feats), groups,
            spec.get("weights"), seeds, spec.get("specialists"), spec.get("residual_of"), split_by, n_round)
    for k, tr, te, fit_i, es_i in folds_idx:
        if mu_code is not None:
            X[:, -1] = np.asarray(mu_table(mu_code[fit_i], y[fit_i]), dtype=np.float32)[mu_code]
        for part in part_vals:
            pm = train_mask if part is None else train_mask & (part_col == part)
            f_i, e_i, t_i = fit_i[pm[fit_i]], es_i[pm[es_i]], te[pm[te]]
            for j, seed in enumerate(seeds):
                b = _train_one(dict(params, seed=seed), n_round, esr, X, y, w, feats, f_i, e_i)
                oof[t_i] += b.predict(X[t_i], num_iteration=b.best_iteration) / len(seeds)
                b.save_model(str(rd / "models" / _model_name("model", k, j, len(seeds), part)),
                             num_iteration=b.best_iteration)
                gain += b.feature_importance("gain", iteration=b.best_iteration) / len(seeds) / len(part_vals)
                log_fold("main" if part is None else f"main_part{int(part)}", k, seed, b)

    if residual:
        # band rows: residual model calibrated on the band; other rows keep the parent's calibrated p
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof[train_mask], y[train_mask])
        calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
        p_cal = par_cal.astype(np.float32).copy()
        p_cal[train_mask] = match.calibrate(oof[train_mask], calib)
        oof = np.where(train_mask, oof, par_raw)
    else:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
        calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
        p_cal = match.calibrate(oof, calib)
    decision, per_entity, report, rule_scores, sweep = choose_decision(queries, s1, pool, y, p_cal, mcfg)

    doc = {"features": match.FEATURES, "extra_groups": groups, "feature_names": names, "features_used": feats,
           "n_folds": mcfg["n_folds"], "n_seeds": len(seeds), "bucketing_fingerprint": fp, "run": run,
           "train_set": "MATCH_TRAIN", "train_limited": rows is not None, "calibration": calib, "decision": decision,
           "floor_threshold": None, "oof_f05": report["f05"], "weights": spec.get("weights"),
           "stack": stack, "residual": residual, "split_by": split_by, "part_vals": part_vals, "mu_table": mu_full,
           "best_iteration": [f["best_iteration"] for f in folds], "num_boost_round": n_round,
           "truncated": any(f["truncated"] for f in folds), "exp": spec,
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    io_utils.atomic_write_json(rd / "models" / "matcher.json", doc)
    np.save(rd / "queries.npy", queries)
    np.save(rd / "OOF_per_entity.npy", per_entity)
    pq.write_table(pa.table({"s1_idx": s1, "pool_idx": pool, "y": y, "p_raw": oof.astype(np.float32),
                             "p_cal": p_cal}), rd / "oof.parquet", compression="zstd")

    payload = {"run": run, "exp": spec, "n_rows": int(y.size), "features_used": feats, "folds": folds,
               "any_fold_truncated": doc["truncated"], "decision": decision, "oof": report,
               "oof_rule_scores": rule_scores, "oof_threshold_sweep": sweep,
               "calibration_reliability": match.reliability(p_cal, y),
               "feature_gain": dict(sorted(((f, float(g / mcfg["n_folds"])) for f, g in zip(feats, gain)),
                                           key=lambda kv: -kv[1])),
               "vs_parent": {} if rows is not None else compare_to_parent(spec.get("parent"), queries, per_entity),
               "wall_s": round(time.time() - t0, 1)}
    io_utils.write_metrics(f"exp_{run}", payload, queries.size, int(match.true_counts(queries).sum()), cfg)
    vp = payload["vs_parent"]
    lg.info("exp %s: OOF F0.5 %.5f; vs %s: %s; band %s; truncated folds %d", run, report["f05"],
            spec.get("parent"), vp.get("all"), vp.get("band_s1"), sum(f["truncated"] for f in folds))


def choose_decision(queries, s1, pool, y, p_cal, mcfg):
    """Same rule selection as match-train: threshold vs expected-F, one-owner on/off, adopt only on a paired
    gain with CI > 0."""
    scorer = match.Scorer(queries, s1, y, match.true_counts(queries))
    dcfg = mcfg["decision"]
    grid = match.threshold_grid(dcfg)
    cands = {}
    for owner in (False, True):
        pc = match.one_owner(pool, p_cal) if owner else p_cal
        sw = {float(t): scorer.f05(pc >= t) for t in grid}
        tb = max(sw, key=sw.get)
        for rule, dec in (("threshold", {"rule": "threshold", "threshold": tb, "one_owner": owner}),
                          ("expected_f", {"rule": "expected_f", "threshold": None, "one_owner": owner})):
            cands[(rule, owner)] = (dec, scorer.per_entity(match.apply_decision(dec, s1, pool, p_cal))[0],
                                    sw if rule == "threshold" else None)
    ef = match.paired_delta(cands[("expected_f", False)][1], cands[("threshold", False)][1])
    base = "expected_f" if match.adopt(ef, dcfg["expected_f_min_gain"]) else "threshold"
    oo = match.paired_delta(cands[(base, True)][1], cands[(base, False)][1])
    owner = match.adopt(oo, dcfg["one_owner_min_gain"])
    decision = cands[(base, owner)][0]
    report, per_entity = scorer.report(match.apply_decision(decision, s1, pool, p_cal), p_cal)
    scores = {f"{r}{'+one_owner' if o else ''}": float(v[1].mean()) for (r, o), v in cands.items()}
    sweep = {str(t): v for t, v in cands[("threshold", False)][2].items()}
    return decision, per_entity, report, scores, sweep


def band_mask(queries, ref_run="x1_strsim"):
    """S1 of MATCH_TRAIN with >= 1 candidate whose calibrated OOF p (reference run) is in the uncertain band.
    Fixed across experiments so every run is compared on the same S1."""
    t = pq.read_table(match.runs_dir(ref_run) / "oof.parquet", columns=["s1_idx", "p_cal"])
    s, p = t.column("s1_idx").to_numpy(), t.column("p_cal").to_numpy()
    inb = s[(p >= BAND[0]) & (p < BAND[1])]
    return np.isin(queries, inb)


def compare_to_parent(parent, queries, per_entity):
    if not parent:
        return {}
    rd = match.runs_dir(parent)
    fa = np.load(rd / "OOF_per_entity.npy")
    if not np.array_equal(np.load(rd / "queries.npy"), queries):
        raise RuntimeError(f"parent {parent} was scored on different queries")
    from ber.evaluate import segment_data
    masks = dict(segment_data(queries, np.zeros(0, dtype=np.uint32))["query"])
    masks["singleton"] = match.true_counts(queries) == 0
    band = band_mask(queries)
    out = {"parent": parent, "f05_parent": float(fa.mean()), "f05_new": float(per_entity.mean()),
           "all": match.paired_delta(per_entity, fa, n_boot=2000),
           "band_s1": dict(match.paired_delta(per_entity[band], fa[band], n_boot=1000),
                           share=float(band.mean()), f05_parent=float(fa[band].mean())),
           "non_band_s1": match.paired_delta(per_entity[~band], fa[~band], n_boot=1000),
           "segments": {k: dict(match.paired_delta(per_entity[m], fa[m], n_boot=500), f05_parent=float(fa[m].mean()))
                        for k, m in masks.items() if m.any()}}
    return out


# ---------------------------------------------------------------- predict

def _avg_predict(files, X):
    import lightgbm as lgb
    if not files:
        raise RuntimeError("no model files")
    p = np.zeros(X.shape[0])
    for f in files:
        p += lgb.Booster(model_file=str(f)).predict(X) / len(files)
    return p


def predict_run(set_name, doc, models_dir, lg=None):
    """(Xf, names, s1, pool, meta, p_cal, p_raw) for every feature row of a set with the run's models:
    fold (and seed) average, specialists stacked, split_by routing, residual band on top of the parent."""
    Xf, names, s1, pool, meta = full_matrix(set_name, doc.get("extra_groups", []), doc["bucketing_fingerprint"], lg)
    if names != doc.get("feature_names", names):
        raise RuntimeError("feature/group columns differ from the run's")
    ci = {n: i for i, n in enumerate(names)}
    base = [f for f in doc["features_used"] if f in ci]
    cols = {f: Xf[:, ci[f]] for f in base}
    stack = doc.get("stack") or {}
    if stack:
        S = {}
        for sname in stack["specialists"]:
            Xs = np.column_stack([cols[c] for c in stack["cols"][sname]]).astype(np.float32)
            S[f"sp_{sname}"] = _avg_predict(sorted(models_dir.glob(f"spec_{sname}_fold*.txt")), Xs)
        if "sp_name_minus_addr" in stack["names"]:
            S["sp_name_minus_addr"] = S["sp_name"] - S["sp_addr"]
        cols.update(S)
    residual = doc.get("residual")
    band = None
    if residual:
        pdir = match.runs_dir(residual["parent"]) / "models"
        pdoc = io_utils.read_json(pdir / "matcher.json")
        _, _, ps1, ppool, _, par_cal, par_raw = predict_run(set_name, pdoc, pdir, lg)
        if not (np.array_equal(ps1, s1) and np.array_equal(ppool, pool)):
            raise RuntimeError("parent rows differ")
        cols["par_p"] = par_raw
        band = (par_cal >= residual["band"][0]) & (par_cal < residual["band"][1])
    if doc.get("mu_table"):
        cols["mu_llr"] = np.asarray(doc["mu_table"])[mu_pattern({f: Xf[:, ci[f]] for f in _MU_BINS})]
    X = np.column_stack([cols[f] for f in doc["features_used"]]).astype(np.float32)
    n_seeds = doc.get("n_seeds", 1)
    raw = np.zeros(X.shape[0])
    split_by = doc.get("split_by")
    if split_by:
        pv = Xf[:, ci[split_by]]
        for part in doc["part_vals"]:
            m = pv == part
            files = sorted(models_dir.glob(f"model_fold*_part{int(part)}.txt"))
            if len(files) != doc["n_folds"] * n_seeds:
                raise RuntimeError(f"part {part}: expected {doc['n_folds'] * n_seeds} model files")
            if m.any():
                raw[m] = _avg_predict(files, X[m])
        unseen = ~np.isin(pv, doc["part_vals"])
        if unseen.any():
            raise RuntimeError(f"{int(unseen.sum())} rows have a {split_by} value unseen in training")
    else:
        files = sorted(models_dir.glob("model_fold*.txt"))
        if len(files) != doc["n_folds"] * n_seeds:
            raise RuntimeError(f"{models_dir}: expected {doc['n_folds'] * n_seeds} model files")
        m = band if band is not None else np.ones(X.shape[0], bool)
        raw[m] = _avg_predict(files, X[m])
        blend = doc.get("xgb_blend")
        if blend:  # (1 - w) * LightGBM average + w * XGBoost average (plan/checks/train_xgb_blend.py)
            import xgboost as xgb
            dm = xgb.DMatrix(X[m], feature_names=list(doc["features_used"]))
            px = np.zeros(int(m.sum()))
            for f in blend["files"]:
                b = xgb.Booster(model_file=str(models_dir / f))
                px += b.predict(dm) / len(blend["files"])
            raw[m] = (1 - blend["weight"]) * raw[m] + blend["weight"] * px
    if residual:
        p_cal = par_cal.astype(np.float32).copy()
        p_cal[band] = match.calibrate(raw[band], doc["calibration"])
        raw = np.where(band, raw, par_raw)
    else:
        p_cal = match.calibrate(raw, doc["calibration"])
    return Xf, names, s1, pool, meta, p_cal, raw


def run_exp_predict(args):
    t0 = time.time()
    lg = io_utils.get_logger(f"exp-predict_{args.run}")
    cfg = io_utils.load_config()
    models_dir = match.runs_dir(args.run) / "models"
    doc = io_utils.read_json(models_dir / "matcher.json")
    if doc["bucketing_fingerprint"] != match.bucketing_fingerprint(cfg):
        raise RuntimeError("bucketing fingerprint changed since this run was trained")
    if doc.get("train_limited"):
        raise RuntimeError("smoke run (smoke_s1): not a submittable model")
    io_utils.log_test_usage(f"exp-predict {args.run}: matcher inference over test candidates (inputs only)",
                            "submission candidate from an experiment run", str(match.features_dir("TEST")))
    _, _, s1, pool, meta, p_cal, _ = predict_run("TEST", doc, models_dir, lg)
    if meta["limited"]:
        raise RuntimeError("TEST features were built with --limit")
    keep = match.apply_decision(doc["decision"], s1, pool, p_cal)
    rec = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["s1_idx", "entity_id"])
    if not np.array_equal(rec["s1_idx"].to_numpy(), np.arange(len(rec))):
        raise RuntimeError("records/test_s1.parquet: s1_idx is not 0..n-1 in file order")
    s1_ids = rec["entity_id"].tolist()
    store = io_utils.paths()["work_root"] / "norm" / "test_pool_store"
    src = np.load(store / "src.npy", mmap_mode="r")
    num = np.load(store / "num.npy", mmap_mode="r")
    out_path = io_utils.work_path("outputs", args.run, "matching_results.tsv")
    k = np.flatnonzero(keep)
    k = k[np.lexsort((-p_cal[k], s1[k]))]
    match.write_matching_file(out_path, s1_ids, s1[k], pool[k], src, num)
    cand_keys = np.sort(match.pair_keys(s1, np.asarray(src)[pool], np.asarray(num)[pool]))
    issues = match.check_matching_file(out_path, s1_ids, cand_keys)
    n_pred = np.bincount(s1, weights=keep, minlength=len(s1_ids))
    country = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["country_raw"])["country_raw"].to_numpy()
    per_country = {c: {"n_s1": int((country == c).sum()), "mean_pred_size": float(n_pred[country == c].mean()),
                       "share_empty": float((n_pred[country == c] == 0).mean())} for c in sorted(set(country.tolist()))}
    payload = {"run": args.run, "decision": doc["decision"], "pairs": int(keep.sum()), "per_country": per_country,
               "file": str(out_path), "check": {"passed": not issues, "issues": issues},
               "wall_s": round(time.time() - t0, 1)}
    io_utils.write_metrics(f"exp_predict_TEST_{args.run}", payload, len(s1_ids), 0, cfg)
    if issues:
        raise RuntimeError("matching_results.tsv failed checks:\n" + "\n".join(issues))
    lg.info("wrote %s: %d pairs (%.0fs)", out_path, int(keep.sum()), time.time() - t0)

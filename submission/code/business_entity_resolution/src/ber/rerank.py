"""M7 stage-2 re-ranker (plan §5.8): feature computation + the five CLI entry points
`rerank-build`, `rerank-train`, `rerank-cut`, `rerank-loco`, `rerank-eval`.

Feature count: the plan's default listing assumes the 10 default stage-1 families (incl. PH). E2
dropped PH and never adopted SK as a default family (PROGRESS.md), so the current kept family set is
`cfg["families"]` = EN, NT, NG, NP, SP, NC, AT, AB, XK (9 families; COV_FAMILIES NT/NG/NP/AT/XK all
present, giving 5 `cov_` columns). `feature_names(r)` builds the column list from the live Retriever
instead of hard-coding a count; with the current 9-family config this is 9 (`s_<F>`) + 5 (`cov_<F>`)
+ 1 (`learn`) + 3 (`rk_LEARN/rk_NAME/rk_ADDR`) + 3 (`log_union/log_dEN/log_chain`) + 4 (name fuzzy)
+ 1 (`sk_jacc`) + 2 (address fuzzy) + 1 (`addr_empty`) + 1 (`postal`) + 1 (`country`) + 1 (`src`)
+ 1 (`pool_indic`) + 2 (`q_ntok`/`p_ntok`) = 35 columns (the plan's 36 default minus the one dropped
`s_PH` column; `log1p_s_<F>` stage-1-only columns are never copied into the stage-2 feature vector,
matching the plan's own count: 10 s_ + 5 cov_ + ... = 36, not 10 + 5 + 10 + ... = 46).

`PoolTextReader` mmaps `norm/{split}_pool_store/` (name/addr text, country, src, indic) the same way
`write_outputs.py` mmaps src/num; text is decoded batched over the sorted-unique pool_idx of a call.
"""

import math
import mmap as _mmap
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupShuffleSplit

from ber import evaluate, io_utils, retrieve, splits
from ber.textnorm import skeleton

SEED_BASE_BUILD = 1000  # plan §5.8: rand_neg rng seed = 1000 + s1_idx
CANDIDATE_SCHEMA = pa.schema([("s1_idx", pa.int32()), ("pool_idx", pa.uint32()),
                              ("rank", pa.int16()), ("p_rerank", pa.float32())])
NOT_LGB_PARAMS = ("num_boost_round", "early_stopping_rounds", "holdout_frac")


def _metrics_path(name):
    return io_utils.paths()["metrics_root"] / f"{name}.json"


# ---------------------------------------------------------------- pool text store (batched reader)

class PoolTextReader:
    """Batched reader over `norm/{split}_pool_store/` (plan §4): name/addr text (mmap'd .bin + .off.npy),
    country (raw label via country.npy + countries.json), src, indic. Construct once per worker process
    and reuse; `name`/`addr` decode only the distinct pool_idx of each call."""

    def __init__(self, split):
        self.dir = io_utils.paths()["work_root"] / "norm" / f"{split}_pool_store"
        self._off = {f: np.load(self.dir / f"{f}.off.npy", mmap_mode="r") for f in ("name", "addr")}
        self._fh = {f: open(self.dir / f"{f}.bin", "rb") for f in ("name", "addr")}
        self._mm = {f: _mmap.mmap(self._fh[f].fileno(), 0, access=_mmap.ACCESS_READ)
                    if os.path.getsize(self.dir / f"{f}.bin") else b"" for f in ("name", "addr")}
        self.src_arr = np.load(self.dir / "src.npy", mmap_mode="r")
        self.num_arr = np.load(self.dir / "num.npy", mmap_mode="r")
        self.country_codes = np.load(self.dir / "country.npy", mmap_mode="r")
        self.countries = io_utils.read_json(self.dir / "countries.json")
        self.indic_arr = np.load(self.dir / "indic.npy", mmap_mode="r")

    def close(self):
        for f, mm in self._mm.items():
            if mm:
                mm.close()
        for fh in self._fh.values():
            fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _text(self, field, pool_idx):
        pool_idx = np.asarray(pool_idx)
        if pool_idx.size == 0:
            return []
        uniq, inv = np.unique(pool_idx, return_inverse=True)
        off, mm = self._off[field], self._mm[field]
        vals = [bytes(mm[int(off[i]):int(off[i + 1])]).decode("utf-8") for i in uniq]
        return [vals[j] for j in inv]

    def name(self, pool_idx):
        return self._text("name", pool_idx)

    def addr(self, pool_idx):
        return self._text("addr", pool_idx)

    def country(self, pool_idx):
        codes = np.asarray(self.country_codes[np.asarray(pool_idx)])
        return [self.countries[int(c)] for c in codes]

    def src(self, pool_idx):
        return np.asarray(self.src_arr[np.asarray(pool_idx)])

    def indic(self, pool_idx):
        return np.asarray(self.indic_arr[np.asarray(pool_idx)])


_PTR_CACHE = {}


def _pool_text_reader(split):
    if _PTR_CACHE.get("split") != split:
        _PTR_CACHE["reader"] = PoolTextReader(split)
        _PTR_CACHE["split"] = split
    return _PTR_CACHE["reader"]


# ---------------------------------------------------------------- feature computation (plan §5.8 A)

def feature_names(r):
    """Column names in the fixed §5.8 order, derived from the live Retriever `r` (not hard-coded)."""
    n_s_cov = len(r.families) + len(r.cov_rows)
    names = list(r.feature_names[:n_s_cov])  # s_<F> ... cov_<F> ...
    names += ["learn"]
    names += [f"rk_{c}" for c in retrieve.STAGE2_RANKS]
    names += ["log_union", "log_dEN", "log_chain"]
    names += ["fz_tset", "fz_ratio", "fz_partial", "jw_concat"]
    names += ["sk_jacc"]
    names += ["fa_tset", "fa_tsort"]
    names += ["addr_empty", "postal", "country", "src", "pool_indic", "q_ntok", "p_ntok"]
    return names


_POSTAL_LEN = (4, 6)


def _postal_tokens(tokens):
    return {t for t in tokens if t.isdigit() and _POSTAL_LEN[0] <= len(t) <= _POSTAL_LEN[1]}


def _skeleton_set(tokens):
    return {k for k in (skeleton(t) for t in tokens if len(t) >= 2) if len(k) >= 2}


def compute_features(r, ptr, out, rows, chain, q_name, q_addr, q_country):
    """Rerank feature matrix (float32, `feature_names(r)` order) for a subset `rows` of one query's
    `retrieve()` output `out`. Rapidfuzz/skeleton features are computed only for the given rows."""
    names = feature_names(r)
    rows = np.asarray(rows, dtype=np.intp)
    n = rows.size
    if n == 0:
        return np.zeros((0, len(names)), dtype=np.float32)
    n_s_cov = len(r.families) + len(r.cov_rows)
    pool_idx = out["pool_idx"][rows]
    parts = [out["X"][rows, :n_s_cov], out["learn"][rows].reshape(-1, 1).astype(np.float32),
             out["ranks"][rows].astype(np.float32)]
    log_union = math.log1p(out["n_union"])
    log_dEN = math.log1p(out["d_en"])
    log_chain = math.log1p(chain)
    parts.append(np.tile(np.array([log_union, log_dEN, log_chain], dtype=np.float32), (n, 1)))

    p_name = ptr.name(pool_idx)
    p_addr = ptr.addr(pool_idx)
    p_country = ptr.country(pool_idx)
    q_tokens = q_name.split()
    q_addr_tokens = q_addr.split()
    q_sk = _skeleton_set(q_tokens)
    q_concat = "".join(sorted(q_tokens))
    q_postal = _postal_tokens(q_tokens) | _postal_tokens(q_addr_tokens)
    q_nonempty = bool(q_addr)

    fz = np.empty((n, 4), dtype=np.float32)
    sk_jacc = np.empty(n, dtype=np.float32)
    fa = np.empty((n, 2), dtype=np.float32)
    addr_empty = np.empty(n, dtype=np.float32)
    postal = np.empty(n, dtype=np.float32)
    country = np.empty(n, dtype=np.float32)
    p_ntok = np.empty(n, dtype=np.float32)
    for i in range(n):
        pn, pa, pc = p_name[i], p_addr[i], p_country[i]
        fz[i, 0] = fuzz.token_set_ratio(q_name, pn) / 100.0
        fz[i, 1] = fuzz.ratio(q_name, pn) / 100.0
        fz[i, 2] = fuzz.partial_ratio(q_name, pn) / 100.0
        p_tokens = pn.split()
        p_concat = "".join(sorted(p_tokens))
        fz[i, 3] = JaroWinkler.normalized_similarity(q_concat, p_concat)
        p_sk = _skeleton_set(p_tokens)
        union = q_sk | p_sk
        sk_jacc[i] = (len(q_sk & p_sk) / len(union)) if union else 0.0
        p_nonempty = bool(pa)
        both = q_nonempty and p_nonempty
        fa[i, 0] = (fuzz.token_set_ratio(q_addr, pa) / 100.0) if both else 0.0
        fa[i, 1] = (fuzz.token_sort_ratio(q_addr, pa) / 100.0) if both else 0.0
        addr_empty[i] = (0 if both else (1 if q_nonempty and not p_nonempty else
                         (2 if p_nonempty and not q_nonempty else 3)))
        p_postal = _postal_tokens(p_tokens) | _postal_tokens(pa.split())
        if not q_postal and not p_postal:
            postal[i] = 0
        elif q_postal & p_postal:
            postal[i] = 2
        else:
            postal[i] = 1
        q_empty_c, p_empty_c = not q_country, not pc
        country[i] = 2 if (q_empty_c or p_empty_c) else (1 if pc == q_country else 0)
        p_ntok[i] = len(p_tokens)
    parts += [fz, sk_jacc.reshape(-1, 1), fa, addr_empty.reshape(-1, 1), postal.reshape(-1, 1),
              country.reshape(-1, 1), ptr.src(pool_idx).astype(np.float32).reshape(-1, 1),
              ptr.indic(pool_idx).astype(np.float32).reshape(-1, 1),
              np.full((n, 1), len(q_tokens), dtype=np.float32), p_ntok.reshape(-1, 1)]
    X = np.concatenate(parts, axis=1).astype(np.float32)
    assert X.shape[1] == len(names), (X.shape[1], len(names))
    return X


# ---------------------------------------------------------------- rank / tie-break (plan §5.8 cut)

def rank_within_groups(qcode, p, learn, pool_idx):
    """1-based rank of each row within its group `qcode` (arbitrary int codes), by (p desc, learn
    desc, pool_idx asc) -- the cut's tie-break, and `retrieve.top_q`'s convention (plan §3.3) applied
    to the (score, index) pair generalized to (p, learn, pool_idx)."""
    qcode = np.asarray(qcode)
    p, learn, pool_idx = np.asarray(p), np.asarray(learn), np.asarray(pool_idx)
    n = qcode.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.lexsort((pool_idx, -learn, -p, qcode))
    sorted_q = qcode[order]
    new_grp = np.empty(n, dtype=bool)
    new_grp[0] = True
    new_grp[1:] = sorted_q[1:] != sorted_q[:-1]
    starts = np.flatnonzero(new_grp)
    grp_id = np.cumsum(new_grp) - 1
    grp_start_per_row = starts[grp_id]
    rank_sorted = (np.arange(n) - grp_start_per_row + 1).astype(np.int32)
    rank = np.empty(n, dtype=np.int32)
    rank[order] = rank_sorted
    return rank


def cut_select(rank, p, K_min, K_max, p_min):
    return (rank <= K_min) | ((p >= p_min) & (rank <= K_max))


# ---------------------------------------------------------------- negative sampling (plan §5.8 build)

def sample_negatives(y, score, s1_idx, top_neg, rand_neg, seed_base):
    """(row indices ascending, weights) for one query: all positives, top `top_neg` by `score`
    (weight 1), plus `rand_neg` uniform random from the rest (weight n_rest/n_sampled, capped at
    n_rest); rng seed = seed_base + s1_idx."""
    y = np.asarray(y, dtype=bool)
    keep = np.zeros(y.size, dtype=bool)
    keep[retrieve.top_q(score, top_neg)] = True
    keep |= y
    w = np.ones(y.size, dtype=np.float32)
    rest = np.flatnonzero(~keep)
    if rest.size:
        k = min(rand_neg, rest.size)
        samp = np.random.default_rng(seed_base + int(s1_idx)).choice(rest, k, replace=False)
        keep[samp] = True
        w[samp] = rest.size / k
    rows = np.flatnonzero(keep)
    return rows, w[rows]


# ---------------------------------------------------------------- spawn-pool worker functions

_BOOSTER_CACHE = {}


def _booster(path):
    if _BOOSTER_CACHE.get("path") != path:
        _BOOSTER_CACHE["booster"] = lgb.Booster(model_file=path)
        _BOOSTER_CACHE["path"] = path
    return _BOOSTER_CACHE["booster"]


def _build_row_chunk(r, idx, names, addrs, chains, per_query, shared):
    """rerank-build worker: retrieve + (sample or keep-all) + features, per query. per_query =
    [(truth pool_idx array, country_raw)]; shared = (split, sample) with sample = None (keep all,
    RR_CAL) or (top_neg, rand_neg, seed_base) (RR_TRAIN)."""
    split, sample = shared
    ptr = _pool_text_reader(split)
    out = []
    for s1i, name, addr, chain, (truth, country) in zip(idx, names, addrs, chains, per_query):
        res = r.retrieve(name, addr, int(chain))
        y = np.isin(res["pool_idx"], truth)
        if sample is None:
            rows = np.arange(y.size)
            w = np.ones(y.size, dtype=np.float32)
        else:
            top_neg, rand_neg, seed_base = sample
            rows, w = sample_negatives(y, res["learn"], s1i, top_neg, rand_neg, seed_base)
        X = compute_features(r, ptr, res, rows, int(chain), name, addr, country)
        # sel_true is per TRUE pair (length n_true: was this true pool_idx reached by the stage-1
        # selection?), matching the shape evaluate.segment_metrics expects (pair_pool/pair_q/pair_hit
        # are all one entry per true pair) -- not `y`, which is per selected candidate row.
        truth_arr = np.asarray(truth, dtype=np.uint32)
        out.append({"s1_idx": int(s1i), "pool_idx": res["pool_idx"][rows],
                    "label": y[rows].astype(np.int8), "weight": w.astype(np.float32), "X": X,
                    "n_true": int(truth.size), "n_hit": int(y.sum()), "size": int(y.size),
                    "truth": truth_arr, "sel_true": np.isin(truth_arr, res["pool_idx"])})
    return out


def _infer_chunk(r, idx, names, addrs, chains, per_query, shared):
    """rerank-eval / rerank-loco / `candidates` worker: retrieve + stage-2 score (given booster) + cut,
    ordered by (p desc, learn desc, pool_idx asc). per_query = [country_raw]; shared = (split,
    model_path, feature_names, K_min, K_max, p_min). Also returns n_union/n_dropped (stage-1 union size
    and postings-limit-drop count, plan §5.10 monitor.py "share of queries hitting the postings limit")
    and max_p_at_rank (p at ranks 1-5 before the cut, padded with nan) for monitor.py; unused elsewhere."""
    split, model_path, feat_names, K_min, K_max, p_min = shared
    ptr = _pool_text_reader(split)
    booster = _booster(model_path)
    out = []
    for s1i, name, addr, chain, country in zip(idx, names, addrs, chains, per_query):
        res = r.retrieve(name, addr, int(chain))
        n = res["pool_idx"].size
        rows = np.arange(n)
        X = compute_features(r, ptr, res, rows, int(chain), name, addr, country)
        p = booster.predict(X, num_threads=1) if n else np.zeros(0, dtype=np.float64)
        learn, pool_idx = res["learn"], res["pool_idx"]
        rank = rank_within_groups(np.zeros(n, dtype=np.int64), p, learn, pool_idx)
        pre_order = np.argsort(-p, kind="stable")[:5] if n else np.zeros(0, dtype=np.int64)
        top5_p = np.full(5, np.nan, dtype=np.float32)
        top5_p[:pre_order.size] = p[pre_order]
        sel = np.flatnonzero(cut_select(rank, p, K_min, K_max, p_min))
        order = sel[np.lexsort((pool_idx[sel], -learn[sel], -p[sel]))]
        out.append({"s1_idx": int(s1i), "pool_idx": pool_idx[order].astype(np.uint32),
                    "p": p[order].astype(np.float32), "rank": rank[order].astype(np.int16),
                    "country": country, "n_union": res["n_union"], "n_dropped": res["n_dropped"],
                    "top5_p": top5_p})
    return out


def generate_candidates(split, s1_idx, country_raw, cfg, model, feat_names, cut, out_dir,
                        collect_query_stats=False):
    """Stage-1+2 inference for a query set (plan M6+M7): reuses `_infer_chunk`/`write_candidates` so
    `run_rerank_eval` and `ber.run_candidates.run_candidates` share one code path. Returns
    (pool_lists, rank_lists, p_lists, query_stats_or_None); query_stats has one row per query
    (s1_idx, country, n_union, n_dropped, size, max_p, p_rank1..5) for `monitor.py` (only collected
    when asked, since it costs an extra pass' worth of small per-query bookkeeping)."""
    model_path = str(io_utils.work_path("rerank", "model_v1.txt"))
    shared = (split, model_path, feat_names, cut["K_min"], cut["K_max"], cut["p_min"])
    pool_lists, rank_lists, p_lists = [], [], []
    stats = [] if collect_query_stats else None
    with retrieve.QueryRunner(split, cfg) as qr:
        for chunk in qr.map(cfg, model, s1_idx, _infer_chunk, per_query=country_raw, shared=shared):
            for res in chunk:
                pool_lists.append(res["pool_idx"])
                rank_lists.append(res["rank"])
                p_lists.append(res["p"])
                if collect_query_stats:
                    stats.append((res["s1_idx"], res["country"], res["n_union"], res["n_dropped"],
                                 res["pool_idx"].size,
                                 float(res["top5_p"][0]) if res["pool_idx"].size else float("nan"),
                                 *[float(x) for x in res["top5_p"]]))
    if stats is not None:
        stats.sort(key=lambda t: t[0])
        stats = pa.table({
            "s1_idx": pa.array([t[0] for t in stats], pa.int32()),
            "country_raw": [t[1] for t in stats],
            "n_union": pa.array([t[2] for t in stats], pa.int32()),
            "n_dropped": pa.array([t[3] for t in stats], pa.int32()),
            "size": pa.array([t[4] for t in stats], pa.int32()),
            "max_p": pa.array([t[5] for t in stats], pa.float32()),
            "p_rank1": pa.array([t[6] for t in stats], pa.float32()),
            "p_rank2": pa.array([t[7] for t in stats], pa.float32()),
            "p_rank3": pa.array([t[8] for t in stats], pa.float32()),
            "p_rank4": pa.array([t[9] for t in stats], pa.float32()),
            "p_rank5": pa.array([t[10] for t in stats], pa.float32()),
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / "query_stats.parquet"
        tmp = stats_path.with_name(stats_path.name + ".tmp")
        pq.write_table(stats, tmp, compression="zstd")
        os.replace(tmp, stats_path)
    write_candidates(out_dir.name, s1_idx, pool_lists, rank_lists, p_lists, out_dir)
    return pool_lists, rank_lists, p_lists, stats


# ---------------------------------------------------------------- rerank-build (plan §5.8 B)

def _rows_schema(feat_names):
    fields = [("s1_idx", pa.int32()), ("pool_idx", pa.uint32()), ("label", pa.int8()),
              ("weight", pa.float32())] + [(f, pa.float32()) for f in feat_names]
    return pa.schema(fields)


def _write_rows(path, feat_names, row_chunks):
    """Streams QueryRunner chunks straight to parquet (one RecordBatch per chunk); returns (row/query
    counts, the light per-query stage-1 stats needed for rr_cal_stage1.json)."""
    schema = _rows_schema(feat_names)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    n_rows = n_pos = n_queries = 0
    stage1_rows = []
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for chunk in row_chunks:
            if not chunk:
                continue
            s1c = np.concatenate([np.full(c["pool_idx"].size, c["s1_idx"], dtype=np.int32)
                                  for c in chunk])
            pool = np.concatenate([c["pool_idx"] for c in chunk]).astype(np.uint32)
            label = np.concatenate([c["label"] for c in chunk])
            weight = np.concatenate([c["weight"] for c in chunk])
            X = (np.concatenate([c["X"] for c in chunk], axis=0) if chunk else
                 np.zeros((0, len(feat_names)), dtype=np.float32))
            cols = [pa.array(s1c), pa.array(pool), pa.array(label), pa.array(weight)]
            cols += [pa.array(X[:, j]) for j in range(len(feat_names))]
            w.write_batch(pa.record_batch(cols, schema=schema))
            n_rows += int(pool.size)
            n_pos += int(label.sum())
            n_queries += len(chunk)
            stage1_rows.extend(chunk)
    os.replace(tmp, path)
    return ({"n_rows": n_rows, "n_positive_rows": n_pos, "n_negative_rows": n_rows - n_pos,
             "n_queries": n_queries}, stage1_rows)


def _stage1_metrics_from_rows(stage1_rows, n_pool):
    """§5.9 full metric set (incl. segments) of the RR_CAL stage-1 selection, from the per-query stats
    `_write_rows` collected while writing rr_cal.parquet (no second retrieval pass)."""
    s1_idx = np.array([r["s1_idx"] for r in stage1_rows], dtype=np.int64)
    n_true = np.array([r["n_true"] for r in stage1_rows], dtype=np.int64)
    n_hit = np.array([r["n_hit"] for r in stage1_rows], dtype=np.int64)
    sizes = np.array([r["size"] for r in stage1_rows], dtype=np.int64)
    pair_pool = (np.concatenate([r["truth"] for r in stage1_rows]) if stage1_rows
                else np.zeros(0, dtype=np.uint32))
    pair_q = np.repeat(np.arange(s1_idx.size), n_true)
    pair_hit = (np.concatenate([r["sel_true"] for r in stage1_rows]) if stage1_rows
               else np.zeros(0, dtype=bool))
    seg = evaluate.segment_data(s1_idx, pair_pool)
    summ = evaluate.summarize(n_true, n_hit, sizes, n_pool)
    summ["segments"] = evaluate.segment_metrics(n_true, n_hit, sizes, n_pool, seg, pair_q, pair_hit)
    return summ


def run_rerank_build(args):
    """CLI `rerank-build`: RR_TRAIN (sampled) -> rr_train.parquet; RR_CAL (all selected) ->
    rr_cal.parquet + rr_cal_stage1.json (plan §5.8)."""
    t0 = time.time()
    lg = io_utils.get_logger("rerank-build")
    cfg = io_utils.load_config()
    model = retrieve.load_stage1_model(cfg)
    if model is None:
        raise RuntimeError("rerank-build requires channels.LEARN == stage1_model; run stage1-train first")
    p = cfg["rerank"]["sample"]
    sets = splits.load_query_sets()
    n_pool = int(io_utils.read_json(io_utils.work_path("index", "train", "meta.json"))["n_pool"])
    country_raw = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                                  columns=["country_raw"])["country_raw"].to_numpy()

    with retrieve.QueryRunner("train", cfg) as qr:
        feat_names = feature_names(retrieve.Retriever("train", cfg, model))

        rr_train_s1 = sets["RR_TRAIN"]
        truth_train = evaluate.truth_lists(rr_train_s1)
        pq_train = list(zip(truth_train, country_raw[rr_train_s1]))
        chunks = qr.map(cfg, model, rr_train_s1, _build_row_chunk, per_query=pq_train,
                        shared=("train", (p["top_neg"], p["rand_neg"], SEED_BASE_BUILD)))
        train_stats, _ = _write_rows(io_utils.work_path("rerank", "rr_train.parquet"), feat_names, chunks)
        lg.info("rr_train written: %s", train_stats)

        rr_cal_s1 = sets["RR_CAL"]
        truth_cal = evaluate.truth_lists(rr_cal_s1)
        pq_cal = list(zip(truth_cal, country_raw[rr_cal_s1]))
        chunks = qr.map(cfg, model, rr_cal_s1, _build_row_chunk, per_query=pq_cal, shared=("train", None))
        cal_stats, cal_rows = _write_rows(io_utils.work_path("rerank", "rr_cal.parquet"), feat_names, chunks)
        lg.info("rr_cal written: %s", cal_stats)

    stage1_metrics = _stage1_metrics_from_rows(cal_rows, n_pool)
    io_utils.write_metrics("rr_cal_stage1", stage1_metrics, rr_cal_s1.size,
                           int(sum(t.size for t in truth_cal)), cfg)

    payload = {"feature_names": feat_names, "n_features": len(feat_names),
               "rr_train": train_stats, "rr_cal": cal_stats,
               "wall_s": round(time.time() - t0, 1), "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics("rerank_build", payload, rr_train_s1.size + rr_cal_s1.size,
                           int(sum(t.size for t in truth_train) + sum(t.size for t in truth_cal)), cfg)
    lg.info("rerank-build done in %.1fs: %s", time.time() - t0,
            {k: payload[k] for k in ("rr_train", "rr_cal", "n_features")})


# ---------------------------------------------------------------- rerank-train (plan §5.8 C)

def _train_lgb(X, y, w, groups, feat_names, lgb_cfg, num_threads):
    """GroupShuffleSplit holdout + LightGBM binary classifier with early stopping (plan §5.8)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=lgb_cfg["holdout_frac"], random_state=42)
    tr_idx, ho_idx = next(gss.split(X, y, groups))
    params = {k: v for k, v in lgb_cfg.items() if k not in NOT_LGB_PARAMS}
    params["num_threads"] = num_threads
    params.setdefault("verbosity", -1)
    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=w[tr_idx], feature_name=feat_names,
                         free_raw_data=False)
    dvalid = lgb.Dataset(X[ho_idx], label=y[ho_idx], weight=w[ho_idx], feature_name=feat_names,
                         reference=dtrain, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=lgb_cfg["num_boost_round"],
                        valid_sets=[dvalid], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=False),
                                  lgb.log_evaluation(period=0)])
    return booster, tr_idx, ho_idx


def run_rerank_train(args):
    """CLI `rerank-train`: LightGBM on rr_train.parquet -> model_v1.txt + model_v1_meta.json."""
    t0 = time.time()
    lg = io_utils.get_logger("rerank-train")
    cfg = io_utils.load_config()
    model = retrieve.load_stage1_model(cfg)
    if model is None:
        raise RuntimeError("rerank-train requires channels.LEARN == stage1_model")
    feat_names = feature_names(retrieve.Retriever("train", cfg, model))
    table = pq.read_table(io_utils.work_path("rerank", "rr_train.parquet"))
    X = np.column_stack([table.column(f).to_numpy() for f in feat_names]).astype(np.float32)
    y = table.column("label").to_numpy()
    w = table.column("weight").to_numpy()
    groups = table.column("s1_idx").to_numpy()

    lgb_cfg = cfg["rerank"]["lgb"]
    booster, tr_idx, ho_idx = _train_lgb(X, y, w, groups, feat_names, lgb_cfg, io_utils.n_workers(cfg))

    model_path = io_utils.work_path("rerank", "model_v1.txt")
    tmp = model_path.with_name(model_path.name + ".tmp")
    booster.save_model(str(tmp), num_iteration=booster.best_iteration)
    os.replace(tmp, model_path)

    meta = {"features": feat_names, "config_hash": io_utils.config_hash(cfg),
            "stage1_config_hash": io_utils.config_hash(retrieve.stage1_config(cfg)),
            "stage1_model_hash": io_utils.config_hash(model),
            "best_iteration": int(booster.best_iteration),
            "holdout_logloss": float(booster.best_score["holdout"]["binary_logloss"])}
    io_utils.atomic_write_json(io_utils.work_path("rerank", "model_v1_meta.json"), meta)

    payload = {"n_rows": int(y.size), "n_train_rows": int(tr_idx.size),
               "n_holdout_rows": int(ho_idx.size),
               "n_train_queries": int(np.unique(groups[tr_idx]).size),
               "n_holdout_queries": int(np.unique(groups[ho_idx]).size),
               "best_iteration": meta["best_iteration"], "holdout_logloss": meta["holdout_logloss"],
               "wall_s": round(time.time() - t0, 1), "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics("rerank_train", payload, int(np.unique(groups).size), int(y.sum()), cfg)
    lg.info("rerank-train done: %s", payload)


# ---------------------------------------------------------------- rerank-cut (plan §5.8 D)

def _grid_search_cut(qcode, nq, n_true, label, p, learn, pool_idx, cut_grid, n_pool, budget,
                     stage1_recall, stage1_f05):
    """Every (K_min, K_max, p_min) in `cut_grid`, evaluated with evaluate.summarize; the plan's
    decision rule; returns (all results, chosen, fallback_used)."""
    rank = rank_within_groups(qcode, p, learn, pool_idx)
    label1 = label == 1
    results = []
    for K_min in cut_grid["K_min"]:
        for K_max in cut_grid["K_max"]:
            for p_min in cut_grid["p_min"]:
                selected = cut_select(rank, p, K_min, K_max, p_min)
                hit = np.bincount(qcode, weights=(selected & label1).astype(np.float64), minlength=nq)
                size = np.bincount(qcode, weights=selected.astype(np.float64), minlength=nq)
                summ = evaluate.summarize(n_true, hit.astype(np.int64), size.astype(np.int64), n_pool)
                results.append({"K_min": int(K_min), "K_max": int(K_max), "p_min": float(p_min), **summ})
    ok = [r for r in results if r["sizes"]["mean"] is not None and r["sizes"]["mean"] <= budget
          and r["entity_recall"] is not None and r["entity_recall"] >= stage1_recall - 0.002
          and r["f05_ceiling"] is not None and r["f05_ceiling"] >= stage1_f05 - 0.002]
    fallback = False
    if ok:
        chosen = min(ok, key=lambda r: (r["sizes"]["mean"], -r["f05_ceiling"]))
    else:
        fallback = True
        under = [r for r in results if r["sizes"]["mean"] is not None and r["sizes"]["mean"] <= budget]
        pool_ = under or results
        chosen = max(pool_, key=lambda r: (r["f05_ceiling"] if r["f05_ceiling"] is not None else -1.0))
    return results, chosen, fallback


def run_rerank_cut(args):
    """CLI `rerank-cut`: score rr_cal.parquet, grid-search the cut, apply the decision (plan §5.8)."""
    t0 = time.time()
    lg = io_utils.get_logger("rerank-cut")
    cfg = io_utils.load_config()
    meta = io_utils.read_json(io_utils.work_path("rerank", "model_v1_meta.json"))
    feat_names = meta["features"]
    n_pool = int(io_utils.read_json(io_utils.work_path("index", "train", "meta.json"))["n_pool"])
    booster = lgb.Booster(model_file=str(io_utils.work_path("rerank", "model_v1.txt")))
    table = pq.read_table(io_utils.work_path("rerank", "rr_cal.parquet"))
    s1_idx_rows = table.column("s1_idx").to_numpy()
    pool_idx_rows = table.column("pool_idx").to_numpy()
    label_rows = table.column("label").to_numpy()
    learn_rows = table.column("learn").to_numpy()
    X = np.column_stack([table.column(f).to_numpy() for f in feat_names]).astype(np.float32)
    p_rows = booster.predict(X, num_threads=1) if X.shape[0] else np.zeros(0)

    rr_cal_s1 = splits.load_query_sets()["RR_CAL"]
    qcode = np.searchsorted(rr_cal_s1, s1_idx_rows)
    n_true = np.array([t.size for t in evaluate.truth_lists(rr_cal_s1)], dtype=np.int64)
    stage1_doc = io_utils.read_json(_metrics_path("rr_cal_stage1"))

    results, chosen, fallback = _grid_search_cut(
        qcode, rr_cal_s1.size, n_true, label_rows, p_rows, learn_rows, pool_idx_rows,
        cfg["rerank"]["cut_grid"], n_pool, cfg["rerank"]["mean_size_budget"],
        stage1_doc["entity_recall"], stage1_doc["f05_ceiling"])

    doc = {"grid": results, "chosen": chosen, "fallback_used": fallback,
           "stage1_baseline": {"entity_recall": stage1_doc["entity_recall"],
                               "f05_ceiling": stage1_doc["f05_ceiling"]}}
    io_utils.atomic_write_json(io_utils.work_path("rerank", "cut_v1.json"), doc)
    path = io_utils.write_metrics("rerank_cut", doc, rr_cal_s1.size, int(n_true.sum()), cfg)
    io_utils.apply_tuned({"rerank.cut.K_min": chosen["K_min"], "rerank.cut.K_max": chosen["K_max"],
                          "rerank.cut.p_min": chosen["p_min"]}, "rerank-cut", path)
    lg.info("rerank-cut done in %.1fs: chosen=%s fallback_used=%s", time.time() - t0, chosen, fallback)


# ---------------------------------------------------------------- rerank-loco (plan §5.8 E)

def _country_subset(table, country, s1_country_raw):
    s1c = table.column("s1_idx").to_numpy()
    keep = s1_country_raw[s1c] == country
    return table.filter(pa.array(keep))


def run_rerank_loco(args):
    """CLI `rerank-loco`: stage-2 trained + cut India-only, evaluated on DEV_B US-only, and the
    reverse; drop vs the all-country model+cut on the same slices (plan §5.8 LOCO)."""
    t0 = time.time()
    lg = io_utils.get_logger("rerank-loco")
    cfg = io_utils.load_config()
    meta = io_utils.read_json(io_utils.work_path("rerank", "model_v1_meta.json"))
    feat_names = meta["features"]
    n_pool = int(io_utils.read_json(io_utils.work_path("index", "train", "meta.json"))["n_pool"])
    country_raw = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                                  columns=["country_raw"])["country_raw"].to_numpy()
    sets = splits.load_query_sets()
    model = retrieve.load_stage1_model(cfg)
    cut_all = cfg["rerank"]["cut"]
    all_model_path = str(io_utils.work_path("rerank", "model_v1.txt"))
    stage1_doc = io_utils.read_json(_metrics_path("rr_cal_stage1"))

    train_tab = pq.read_table(io_utils.work_path("rerank", "rr_train.parquet"))
    cal_tab = pq.read_table(io_utils.work_path("rerank", "rr_cal.parquet"))

    def train_and_cut(country, tag):
        tr = _country_subset(train_tab, country, country_raw)
        X = np.column_stack([tr.column(f).to_numpy() for f in feat_names]).astype(np.float32)
        y = tr.column("label").to_numpy()
        w = tr.column("weight").to_numpy()
        groups = tr.column("s1_idx").to_numpy()
        booster, _, _ = _train_lgb(X, y, w, groups, feat_names, cfg["rerank"]["lgb"],
                                   io_utils.n_workers(cfg))
        model_dir = io_utils.work_path("rerank", tag, "model.txt").parent
        model_path = model_dir / "model.txt"
        booster.save_model(str(model_path), num_iteration=booster.best_iteration)

        cal = _country_subset(cal_tab, country, country_raw)
        cal_s1 = np.array(sorted(set(cal.column("s1_idx").to_numpy().tolist())), dtype=np.int32)
        Xc = np.column_stack([cal.column(f).to_numpy() for f in feat_names]).astype(np.float32)
        p = booster.predict(Xc, num_threads=1) if Xc.shape[0] else np.zeros(0)
        label, learn = cal.column("label").to_numpy(), cal.column("learn").to_numpy()
        pool_idx, s1_rows = cal.column("pool_idx").to_numpy(), cal.column("s1_idx").to_numpy()
        qcode = np.searchsorted(cal_s1, s1_rows)
        n_true = np.array([t.size for t in evaluate.truth_lists(cal_s1)], dtype=np.int64)
        seg = stage1_doc["segments"].get(f"country_{country}", {})
        base_er, base_f05 = seg.get("entity_recall") or 0.0, seg.get("f05_ceiling") or 0.0
        results, chosen, fallback = _grid_search_cut(
            qcode, cal_s1.size, n_true, label, p, learn, pool_idx, cfg["rerank"]["cut_grid"], n_pool,
            cfg["rerank"]["mean_size_budget"], base_er, base_f05)
        cut_doc = {"K_min": chosen["K_min"], "K_max": chosen["K_max"], "p_min": chosen["p_min"],
                   "fallback_used": fallback}
        io_utils.atomic_write_json(model_dir / "cut.json", dict(cut_doc, grid=results))
        return str(model_path), cut_doc

    def eval_on(model_path, cut, eval_s1):
        truth = evaluate.truth_lists(eval_s1)
        n_true = np.array([t.size for t in truth], dtype=np.int64)
        shared = ("train", model_path, feat_names, cut["K_min"], cut["K_max"], cut["p_min"])
        pool_lists = []
        with retrieve.QueryRunner("train", cfg) as qr:
            for chunk in qr.map(cfg, model, eval_s1, _infer_chunk,
                                per_query=country_raw[eval_s1], shared=shared):
                pool_lists.extend(res["pool_idx"] for res in chunk)
        n_hit = np.array([np.isin(t, pool_lists[i]).sum() for i, t in enumerate(truth)], dtype=np.int64)
        sizes = np.array([len(x) for x in pool_lists], dtype=np.int64)
        return evaluate.summarize(n_true, n_hit, sizes, n_pool)

    dev_b = sets["DEV_B"]
    us_idx = dev_b[country_raw[dev_b] == "US"]
    india_idx = dev_b[country_raw[dev_b] == "India"]

    india_model, india_cut = train_and_cut("India", "loco_india_us")
    us_model, us_cut = train_and_cut("US", "loco_us_india")

    result = {}
    for tag, model_path, cut, eval_idx in (
            ("india_train_us_eval", india_model, india_cut, us_idx),
            ("us_train_india_eval", us_model, us_cut, india_idx)):
        loco_metrics = eval_on(model_path, cut, eval_idx)
        all_metrics = eval_on(all_model_path, cut_all, eval_idx)
        drop = (None if loco_metrics["f05_ceiling"] is None or all_metrics["f05_ceiling"] is None else
                all_metrics["f05_ceiling"] - loco_metrics["f05_ceiling"])
        result[tag] = {"loco": loco_metrics, "all_country": all_metrics, "f05_ceiling_drop": drop}

    payload = {**result, "wall_s": round(time.time() - t0, 1), "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics("rerank_loco", payload, int(dev_b.size), 0, cfg)
    lg.info("rerank-loco done: %s", {k: result[k]["f05_ceiling_drop"] for k in result})


# ---------------------------------------------------------------- candidates writer (plan §5.10/§4)

def write_candidates(split, s1_idx_array, pool_idx_lists, rank_lists, p_lists, out_dir, part_size=200_000):
    """Writes `candidates/{split}/part-XXXXX.parquet` + `part-XXXXX._SUCCESS` (plan §4/§5.10
    convention), atomically, chunked by `part_size` queries so the whole set is never held in memory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(s1_idx_array)
    part = 0
    for start in range(0, n, part_size):
        stop = min(start + part_size, n)
        s1c, poolc, rankc, pc = [], [], [], []
        for i in range(start, stop):
            m = len(pool_idx_lists[i])
            if m == 0:
                continue
            s1c.append(np.full(m, s1_idx_array[i], dtype=np.int32))
            poolc.append(np.asarray(pool_idx_lists[i], dtype=np.uint32))
            rankc.append(np.asarray(rank_lists[i], dtype=np.int16))
            pc.append(np.asarray(p_lists[i], dtype=np.float32))
        table = pa.table({
            "s1_idx": np.concatenate(s1c) if s1c else np.zeros(0, dtype=np.int32),
            "pool_idx": np.concatenate(poolc) if poolc else np.zeros(0, dtype=np.uint32),
            "rank": np.concatenate(rankc) if rankc else np.zeros(0, dtype=np.int16),
            "p_rerank": np.concatenate(pc) if pc else np.zeros(0, dtype=np.float32),
        }, schema=CANDIDATE_SCHEMA)
        name = f"part-{part:05d}.parquet"
        path = out_dir / name
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
        success = path.with_suffix("._SUCCESS")
        success_tmp = success.with_name(success.name + ".tmp")
        open(success_tmp, "w", encoding="utf-8").close()
        os.replace(success_tmp, success)
        part += 1


def _candidates_ready(out_dir):
    parts = sorted(Path(out_dir).glob("part-*.parquet"))
    return bool(parts) and all(p.with_suffix("._SUCCESS").exists() for p in parts)


def _read_candidates(out_dir, s1_idx_array):
    parts = sorted(Path(out_dir).glob("part-*.parquet"))
    if parts:
        table = pa.concat_tables([pq.read_table(p) for p in parts])
    else:
        table = CANDIDATE_SCHEMA.empty_table()
    s1c = table.column("s1_idx").to_numpy()
    pool = table.column("pool_idx").to_numpy()
    rank = table.column("rank").to_numpy()
    p = table.column("p_rerank").to_numpy()
    order = np.lexsort((rank, s1c))
    s1c, pool, rank, p = s1c[order], pool[order], rank[order], p[order]
    starts = np.searchsorted(s1c, s1_idx_array, side="left")
    stops = np.searchsorted(s1c, s1_idx_array, side="right")
    pool_lists = [pool[a:b] for a, b in zip(starts, stops)]
    rank_lists = [rank[a:b] for a, b in zip(starts, stops)]
    p_lists = [p[a:b] for a, b in zip(starts, stops)]
    return pool_lists, rank_lists, p_lists


# ---------------------------------------------------------------- rerank-eval (plan §5.8 F / §7)

def run_rerank_eval(args):
    """CLI `rerank-eval --set {DEV_B, HOLDOUT_TEST|val-S1}`: staleness guard, run stage-1+2 for the
    set if `work/candidates/{SET}/` has no finished parts, write full §5.9 metrics on the stage-2 cut
    candidates (plan §5.8 F)."""
    t0 = time.time()
    set_name = "HOLDOUT_TEST" if args.set == "val-S1" else args.set
    if set_name not in ("DEV_B", "HOLDOUT_TEST"):
        raise ValueError("rerank-eval only supports --set DEV_B or HOLDOUT_TEST/val-S1")
    lg = io_utils.get_logger(f"rerank-eval_{set_name}")
    cfg = io_utils.load_config()
    model = retrieve.load_stage1_model(cfg)
    if model is None:
        raise RuntimeError("rerank-eval requires channels.LEARN == stage1_model")
    meta = io_utils.read_json(io_utils.work_path("rerank", "model_v1_meta.json"))
    cur_stage1_hash = io_utils.config_hash(retrieve.stage1_config(cfg))
    cur_model_hash = io_utils.config_hash(model)
    if meta["stage1_config_hash"] != cur_stage1_hash or meta["stage1_model_hash"] != cur_model_hash:
        raise RuntimeError(
            "work/rerank/model_v1_meta.json is stale: the stage-1 config or model changed since "
            "rerank-train (plan §5.8 staleness guard) -- re-run rerank-build/rerank-train/rerank-cut")
    feat_names = meta["features"]
    n_pool = int(io_utils.read_json(io_utils.work_path("index", "train", "meta.json"))["n_pool"])
    cut = io_utils.load_config()["rerank"]["cut"]
    sets = splits.load_query_sets()
    s1_idx = sets[set_name]
    country_raw = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                                  columns=["country_raw"])["country_raw"].to_numpy()

    out_dir = io_utils.work_path("candidates", set_name)
    reused = _candidates_ready(out_dir)
    if reused:
        pool_lists, rank_lists, p_lists = _read_candidates(out_dir, s1_idx)
    else:
        pool_lists, rank_lists, p_lists, _ = generate_candidates(
            "train", s1_idx, country_raw[s1_idx], cfg, model, feat_names, cut, out_dir)

    truth = evaluate.truth_lists(s1_idx)
    n_true = np.array([t.size for t in truth], dtype=np.int64)
    n_hit = np.array([np.isin(t, pool_lists[i]).sum() for i, t in enumerate(truth)], dtype=np.int64)
    sizes = np.array([len(pool_lists[i]) for i in range(s1_idx.size)], dtype=np.int64)
    summ = evaluate.summarize(n_true, n_hit, sizes, n_pool)
    pair_pool = np.concatenate(truth) if truth else np.zeros(0, dtype=np.uint32)
    pair_q = np.repeat(np.arange(s1_idx.size), n_true)
    pair_hit = (np.concatenate([np.isin(t, pool_lists[i]) for i, t in enumerate(truth)])
               if truth else np.zeros(0, dtype=bool))
    seg = evaluate.segment_data(s1_idx, pair_pool)
    summ["segments"] = evaluate.segment_metrics(n_true, n_hit, sizes, n_pool, seg, pair_q, pair_hit)

    payload = {"set": set_name, **summ, "reused_candidates": reused,
               "wall_s": round(time.time() - t0, 1), "peak_rss_mb": round(io_utils.peak_rss_mb(), 1)}
    io_utils.write_metrics(f"rerank_eval_{set_name}", payload, int(s1_idx.size), int(n_true.sum()), cfg)
    if set_name == "HOLDOUT_TEST":
        lg.info("rerank-eval HOLDOUT_TEST: used val-S1 labels for scoring only, once (plan R4/§6)")
    lg.info("rerank-eval %s done in %.1fs: f05_ceiling=%s", set_name, time.time() - t0,
            summ.get("f05_ceiling"))

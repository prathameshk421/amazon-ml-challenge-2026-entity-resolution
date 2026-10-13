"""M6 stage-1 retrieval (plan §5.7).

Per query: keys per family with df caps and the postings limit -> union with per-family scores S[F] ->
stage-1 features -> LEARN + hand channel scores -> chain-aware quotas -> selected union ordered by LEARN.
Evaluation mode ranks every union item in every channel (plus the hand_both scorer) so many quota
configs are evaluated at once. QueryRunner is the spawn-pool driver over a query set (§3.3) that
s1-eval, stage1-train, stage 2 and `candidates` share.
"""

import math
import time
from collections import namedtuple

import numpy as np
import pyarrow.parquet as pq

from ber import hashing, io_utils, keys
from ber.index_store import IndexStore

FAMILY_ORDER = keys.FAMILIES + ("SK",)  # SK only when E2 keeps it
COV_FAMILIES = ("NT", "NG", "NP", "AT", "XK")
CHANNELS = ("LEARN", "NAME", "ADDR", "PAIR", "XK", "CAT")
EVAL_COLUMNS = CHANNELS + ("HAND",)  # HAND = hand_both over the whole union (E1, E3b top-K)
STAGE2_RANKS = ("LEARN", "NAME", "ADDR")  # stage-2 features rk_LEARN, rk_NAME, rk_ADDR (§5.8)
RANK_CAP = 10_000
RANK_INF = np.iinfo(np.int32).max
CHUNK = 256

Union = namedtuple("Union", "u S Q d_en n_dropped")


def kept_families(cfg):
    return [F for F in FAMILY_ORDER if F in cfg["families"]]


def feature_names(families):
    """§5.7 step 4 order: S[F] for the kept families, cov for NT NG NP AT XK among them, log1p(S[F])."""
    return ([f"s_{F}" for F in families] + [f"cov_{F}" for F in COV_FAMILIES if F in families]
            + [f"log1p_s_{F}" for F in families])


def top_q(score, q):
    """Indices of the top q by (score desc, index asc), in that order (§3.3). Index order must be
    pool_idx order (true for union arrays, which come from np.unique)."""
    score = np.asarray(score)
    n = score.size
    if q <= 0:
        return np.zeros(0, dtype=np.intp)
    if n <= q:
        sel = np.arange(n)
    else:
        v = np.partition(score, n - q)[n - q]
        gt = np.flatnonzero(score > v)
        sel = np.concatenate([gt, np.flatnonzero(score == v)[:q - gt.size]])
    return sel[np.lexsort((sel, -score[sel]))]


def _order(score, eligible, L):
    """Top L eligible items (eligible None = all) in (score desc, index asc) order."""
    if eligible is None:
        return top_q(score, L)
    e = np.flatnonzero(eligible)
    return e[top_q(score[e], L)]


# ---------------------------------------------------------------- stage-1 model file

def model_path():
    return io_utils.work_path("stage1", "model_v1.json")


def stage1_config(cfg):
    """The config parts that define the stage-1 features (staleness key of model_v1.json)."""
    fams = kept_families(cfg)
    return {"families": fams, "cap_frac": {F: cfg["cap_frac"][F] for F in fams},
            "max_postings_per_query": cfg["max_postings_per_query"]}


def load_stage1_model(cfg):
    """model_v1.json when channels.LEARN uses it, else None; refuses a missing or stale model."""
    if cfg["channels"]["LEARN"] != "stage1_model":
        return None
    p = model_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m ber.cli stage1-train` first")
    doc = io_utils.read_json(p)
    if doc["stage1_config"] != stage1_config(cfg):
        raise RuntimeError(f"{p} was trained with other families/caps: re-run stage1-train")
    return doc


# ---------------------------------------------------------------- retriever

class Retriever:
    """Stage-1 retrieval over index/{split}/ (mmap; construct inside each worker)."""

    def __init__(self, split, cfg, model=None):
        self.families = kept_families(cfg)
        self.store = IndexStore(split, self.families)
        self.n_pool = self.store.n_pool
        self.caps = np.array([math.floor(cfg["cap_frac"][F] * self.n_pool) for F in self.families],
                             dtype=np.int64)
        self.is_en = np.array([F == "EN" for F in self.families])
        self.bits = np.array([1 << FAMILY_ORDER.index(F) for F in self.families], dtype=np.int64)
        self.max_post = cfg["max_postings_per_query"]
        self.feature_names = feature_names(self.families)
        self.cov_rows = [self.families.index(F) for F in COV_FAMILIES if F in self.families]
        self.hand = self._weights(cfg["hand_both"])
        self.channel_w = {C: self._weights(cfg["channels"][C]) for C in CHANNELS[1:]}
        self.model = model if model is not None and cfg["channels"]["LEARN"] != "hand_both" else None
        if self.model is not None:
            if self.model["features"] != self.feature_names:
                raise RuntimeError("stage-1 model feature order differs from the current families")
            mean, std = np.asarray(self.model["mean"]), np.asarray(self.model["std"])
            coef = np.asarray(self.model["coef"])
            self.w_eff = coef / std
            self.b_eff = float(self.model["intercept"]) - float(mean / std @ coef)
        self.quota = {C: int(cfg["quota"][C]) for C in CHANNELS}
        cr = cfg["chain_rule"]
        self.chain_on, self.min_chain, self.mult = cr["enabled"], cr["min_chain"], cr["mult"]
        self.chain_channels = frozenset(cr["channels"])

    def _weights(self, wdict):
        """Weight vector over the kept families; None if the channel has no kept family (§3.5)."""
        w = np.array([float(wdict.get(F, 0.0)) for F in self.families])
        return w if any(F in wdict for F in self.families) else None

    def quotas(self, chain):
        """Per-channel quota for a query with S1 chain size `chain` (§5.7 step 5)."""
        m = self.mult if self.chain_on and chain >= self.min_chain else 1
        return {C: 0 if C != "LEARN" and self.channel_w[C] is None
                else self.quota[C] * (m if C in self.chain_channels else 1) for C in CHANNELS}

    def union(self, name, addr):
        """Steps 1-3: used keys, postings limit, union u (ascending pool_idx) and S[F] (float64)."""
        kd = keys.record_keys(name, addr, self.families, self.store.G, self.store.at_df_lookup,
                              self.store.at_cap, "query")
        k64 = [hashing.keys64(F, kd[F]) for F in self.families]
        fam = np.concatenate([np.full(k.size, j, dtype=np.int64) for j, k in enumerate(k64)])
        df = np.concatenate([self.store.df(F, k) for F, k in zip(self.families, k64)])
        k64 = np.concatenate(k64)
        en = np.flatnonzero(self.is_en[fam])
        d_en = int(df[en[0]]) if en.size else 0
        used = df <= self.caps[fam]
        n_dropped = 0
        total = int(df[used].sum())
        if total > self.max_post:
            iu = np.flatnonzero(used)
            order = iu[np.lexsort((k64[iu], fam[iu], -df[iu]))]
            n_dropped = int(np.argmax(total - np.cumsum(df[order]) <= self.max_post)) + 1
            used[order[:n_dropped]] = False
        w = np.where(self.is_en[fam], 1.0, np.log((self.n_pool + 1) / (df + 1.0)))
        nF = len(self.families)
        Q = np.bincount(fam[used], weights=w[used], minlength=nF)
        hit = used & (df > 0)
        idx, posts = [], []
        for j, F in enumerate(self.families):
            sel = np.flatnonzero(hit & (fam == j))
            idx.append(sel)
            posts.extend(self.store.postings(F, k64[sel]))
        idx = np.concatenate(idx)
        if idx.size == 0:
            return Union(np.zeros(0, dtype=np.uint32), np.zeros((nF, 0)), Q, d_en, n_dropped)
        P = np.concatenate(posts)
        u, inv = np.unique(P, return_inverse=True)
        cnt = df[idx]
        S = np.bincount(np.repeat(fam[idx], cnt) * u.size + inv, weights=np.repeat(w[idx], cnt),
                        minlength=nF * u.size).reshape(nF, u.size)
        return Union(u.astype(np.uint32), S, Q, d_en, n_dropped)

    def features(self, un):
        """Stage-1 feature matrix [|U|, n_features] float64 (step 4)."""
        cov = [un.S[r] / un.Q[r] if un.Q[r] > 0 else np.zeros(un.u.size) for r in self.cov_rows]
        return np.vstack([un.S, *cov, np.log1p(un.S)]).T

    def hand_score(self, un):
        return self.hand @ un.S if self.hand is not None else np.zeros(un.u.size)

    def scores(self, un, X):
        """{column: (score, eligible mask or None = all)} for EVAL_COLUMNS; None if the channel has
        no kept family. LEARN = model score, or hand_both without a model / with LEARN = hand_both."""
        hand = self.hand_score(un)
        learn = X @ self.w_eff + self.b_eff if self.model is not None else hand
        out = {"LEARN": (learn, None), "HAND": (hand, None)}
        for C, w in self.channel_w.items():
            out[C] = None
            if w is not None:
                s = w @ un.S
                out[C] = (s, s > 0)
        return out

    def retrieve(self, name, addr, chain):
        """Selected candidates of one query, ordered by LEARN desc, pool_idx asc (steps 1-6)."""
        t0 = time.perf_counter()
        un = self.union(name, addr)
        X = self.features(un)
        cols = self.scores(un, X)
        q = self.quotas(chain)
        rk = np.full((un.u.size, len(STAGE2_RANKS)), RANK_CAP, dtype=np.int32)
        picks = []
        for C in CHANNELS:
            if cols[C] is None:
                continue
            score, elig = cols[C]
            ranked = C in STAGE2_RANKS
            order = _order(score, elig, max(q[C], RANK_CAP) if ranked else q[C])
            picks.append(order[:q[C]])
            if ranked:
                top = order[:RANK_CAP]
                rk[top, STAGE2_RANKS.index(C)] = np.arange(1, top.size + 1, dtype=np.int32)
        learn = cols["LEARN"][0]
        sel = np.unique(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.intp)
        sel = sel[np.lexsort((sel, -learn[sel]))]
        return {"pool_idx": un.u[sel], "X": X[sel].astype(np.float32),
                "learn": learn[sel].astype(np.float32), "ranks": rk[sel], "n_union": int(un.u.size),
                "d_en": un.d_en, "n_dropped": un.n_dropped,
                "ms": (time.perf_counter() - t0) * 1000.0}

    def evaluate(self, name, addr, chain, truth, grid):
        """Evaluation mode: rank union items per column (up to the largest effective quota of that
        column in `grid`, see make_grid), then apply every config at once: an item is selected iff
        some column c has rank_c <= effective q_c. Also reachability of `truth` in the whole union and
        the families that reach each true item (bit FAMILY_ORDER.index(F))."""
        t0 = time.perf_counter()
        un = self.union(name, addr)
        cols = self.scores(un, self.features(un))
        fac = np.where(chain >= grid["min_chain"], grid["mult"], 1)
        qeff = grid["quota"] * np.where(grid["chain_mask"][None, :], fac[:, None], 1)
        cap = qeff.max(axis=0) if qeff.size else np.zeros(len(EVAL_COLUMNS), dtype=np.int64)
        R = np.full((un.u.size, len(EVAL_COLUMNS)), RANK_INF, dtype=np.int32)
        for c, C in enumerate(EVAL_COLUMNS):
            if cols[C] is not None and cap[c] > 0:
                order = _order(*cols[C], int(cap[c]))
                R[order, c] = np.arange(1, order.size + 1, dtype=np.int32)
        ms = (time.perf_counter() - t0) * 1000.0
        cand = np.flatnonzero((R < RANK_INF).any(axis=1))
        sel = (R[cand][:, None, :] <= qeff[None, :, :]).any(axis=2)
        truth = np.asarray(truth, dtype=np.uint32)
        pos = np.searchsorted(un.u, truth)
        reached = pos < un.u.size
        reached[reached] = un.u[pos[reached]] == truth[reached]
        famask = np.zeros(truth.size, dtype=np.int64)
        famask[reached] = self.bits @ (un.S[:, pos[reached]] > 0)
        sel_true = np.zeros((truth.size, qeff.shape[0]), dtype=bool)
        if cand.size and reached.any():
            cp = np.minimum(np.searchsorted(cand, pos[reached]), cand.size - 1)
            inc = cand[cp] == pos[reached]
            sel_true[np.flatnonzero(reached)[inc]] = sel[cp[inc]]
        return {"n_true": int(truth.size), "reached": reached, "famask": famask, "sel_true": sel_true,
                "sizes": sel.sum(axis=0).astype(np.int32), "n_union": int(un.u.size),
                "d_en": un.d_en, "n_dropped": un.n_dropped, "ms": ms}


def make_grid(labels, quotas, chain_channels, chain=None):
    """Quota configs for Retriever.evaluate. quotas: [{column: q}] (missing columns 0);
    chain: [(min_chain, mult)] per config (mult 1 = off); chain_channels: the chain rule's channels."""
    n = len(labels)
    chain = chain or [(0, 1)] * n
    return {"labels": list(labels),
            "quota": np.array([[int(q.get(C, 0)) for C in EVAL_COLUMNS] for q in quotas], dtype=np.int64),
            "min_chain": np.array([c[0] for c in chain], dtype=np.int64),
            "mult": np.array([c[1] for c in chain], dtype=np.int64),
            "chain_mask": np.array([C in chain_channels for C in EVAL_COLUMNS])}


def config_chain(cfg):
    """(min_chain, mult) of the configured chain rule (mult 1 when disabled)."""
    cr = cfg["chain_rule"]
    return (cr["min_chain"], cr["mult"]) if cr["enabled"] else (0, 1)


# ---------------------------------------------------------------- spawn-pool driver

_W = {}


def _init_worker(split):
    _W.update(split=split, key=None, retriever=None)


def _retriever(cfg, model):
    key = (io_utils.config_hash(cfg), None if model is None else io_utils.config_hash(model))
    if _W["key"] != key:
        _W["retriever"], _W["key"] = Retriever(_W["split"], cfg, model), key
    return _W["retriever"]


def _run_task(task):
    cfg, model, fn, idx, names, addrs, chains, per_query, shared = task
    return fn(_retriever(cfg, model), idx, names, addrs, chains, per_query, shared)


def retrieve_chunk(r, idx, names, addrs, chains, per_query, shared):
    """Worker function: Retriever.retrieve per query."""
    return [r.retrieve(n, a, int(c)) for n, a, c in zip(names, addrs, chains)]


def evaluate_chunk(r, idx, names, addrs, chains, per_query, shared):
    """Worker function: Retriever.evaluate per query; per_query = truth arrays, shared = grid."""
    return [r.evaluate(n, a, int(c), t, shared) for n, a, c, t in zip(names, addrs, chains, per_query)]


class QueryRunner:
    """Spawn pool of W workers over the S1 queries of a split. Workers build (and cache) a Retriever
    per (cfg, model); the parent sends query texts in chunks.

    with QueryRunner("train", cfg) as qr:
        for chunk_results in qr.map(cfg, model, s1_idx, retrieve_chunk): ...
    fn(retriever, s1_idx, names, addrs, chains, per_query, shared) must be a module-level function.
    """

    def __init__(self, split, cfg):
        self.split = split
        self.W = io_utils.n_workers(cfg)
        t = pq.read_table(io_utils.work_path("norm", f"{split}_s1.parquet"),
                          columns=["s1_idx", "name_norm", "addr_norm"])
        if not np.array_equal(t.column("s1_idx").to_numpy(), np.arange(t.num_rows)):
            raise ValueError(f"norm/{split}_s1.parquet: s1_idx is not 0..n-1 in file order")
        self.names, self.addrs = t.column("name_norm"), t.column("addr_norm")
        self.chain = np.load(io_utils.work_path("norm", f"{split}_s1_chain.npy"), mmap_mode="r")
        self.pool = None

    def __enter__(self):
        self.pool = io_utils.spawn_pool(self.W, _init_worker, (self.split,))
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.pool.close()
        else:
            self.pool.terminate()
        self.pool.join()

    def map(self, cfg, model, s1_idx, fn, per_query=None, shared=None, chunk=CHUNK):
        """Yield fn's result for each chunk of s1_idx, in order."""
        s1_idx = np.asarray(s1_idx, dtype=np.int64)

        def tasks():
            for a in range(0, s1_idx.size, chunk):
                idx = s1_idx[a:a + chunk]
                yield (cfg, model, fn, idx, self.names.take(idx).to_pylist(),
                       self.addrs.take(idx).to_pylist(), np.asarray(self.chain[idx]),
                       None if per_query is None else per_query[a:a + chunk], shared)

        yield from self.pool.imap(_run_task, tasks())

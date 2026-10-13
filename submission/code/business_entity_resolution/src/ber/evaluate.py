"""M8 metrics (plan §5.9): per-query recall and F0.5 ceiling, set summaries, segment summaries;
stage-1 evaluation and experiments E1-E6 (`s1-eval`, plan §5.7 evaluation mode, §6, §8).

Per query: r = hits / n_true (non-singletons); ceiling = 1.25 r / (0.25 + r), 0 if r = 0; a singleton
(n_true = 0) has ceiling 1.0 because a perfect matcher predicts the empty list. The F0.5 ceiling (the
primary objective) is the mean over ALL queries.
"""

import copy
import itertools

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import io_utils, retrieve, splits


def query_metrics(n_true, n_hit):
    """Per-query (recall, ceiling) as float64 arrays; recall is NaN for singletons."""
    n_true = np.asarray(n_true, dtype=np.int64)
    n_hit = np.asarray(n_hit, dtype=np.int64)
    if np.any(n_hit < 0) or np.any(n_hit > n_true):
        raise ValueError("need 0 <= n_hit <= n_true")
    single = n_true == 0
    r = np.full(n_true.shape, np.nan)
    r[~single] = n_hit[~single] / n_true[~single]
    ceiling = np.ones(n_true.shape)
    ceiling[~single] = 1.25 * r[~single] / (0.25 + r[~single])  # r = 0 gives 0
    return r, ceiling


def size_stats(sizes, n_pool):
    """Candidate-list size distribution and reduction ratio 1 - mean size / n_pool."""
    sizes = np.asarray(sizes, dtype=np.float64)
    if sizes.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p99": None, "max": None,
                "reduction_ratio": None}
    p50, p90, p99 = np.percentile(sizes, [50, 90, 99])
    mean = float(sizes.mean())
    return {"mean": mean, "p50": float(p50), "p90": float(p90), "p99": float(p99),
            "max": int(sizes.max()), "reduction_ratio": 1.0 - mean / n_pool}


def summarize(n_true, n_hit, sizes, n_pool):
    """Set-level metrics over ALL queries (§5.9); recall shares are over non-singleton queries."""
    n_true = np.asarray(n_true, dtype=np.int64)
    n_hit = np.asarray(n_hit, dtype=np.int64)
    r, ceiling = query_metrics(n_true, n_hit)
    multi = n_true > 0
    n_multi = int(multi.sum())
    total_true = int(n_true.sum())
    return {
        "n_queries": int(n_true.size),
        "n_singletons": int(n_true.size - n_multi),
        "n_true_pairs": total_true,
        "n_hit_pairs": int(n_hit.sum()),
        "f05_ceiling": float(ceiling.mean()) if n_true.size else None,
        "entity_recall": float((n_hit[multi] > 0).mean()) if n_multi else None,
        "full_recall": float((n_hit[multi] == n_true[multi]).mean()) if n_multi else None,
        "pair_recall": int(n_hit.sum()) / total_true if total_true else None,
        "sizes": size_stats(sizes, n_pool),
    }


def summarize_segments(n_true, n_hit, sizes, n_pool, masks):
    """{segment: summarize(...) restricted to the queries where masks[segment] is True}."""
    n_true, n_hit, sizes = (np.asarray(a) for a in (n_true, n_hit, sizes))
    return {name: summarize(n_true[m], n_hit[m], sizes[m], n_pool)
            for name, m in ((k, np.asarray(v, dtype=bool)) for k, v in masks.items())}


# ---------------------------------------------------------------- stage-1 evaluation inputs

E1_FAMILIES = ("EN", "NT", "NG", "PH", "AT")
E1_TOPK = (50, 150, 500, 2000)
E1_GATE = 0.91
E2_CANDIDATES = ("NP", "SP", "NC", "AB", "XK", "SK")
E2_EXCL_SHARE = 0.0005
E2_GAIN = 0.001
E3_ORDER = ("NT", "NG", "AT", "NP", "SP", "AB", "XK")
E3_MULT = (0.5, 1, 2.5)
E3B_TOPK = (150, 300)
E4_GRID = (("LEARN", (150, 300, 500)), ("NAME", (0, 50, 100)), ("ADDR", (50, 100, 200)),
           ("PAIR", (0, 50, 100)), ("XK", (0, 50, 100)), ("CAT", (0, 20)))
E4_MAX_MEAN = 1000
E4_TOL = 0.001
E5_MIN_CHAIN = (5, 10, 20)
E5_MULT = (1, 2, 3)
E5_SEG_GAIN = 0.005
E5_MAX_GROWTH = 0.15
E6_TOL = 0.003
ADHOC_TOPK = (50, 150, 300, 500, 2000)
RAW_BATCH = 1_000_000


def truth_lists(s1_idx):
    """True pool_idx arrays (uint32, ascending) per query from records/train_gt.parquet."""
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"), columns=["s1_idx", "pool_idx"])
    s, p = gt.column("s1_idx").to_numpy(), gt.column("pool_idx").to_numpy()
    s1_idx = np.asarray(s1_idx)
    lo, hi = np.searchsorted(s, s1_idx, "left"), np.searchsorted(s, s1_idx, "right")
    return [p[a:b].astype(np.uint32) for a, b in zip(lo, hi)]


def _pool_raw_flags(pool_idx):
    """(raw name is ASCII, raw address empty) per given train pool record."""
    need = np.unique(pool_idx)
    asc, emp = np.zeros(need.size, dtype=bool), np.zeros(need.size, dtype=bool)
    pf = pq.ParquetFile(io_utils.work_path("records", "train_pool.parquet"))
    start = 0
    for b in pf.iter_batches(batch_size=RAW_BATCH, columns=["name_raw", "addr_raw"]):
        stop = start + b.num_rows
        lo, hi = np.searchsorted(need, [start, stop])
        if hi > lo:
            loc = need[lo:hi] - start
            asc[lo:hi] = [x.isascii() for x in b.column(0).take(loc).to_pylist()]
            emp[lo:hi] = [x == "" for x in b.column(1).take(loc).to_pylist()]
        start = stop
    j = np.searchsorted(need, pool_idx)
    return asc[j], emp[j]


def segment_data(s1_idx, pair_pool):
    """§5.9 segments. Query masks: chain_S1 {1, 2-9, >=10}, country, name unseen (no DEV S1 = non-HOLDOUT
    train S1 other than the query itself has the same name_norm), match count {1, 2-3, 4-5, 6+}.
    True-pair masks (pool side; singletons have none): pool name class {indic, ascii, other}, pool
    address empty (raw)."""
    s1_idx = np.asarray(s1_idx)
    chain = np.load(io_utils.work_path("norm", "train_s1_chain.npy"), mmap_mode="r")[s1_idx]
    country = pd.read_parquet(io_utils.work_path("records", "train_s1.parquet"),
                              columns=["country_raw"])["country_raw"].to_numpy()[s1_idx]
    nmatch = np.load(io_utils.work_path("records", "train_s1_nmatch.npy"))[s1_idx]
    names = pq.read_table(io_utils.work_path("norm", "train_s1.parquet"), columns=["name_norm"])
    codes, _ = pd.factorize(names.column(0).to_pandas())
    del names
    dev = np.ones(codes.size, dtype=bool)
    dev[splits.load_query_sets()["HOLDOUT_TEST"]] = False
    others = np.bincount(codes[dev], minlength=codes.max() + 1)[codes[s1_idx]] - dev[s1_idx]
    q = {"chain_1": chain == 1, "chain_2_9": (chain >= 2) & (chain <= 9), "chain_ge10": chain >= 10}
    q.update({f"country_{c}": country == c for c in sorted(set(country.tolist()))})
    q["name_unseen"] = others == 0
    q.update({"match_1": nmatch == 1, "match_2_3": (nmatch >= 2) & (nmatch <= 3),
              "match_4_5": (nmatch >= 4) & (nmatch <= 5), "match_6p": nmatch >= 6})
    indic = np.load(io_utils.paths()["work_root"] / "norm" / "train_pool_store" / "indic.npy",
                    mmap_mode="r")[pair_pool]
    asc, emp = _pool_raw_flags(pair_pool)
    pair = {"pool_indic": indic, "pool_ascii": ~indic & asc, "pool_other": ~indic & ~asc,
            "pool_addr_empty": emp}
    return {"query": q, "pair": pair}


def segment_metrics(n_true, n_hit, sizes, n_pool, seg, pair_q, pair_hit):
    """All metrics per segment. Pool-side segment X: queries with >= 1 true pair having X, plus the
    pair recall over the true pairs having X."""
    n_true, n_hit, sizes = (np.asarray(a) for a in (n_true, n_hit, sizes))
    out = summarize_segments(n_true, n_hit, sizes, n_pool, seg["query"])
    for name, pm in seg["pair"].items():
        qm = np.zeros(n_true.size, dtype=bool)
        qm[pair_q[pm]] = True
        d = summarize(n_true[qm], n_hit[qm], sizes[qm], n_pool)
        d["n_segment_pairs"] = int(pm.sum())
        d["segment_pair_recall"] = float(pair_hit[pm].mean()) if pm.any() else None
        out[name] = d
    return out


# ---------------------------------------------------------------- stage-1 evaluation results

class S1Result:
    """Evaluation-mode results of one query set under one grid of quota configs."""

    def __init__(self, rows, grid, n_pool, seg):
        self.grid, self.n_pool, self.seg = grid, n_pool, seg
        nc = len(grid["labels"])
        self.n_true = np.array([r["n_true"] for r in rows], dtype=np.int64)
        self.pair_q = np.repeat(np.arange(self.n_true.size), self.n_true)
        self.pair_sel = np.vstack([r["sel_true"] for r in rows]).reshape(-1, nc)
        self.hits = np.vstack([r["sel_true"].sum(axis=0) for r in rows]).reshape(-1, nc)
        self.sizes = np.vstack([r["sizes"] for r in rows]).reshape(-1, nc)
        self.reached = np.concatenate([r["reached"] for r in rows])
        self.famask = np.concatenate([r["famask"] for r in rows])
        for k in ("n_union", "d_en", "n_dropped", "ms"):
            setattr(self, k, np.array([r[k] for r in rows]))

    def summary(self, j):
        return summarize(self.n_true, self.hits[:, j], self.sizes[:, j], self.n_pool)

    def f05(self, j, mask=None):
        m = slice(None) if mask is None else mask
        return float(query_metrics(self.n_true[m], self.hits[m, j])[1].mean())

    def segments(self, j):
        return segment_metrics(self.n_true, self.hits[:, j], self.sizes[:, j], self.n_pool, self.seg,
                               self.pair_q, self.pair_sel[:, j])

    def reachability(self):
        hits = np.bincount(self.pair_q, weights=self.reached, minlength=self.n_true.size).astype(np.int64)
        d = summarize(self.n_true, hits, self.n_union, self.n_pool)
        d["segments"] = segment_metrics(self.n_true, hits, self.n_union, self.n_pool, self.seg,
                                        self.pair_q, self.reached)
        return d

    def exclusive_pairs(self):
        """True pairs in the union reachable through exactly one family."""
        tot = int(self.n_true.sum())
        out = {}
        for b, F in enumerate(retrieve.FAMILY_ORDER):
            n = int((self.famask == (1 << b)).sum())
            if n or (self.famask & (1 << b)).any():
                out[F] = {"n": n, "share_of_true_pairs": n / tot if tot else None}
        return out

    def ms_median(self):
        return float(np.median(self.ms))

    def payload(self, primary=None):
        cfgs = [dict(label=lab, **self.summary(j)) for j, lab in enumerate(self.grid["labels"])]
        out = {"configs": cfgs, "reachability": self.reachability(),
               "exclusive_pairs": self.exclusive_pairs(),
               "ms": {"p50": self.ms_median(), "p99": float(np.percentile(self.ms, 99)),
                      "mean": float(self.ms.mean())},
               "union_size": size_stats(self.n_union, self.n_pool),
               "d_en": {"p50": float(np.median(self.d_en)), "p90": float(np.percentile(self.d_en, 90))},
               "dropped_keys": {"queries_with_drops": int((self.n_dropped > 0).sum()),
                                "total": int(self.n_dropped.sum()), "max": int(self.n_dropped.max())}}
        if primary is not None:
            out["primary"] = self.grid["labels"][primary]
            out["primary_segments"] = self.segments(primary)
        return out


class _Session:
    """One s1-eval command: a QueryRunner over train plus per-set truth and segment data."""

    def __init__(self, qr, cfg, lg):
        self.qr, self.cfg, self.lg = qr, cfg, lg
        self.n_pool = int(io_utils.read_json(io_utils.work_path("index", "train", "meta.json"))["n_pool"])
        self.sets = splits.load_query_sets()
        self._data = {}

    def data(self, set_name):
        if set_name not in self._data:
            s1 = self.sets[set_name]
            truth = truth_lists(s1)
            pair_pool = np.concatenate(truth) if truth else np.zeros(0, dtype=np.uint32)
            self._data[set_name] = (s1, truth, segment_data(s1, pair_pool))
        return self._data[set_name]

    def run(self, set_name, cfg_v, model, grid, tag):
        s1, truth, seg = self.data(set_name)
        rows = []
        for chunk in self.qr.map(cfg_v, model, s1, retrieve.evaluate_chunk, per_query=truth, shared=grid):
            rows.extend(chunk)
        res = S1Result(rows, grid, self.n_pool, seg)
        self.lg.info("%s %s: families=%s ms p50 %.2f |U| mean %.0f; %s", set_name, tag,
                     retrieve.kept_families(cfg_v), res.ms_median(), res.n_union.mean(),
                     ", ".join(f"{lab} f05={res.f05(j):.4f}" for j, lab in enumerate(grid["labels"][:6])))
        return res

    def write(self, args, payload):
        s1, truth, _ = self.data(args.set)
        doc = {"exp": args.exp, "set": args.set, **payload}
        return io_utils.write_metrics(f"s1_{args.exp}", doc, s1.size, sum(t.size for t in truth), self.cfg)


# ---------------------------------------------------------------- experiment helpers

def _variant(cfg, families=None, caps=None, learn=None, quota=None):
    v = copy.deepcopy(cfg)
    if families is not None:
        v["families"] = [F for F in retrieve.FAMILY_ORDER if F in families]
    if caps:
        v["cap_frac"].update(caps)
    if learn is not None:
        v["channels"]["LEARN"] = learn
    if quota is not None:
        v["quota"] = dict(quota)
    return v


def _quota_of(cfg):
    return {C: cfg["quota"][C] for C in retrieve.CHANNELS}


def _config_grid(cfg):
    """The configured quotas and chain rule as a one-config grid."""
    return retrieve.make_grid(["config"], [_quota_of(cfg)], cfg["chain_rule"]["channels"],
                              [retrieve.config_chain(cfg)])


def _topk_grid(column, Ks, prefix):
    return retrieve.make_grid([f"{prefix}_top{K}" for K in Ks], [{column: K} for K in Ks], ())


def _e4_grid(cfg):
    names = [c for c, _ in E4_GRID]
    quotas = [dict(zip(names, vals)) for vals in itertools.product(*(v for _, v in E4_GRID))]
    labels = ["_".join(f"{c}{q[c]}" for c in names) for q in quotas]
    return retrieve.make_grid(labels, quotas, cfg["chain_rule"]["channels"],
                              [retrieve.config_chain(cfg)] * len(quotas))


def pareto_front(f05, mean):
    """Indices of configs not dominated in (higher F0.5 ceiling, smaller mean size), by mean asc."""
    front, best = [], -np.inf
    for j in np.lexsort((-np.asarray(f05), np.asarray(mean))):
        if f05[j] > best:
            front.append(int(j))
            best = f05[j]
    return front


def _metrics_path(name):
    return io_utils.paths()["metrics_root"] / f"{name}.json"


# ---------------------------------------------------------------- experiments (plan §8)

def _e1(ss, args):
    cfg_v = _variant(ss.cfg, families=E1_FAMILIES, learn="hand_both")
    grid = _topk_grid("HAND", E1_TOPK, "hand")
    res = ss.run(args.set, cfg_v, None, grid, "E1")
    j = E1_TOPK.index(150)
    pr = res.summary(j)["pair_recall"]
    passed = pr is not None and pr >= E1_GATE
    ss.write(args, {"runs": {"E1": res.payload(primary=j)},
                    "decision": {"rule": "pair recall@150 >= 0.91", "pair_recall_at_150": pr,
                                 "passed": passed}})
    if not passed:
        raise RuntimeError(f"BLOCKER: E1 gate failed: hand_both pair recall@150 = {pr} < {E1_GATE} "
                           "(plan §8: bug hunt, stop)")


def _e2(ss, args):
    cfg = ss.cfg
    runs, cache = {}, {}

    def ev(fams, tag):
        key = tuple(F for F in retrieve.FAMILY_ORDER if F in fams)
        if key not in cache:
            cfg_v = _variant(cfg, families=key, learn="hand_both")
            cache[key] = ss.run(args.set, cfg_v, None, _config_grid(cfg_v), tag)
        runs[tag] = cache[key].payload(primary=0)
        return cache[key]

    n_pairs = sum(t.size for t in ss.data(args.set)[1])
    thr = E2_EXCL_SHARE * n_pairs
    base = ev(E1_FAMILIES, "E1")
    kept, per_family = list(E1_FAMILIES), {}
    for F in E2_CANDIDATES:
        res = ev(list(E1_FAMILIES) + [F], f"E1+{F}")
        excl = res.exclusive_pairs().get(F, {"n": 0})["n"]
        gain = res.f05(0) - base.f05(0)
        keep = excl >= thr or gain >= E2_GAIN
        per_family[F] = {"exclusive_pairs": excl, "f05_gain": gain, "ms_p50": res.ms_median(), "keep": keep}
        if keep:
            kept.append(F)
    full = ev(kept, "kept")
    no_ph = ev([F for F in kept if F != "PH"], "kept-PH")
    excl_ph = full.exclusive_pairs().get("PH", {"n": 0})["n"]
    ph_gain = full.f05(0) - no_ph.f05(0)
    keep_ph = excl_ph >= thr or ph_gain >= E2_GAIN
    final = [F for F in retrieve.FAMILY_ORDER if F in kept and (keep_ph or F != "PH")]
    meta = io_utils.read_json(io_utils.work_path("index", "train", "meta.json"))["families"]
    decision = {"rule": "keep F if exclusive pairs >= 0.05% of true pairs or F0.5 ceiling gain >= 0.001",
                "exclusive_threshold": thr, "candidates": per_family,
                "PH": {"exclusive_pairs": excl_ph, "f05_gain": ph_gain, "keep": keep_ph},
                "families": final, "index_bytes": {F: meta[F]["bytes"] for F in meta},
                "ms_p50_per_run": {k: v["ms"]["p50"] for k, v in runs.items()}}
    path = ss.write(args, {"runs": runs, "decision": decision})
    io_utils.apply_tuned({"families": final}, "E2", path)


def _e3(ss, args):
    cfg = ss.cfg
    limit = cfg["build_df_max_frac"]
    caps, runs, decisions = {}, {}, {}
    for F in (F for F in E3_ORDER if F in cfg["families"]):
        base = cfg["cap_frac"][F]
        vals = {m: round(base * m, 12) for m in E3_MULT if round(base * m, 12) <= limit}
        res = {}
        for m, v in vals.items():
            cfg_v = _variant(cfg, caps={**caps, F: v}, learn="hand_both")
            res[m] = ss.run(args.set, cfg_v, None, _config_grid(cfg_v), f"E3 {F}={v}")
            runs[f"{F}={v}"] = res[m].payload(primary=0)
        f05 = {m: r.f05(0) for m, r in res.items()}
        ms = {m: r.ms_median() for m, r in res.items()}
        choice = 1
        if 2.5 in res and f05[2.5] - f05[1] >= 0.001 and ms[2.5] <= 1.3 * ms[1]:
            choice = 2.5
        elif 0.5 in res and f05[1] - f05[0.5] < 0.0005 and ms[0.5] <= 0.85 * ms[1]:
            choice = 0.5
        caps[F] = vals[choice]
        decisions[F] = {"values": {str(vals[m]): {"f05": f05[m], "ms_p50": ms[m]} for m in vals},
                        "chosen": caps[F]}
    decision = {"rule": "larger cap if F0.5 gain >= 0.001 and median ms +<= 30%; smaller if loss < 0.0005 "
                        "and median ms -15% or more", "per_family": decisions}
    path = ss.write(args, {"runs": runs, "decision": decision})
    if caps:
        io_utils.apply_tuned({f"cap_frac.{F}": v for F, v in caps.items()}, "E3", path)


def _e3b(ss, args):
    cfg = ss.cfg
    p = retrieve.model_path()
    model = io_utils.read_json(p) if p.exists() else None
    if model is None or model["config_hash"] != io_utils.config_hash(cfg):
        raise RuntimeError(f"E3b needs {p} trained with the current config: "
                           "run `python -m ber.cli stage1-train` first")
    cfg_v = _variant(cfg, learn="stage1_model")
    grid = retrieve.make_grid([f"learn_top{K}" for K in E3B_TOPK] + [f"hand_top{K}" for K in E3B_TOPK],
                              [{"LEARN": K} for K in E3B_TOPK] + [{"HAND": K} for K in E3B_TOPK], ())
    res = ss.run(args.set, cfg_v, model, grid, "E3b")
    learn300, hand300 = res.f05(1), res.f05(3)
    keep = learn300 >= hand300
    decision = {"rule": "LEARN main channel iff learned F0.5 ceiling at K=300 >= hand_both",
                "learn_f05_at_300": learn300, "hand_f05_at_300": hand300, "keep_learn": keep}
    path = ss.write(args, {"runs": {"E3b": res.payload(primary=1)}, "decision": decision})
    io_utils.apply_tuned({"channels.LEARN": "stage1_model" if keep else "hand_both"}, "E3b", path)


def _e4(ss, args):
    cfg = ss.cfg
    grid = _e4_grid(cfg)
    res = ss.run(args.set, cfg, retrieve.load_stage1_model(cfg), grid, "E4")
    summ = [res.summary(j) for j in range(len(grid["labels"]))]
    f05 = np.array([s["f05_ceiling"] for s in summ])
    er = np.array([s["entity_recall"] for s in summ], dtype=float)
    mean = np.array([s["sizes"]["mean"] for s in summ])
    ok = np.flatnonzero(mean <= E4_MAX_MEAN)
    if ok.size == 0:
        raise RuntimeError(f"BLOCKER: E4 no quota config with mean size <= {E4_MAX_MEAN}")
    b = int(ok[np.lexsort((ok, mean[ok], -f05[ok]))[0]])
    cand = np.flatnonzero((f05 >= f05[b] - E4_TOL) & (er >= er[b] - E4_TOL))
    c = int(cand[np.lexsort((cand, -f05[cand], mean[cand]))[0]])
    quota = {C: int(q) for C, q in zip(retrieve.EVAL_COLUMNS, grid["quota"][c]) if C != "HAND"}

    def point(j):
        return {"label": grid["labels"][j], "f05_ceiling": float(f05[j]), "entity_recall": float(er[j]),
                "mean_size": float(mean[j]),
                "quota": {C: int(q) for C, q in zip(retrieve.EVAL_COLUMNS, grid["quota"][j]) if C != "HAND"}}

    decision = {"rule": "best = max F0.5 ceiling with mean <= 1000; smallest mean with F0.5 >= best - 0.001 "
                        "and entity recall >= best's - 0.001",
                "best": point(b), "chosen": point(c),
                "pareto": [point(j) for j in pareto_front(f05, mean)]}
    path = ss.write(args, {"runs": {"E4": res.payload(primary=c)}, "decision": decision})
    io_utils.apply_tuned({"quota": quota}, "E4", path)


def _e5(ss, args):
    cfg = ss.cfg
    settings = [(mc, m) for mc in E5_MIN_CHAIN for m in E5_MULT]
    labels = [f"min_chain{mc}_mult{m}" for mc, m in settings]
    grid = retrieve.make_grid(labels, [_quota_of(cfg)] * len(settings), cfg["chain_rule"]["channels"],
                              settings)
    res = ss.run(args.set, cfg, retrieve.load_stage1_model(cfg), grid, "E5")
    seg = ss.data(args.set)[2]["query"]["chain_ge10"]
    off = settings.index(next(s for s in settings if s[1] == 1))
    mean = [res.summary(j)["sizes"]["mean"] for j in range(len(settings))]
    per, ok = {}, []
    for j, (mc, m) in enumerate(settings):
        gain = res.f05(j, seg) - res.f05(off, seg)
        growth = mean[j] / mean[off] - 1.0
        passes = m > 1 and gain >= E5_SEG_GAIN and growth <= E5_MAX_GROWTH
        per[labels[j]] = {"f05": res.f05(j), "chain_ge10_f05": res.f05(j, seg), "chain_ge10_gain": gain,
                          "mean_size": mean[j], "mean_growth": growth, "passes": passes}
        if passes:
            ok.append(j)
    best = min(ok, key=lambda j: (-res.f05(j), mean[j], j)) if ok else None
    chosen = off if best is None else best
    decision = {"rule": "among settings with chain>=10 F0.5 gain >= 0.005 and mean growth <= 15%: highest "
                        "F0.5 ceiling (ties smaller mean); else off",
                "settings": per, "chosen": None if best is None else labels[best],
                "f05_ceiling_chosen": res.f05(chosen), "mean_size_chosen": mean[chosen]}
    path = ss.write(args, {"runs": {"E5": res.payload(primary=chosen)}, "decision": decision})
    if best is None:
        io_utils.apply_tuned({"chain_rule.enabled": False}, "E5", path)
    else:
        mc, m = settings[best]
        io_utils.apply_tuned({"chain_rule.enabled": True, "chain_rule.min_chain": mc,
                              "chain_rule.mult": m}, "E5", path)


def _without(cfg, *drop):
    c = copy.deepcopy(cfg)
    for k in drop:
        c.pop(k, None)
    return c


def _e6(ss, args):
    cfg = ss.cfg
    e5 = io_utils.read_json(_metrics_path("s1_E5"))
    e4 = io_utils.read_json(_metrics_path("s1_E4"))
    ch = e5["decision"]["chosen"]
    cr = cfg["chain_rule"]
    chain_ok = (not cr["enabled"]) if ch is None else (
        cr["enabled"] and f"min_chain{cr['min_chain']}_mult{cr['mult']}" == ch)
    if _without(e5["config"], "tuned_by", "chain_rule") != _without(cfg, "tuned_by", "chain_rule") or not chain_ok:
        raise RuntimeError("E6 needs s1_E5.json computed with the current config: run E5 first")
    dev_a = e5["decision"]["f05_ceiling_chosen"]
    model = retrieve.load_stage1_model(cfg)
    res = ss.run(args.set, cfg, model, _config_grid(cfg), "E6")
    runs = {"chosen": res.payload(primary=0)}
    dev_b = res.f05(0)
    decision = {"rule": "switch to the next E4 Pareto point (larger mean) if DEV_B F0.5 < DEV_A - 0.003; "
                        "re-check once", "dev_a_f05": dev_a, "dev_b_f05": dev_b,
                "passed": dev_b >= dev_a - E6_TOL}
    switch = None
    if not decision["passed"]:
        chosen_mean = e4["decision"]["chosen"]["mean_size"]
        nxt = [p for p in e4["decision"]["pareto"] if p["mean_size"] > chosen_mean]
        if nxt:
            switch = nxt[0]
            cfg2 = _variant(cfg, quota=switch["quota"])
            ra = ss.run("DEV_A", cfg2, model, _config_grid(cfg2), "E6 recheck DEV_A")
            rb = ss.run(args.set, cfg2, model, _config_grid(cfg2), "E6 recheck DEV_B")
            runs["recheck_DEV_A"], runs["recheck_DEV_B"] = ra.payload(primary=0), rb.payload(primary=0)
            decision["recheck"] = {"pareto_point": switch, "dev_a_f05": ra.f05(0), "dev_b_f05": rb.f05(0),
                                   "passed": rb.f05(0) >= ra.f05(0) - E6_TOL}
        else:
            decision["recheck"] = {"pareto_point": None, "passed": False}
    path = ss.write(args, {"runs": runs, "decision": decision})
    if switch is not None:
        io_utils.apply_tuned({"quota": switch["quota"]}, "E6", path)
    if not decision["passed"] and not decision["recheck"]["passed"]:
        raise RuntimeError(f"BLOCKER: E6 DEV_B confirmation failed after the re-check ({path})")


def _adhoc(ss, args):
    cfg = ss.cfg
    fams = args.families or cfg["families"]
    bad = [F for F in fams if F not in retrieve.FAMILY_ORDER]
    if bad:
        raise ValueError(f"unknown families: {bad}")
    caps = {}
    for item in args.cap:
        F, x = item.split("=", 1)
        caps[F] = float(x)
    learn = "hand_both" if args.scorer == "hand" else "stage1_model"
    cfg_v = _variant(cfg, families=fams, caps=caps, learn=learn)
    model = retrieve.load_stage1_model(cfg_v)
    if args.quota_grid:
        grid = _e4_grid(cfg_v)
    else:
        top = _topk_grid("LEARN", ADHOC_TOPK, args.scorer)
        base = _config_grid(cfg_v)
        grid = {k: (base[k] + top[k] if k == "labels" else
                    base[k] if k == "chain_mask" else np.concatenate([base[k], top[k]]))
                for k in base}
    res = ss.run(args.set, cfg_v, model, grid, args.exp)
    ss.write(args, {"runs": {args.exp: res.payload(primary=0)},
                    "setup": {"families": cfg_v["families"], "caps": caps, "scorer": args.scorer,
                              "quota_grid": bool(args.quota_grid)}})


EXPERIMENTS = {"E1": _e1, "E2": _e2, "E3": _e3, "E3b": _e3b, "E4": _e4, "E5": _e5, "E6": _e6}


def run_s1_eval(args):
    """CLI `s1-eval`: E1-E6 (plan §8; decisions written to blocking.json) or one ad-hoc evaluation."""
    lg = io_utils.get_logger(f"s1-eval_{args.exp}")
    if args.set in ("HOLDOUT_TEST", "val-S1"):
        raise ValueError("val-S1 is never used for stage-1 evaluation or tuning")
    if args.exp in EXPERIMENTS and args.set != ("DEV_B" if args.exp == "E6" else "DEV_A"):
        raise ValueError(f"{args.exp} runs on {'DEV_B' if args.exp == 'E6' else 'DEV_A'} (plan §8)")
    cfg = io_utils.load_config()
    with retrieve.QueryRunner("train", cfg) as qr:
        EXPERIMENTS.get(args.exp, _adhoc)(_Session(qr, cfg, lg), args)
    lg.info("s1-eval %s done", args.exp)

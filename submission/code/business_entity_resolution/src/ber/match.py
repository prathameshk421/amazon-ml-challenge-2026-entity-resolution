"""Stage-3 matcher (plan/MATCHER_PLAN.md): stage-2 candidates -> pair + list features -> LightGBM
(5-fold, grouped by S1) -> isotonic calibration -> per-S1 decision -> matching_results.tsv.

CLI (repository root; see README.md):
  match-features --set {MATCH_TRAIN,MATCH_VALID,HOLDOUT_TEST,TEST} [--limit N] [--workers W]
  match-train [--workers W]            OOF on MATCH_TRAIN, calibration, decision rule, error CSVs
  match-eval --set {MATCH_VALID,HOLDOUT_TEST} [--force]   score the frozen matcher (HOLDOUT_TEST once)
  match-predict                        TEST -> work/outputs/matching_results.tsv

Label firewall (split.md rules 2-4): feature files never hold labels. Labels are joined from
records/train_gt.parquet only in `match-train` (MATCH_TRAIN: fit, calibration, decision choice) and in
`match-eval` (scoring only). MATCH_VALID confirms, HOLDOUT_TEST (= val-S1) is scored once and nothing is
tuned on either. The test split has no labels; `match-predict` uses test inputs only.

The one-owner rule (optional, chosen on MATCH_TRAIN OOF) uses the verified ground-truth fact that no
pool record has two owners: a pool record listed by several S1 of the scored set is kept only for the S1
with the highest calibrated p. It reads inputs only, never labels.
"""

import hashlib
import math
import time
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ber import hashing, io_utils, splits
from ber.index_store import IndexStore
from ber.textnorm import skeleton

MATCH_JSON = io_utils.CONFIG_DIR / "match.json"
TRAIN_SETS = ("MATCH_TRAIN", "MATCH_VALID", "HOLDOUT_TEST")
ALL_SETS = TRAIN_SETS + ("TEST",)
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
NAN = float("nan")

PAIR_FEATURES = [
    "p_rerank", "rank",
    # names (normalized text)
    "n_tset", "n_tsort", "n_ratio", "n_partial", "n_jw_sorted",
    "n_jacc", "n_idf_jacc", "n_idf_qcov", "n_idf_pcov", "n_q_miss_maxidf", "n_p_miss_maxidf",
    "n_eq", "n_eq_core", "n_eq_nospace", "n_acronym", "n_first_eq", "n_shared_maxlen", "n_sk_jacc",
    "q_ntok", "p_ntok",
    # addresses (normalized text)
    "a_tset", "a_tsort", "a_idf_jacc", "a_q_miss_maxidf", "a_p_miss_maxidf",
    "a_num_jacc", "a_num_q_only", "a_num_p_only", "a_longnum", "a_house", "a_empty",
    # record meta
    "country3", "src", "pool_indic", "log_chain",
    # rarest shared token, address shape (substring, char 3-grams, near-equal numbers, number + street)
    "n_shared_maxidf", "a_shared_maxidf", "a_partial", "a_char3_jacc", "a_num_minlev", "a_numstreet",
    # log1p pool count of records with the same core name / the same address token set (split pool)
    "q_name_cnt", "p_name_cnt", "q_addr_cnt", "p_addr_cnt",
    # Fellegi-Sunter-style 4-state digit agreement code (0 one-side, 1 conflict, 2 overlap, 3 match; NaN
    # neither side has digits), distinct from a_longnum which only covers >=3-digit tokens
    "a_digits_state",
    # X1 string-similarity group (plan/FEATURE_EXPERIMENTS.md): Monge-Elkan (Jaro-Winkler) and phonetic
    # set Jaccard on name tokens, computed per pair in pair_features(); char n-gram TF-IDF cosine on name
    # and address text, computed vectorized per chunk in _feature_chunk() and written directly into these
    # last two columns (not part of pair_features()'s return value).
    "n_me_mean", "n_me_min", "n_metaphone_jacc", "n_soundex_jacc",
    "n_tfidf_cos", "a_tfidf_cos",
]
TFIDF_N_FEATURES = 2 ** 20
_PHONETIC_MIN_LEN = 2
LIST_FEATURES = [
    "l_n", "l_p_max", "l_p_gap_max", "l_p_ratio_max", "l_p_gap_next", "l_p_sum",
    "l_rank_ntset", "l_rank_atset", "l_ntset_max_other", "l_nidf_gap_max",
    "l_n_name_close", "l_n_longnum_shared", "l_share_same_src", "l_n_addr_close",
]
FEATURES = PAIR_FEATURES + LIST_FEATURES
_COL = {f: i for i, f in enumerate(FEATURES)}
_NAME_CLOSE = 0.9
_ADDR_CLOSE = 0.9
_LONGNUM_MIN = 3  # digit tokens of >= 3 digits: house / plot / door numbers, PIN / ZIP when present


# ---------------------------------------------------------------- config, paths, fingerprints

def load_match_config():
    return io_utils.read_json(MATCH_JSON)


def features_dir(set_name):
    return io_utils.work_path("match", "features", set_name, "meta.json").parent


def _file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bucketing_fingerprint(cfg):
    """Identity of the candidate generator whose output the matcher consumes: stage-2 model file, its
    stage-1 hashes, the cut and the matcher-side candidate filter. Features, models and predictions must
    all share it."""
    meta = io_utils.read_json(io_utils.work_path("rerank", "model_v1_meta.json"))
    doc = {"rerank_model_sha1": _file_sha1(io_utils.work_path("rerank", "model_v1.txt")),
           "stage1_config_hash": meta["stage1_config_hash"],
           "stage1_model_hash": meta["stage1_model_hash"], "cut": cfg["rerank"]["cut"]}
    cand_filter = load_match_config().get("cand_filter")
    if cand_filter:
        doc["cand_filter"] = cand_filter
    return io_utils.config_hash(doc)


def used_features(mcfg):
    """Features the model is trained on: FEATURES minus match.json `drop_features` (ablations)."""
    drop = set(mcfg.get("drop_features", []))
    unknown = drop - set(FEATURES)
    if unknown:
        raise ValueError(f"match.json drop_features not in FEATURES: {sorted(unknown)}")
    return [f for f in FEATURES if f not in drop]


def _set_split(set_name):
    if set_name not in ALL_SETS:
        raise ValueError(f"unknown set {set_name!r}; expected one of {ALL_SETS}")
    return "test" if set_name == "TEST" else "train"


def set_queries(set_name):
    """s1_idx (int32, ascending) of a set; TEST = every test S1 in file order."""
    if set_name == "TEST":
        n = pq.ParquetFile(io_utils.work_path("records", "test_s1.parquet")).metadata.num_rows
        return np.arange(n, dtype=np.int32)
    return np.sort(splits.load_query_sets()[set_name]).astype(np.int32)


# ---------------------------------------------------------------- candidates

def read_candidates(set_name, s1_idx):
    """Candidate rows of the given queries, sorted by (s1_idx, rank): dict of numpy arrays.

    match.json `cand_filter` {"K_min", "p_min"} keeps rows with rank <= K_min or p_rerank >= p_min, i.e. a
    narrower stage-2 cut than the one the candidates were generated with (same K_max). It lets one wide
    candidate run serve every narrower cut; the final pipeline sets the cut in blocking.json instead."""
    d = io_utils.paths()["work_root"] / "candidates" / set_name
    parts = sorted(d.glob("part-*.parquet"))
    if not parts:
        raise RuntimeError(f"no candidates in {d}; run `candidates --set {set_name}` first")
    unfinished = [p.name for p in parts if not p.with_suffix("._SUCCESS").exists()]
    if unfinished:
        raise RuntimeError(f"{d}: unfinished parts {unfinished} (candidates run still going?)")
    t = pa.concat_tables([pq.read_table(p) for p in parts])
    s1 = t.column("s1_idx").to_numpy().astype(np.int32)
    rank = t.column("rank").to_numpy().astype(np.int32)
    p_rr = t.column("p_rerank").to_numpy().astype(np.float32)
    keep = np.isin(s1, s1_idx)
    cand_filter = load_match_config().get("cand_filter")
    if cand_filter:
        keep &= (rank <= cand_filter["K_min"]) | (p_rr >= cand_filter["p_min"])
    out = {"s1_idx": s1[keep], "pool_idx": t.column("pool_idx").to_numpy()[keep].astype(np.uint32),
           "rank": rank[keep], "p_rerank": p_rr[keep]}
    order = np.lexsort((out["rank"], out["s1_idx"]))
    return {k: v[order] for k, v in out.items()}


def group_bounds(s1_sorted):
    """(unique s1, start, stop) of runs in an s1-sorted array."""
    if s1_sorted.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z.astype(np.int32), z, z
    uniq, start = np.unique(s1_sorted, return_index=True)
    stop = np.append(start[1:], s1_sorted.size)
    return uniq, start, stop


# ---------------------------------------------------------------- pair features

class TokenIdf:
    """idf(token) = ln(1 + n_pool / df) from the split's M5 index: name tokens via NT (tokens of
    length >= 2), address tokens via AT. df is unsupervised (split.md rule 5). Absent tokens get
    df = 1 (rarest); tokens dropped at index build (df over the build limit) get df = build limit;
    one-character name tokens (not in NT) get idf 0."""

    def __init__(self, split):
        self.store = IndexStore(split, ("NT", "AT"))
        meta = io_utils.read_json(self.store.dir / "meta.json")
        self.n_pool = self.store.n_pool
        self.limit = int(meta["build_df_limit"])
        self.G = self.store.G

    def table(self, family, tokens):
        toks = sorted(set(tokens))
        if family == "NT":
            toks = [t for t in toks if len(t) >= 2]
        if not toks:
            return {}
        df = self.store.df(family, hashing.keys64(family, toks))
        df = np.where(df == self.store.sentinel, self.limit, np.maximum(df, 1))
        idf = np.log1p(self.n_pool / df.astype(np.float64))
        return dict(zip(toks, idf.tolist()))


def _tok_stats(q_toks, p_toks, idf):
    """(jaccard, idf-jaccard, idf share of q covered, idf share of p covered, max idf of a q token
    missing in p, max idf of a p token missing in q); NaN when either side has no tokens."""
    qs, ps = set(q_toks), set(p_toks)
    if not qs or not ps:
        return NAN, NAN, NAN, NAN, NAN, NAN
    sh, un = qs & ps, qs | ps
    w = lambda ts: sum(idf.get(t, 0.0) for t in ts)  # noqa: E731
    w_sh, w_un, w_q, w_p = w(sh), w(un), w(qs), w(ps)
    return (len(sh) / len(un),
            w_sh / w_un if w_un > 0 else NAN,
            w_sh / w_q if w_q > 0 else NAN,
            w_sh / w_p if w_p > 0 else NAN,
            max((idf.get(t, 0.0) for t in qs - ps), default=0.0),
            max((idf.get(t, 0.0) for t in ps - qs), default=0.0))


def _acronym(a, b):
    """1 if the initials of one multi-token name equal a token (or the concatenation) of the other."""
    for x, y in ((a, b), (b, a)):
        if len(x) >= 2:
            ini = "".join(t[0] for t in x)
            if ini in y or ini == "".join(y):
                return 1.0
    return 0.0


def _core(tokens, G):
    core = [t for t in tokens if t not in G]
    return sorted(core if core else tokens)


def _skeletons(tokens):
    return {k for k in (skeleton(t) for t in tokens if len(t) >= 2) if len(k) >= 2}


def _canon_num(t):
    """Digit token without leading zeros: pool addresses write 0412 / door no 0381 for 412 / 381."""
    return t.lstrip("0") or "0"


def _numbers(addr_toks):
    """(all digit tokens, digit tokens with >= _LONGNUM_MIN digits, first digit token or None), each
    without leading zeros."""
    digits = [_canon_num(t) for t in addr_toks if t.isdigit()]
    longnum = {t for t in digits if len(t) >= _LONGNUM_MIN}
    return set(digits), longnum, (digits[0] if digits else None)


def _num_street(toks):
    """(number, following word) pairs of an address: 1612 grandview avenue -> {("1612", "grandview")}."""
    return {(_canon_num(a), b) for a, b in zip(toks, toks[1:])
            if a.isdigit() and len(b) >= 2 and b.isalpha()}


def _char3(s):
    s = s.replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _digits_state(q_dig, p_dig):
    """Fellegi-Sunter-style 4-state code on stripped digit token sets: 3 match (equal sets), 2 overlap
    (both nonempty, share >=1 token, sets differ), 1 conflict (both nonempty, disjoint), 0 one-side
    (exactly one side has digits); NaN when neither side has any digit token (no information, not 0)."""
    if not q_dig and not p_dig:
        return NAN
    if not q_dig or not p_dig:
        return 0.0
    if q_dig == p_dig:
        return 3.0
    return 2.0 if q_dig & p_dig else 1.0


def _monge_elkan(qt, pt):
    """F3: Monge-Elkan on name tokens with rapidfuzz JaroWinkler.normalized_similarity as the inner sim:
    for each token of one side, take its best match on the other side, then average (Monge-Elkan is not
    symmetric); returns (mean of both directions, min of both directions). NaN if either side has no
    tokens."""
    if not qt or not pt:
        return NAN, NAN
    m = process.cdist(qt, pt, scorer=JaroWinkler.normalized_similarity)
    q_to_p = float(m.max(axis=1).mean())
    p_to_q = float(m.max(axis=0).mean())
    return (q_to_p + p_to_q) / 2.0, min(q_to_p, p_to_q)


def _phonetic_tokens(tokens):
    """ASCII alphabetic tokens of length >= 2 (jellyfish's metaphone/soundex are undefined on digits and
    are unreliable on very short or non-ASCII tokens)."""
    return [t for t in tokens if len(t) >= _PHONETIC_MIN_LEN and t.isascii() and t.isalpha()]


def _phonetic_jacc(qt, pt, code_fn):
    """F4: Jaccard of phonetic-code sets (Metaphone or Soundex, via jellyfish) over eligible name tokens.
    NaN if either side has no eligible token."""
    q_elig, p_elig = _phonetic_tokens(qt), _phonetic_tokens(pt)
    if not q_elig or not p_elig:
        return NAN
    qs = {code_fn(t) for t in q_elig}
    ps = {code_fn(t) for t in p_elig}
    return len(qs & ps) / len(qs | ps)


def _shared_maxidf(q_toks, p_toks, idf):
    """Largest idf among tokens both sides share (0 if none); NaN when either side has no tokens."""
    qs, ps = set(q_toks), set(p_toks)
    if not qs or not ps:
        return NAN
    return max((idf.get(t, 0.0) for t in qs & ps), default=0.0)


def pair_features(q, p, name_idf, addr_idf, G):
    """Feature values (PAIR_FEATURES order, without p_rerank/rank) for one S1 record `q` and one pool
    record `p`; both are dicts with name, addr, country (+ p: src, indic; q: chain)."""
    qn, pn, qa, pa_ = q["name"], p["name"], q["addr"], p["addr"]
    qt, pt, qat, pat = qn.split(), pn.split(), qa.split(), pa_.split()

    n_jacc, n_idf_jacc, n_qcov, n_pcov, n_qmiss, n_pmiss = _tok_stats(qt, pt, name_idf)
    shared = set(qt) & set(pt)
    q_sk, p_sk = _skeletons(qt), _skeletons(pt)
    sk_un = q_sk | p_sk
    name = [
        fuzz.token_set_ratio(qn, pn) / 100.0, fuzz.token_sort_ratio(qn, pn) / 100.0,
        fuzz.ratio(qn, pn) / 100.0, fuzz.partial_ratio(qn, pn) / 100.0,
        JaroWinkler.normalized_similarity("".join(sorted(qt)), "".join(sorted(pt))),
        n_jacc, n_idf_jacc, n_qcov, n_pcov, n_qmiss, n_pmiss,
        float(qn == pn), float(_core(qt, G) == _core(pt, G)), float("".join(qt) == "".join(pt)),
        _acronym(qt, pt), float(bool(qt) and bool(pt) and qt[0] == pt[0]),
        float(max((len(t) for t in shared), default=0)),
        len(q_sk & p_sk) / len(sk_un) if sk_un else NAN,
        float(len(qt)), float(len(pt)),
    ]

    both_addr = bool(qat) and bool(pat)
    _, a_idf_jacc, _, _, a_qmiss, a_pmiss = _tok_stats(qat, pat, addr_idf)
    q_dig, q_long, q_house = _numbers(qat)
    p_dig, p_long, p_house = _numbers(pat)
    if not q_long and not p_long:
        longnum = NAN            # no information
    elif not q_long or not p_long:
        longnum = 0.0            # one side only
    else:
        longnum = 2.0 if q_long & p_long else 1.0  # shared / conflict
    addr = [
        fuzz.token_set_ratio(qa, pa_) / 100.0 if both_addr else NAN,
        fuzz.token_sort_ratio(qa, pa_) / 100.0 if both_addr else NAN,
        a_idf_jacc, a_qmiss, a_pmiss,
        len(q_dig & p_dig) / len(q_dig | p_dig) if q_dig and p_dig else NAN,
        float(len(q_dig - p_dig)), float(len(p_dig - q_dig)),
        longnum,
        NAN if q_house is None or p_house is None else float(q_house == p_house),
        0.0 if both_addr else (1.0 if qat else (2.0 if pat else 3.0)),
    ]

    qc, pc = q["country"], p["country"]
    meta = [2.0 if not qc or not pc else float(qc == pc), float(p["src"]), float(p["indic"]),
            math.log1p(q["chain"])]

    if q_long and p_long:
        num_minlev = min(Levenshtein.distance(a, b) for a in q_long for b in p_long)
    else:
        num_minlev = NAN
    q_ns, p_ns = _num_street(qat), _num_street(pat)
    q3, p3 = (_char3(qa), _char3(pa_)) if both_addr else (set(), set())
    extra = [
        _shared_maxidf(qt, pt, name_idf), _shared_maxidf(qat, pat, addr_idf),
        fuzz.partial_ratio(qa, pa_) / 100.0 if both_addr else NAN,
        len(q3 & p3) / len(q3 | p3) if q3 and p3 else NAN,
        float(num_minlev),
        float(bool(q_ns & p_ns)) if q_ns and p_ns else NAN,
        q.get("name_cnt", NAN), p.get("name_cnt", NAN), q.get("addr_cnt", NAN), p.get("addr_cnt", NAN),
        _digits_state(q_dig, p_dig),
    ]

    me_mean, me_min = _monge_elkan(qt, pt)
    sim = [me_mean, me_min, _phonetic_jacc(qt, pt, jellyfish.metaphone), _phonetic_jacc(qt, pt, jellyfish.soundex)]
    return name + addr + meta + extra + sim


def list_features(X):
    """Fill LIST_FEATURES columns of one S1's rows in place (X rows = that S1's candidates)."""
    n = X.shape[0]
    c = _COL
    p = X[:, c["p_rerank"]].astype(np.float64)
    p_max = p.max()
    order = np.argsort(-p, kind="stable")
    nxt = np.zeros(n)
    nxt[order[:-1]] = p[order[1:]]
    ntset, atset = X[:, c["n_tset"]], X[:, c["a_tset"]]
    nidf = X[:, c["n_idf_jacc"]]
    src = X[:, c["src"]]
    X[:, c["l_n"]] = n
    X[:, c["l_p_max"]] = p_max
    X[:, c["l_p_gap_max"]] = p_max - p
    X[:, c["l_p_ratio_max"]] = p / p_max if p_max > 0 else NAN
    X[:, c["l_p_gap_next"]] = p - nxt
    X[:, c["l_p_sum"]] = p.sum()
    X[:, c["l_rank_ntset"]] = 1 + (ntset[None, :] > ntset[:, None]).sum(1)
    a = np.where(np.isnan(atset), -1.0, atset)
    X[:, c["l_rank_atset"]] = 1 + (a[None, :] > a[:, None]).sum(1)
    if n > 1:
        other = np.where(np.eye(n, dtype=bool), -np.inf, ntset[None, :]).max(1)
        X[:, c["l_ntset_max_other"]] = other
    else:
        X[:, c["l_ntset_max_other"]] = NAN
    X[:, c["l_nidf_gap_max"]] = (np.nanmax(nidf) - nidf) if np.any(~np.isnan(nidf)) else NAN
    X[:, c["l_n_name_close"]] = float((ntset >= _NAME_CLOSE).sum())
    X[:, c["l_n_longnum_shared"]] = float((X[:, c["a_longnum"]] == 2.0).sum())
    X[:, c["l_share_same_src"]] = (src[None, :] == src[:, None]).mean(1)
    X[:, c["l_n_addr_close"]] = float((a >= _ADDR_CLOSE).sum())


# ---------------------------------------------------------------- pool count tables (unsupervised)

COUNT_KINDS = ("name", "addr")


def _count_key(kind, text, G):
    """Key whose pool frequency is counted: core name tokens (edge tokens G removed, as `_core`) or the
    address token set; both sorted, so word-order transposition and component reordering share a key."""
    toks = text.split()
    if kind == "name":
        return " ".join(sorted(set(_core(toks, G))))
    return " ".join(sorted(set(toks)))


def _hash_keys(keys):
    return pd.util.hash_array(np.asarray(keys, dtype=object), categorize=False).astype(np.uint64)


def count_tables_dir(split):
    return io_utils.work_path("match", "aux", split, "counts.json").parent


def build_count_tables(split, lg=None):
    """work/match/aux/{split}/: per kind, sorted unique key hashes + counts over the split's pool
    (S2 + S3 text only, split.md rule 5) and the count of every pool record's own key. Rebuilt when the
    normalized pool file changes."""
    src = io_utils.work_path("norm", f"{split}_pool.parquet")
    st = src.stat()
    stamp = {"norm_pool_size": st.st_size, "norm_pool_mtime_ns": st.st_mtime_ns}
    d = count_tables_dir(split)
    done = d / "counts.json"
    if done.exists() and {k: io_utils.read_json(done).get(k) for k in stamp} == stamp:
        return d
    t0 = time.time()
    G = frozenset(io_utils.read_json(io_utils.work_path("index", split, "meta.json"))["G"])
    hashes = {k: [] for k in COUNT_KINDS}
    pf = pq.ParquetFile(src)
    for b in pf.iter_batches(batch_size=1_000_000, columns=["name_norm", "addr_norm"]):
        for kind, col in zip(COUNT_KINDS, (b.column(0), b.column(1))):
            hashes[kind].append(_hash_keys([_count_key(kind, s or "", G) for s in col.to_pylist()]))
    d.mkdir(parents=True, exist_ok=True)
    for kind in COUNT_KINDS:
        h = np.concatenate(hashes[kind])
        uniq, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
        np.save(d / f"{kind}_uniq.npy", uniq)
        np.save(d / f"{kind}_cnt.npy", cnt.astype(np.uint32))
        np.save(d / f"pool_{kind}_cnt.npy", cnt[inv].astype(np.uint32))
    io_utils.atomic_write_json(done, dict(stamp, n_pool=int(pf.metadata.num_rows),
                                          wall_s=round(time.time() - t0, 1)))
    if lg:
        lg.info("count tables %s built in %.0fs", split, time.time() - t0)
    return d


def query_counts(split, kind, texts):
    """log1p(pool count of each text's key); NaN for empty text."""
    d = count_tables_dir(split)
    G = frozenset(io_utils.read_json(io_utils.work_path("index", split, "meta.json"))["G"])
    uniq, cnt = np.load(d / f"{kind}_uniq.npy"), np.load(d / f"{kind}_cnt.npy")
    h = _hash_keys([_count_key(kind, s, G) for s in texts])
    pos = np.minimum(np.searchsorted(uniq, h), max(uniq.size - 1, 0))
    c = np.where(uniq[pos] == h, cnt[pos], 0).astype(np.float64)
    out = np.log1p(c)
    out[np.array([not s for s in texts], dtype=bool)] = NAN
    return out


# ---------------------------------------------------------------- TF-IDF char n-gram IDF (unsupervised)

def tfidf_dir(split):
    return io_utils.work_path("match", "aux", split, "tfidf_meta.json").parent


def _tfidf_hv():
    """HashingVectorizer shared by the IDF build and by every worker's per-chunk transform: char n-grams
    (word-boundary-padded) 2-4, hashed to TFIDF_N_FEATURES dims, raw (unweighted, unnormalized) counts."""
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=TFIDF_N_FEATURES,
                             alternate_sign=False, norm=None)


def build_tfidf_idf(split, lg=None):
    """F2: char n-gram document frequency, counted unsupervised over the split's pool (S2 + S3 text only,
    split.md rule 5), for name_norm and addr_norm separately. idf(feature) = ln((1 + N) / (1 + df)) + 1,
    N = pool record count, df = number of pool records whose hashed n-gram set contains that feature.
    Saved as work/match/aux/{split}/tfidf_{name,addr}_idf.npy (float32, length TFIDF_N_FEATURES) plus a
    stamp json; rebuilt when the normalized pool file changes (same staleness check as
    build_count_tables)."""
    src = io_utils.work_path("norm", f"{split}_pool.parquet")
    st = src.stat()
    stamp = {"norm_pool_size": st.st_size, "norm_pool_mtime_ns": st.st_mtime_ns, "n_features": TFIDF_N_FEATURES}
    d = tfidf_dir(split)
    done = d / "tfidf_meta.json"
    if done.exists() and {k: io_utils.read_json(done).get(k) for k in stamp} == stamp:
        return d
    t0 = time.time()
    hv = _tfidf_hv()
    pf = pq.ParquetFile(src)
    n_pool = pf.metadata.num_rows
    df = {kind: np.zeros(TFIDF_N_FEATURES, dtype=np.int64) for kind in ("name", "addr")}
    for b in pf.iter_batches(batch_size=200_000, columns=["name_norm", "addr_norm"]):
        for kind, col in zip(("name", "addr"), (b.column(0), b.column(1))):
            texts = [s or "" for s in col.to_pylist()]
            X = hv.transform(texts)
            X.data[:] = 1.0  # binarize term counts -> per-doc feature presence, so the row sum is df
            df[kind] += np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    d.mkdir(parents=True, exist_ok=True)
    for kind in ("name", "addr"):
        idf = (np.log((1.0 + n_pool) / (1.0 + df[kind])) + 1.0).astype(np.float32)
        np.save(d / f"tfidf_{kind}_idf.npy", idf)
    io_utils.atomic_write_json(done, dict(stamp, n_pool=int(n_pool), wall_s=round(time.time() - t0, 1)))
    if lg:
        lg.info("tfidf idf %s built in %.0fs (n_pool=%d)", split, time.time() - t0, n_pool)
    return d


def tfidf_pair_cosine(hv, idf, q_texts, p_texts, bounds):
    """Cosine similarity of the IDF-weighted hashed char n-gram vectors, one value per pool row of a
    _feature_chunk task. `q_texts` is one text per S1 group (len = len(bounds)); `p_texts` is one text per
    pool row of the whole chunk (len = sum of group sizes). Vectorizes the chunk's DISTINCT query and pool
    strings once each and computes each pair's cosine as a row-wise sparse dot product -- no S1 x pool
    dense matrix is built, per plan/FEATURE_EXPERIMENTS.md X1 F2. Zero-vector rows (e.g. empty text) get
    cosine 0.0 here; callers that need NaN for empty text (addresses) mask it in after the call."""
    from sklearn.preprocessing import normalize
    n_rows = len(p_texts)
    q_uniq, q_inv = np.unique(np.asarray(q_texts, dtype=object), return_inverse=True)
    p_uniq, p_inv = np.unique(np.asarray(p_texts, dtype=object), return_inverse=True)
    Xq = normalize(hv.transform(q_uniq.tolist()).multiply(idf).tocsr(), norm="l2", copy=False)
    Xp = normalize(hv.transform(p_uniq.tolist()).multiply(idf).tocsr(), norm="l2", copy=False)
    row_group = np.empty(n_rows, dtype=np.int64)
    for k, (a, b) in enumerate(bounds):
        row_group[a:b] = k
    Xq_full = Xq[q_inv[row_group]]
    Xp_full = Xp[p_inv]
    return np.asarray(Xq_full.multiply(Xp_full).sum(axis=1)).ravel().astype(np.float32)


def tfidf_addr_cosine(hv, idf, q_addrs, p_addr, bounds):
    """tfidf_pair_cosine for addresses, with the empty-address rule: NaN (not 0.0) wherever either side's
    address is empty, since an empty address carries no evidence either way (plan/FEATURE_EXPERIMENTS.md
    X1 F2)."""
    cos = tfidf_pair_cosine(hv, idf, q_addrs, p_addr, bounds)
    p_empty = np.array([not s for s in p_addr], dtype=bool)
    q_empty_row = np.zeros(len(p_addr), dtype=bool)
    for qaddr, (a, b) in zip(q_addrs, bounds):
        if not qaddr:
            q_empty_row[a:b] = True
    cos[q_empty_row | p_empty] = NAN
    return cos


# ---------------------------------------------------------------- feature build (workers)

_W = {}


def _init_worker(split):
    from ber.rerank import PoolTextReader  # heavy import only in workers that need it
    _W["ptr"] = PoolTextReader(split)
    _W["idf"] = TokenIdf(split)
    d = count_tables_dir(split)
    _W["pool_cnt"] = {k: np.load(d / f"pool_{k}_cnt.npy", mmap_mode="r") for k in COUNT_KINDS}
    _W["hv"] = _tfidf_hv()
    td = tfidf_dir(split)
    _W["tfidf_idf"] = {k: np.load(td / f"tfidf_{k}_idf.npy") for k in ("name", "addr")}


def _pool_cnt(kind, pool_idx, texts):
    c = np.log1p(np.asarray(_W["pool_cnt"][kind][pool_idx], dtype=np.float64))
    c[np.array([not s for s in texts], dtype=bool)] = NAN
    return c


def _feature_chunk(task):
    """task = (per-s1 (name, addr, country, chain, name_cnt, addr_cnt), pool_idx, p_rerank, rank, bounds)."""
    q_rows, pool_idx, p_rerank, rank, bounds = task
    ptr, idf = _W["ptr"], _W["idf"]
    p_name, p_addr, p_country = ptr.name(pool_idx), ptr.addr(pool_idx), ptr.country(pool_idx)
    p_src, p_indic = ptr.src(pool_idx), ptr.indic(pool_idx)
    p_ncnt, p_acnt = _pool_cnt("name", pool_idx, p_name), _pool_cnt("addr", pool_idx, p_addr)
    name_idf = idf.table("NT", [t for s in p_name for t in s.split()]
                         + [t for q in q_rows for t in q[0].split()])
    addr_idf = idf.table("AT", [t for s in p_addr for t in s.split()]
                         + [t for q in q_rows for t in q[1].split()])
    X = np.full((pool_idx.size, len(FEATURES)), np.nan, dtype=np.float32)
    X[:, _COL["p_rerank"]] = p_rerank
    X[:, _COL["rank"]] = rank
    lo_pair = _COL["n_tset"]
    hi_pair = _COL["n_tfidf_cos"]  # last two PAIR_FEATURES (tfidf cosines) are filled separately below
    for (qname, qaddr, qcountry, qchain, qncnt, qacnt), (a, b) in zip(q_rows, bounds):
        q = {"name": qname, "addr": qaddr, "country": qcountry, "chain": qchain,
             "name_cnt": qncnt, "addr_cnt": qacnt}
        for i in range(a, b):
            p = {"name": p_name[i], "addr": p_addr[i], "country": p_country[i],
                 "src": int(p_src[i]), "indic": bool(p_indic[i]),
                 "name_cnt": float(p_ncnt[i]), "addr_cnt": float(p_acnt[i])}
            X[i, lo_pair:hi_pair] = pair_features(q, p, name_idf, addr_idf, idf.G)
        list_features(X[a:b])

    # F2 TF-IDF char n-gram cosine (name, address): vectorized once per chunk over distinct strings, not
    # per pair and never as a dense S1 x pool matrix (plan/FEATURE_EXPERIMENTS.md X1).
    hv = _W["hv"]
    q_names = [q[0] for q in q_rows]
    q_addrs = [q[1] for q in q_rows]
    X[:, _COL["n_tfidf_cos"]] = tfidf_pair_cosine(hv, _W["tfidf_idf"]["name"], q_names, p_name, bounds)
    X[:, _COL["a_tfidf_cos"]] = tfidf_addr_cosine(hv, _W["tfidf_idf"]["addr"], q_addrs, p_addr, bounds)
    return X


def _s1_inputs(split, s1_idx):
    """name_norm, addr_norm, country_raw, chain for the given s1_idx."""
    norm = pd.read_parquet(io_utils.work_path("norm", f"{split}_s1.parquet"),
                           columns=["s1_idx", "name_norm", "addr_norm"])
    if not np.array_equal(norm["s1_idx"].to_numpy(), np.arange(len(norm))):
        raise RuntimeError(f"norm/{split}_s1.parquet: s1_idx is not 0..n-1 in row order")
    names = norm["name_norm"].to_numpy()[s1_idx]
    addrs = norm["addr_norm"].to_numpy()[s1_idx]
    del norm
    country = pd.read_parquet(io_utils.work_path("records", f"{split}_s1.parquet"),
                              columns=["country_raw"])["country_raw"].to_numpy()[s1_idx]
    chain = np.load(io_utils.work_path("norm", f"{split}_s1_chain.npy"), mmap_mode="r")[s1_idx]
    return names, addrs, country, np.asarray(chain)


def _tasks(cand, qpos, names, addrs, country, chain, ncnt, acnt, uniq, start, stop, chunk):
    for lo in range(0, uniq.size, chunk):
        hi = min(lo + chunk, uniq.size)
        a0, b0 = int(start[lo]), int(stop[hi - 1])
        rows = [(names[j], addrs[j], country[j], int(chain[j]), float(ncnt[j]), float(acnt[j]))
                for j in qpos[lo:hi]]
        bounds = list(zip((start[lo:hi] - a0).tolist(), (stop[lo:hi] - a0).tolist()))
        yield (rows, cand["pool_idx"][a0:b0], cand["p_rerank"][a0:b0], cand["rank"][a0:b0], bounds)


def run_match_features(args):
    t0 = time.time()
    set_name = args.set
    split = _set_split(set_name)
    lg = io_utils.get_logger(f"match-features_{set_name}")
    cfg, mcfg = io_utils.load_config(), load_match_config()
    s1_idx = set_queries(set_name)
    if args.limit is not None:
        s1_idx = s1_idx[:args.limit]
    if set_name == "TEST":
        io_utils.log_test_usage("matcher features over all test candidates (inputs only)",
                                "stage-3 inference (match-features --set TEST)",
                                str(io_utils.paths()["work_root"] / "candidates" / "TEST"))
    cand = read_candidates(set_name, s1_idx)
    names, addrs, country, chain = _s1_inputs(split, s1_idx)
    build_count_tables(split, lg)
    build_tfidf_idf(split, lg)
    ncnt, acnt = query_counts(split, "name", names), query_counts(split, "addr", addrs)
    uniq, start, stop = group_bounds(cand["s1_idx"])
    qpos = np.searchsorted(s1_idx, uniq)

    out_dir = features_dir(set_name)
    (out_dir / "meta.json").unlink(missing_ok=True)  # a crashed run must not leave a valid-looking set
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    workers = args.workers or io_utils.n_workers(cfg)
    tasks = _tasks(cand, qpos, names, addrs, country, chain, ncnt, acnt, uniq, start, stop,
                   mcfg["chunk_queries"])
    part_rows = mcfg["part_queries"] // mcfg["chunk_queries"]
    buf, n_part, row0, n_rows = [], 0, 0, 0
    with io_utils.spawn_pool(workers, _init_worker, (split,)) as pool:
        for X in pool.imap(_feature_chunk, tasks):
            buf.append(X)
            if len(buf) >= part_rows:
                row0 = _write_part(out_dir, n_part, cand, row0, buf)
                n_part, buf = n_part + 1, []
                lg.info("features %s: %d / %d rows (%.0fs)", set_name, row0, cand["s1_idx"].size,
                        time.time() - t0)
        if buf:
            row0 = _write_part(out_dir, n_part, cand, row0, buf)
            n_part += 1
    n_rows = row0
    meta = {"set": set_name, "split": split, "features": FEATURES, "n_queries": int(s1_idx.size),
            "n_rows": int(n_rows), "n_parts": n_part, "limited": args.limit is not None,
            "bucketing_fingerprint": bucketing_fingerprint(cfg), "match_config_version": mcfg["version"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    io_utils.atomic_write_json(out_dir / "meta.json", meta)
    payload = dict(meta, wall_s=round(time.time() - t0, 1), rows_per_s=round(n_rows / max(time.time() - t0, 1e-9)),
                   peak_rss_mb=round(io_utils.peak_rss_mb(), 1),
                   nan_share={f: float(v) for f, v in zip(FEATURES, _nan_share(out_dir))})
    io_utils.write_metrics(f"match_features_{set_name}", payload, int(s1_idx.size), 0, cfg)
    lg.info("match-features %s done: %d queries, %d rows, %.1fs", set_name, s1_idx.size, n_rows,
            time.time() - t0)


def _write_part(out_dir, k, cand, row0, buf):
    X = np.concatenate(buf)
    row1 = row0 + X.shape[0]
    cols = {"s1_idx": pa.array(cand["s1_idx"][row0:row1]), "pool_idx": pa.array(cand["pool_idx"][row0:row1])}
    cols.update({f: pa.array(X[:, i]) for i, f in enumerate(FEATURES)})
    path = out_dir / f"part-{k:05d}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    tmp.replace(path)
    return row1


def _nan_share(out_dir):
    tot, nan = 0, np.zeros(len(FEATURES))
    for p in sorted(out_dir.glob("part-*.parquet")):
        t = pq.read_table(p, columns=FEATURES)
        tot += t.num_rows
        nan += [t.column(f).null_count + int(np.isnan(t.column(f).to_numpy()).sum()) for f in FEATURES]
    return nan / max(tot, 1)


def load_features(set_name, expected_fingerprint=None):
    """(X float32, s1_idx, pool_idx, meta) of a feature set, rows sorted by (s1_idx, p_rerank rank)."""
    d = features_dir(set_name)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"no features for {set_name}; run `match-features --set {set_name}`")
    meta = io_utils.read_json(meta_path)
    if meta["features"] != FEATURES:
        raise RuntimeError(f"{meta_path}: feature list differs from this code; re-run match-features")
    if expected_fingerprint and meta["bucketing_fingerprint"] != expected_fingerprint:
        raise RuntimeError(f"{set_name} features were built from different bucketing outputs than the "
                           "trained matcher (bucketing_fingerprint mismatch); rebuild features / retrain")
    t = pa.concat_tables([pq.read_table(p) for p in sorted(d.glob("part-*.parquet"))])
    X = np.column_stack([t.column(f).to_numpy() for f in FEATURES]).astype(np.float32)
    return X, t.column("s1_idx").to_numpy().astype(np.int32), t.column("pool_idx").to_numpy(), meta


# ---------------------------------------------------------------- labels and scoring

def pair_labels(s1_idx, pool_idx):
    """1 if (s1, pool) is a train ground-truth link (train split only)."""
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"), columns=["s1_idx", "pool_idx"])
    key_gt = gt.column("s1_idx").to_numpy().astype(np.int64) << 32 | gt.column("pool_idx").to_numpy().astype(np.int64)
    key = s1_idx.astype(np.int64) << 32 | pool_idx.astype(np.int64)
    return np.isin(key, key_gt)


def true_counts(queries):
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"), columns=["s1_idx"])
    n = np.bincount(gt.column("s1_idx").to_numpy(), minlength=int(queries.max()) + 1 if queries.size else 1)
    return n[queries]


def entity_f05(n_pred, n_true, n_tp):
    """Per-entity F0.5 with the contest rule: singleton scores 1 iff the prediction is empty."""
    n_pred, n_true, n_tp = (np.asarray(a, dtype=np.float64) for a in (n_pred, n_true, n_tp))
    out = np.zeros(n_pred.shape)
    single = n_true == 0
    out[single] = (n_pred[single] == 0).astype(np.float64)
    m = ~single & (n_tp > 0)
    out[m] = 1.25 * n_tp[m] / (n_pred[m] + 0.25 * n_true[m])
    return out


class Scorer:
    """Entity F0.5 of row-level keep masks over ALL queries of a set (queries without candidates count)."""

    def __init__(self, queries, s1_idx, y, n_true):
        self.queries = queries
        self.qpos = np.minimum(np.searchsorted(queries, s1_idx), max(queries.size - 1, 0))
        if queries.size == 0 or not np.array_equal(queries[self.qpos], s1_idx):
            raise RuntimeError("candidate rows reference S1 outside the scored query set")
        self.y = y.astype(bool)
        self.n_true = n_true
        self.n_cand_pos = np.bincount(self.qpos, weights=self.y, minlength=queries.size)

    def per_entity(self, keep):
        n_pred = np.bincount(self.qpos, weights=keep, minlength=self.queries.size)
        n_tp = np.bincount(self.qpos, weights=keep & self.y, minlength=self.queries.size)
        return entity_f05(n_pred, self.n_true, n_tp), n_pred, n_tp

    def f05(self, keep):
        return float(self.per_entity(keep)[0].mean())

    def report(self, keep, p=None):
        f, n_pred, n_tp = self.per_entity(keep)
        loss = 1.0 - f
        nt, fp = self.n_true, n_pred - n_tp
        fn_in = self.n_cand_pos - n_tp
        fn_out = nt - self.n_cand_pos
        buckets = {
            "singleton_with_fp": (nt == 0) & (fp > 0),
            "matched_fp_only": (nt > 0) & (fp > 0) & (fn_in == 0),
            "matched_missed_only": (nt > 0) & (fp == 0) & (fn_in > 0),
            "matched_fp_and_missed": (nt > 0) & (fp > 0) & (fn_in > 0),
            "blocking_miss_only": (nt > 0) & (fp == 0) & (fn_in == 0) & (fn_out > 0),
        }
        single = nt == 0
        tp, pred, pos = float(n_tp.sum()), float(n_pred.sum()), float(self.y.sum())
        out = {
            "f05": float(f.mean()),
            "ceiling": float(entity_f05(self.n_cand_pos, nt, self.n_cand_pos).mean()),
            "n_queries": int(self.queries.size), "n_singletons": int(single.sum()),
            "singleton_accuracy": float((n_pred[single] == 0).mean()) if single.any() else None,
            "candidate_recall_retained": tp / pos if pos else None,
            "pairwise": {"precision": tp / pred if pred else None, "recall_in_candidates": tp / pos if pos else None,
                         "recall_all_true": tp / float(nt.sum()) if nt.sum() else None,
                         "n_predicted": int(pred)},
            "pred_size": {"mean": float(n_pred.mean()), "share_empty": float((n_pred == 0).mean())},
            "loss_total": float(loss.mean()),
            "loss_breakdown": {k: {"n_entities": int(m.sum()), "loss": float(loss[m].sum() / loss.size)}
                               for k, m in buckets.items()},
        }
        if p is not None:
            out["pairwise"].update(_ranking_metrics(self.y, p))
        return out, f


def _ranking_metrics(y, p):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    if y.all() or not y.any():
        return {}
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, pc))}


# ---------------------------------------------------------------- decision rules

def threshold_grid(dcfg):
    return np.round(np.arange(dcfg["threshold_start"], dcfg["threshold_stop"] + 1e-9,
                              dcfg["threshold_step"]), 4)


def expected_f05_table(P):
    """E[F0.5 of predicting the top-k] for k = 0..n, for rows of P (G, n) sorted descending, assuming
    independent Bernoulli(p) labels. Exact: Poisson-binomial distributions of TP (top-k) and of the rest.
    k = 0 scores P(no true match among the candidates)."""
    G, n = P.shape
    pre = [np.ones((G, 1))]
    for j in range(n):
        d, pj = pre[-1], P[:, j:j + 1]
        new = np.zeros((G, j + 2))
        new[:, :-1] += d * (1 - pj)
        new[:, 1:] += d * pj
        pre.append(new)
    suf = [None] * (n + 1)
    suf[n] = np.ones((G, 1))
    for j in range(n - 1, -1, -1):
        d, pj = suf[j + 1], P[:, j:j + 1]
        new = np.zeros((G, n - j + 1))
        new[:, :-1] += d * (1 - pj)
        new[:, 1:] += d * pj
        suf[j] = new
    E = np.zeros((G, n + 1))
    E[:, 0] = suf[0][:, 0]
    for k in range(1, n + 1):
        a = np.arange(k + 1, dtype=np.float64)[:, None]
        b = np.arange(n - k + 1, dtype=np.float64)[None, :]
        W = np.where(a > 0, 1.25 * a / (k + 0.25 * (a + b)), 0.0)
        E[:, k] = np.einsum("ga,ab,gb->g", pre[k], W, suf[k])
    return E


def decide_expected_f(s1_idx, p):
    """Keep mask choosing, per S1, the top-k (k may be 0) with the highest expected F0.5."""
    keep = np.zeros(p.size, dtype=bool)
    order = np.lexsort((-p, s1_idx))
    s_sorted, p_sorted = s1_idx[order], p[order].astype(np.float64)
    _, start, stop = group_bounds(s_sorted)
    lens = stop - start
    for L in np.unique(lens):
        g = np.flatnonzero(lens == L)
        idx = start[g][:, None] + np.arange(L)[None, :]
        k = expected_f05_table(p_sorted[idx]).argmax(1)  # first max -> smallest k on ties
        sel = np.arange(L)[None, :] < k[:, None]
        keep[order[idx[sel]]] = True
    return keep


def one_owner(pool_idx, p):
    """p with every row set to 0 whose pool record has a strictly higher p in another S1's list."""
    best = pd.Series(p).groupby(pool_idx).transform("max").to_numpy()
    return np.where(p >= best, p, 0.0).astype(p.dtype)


def apply_decision(decision, s1_idx, pool_idx, p_cal):
    p = one_owner(pool_idx, p_cal) if decision["one_owner"] else p_cal
    if decision["rule"] == "expected_f":
        return decide_expected_f(s1_idx, p)
    return p >= decision["threshold"]


# ---------------------------------------------------------------- model: train / predict

def _lgb_params(mcfg, threads):
    params = dict(mcfg["lgb"])
    params["num_threads"] = threads
    return params


def calibrate(p_raw, calib):
    return np.interp(p_raw, calib["x"], calib["y"]).astype(np.float32)


def paired_delta(fa, fb, n_boot=1000, seed=0):
    """Mean per-entity difference fa - fb over the same queries, with a 95% bootstrap CI (resampling S1
    rows) and the number of entities whose score changed."""
    d = np.asarray(fa, dtype=np.float64) - np.asarray(fb, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)]) if d.size else [0.0]
    return {"delta": float(d.mean()) if d.size else 0.0, "ci95": [float(np.percentile(boots, 2.5)),
                                                                   float(np.percentile(boots, 97.5))],
            "n_changed": int((d != 0).sum()), "n": int(d.size)}


def adopt(cmp, min_gain):
    """Adopt the alternative only if the paired gain reaches min_gain and its CI excludes zero."""
    return cmp["delta"] >= min_gain and cmp["ci95"][0] > 0


def runs_dir(run):
    return io_utils.work_path("match", "runs", run, "x").parent


def predict_raw(X, models_dir, n_folds):
    import lightgbm as lgb
    p = np.zeros(X.shape[0], dtype=np.float64)
    for k in range(n_folds):
        booster = lgb.Booster(model_file=str(models_dir / f"model_fold{k}.txt"))
        p += booster.predict(X)
    return (p / n_folds).astype(np.float32)


def run_match_train(args):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    t0 = time.time()
    lg = io_utils.get_logger("match-train")
    cfg, mcfg = io_utils.load_config(), load_match_config()
    fp = bucketing_fingerprint(cfg)
    set_name = mcfg["train_set"]
    if set_name != "MATCH_TRAIN":
        raise ValueError("match-train fits on MATCH_TRAIN only: MATCH_VALID confirms, HOLDOUT_TEST (val-S1) "
                         "is scoring-only (split.md rule 4), and other train-S1 slices trained stage 1/2")
    X_all, s1, pool, meta = load_features(set_name, fp)
    feats = used_features(mcfg)
    X = X_all[:, [_COL[f] for f in feats]]
    if meta["limited"]:
        lg.warning("training on a --limit feature set (%d queries): smoke run only", meta["n_queries"])
    y = pair_labels(s1, pool)
    queries = set_queries(set_name)
    if meta["limited"]:
        queries = queries[:meta["n_queries"]]
    threads = args.workers or io_utils.n_workers(cfg)
    run = getattr(args, "run", None)
    lg.info("match-train%s: %d rows, %d queries, %d features, positive rate %.4f",
            f" [{run}]" if run else "", y.size, queries.size, len(feats), y.mean())

    models_dir = io_utils.work_path("match", "models", "x").parent
    oof = np.zeros(y.size, dtype=np.float64)
    folds = []
    # Fold membership depends only on the set of S1 (shuffled unique groups), so two runs over the same
    # queries share folds even when their candidate lists differ: per-entity OOF scores are paired.
    gkf = GroupKFold(n_splits=mcfg["n_folds"], shuffle=True, random_state=mcfg["fold_seed"])
    gain = np.zeros(len(feats))
    for k, (tr, te) in enumerate(gkf.split(X, y, s1)):
        gss = GroupShuffleSplit(n_splits=1, test_size=mcfg["early_stop_frac"],
                                random_state=mcfg["fold_seed"] + k)
        fit_i, es_i = next(gss.split(tr, groups=s1[tr]))
        fit_i, es_i = tr[fit_i], tr[es_i]
        dfit = lgb.Dataset(X[fit_i], label=y[fit_i], feature_name=feats, free_raw_data=False)
        des = lgb.Dataset(X[es_i], label=y[es_i], reference=dfit, free_raw_data=False)
        booster = lgb.train(_lgb_params(mcfg, threads), dfit, num_boost_round=mcfg["num_boost_round"],
                            valid_sets=[des], valid_names=["early_stop"],
                            callbacks=[lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False),
                                       lgb.log_evaluation(period=0)])
        oof[te] = booster.predict(X[te], num_iteration=booster.best_iteration)
        path = models_dir / f"model_fold{k}.txt"
        booster.save_model(str(path.with_name(path.name + ".tmp")), num_iteration=booster.best_iteration)
        path.with_name(path.name + ".tmp").replace(path)
        gain += booster.feature_importance("gain", iteration=booster.best_iteration)
        folds.append({"fold": k, "n_fit_rows": int(fit_i.size), "n_early_stop_rows": int(es_i.size),
                      "n_oof_rows": int(te.size), "best_iteration": int(booster.best_iteration),
                      "early_stop_logloss": float(booster.best_score["early_stop"]["binary_logloss"])})
        lg.info("fold %d: best_iteration %d, early-stop logloss %.5f (%.0fs)", k,
                booster.best_iteration, folds[-1]["early_stop_logloss"], time.time() - t0)
        if booster.best_iteration >= mcfg["num_boost_round"]:
            lg.warning("fold %d hit num_boost_round without early stopping", k)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
    calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    p_cal = calibrate(oof, calib)

    scorer = Scorer(queries, s1, y, true_counts(queries))
    dcfg = mcfg["decision"]
    grid = threshold_grid(dcfg)
    p_rr = X_all[:, _COL["p_rerank"]]
    floor_sweep = {float(t): scorer.f05(p_rr >= t) for t in grid}
    floor_t = max(floor_sweep, key=floor_sweep.get)
    candidates = {}
    for owner in (False, True):
        pc = one_owner(pool, p_cal) if owner else p_cal
        sweep = {float(t): scorer.f05(pc >= t) for t in grid}
        t_best = max(sweep, key=sweep.get)
        dec = {"rule": "threshold", "threshold": t_best, "one_owner": owner}
        candidates[("threshold", owner)] = (dec, scorer.per_entity(apply_decision(dec, s1, pool, p_cal))[0],
                                            sweep)
        dec = {"rule": "expected_f", "threshold": None, "one_owner": owner}
        candidates[("expected_f", owner)] = (dec, scorer.per_entity(apply_decision(dec, s1, pool, p_cal))[0],
                                             None)
    # Paired comparisons on the same OOF queries; an alternative is adopted only when the gain reaches the
    # configured minimum and its 95% bootstrap CI excludes zero (no adoption on noise).
    rule_cmp = {"expected_f_vs_threshold": paired_delta(candidates[("expected_f", False)][1],
                                                        candidates[("threshold", False)][1])}
    base_rule = "expected_f" if adopt(rule_cmp["expected_f_vs_threshold"], dcfg["expected_f_min_gain"]) \
        else "threshold"
    rule_cmp["one_owner"] = paired_delta(candidates[(base_rule, True)][1], candidates[(base_rule, False)][1])
    use_owner = adopt(rule_cmp["one_owner"], dcfg["one_owner_min_gain"])
    decision, _, _ = candidates[(base_rule, use_owner)]

    keep = apply_decision(decision, s1, pool, p_cal)
    report, per_entity = scorer.report(keep, p_cal)
    oof_f05 = report["f05"]
    floor_report, _ = scorer.report(p_rr >= floor_t)
    lg.info("OOF entity F0.5 %.5f (rule %s, one_owner %s) vs re-ranker-score floor %.4f @ %.2f; rule "
            "comparisons %s", oof_f05, decision["rule"], decision["one_owner"], floor_sweep[floor_t], floor_t,
            rule_cmp)

    model_doc = {"features": FEATURES, "features_used": feats, "n_folds": mcfg["n_folds"],
                 "bucketing_fingerprint": fp, "run": run,
                 "train_set": set_name, "train_limited": meta["limited"], "calibration": calib,
                 "decision": decision, "floor_threshold": floor_t, "oof_f05": oof_f05,
                 "oof_floor_f05": floor_sweep[floor_t],
                 "match_config": mcfg, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    io_utils.atomic_write_json(models_dir / "matcher.json", model_doc)
    if run:
        rd = runs_dir(run)
        (rd / "models").mkdir(parents=True, exist_ok=True)
        for k in range(mcfg["n_folds"]):
            (rd / "models" / f"model_fold{k}.txt").write_bytes((models_dir / f"model_fold{k}.txt").read_bytes())
        io_utils.atomic_write_json(rd / "models" / "matcher.json", model_doc)
        np.save(rd / "queries.npy", queries)
        np.save(rd / "OOF_per_entity.npy", per_entity)
        pq.write_table(pa.table({"s1_idx": s1, "pool_idx": pool, "y": y, "p_raw": oof.astype(np.float32),
                                 "p_cal": p_cal}), rd / "oof.parquet", compression="zstd")

    err = export_errors(s1, pool, y, keep, p_cal, p_rr, X_all[:, _COL["rank"]], queries,
                        mcfg["error_export_max_rows"], lg)
    payload = {
        "set": set_name, "run": run, "limited": meta["limited"], "n_rows": int(y.size),
        "positive_rate": float(y.mean()), "features_used": feats,
        "folds": folds, "decision": decision, "oof": report,
        "oof_rule_scores": {f"{r}{'+one_owner' if o else ''}": float(v[1].mean())
                            for (r, o), v in candidates.items()},
        "rule_comparisons": rule_cmp,
        "oof_threshold_sweep": {str(t): v for t, v in candidates[("threshold", False)][2].items()},
        "floor": {"threshold": floor_t, "f05": floor_sweep[floor_t], "report": floor_report},
        "gain_vs_floor": oof_f05 - floor_sweep[floor_t],
        "calibration_reliability": reliability(p_cal, y),
        "feature_gain": dict(sorted(((f, float(g / mcfg["n_folds"])) for f, g in zip(feats, gain)),
                                    key=lambda kv: -kv[1])),
        "segments": segment_scores(queries, per_entity, meta["split"]),
        "error_files": err, "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(io_utils.peak_rss_mb(), 1),
    }
    io_utils.write_metrics(f"match_train_{run}" if run else "match_train", payload, queries.size,
                           int(scorer.n_true.sum()), cfg)
    lg.info("match-train done in %.0fs", time.time() - t0)


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        if m.any():
            out.append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "pos_rate": float(y[m].mean())})
    return out


def segment_scores(queries, per_entity, split):
    """Mean entity F0.5 per bucketing segment (chain, country, name unseen, match count); train only."""
    if split != "train":
        return {}
    from ber.evaluate import segment_data
    seg = segment_data(queries, np.zeros(0, dtype=np.uint32))["query"]
    return {k: {"n_queries": int(m.sum()), "f05": float(per_entity[m].mean()) if m.any() else None}
            for k, m in seg.items()}


# ---------------------------------------------------------------- error export (train OOF only)

def _pool_raw(split, pool_idx):
    """(name_raw, addr_raw, entity id) per pool_idx, streaming records/{split}_pool.parquet."""
    need = np.unique(pool_idx)
    name, addr, eid = np.empty(need.size, object), np.empty(need.size, object), np.empty(need.size, object)
    pf = pq.ParquetFile(io_utils.work_path("records", f"{split}_pool.parquet"))
    start = 0
    for b in pf.iter_batches(batch_size=500_000, columns=["name_raw", "addr_raw", "src", "num"]):
        stop = start + b.num_rows
        lo, hi = np.searchsorted(need, [start, stop])
        if hi > lo:
            loc = pa.array(need[lo:hi] - start)
            name[lo:hi] = b.column(0).take(loc).to_pylist()
            addr[lo:hi] = b.column(1).take(loc).to_pylist()
            eid[lo:hi] = [hashing.format_id(s, n) for s, n in
                          zip(b.column(2).take(loc).to_pylist(), b.column(3).take(loc).to_pylist())]
        start = stop
    j = np.searchsorted(need, pool_idx)
    return name[j], addr[j], eid[j]


def _error_frame(s1, pool, p_cal, p_rr, rank, split):
    rec = pd.read_parquet(io_utils.work_path("records", f"{split}_s1.parquet"),
                          columns=["entity_id", "name_raw", "addr_raw", "country_raw"])
    pn, pa_, pid = _pool_raw(split, pool)
    return pd.DataFrame({
        "Name_S1": rec["name_raw"].to_numpy()[s1], "Address_S1": rec["addr_raw"].to_numpy()[s1],
        "Name_Other": pn, "Address_Other": pa_,
        "source1_entity_id": rec["entity_id"].to_numpy()[s1], "other_entity_id": pid,
        "p_matcher": p_cal, "p_rerank": p_rr, "rerank_rank": rank,
        "country_S1": rec["country_raw"].to_numpy()[s1]})


def export_errors(s1, pool, y, keep, p_cal, p_rr, rank, queries, max_rows, lg):
    """OOF false positives / false negatives + blocking misses of MATCH_TRAIN (never val-S1 or test)."""
    out_dir = Path("reports/match_errors").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    fp_i = np.flatnonzero(keep & ~y)
    fn_i = np.flatnonzero(~keep & y)
    fp_i = fp_i[np.argsort(-p_cal[fp_i], kind="stable")][:max_rows]
    fn_i = fn_i[np.argsort(p_cal[fn_i], kind="stable")][:max_rows]
    for name, idx in (("train_false_positives.csv", fp_i), ("train_false_negatives.csv", fn_i)):
        df = _error_frame(s1[idx], pool[idx], p_cal[idx], p_rr[idx], rank[idx], "train")
        io_utils.atomic_write_text(out_dir / name, df.to_csv(index=False, lineterminator="\n"))
        files[name] = int(len(df))
    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet"), columns=["s1_idx", "pool_idx"])
    gs, gp = gt.column("s1_idx").to_numpy(), gt.column("pool_idx").to_numpy()
    m = np.isin(gs, queries)
    gs, gp = gs[m], gp[m]
    in_cand = np.isin(gs.astype(np.int64) << 32 | gp.astype(np.int64),
                      s1.astype(np.int64) << 32 | pool.astype(np.int64))
    miss = np.flatnonzero(~in_cand)[:max_rows]
    nanv = np.full(miss.size, np.nan)
    df = _error_frame(gs[miss], gp[miss], nanv, nanv, nanv, "train")
    io_utils.atomic_write_text(out_dir / "train_blocking_misses.csv", df.to_csv(index=False, lineterminator="\n"))
    files["train_blocking_misses.csv"] = int(len(df))
    files["n_false_positives_total"] = int((keep & ~y).sum())
    files["n_false_negatives_total"] = int((~keep & y).sum())
    files["n_blocking_misses_total"] = int((~in_cand).sum())
    lg.info("error export: %s", files)
    return files


# ---------------------------------------------------------------- eval / predict

def load_matcher(cfg):
    models_dir = io_utils.work_path("match", "models", "x").parent
    doc_path = models_dir / "matcher.json"
    if not doc_path.exists():
        raise RuntimeError("no trained matcher; run `match-train` first")
    doc = io_utils.read_json(doc_path)
    if doc["features"] != FEATURES:
        raise RuntimeError("matcher.json feature list differs from this code; re-run match-train")
    if doc["bucketing_fingerprint"] != bucketing_fingerprint(cfg):
        raise RuntimeError("bucketing outputs changed since match-train (fingerprint mismatch); retrain")
    return doc, models_dir


def score_set(set_name, doc, models_dir):
    X, s1, pool, meta = load_features(set_name, doc["bucketing_fingerprint"])
    cols = [_COL[f] for f in doc.get("features_used", FEATURES)]
    p_cal = calibrate(predict_raw(X[:, cols], models_dir, doc["n_folds"]), doc["calibration"])
    keep = apply_decision(doc["decision"], s1, pool, p_cal)
    return X, s1, pool, meta, p_cal, keep


def run_match_eval(args):
    t0 = time.time()
    set_name = args.set
    if set_name not in ("MATCH_VALID", "HOLDOUT_TEST"):
        raise ValueError("match-eval scores MATCH_VALID (confirmation) or HOLDOUT_TEST (val-S1, once)")
    lg = io_utils.get_logger(f"match-eval_{set_name}")
    cfg = io_utils.load_config()
    metrics_path = io_utils.paths()["metrics_root"] / f"match_eval_{set_name}.json"
    if set_name == "HOLDOUT_TEST" and metrics_path.exists() and not args.force:
        raise RuntimeError(f"{metrics_path} exists: val-S1 is scored once (split.md rule 4). "
                           "Use --force only if the matcher did not change since that run.")
    doc, models_dir = load_matcher(cfg)
    X, s1, pool, meta, p_cal, keep = score_set(set_name, doc, models_dir)
    queries = set_queries(set_name)
    if meta["limited"]:
        queries = queries[:meta["n_queries"]]
    y = pair_labels(s1, pool)  # scoring only
    scorer = Scorer(queries, s1, y, true_counts(queries))
    report, per_entity = scorer.report(keep, p_cal)
    floor_report, _ = scorer.report(X[:, _COL["p_rerank"]] >= doc["floor_threshold"])
    payload = {"set": set_name, "limited": meta["limited"], "decision": doc["decision"], "eval": report,
               "floor": {"threshold": doc["floor_threshold"], "f05": floor_report["f05"],
                         "note": "threshold chosen on MATCH_TRAIN, not on this set"},
               "gain_vs_floor": report["f05"] - floor_report["f05"],
               "oof_f05": doc["oof_f05"], "drop_vs_oof": doc["oof_f05"] - report["f05"],
               "oof_gain_vs_floor": doc["oof_f05"] - doc["oof_floor_f05"],
               "calibration_reliability": reliability(p_cal, y),
               "segments": segment_scores(queries, per_entity, "train"),
               "wall_s": round(time.time() - t0, 1)}
    if set_name == "MATCH_VALID" and payload["drop_vs_oof"] > doc["match_config"]["confirm_tolerance"]:
        payload["confirm_alarm"] = "MATCH_VALID F0.5 is more than confirm_tolerance below OOF; investigate"
        lg.warning(payload["confirm_alarm"])
    run = doc.get("run")
    payload["run"] = run
    name = f"match_eval_{set_name}"
    if run and set_name == "MATCH_VALID":  # HOLDOUT_TEST keeps one fixed file name (scored-once guard)
        name += f"_{run}"
        np.save(runs_dir(run) / "MATCH_VALID_per_entity.npy", per_entity)
    io_utils.write_metrics(name, payload, queries.size, int(scorer.n_true.sum()), cfg)
    lg.info("match-eval %s: F0.5 %.4f (floor %.4f, ceiling %.4f, OOF %.4f)", set_name, report["f05"],
            floor_report["f05"], report["ceiling"], doc["oof_f05"])


def run_match_compare(args):
    """Paired comparison of two `match-train --run` runs on the same queries (OOF of MATCH_TRAIN or
    MATCH_VALID): mean per-entity F0.5 delta b - a, 95% bootstrap CI, per-segment deltas."""
    cfg = io_utils.load_config()
    lg = io_utils.get_logger("match-compare")
    fa, fb = (np.load(runs_dir(r) / f"{args.set}_per_entity.npy") for r in (args.a, args.b))
    queries = set_queries("MATCH_TRAIN" if args.set == "OOF" else args.set)
    if not (fa.size == fb.size == queries.size):
        raise RuntimeError("runs were scored on different query sets")
    from ber.evaluate import segment_data
    masks = dict(segment_data(queries, np.zeros(0, dtype=np.uint32))["query"])
    masks["singleton"] = true_counts(queries) == 0
    payload = {"a": args.a, "b": args.b, "set": args.set, "f05_a": float(fa.mean()), "f05_b": float(fb.mean()),
               "b_minus_a": paired_delta(fb, fa, n_boot=2000),
               "segments": {k: dict(paired_delta(fb[m], fa[m], n_boot=500), f05_a=float(fa[m].mean()))
                            for k, m in masks.items() if m.any()}}
    io_utils.write_metrics(f"compare_{args.a}_vs_{args.b}_{args.set}", payload, queries.size, 0, cfg)
    lg.info("%s: %s %.5f -> %s %.5f, delta %s", args.set, args.a, fa.mean(), args.b, fb.mean(),
            payload["b_minus_a"])


def write_matching_file(path, s1_ids, s1_idx, pool_idx, src, num):
    """matching_results.tsv: header, one row per S1 in file order, kept IDs in the given row order."""
    path = Path(path)
    order = np.argsort(s1_idx, kind="stable")
    s1s, pools = s1_idx[order], pool_idx[order]
    uniq, start, stop = group_bounds(s1s)
    lists = {}
    for q, a, b in zip(uniq.tolist(), start.tolist(), stop.tolist()):
        ids = [hashing.format_id(src[p], num[p]) for p in pools[a:b].tolist()]
        lists[q] = ",".join(dict.fromkeys(ids))
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(MATCHING_HEADER)
        for i, eid in enumerate(s1_ids):
            f.write(f"{eid}\t{lists.get(i, '')}\n")
    tmp.replace(path)


def pair_keys(s1_pos, src, num):
    """uint64 key (s1 position << 34 | src << 32 | num) identifying an (S1 row, pool ID) pair."""
    return (np.asarray(s1_pos, dtype=np.uint64) << np.uint64(34)) | \
        (np.asarray(src, dtype=np.uint64) << np.uint64(32)) | np.asarray(num, dtype=np.uint64)


def check_matching_file(path, s1_ids, cand_keys):
    """Issues (empty = OK): header, one row per S1 in file order, IDs of form S2-/S3-<n>, no duplicate
    IDs, every ID a candidate of its S1 (`cand_keys` = sorted `pair_keys` of all candidate rows; the
    candidate file itself is validated against the test sources by validate_stream)."""
    issues = []
    rows_s1, rows_src, rows_num = [], [], []
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8")
        if header != MATCHING_HEADER:
            return [f"bad header {header!r}"]
        n = 0
        for i, raw in enumerate(f):
            line = raw.decode("utf-8")
            if not line.endswith("\n") or "\r" in line:
                issues.append(f"row {i + 2}: bad line ending")
            parts = line.rstrip("\n").split("\t")
            if i >= len(s1_ids) or len(parts) != 2 or parts[0] != s1_ids[i]:
                issues.append(f"row {i + 2}: expected S1 {s1_ids[i] if i < len(s1_ids) else '(none)'}")
                break
            ids = parts[1].split(",") if parts[1] else []
            if len(ids) != len(set(ids)):
                issues.append(f"row {i + 2}: duplicate IDs")
            for eid in ids:
                if len(eid) < 4 or eid[:3] not in ("S2-", "S3-") or not eid[3:].isdigit():
                    issues.append(f"row {i + 2}: bad ID {eid!r}")
                    continue
                rows_s1.append(i)
                rows_src.append(int(eid[1]))
                rows_num.append(int(eid[3:]))
            n = i + 1
            if len(issues) > 20:
                return issues
    if n != len(s1_ids):
        issues.append(f"{n} rows, expected {len(s1_ids)}")
    keys = pair_keys(rows_s1, rows_src, rows_num)
    if keys.size and not np.isin(keys, cand_keys).all():
        issues.append(f"{int((~np.isin(keys, cand_keys)).sum())} matched IDs are not candidates of their S1")
    return issues


def run_match_predict(args):
    t0 = time.time()
    lg = io_utils.get_logger("match-predict")
    cfg = io_utils.load_config()
    doc, models_dir = load_matcher(cfg)
    if doc["train_limited"]:
        raise RuntimeError("the matcher was trained on a --limit smoke feature set; retrain on full MATCH_TRAIN")
    io_utils.log_test_usage("matcher inference over test candidates (inputs only)",
                            "stage-3 output (match-predict)", str(features_dir("TEST")))
    X, s1, pool, meta, p_cal, keep = score_set("TEST", doc, models_dir)
    if meta["limited"]:
        raise RuntimeError("TEST features were built with --limit; rebuild on all test S1")
    rec = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["s1_idx", "entity_id"])
    if not np.array_equal(rec["s1_idx"].to_numpy(), np.arange(len(rec))):
        raise RuntimeError("records/test_s1.parquet: s1_idx is not 0..n-1 in file order")
    s1_ids = rec["entity_id"].tolist()
    store = io_utils.paths()["work_root"] / "norm" / "test_pool_store"
    src = np.load(store / "src.npy", mmap_mode="r")
    num = np.load(store / "num.npy", mmap_mode="r")
    out_path = io_utils.work_path("outputs", "matching_results.tsv")
    k_order = np.flatnonzero(keep)
    k_order = k_order[np.lexsort((-p_cal[k_order], s1[k_order]))]
    write_matching_file(out_path, s1_ids, s1[k_order], pool[k_order], src, num)

    cand_keys = np.sort(pair_keys(s1, np.asarray(src)[pool], np.asarray(num)[pool]))
    issues = check_matching_file(out_path, s1_ids, cand_keys)
    lg.info("matching_results.tsv check: %s", "PASS" if not issues else issues)

    n_pred = np.bincount(s1, weights=keep, minlength=len(s1_ids))
    country = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"),
                              columns=["country_raw"])["country_raw"].to_numpy()
    max_p = np.zeros(len(s1_ids))
    np.maximum.at(max_p, s1, p_cal)
    per_country = {c: {"n_s1": int((country == c).sum()),
                       "mean_pred_size": float(n_pred[country == c].mean()),
                       "share_empty": float((n_pred[country == c] == 0).mean()),
                       "mean_max_p": float(max_p[country == c].mean())}
                   for c in sorted(set(country.tolist()))}
    payload = {"decision": doc["decision"], "rows": len(s1_ids), "pairs": int(keep.sum()),
               "per_country": per_country, "file": str(out_path), "file_bytes": out_path.stat().st_size,
               "check": {"passed": not issues, "issues": issues}, "wall_s": round(time.time() - t0, 1)}
    io_utils.write_metrics("match_predict_TEST", payload, len(s1_ids), 0, cfg)
    if issues:
        raise RuntimeError("matching_results.tsv failed checks:\n" + "\n".join(issues))
    lg.info("wrote %s: %d S1, %d matched pairs (%.0fs)", out_path, len(s1_ids), int(keep.sum()),
            time.time() - t0)

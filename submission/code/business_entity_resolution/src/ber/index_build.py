"""M5 index build and index-sanity (plan §5.6).

Per split: G and the full AT df first, then per family: spawn workers over 250,000-row chunks produce
(key64, pool_idx) pairs, the main process appends them to 64 shard files by key64 >> 58, then each shard
is reduced on its own (lexsort, unique, df, build-limit drop) and the shards are concatenated in shard
order into {F}.keys.npy / {F}.indptr.npy / {F}.post.npy (+ {F}.dropped.npy). A family is skipped and
resumed from disk only if its files verify AND the input norm pool parquet has not been rewritten since
(size + mtime_ns fingerprint stored in meta.json) -- otherwise every family is rebuilt, since a changed
normalization (new dictionary/abbreviation map) changes every family's keys, not just the ones that
would fail structural verification.
"""

import logging
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber import hashing, io_utils, keys
from ber.index_store import IndexStore

CHUNK = 250_000
HASH_BATCH = 1_000_000
N_SHARDS = 64
SHARD_SHIFT = np.uint64(58)
VERIFY_BLOCK = 1 << 24
SANITY_N = 1000
SANITY_SEED = 1
FILES = ("keys", "indptr", "post", "dropped")

log = logging.getLogger("ber.build-index")


def split_families(split, cfg):
    """train: every family E2 may evaluate (keys.FAMILIES + SK); test: the E2-kept list."""
    return list(keys.FAMILIES) + ["SK"] if split == "train" else list(cfg["families"])


# ---------------------------------------------------------------- workers

_W = {}


def _init_worker(norm_pool_path):
    pf = pq.ParquetFile(norm_pool_path)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    _W.update(pf=pf, rg_starts=np.concatenate(([0], np.cumsum(sizes))))


def _read_rows(start, stop, columns):
    """Rows [start, stop) of the norm pool parquet as {column: python list}."""
    starts = _W["rg_starts"]
    g0 = int(np.searchsorted(starts, start, side="right")) - 1
    g1 = int(np.searchsorted(starts, stop, side="left"))
    t = _W["pf"].read_row_groups(list(range(g0, g1)), columns=columns)
    t = t.slice(start - int(starts[g0]), stop - start)
    return {c: t.column(c).to_pylist() for c in columns}


def _chunk_at_df(task):
    """Distinct-address-token counts of one chunk."""
    start, stop = task
    c = Counter()
    for a in _read_rows(start, stop, ["addr_norm"])["addr_norm"]:
        c.update(set(a.split()))
    return c, os.getpid(), io_utils.peak_rss_mb()


def _chunk_pairs(task):
    """(key64, pool_idx) pairs of one chunk for family F, grouped by shard; plus shard counts."""
    F, start, stop, G, at_high, at_cap = task
    rows = _read_rows(start, stop, ["pool_idx", "name_norm", "addr_norm"])
    buf, parts, counts = [], [], np.zeros(stop - start, dtype=np.int64)
    for i, (nm, ad) in enumerate(zip(rows["name_norm"], rows["addr_norm"])):
        ks = keys.record_keys(nm, ad, (F,), G, at_high, at_cap, "pool")[F]
        buf.extend(ks)
        counts[i] = len(ks)
        if len(buf) >= HASH_BATCH:  # bound the number of live key strings per worker
            parts.append(hashing.keys64(F, buf))
            buf = []
    parts.append(hashing.keys64(F, buf))
    k64 = np.concatenate(parts)
    del parts
    post = np.repeat(np.asarray(rows["pool_idx"], dtype=np.uint32), counts)
    shard = (k64 >> SHARD_SHIFT).astype(np.intp)
    order = np.argsort(shard, kind="stable")
    return (k64[order], post[order], np.bincount(shard, minlength=N_SHARDS), os.getpid(),
            io_utils.peak_rss_mb())


# ---------------------------------------------------------------- shard reduce and assembly

def _reduce_shard(tmp, s, limit):
    """Shard s: lexsort, unique (key, pool_idx), df, drop df > limit; write red_{s}.* files."""
    kpath, ppath = tmp / f"shard_{s:02d}.k.bin", tmp / f"shard_{s:02d}.p.bin"
    k = np.fromfile(kpath, dtype=np.uint64)
    p = np.fromfile(ppath, dtype=np.uint32)
    order = np.lexsort((p, k))
    k, p = k[order], p[order]
    del order
    if k.size:
        keep = np.ones(k.size, dtype=bool)
        keep[1:] = (k[1:] != k[:-1]) | (p[1:] != p[:-1])
        k, p = k[keep], p[keep]
    starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1]))) if k.size else np.zeros(0, np.intp)
    uk = k[starts]
    cnt = np.diff(np.append(starts, k.size)).astype(np.int64)
    del k
    big = cnt > limit
    p = p[np.repeat(~big, cnt)]
    np.save(tmp / f"red_{s:02d}.k.npy", uk[~big])
    np.save(tmp / f"red_{s:02d}.c.npy", cnt[~big])
    np.save(tmp / f"red_{s:02d}.p.npy", p)
    np.save(tmp / f"red_{s:02d}.d.npy", uk[big])
    os.remove(kpath)
    os.remove(ppath)


def _assemble(tmp, out_dir, F):
    """Concatenate the reduced shards in shard order (= key order) into the final .npy files."""
    def red(s, x):
        return tmp / f"red_{s:02d}.{x}.npy"

    n_keys = sum(np.load(red(s, "k"), mmap_mode="r").size for s in range(N_SHARDS))
    n_post = sum(np.load(red(s, "p"), mmap_mode="r").size for s in range(N_SHARDS))
    final = {x: out_dir / f"{F}.{x}.npy" for x in FILES}
    tmps = {x: p.with_name(p.name + ".tmp") for x, p in final.items()}
    K = np.lib.format.open_memmap(tmps["keys"], mode="w+", dtype=np.uint64, shape=(n_keys,))
    I = np.lib.format.open_memmap(tmps["indptr"], mode="w+", dtype=np.int64, shape=(n_keys + 1,))
    P = np.lib.format.open_memmap(tmps["post"], mode="w+", dtype=np.uint32, shape=(n_post,))
    I[0] = 0
    ko = po = 0
    dropped = []
    for s in range(N_SHARDS):
        k, c, p = np.load(red(s, "k")), np.load(red(s, "c")), np.load(red(s, "p"))
        K[ko:ko + k.size] = k
        I[ko + 1:ko + 1 + k.size] = po + np.cumsum(c)
        P[po:po + p.size] = p
        ko, po = ko + k.size, po + p.size
        dropped.append(np.load(red(s, "d")))
    for m in (K, I, P):
        m.flush()
    del K, I, P
    io_utils.atomic_save_npy(final["dropped"], np.concatenate(dropped).astype(np.uint64))
    for x in ("keys", "indptr", "post"):
        os.replace(tmps[x], final[x])
    for s in range(N_SHARDS):
        for x in ("k", "c", "p", "d"):
            os.remove(red(s, x))
    os.rmdir(tmp)


def _build_family(pool, F, out_dir, chunks, G, at_high, at_cap, limit, rss):
    tmp = out_dir / f"_tmp_{F}"
    tmp.mkdir(exist_ok=True)
    kf = [open(tmp / f"shard_{s:02d}.k.bin", "wb") for s in range(N_SHARDS)]
    pf = [open(tmp / f"shard_{s:02d}.p.bin", "wb") for s in range(N_SHARDS)]
    try:
        tasks = [(F, a, b, G, at_high, at_cap) for a, b in chunks]
        for k64, post, cnt, pid, mb in pool.imap(_chunk_pairs, tasks):
            rss[pid] = max(rss.get(pid, 0.0), mb)
            off = np.concatenate(([0], np.cumsum(cnt)))
            for s in np.flatnonzero(cnt):
                k64[off[s]:off[s + 1]].tofile(kf[s])
                post[off[s]:off[s + 1]].tofile(pf[s])
    finally:
        for f in kf + pf:
            f.close()
    for s in range(N_SHARDS):
        _reduce_shard(tmp, s, limit)
    _assemble(tmp, out_dir, F)


# ---------------------------------------------------------------- verification and stats

def check_family(out_dir, F, n_pool):
    """None if the family's files exist and verify, else a short reason."""
    paths = {x: Path(out_dir) / f"{F}.{x}.npy" for x in FILES}
    if not all(p.exists() for p in paths.values()):
        return "missing files"
    try:
        k, ip, p, d = (np.load(paths[x], mmap_mode="r") for x in FILES)
    except (ValueError, OSError, EOFError) as e:
        return f"unreadable: {e}"
    if (k.dtype, ip.dtype, p.dtype, d.dtype) != (np.uint64, np.int64, np.uint32, np.uint64):
        return "dtypes"
    if ip.size != k.size + 1 or ip[0] != 0 or ip[-1] != p.size:
        return "indptr bounds"
    for a in range(0, k.size, VERIFY_BLOCK):
        blk = np.asarray(k[a:a + VERIFY_BLOCK + 1])
        if np.any(blk[1:] <= blk[:-1]):
            return "keys not strictly increasing"
    for a in range(0, ip.size, VERIFY_BLOCK):
        if np.any(np.diff(ip[a:a + VERIFY_BLOCK + 1]) < 1):
            return "indptr not monotone"
    for a in range(0, p.size, VERIFY_BLOCK):
        seg = np.asarray(p[a:a + VERIFY_BLOCK + 1])
        if seg.size and int(seg.max()) >= n_pool:
            return "posting >= n_pool"
        j = np.flatnonzero(seg[1:] <= seg[:-1]) + a + 1
        pos = np.minimum(np.searchsorted(ip, j), ip.size - 1)
        if np.any(ip[pos] != j):
            return "postings not ascending per key"
    if d.size and (np.any(d[1:] <= d[:-1]) or
                   (k.size and np.any(k[np.minimum(np.searchsorted(k, d), k.size - 1)] == d))):
        return "dropped keys unsorted or indexed"
    return None


def family_stats(out_dir, F):
    """n_keys, n_postings, n_dropped, df quantiles (inverted CDF over kept keys), bytes."""
    k, ip, p, d = (np.load(Path(out_dir) / f"{F}.{x}.npy", mmap_mode="r") for x in FILES)
    hists = [np.bincount(np.diff(ip[a:a + VERIFY_BLOCK + 1])) for a in range(0, k.size, VERIFY_BLOCK)]
    hist = np.zeros(max((h.size for h in hists), default=1), dtype=np.int64)
    for h in hists:
        hist[:h.size] += h
    cum = np.cumsum(hist)
    q = {name: int(np.searchsorted(cum, math.ceil(frac * k.size))) if k.size else 0
         for name, frac in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))}
    q["max"] = int(np.flatnonzero(hist)[-1]) if k.size else 0
    size = sum((Path(out_dir) / f"{F}.{x}.npy").stat().st_size for x in FILES)
    return {"n_keys": int(k.size), "n_postings": int(p.size), "n_dropped": int(d.size),
            "df_quantiles": q, "bytes": int(size)}


# ---------------------------------------------------------------- split build

def _pool_fingerprint(path):
    """Cheap "was this file rewritten" check: size + mtime_ns. atomic_write_parquet always replaces the
    file (never edits in place), so any re-normalize changes this even if the new content happens to be
    logically identical."""
    st = Path(path).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _iter_names(norm_pool_path):
    for b in pq.ParquetFile(norm_pool_path).iter_batches(columns=["name_norm"], batch_size=CHUNK):
        yield from b.column(0).to_pylist()


def build_split_index(norm_pool_path, out_dir, families, cfg, n_workers):
    """Build (or resume) the index of one split under out_dir; returns the meta dict (also meta.json)."""
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pool = pq.ParquetFile(norm_pool_path).metadata.num_rows
    pidx = pq.read_table(norm_pool_path, columns=["pool_idx"]).column(0).to_numpy()
    if not np.array_equal(pidx, np.arange(n_pool, dtype=np.uint32)):
        raise ValueError("norm pool: pool_idx is not 0..n_pool-1 in file order")
    del pidx
    et = cfg["edge_tokens"]
    G = keys.edge_token_set(_iter_names(norm_pool_path), n_pool, et["min_frac"], et["min_edge_share"])
    at_cap = math.floor(cfg["cap_frac"]["AT"] * n_pool)
    limit = math.floor(cfg["build_df_max_frac"] * n_pool)
    chunks = [(a, min(a + CHUNK, n_pool)) for a in range(0, n_pool, CHUNK)]
    meta_path = out_dir / "meta.json"
    fp = _pool_fingerprint(norm_pool_path)
    old_doc = io_utils.read_json(meta_path) if meta_path.exists() else {}
    fresh = old_doc.get("norm_input_fingerprint") == fp
    old = old_doc.get("families", {}) if fresh else {}
    if old_doc and not fresh:
        log.info("norm pool changed since the last build (fingerprint mismatch): rebuilding all families")
    rss = {}
    meta = {"n_pool": n_pool, "G": G, "at_cap": at_cap, "build_df_limit": limit,
            "chunk": CHUNK, "n_shards": N_SHARDS, "norm_input_fingerprint": fp, "families": {}}
    with io_utils.spawn_pool(n_workers, _init_worker, (str(norm_pool_path),)) as pool:
        at_df = Counter()
        for c, pid, mb in pool.imap_unordered(_chunk_at_df, chunks):
            at_df.update(c)
            rss[pid] = max(rss.get(pid, 0.0), mb)
        # Full AT df; xk_keys only compares df with at_cap, so the tokens above at_cap carry all of it
        # (absent -> 0 <= at_cap, exactly like every token with df <= at_cap).
        at_high = {t: c for t, c in at_df.items() if c > at_cap}
        meta.update(at_distinct_tokens=len(at_df), at_tokens_above_cap=len(at_high),
                    g_at_wall_s=round(time.time() - t0, 1))
        del at_df
        Gs = frozenset(G)
        for F in families:
            tf = time.time()
            if fresh and check_family(out_dir, F, n_pool) is None:
                log.info("%s: files verify, skipped", F)
                wall = old.get(F, {}).get("wall_s")
            else:
                _build_family(pool, F, out_dir, chunks, Gs, at_high, at_cap, limit, rss)
                bad = check_family(out_dir, F, n_pool)
                if bad:
                    raise RuntimeError(f"index {F}: verification failed: {bad}")
                wall = round(time.time() - tf, 1)
            meta["families"][F] = dict(family_stats(out_dir, F), wall_s=wall)
            meta["peak_rss_mb"] = {"main": round(io_utils.peak_rss_mb(), 1),
                                   "workers_max": round(max(rss.values(), default=0.0), 1),
                                   "workers_sum": round(sum(rss.values()), 1)}
            io_utils.atomic_write_json(meta_path, meta)
            log.info("%s: %s", F, meta["families"][F])
    meta["wall_s"] = round(time.time() - t0, 1)
    io_utils.atomic_write_json(meta_path, meta)
    return meta


def run_build_index(args):
    lg = io_utils.get_logger("build-index")
    cfg = io_utils.load_config()
    norm = io_utils.work_path("norm", f"{args.split}_pool.parquet")
    out_dir = io_utils.work_path("index", args.split, "meta.json").parent
    if args.split == "test":
        io_utils.log_test_usage("G, AT df and key index of the test pool",
                                "R4: unsupervised statistics + test-pool index (build-index)", norm)
    families = split_families(args.split, cfg)
    meta = build_split_index(norm, out_dir, families, cfg, io_utils.n_workers(cfg))
    io_utils.write_metrics(f"index_{args.split}", dict(meta, n_G=len(meta["G"])), 0, 0, cfg)
    lg.info("build-index %s done in %.1f s", args.split, meta["wall_s"])


# ---------------------------------------------------------------- sanity

def index_sanity(norm_pool_path, store):
    """Sampled pool records must be in their own EN postings and in the postings of each NT key
    (keys over the build limit are exempt)."""
    n = min(SANITY_N, store.n_pool)
    idx = np.sort(np.random.default_rng(SANITY_SEED).choice(store.n_pool, n, replace=False))
    names = pq.read_table(norm_pool_path, columns=["name_norm"]).column(0).take(idx).to_pylist()
    counts = {F: {"checked": 0, "found": 0, "exempt": 0} for F in ("EN", "NT")}
    for i, nm in zip(idx, names):
        nk = keys.name_keys(nm, store.G, "pool")
        for F in ("EN", "NT"):
            k64 = hashing.keys64(F, nk[F])
            df = store.df(F, k64)
            for d, post in zip(df, store.postings(F, k64)):
                c = counts[F]
                if d == store.sentinel:
                    c["exempt"] += 1
                    continue
                c["checked"] += 1
                j = np.searchsorted(post, i)
                c["found"] += int(j < post.size and post[j] == i)
    checked = sum(c["checked"] for c in counts.values())
    found = sum(c["found"] for c in counts.values())
    return {"n_records": int(n), "seed": SANITY_SEED, "per_family": counts, "n_checked": checked,
            "n_found": found, "share": found / checked if checked else 1.0}


def run_index_sanity(args):
    lg = io_utils.get_logger("index-sanity")
    cfg = io_utils.load_config()
    norm = io_utils.work_path("norm", f"{args.split}_pool.parquet")
    if args.split == "test":
        io_utils.log_test_usage("1,000 sampled test-pool names", "R4: index-sanity of the test index", norm)
    res = index_sanity(norm, IndexStore(args.split, ("EN", "NT")))
    io_utils.write_metrics(f"index_sanity_{args.split}", res, res["n_records"], 0, cfg)
    lg.info("index-sanity %s: %d/%d = %.6f", args.split, res["n_found"], res["n_checked"], res["share"])
    if res["share"] < 1.0:
        raise RuntimeError(f"index-sanity {args.split}: share {res['share']:.6f} < 1.0")

"""M1 ingest (plan §5.1): raw TSV -> records parquet (+ train GT), with the F1-F5 / F14 asserts."""

import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber import io_utils

CHUNK = 1_000_000
HEADER = ["entity_id", "business_name", "business_address", "country"]
GT_HEADER = ["source1_entity_id", "matched_entity_ids"]
ID_RE = re.compile(r"S([123])-([0-9]+)")

# Facts F5 and F14 (plan §1); run_ingest always asserts them.
EXPECTED_COUNTS = {
    "train": {"s1": 2_206_821, "s2": 5_034_616, "s3": 5_285_603,
              "gt_links": 7_638_365, "gt_singletons": 123_247, "gt_max_nmatch": 11},
    "test": {"s1": 1_732_544, "s2": 4_887_273, "s3": 5_082_316},
}

S1_SCHEMA = pa.schema([("s1_idx", pa.int32()), ("entity_id", pa.string()),
                       ("name_raw", pa.string()), ("addr_raw", pa.string()),
                       ("country_raw", pa.string())])
POOL_SCHEMA = pa.schema([("pool_idx", pa.uint32()), ("src", pa.int8()), ("num", pa.uint32()),
                         ("name_raw", pa.string()), ("addr_raw", pa.string()),
                         ("country_raw", pa.string())])


class IngestError(ValueError):
    """A data fact the pipeline relies on does not hold."""


def _check(cond, msg):
    if not cond:
        raise IngestError(msg)


def _source_path(data_dir, split, source):
    name = f"{split}_ground_truth.tsv" if source == "gt" else f"{split}_source{source}.tsv"
    return Path(data_dir) / split / name


def _parse_id(eid, srcs, where):
    """'S{src}-{num}' -> (src, num); src must be in `srcs`; exact round-trip, num < 2^32."""
    m = ID_RE.fullmatch(eid)
    _check(m is not None, f"{where}: bad id {eid!r}")
    src, num = int(m.group(1)), int(m.group(2))
    _check(src in srcs, f"{where}: id {eid!r} has wrong source prefix")
    _check(num < 2**32 and f"S{src}-{num}" == eid, f"{where}: id {eid!r} does not round-trip")
    return src, num


def _rows(path, header):
    """Yield (line_no, fields) of a TSV parsed line by line (F1, F2: no quoting reader)."""
    with open(path, encoding="utf-8", newline="") as f:
        first = f.readline().rstrip("\n").split("\t")
        _check(first == header, f"{path}: header {first} != {header}")
        for i, line in enumerate(f, start=2):
            yield i, line.rstrip("\n").split("\t")


def _record_chunks(path, src):
    """Validated chunks of a source file: (ids, nums uint32, names, addrs, countries)."""
    ids, nums, names, addrs, countries = [], [], [], [], []
    for i, fields in _rows(path, HEADER):
        _check(len(fields) == 4, f"{path}:{i}: {len(fields)} fields")
        eid, name, addr, country = fields
        nums.append(_parse_id(eid, (src,), f"{path}:{i}")[1])
        ids.append(eid)
        names.append(name)
        addrs.append(addr)
        countries.append(country)
        if len(ids) == CHUNK:
            yield ids, np.array(nums, dtype=np.uint32), names, addrs, countries
            ids, nums, names, addrs, countries = [], [], [], [], []
    if ids:
        yield ids, np.array(nums, dtype=np.uint32), names, addrs, countries


def _empties(names, addrs, countries, acc):
    acc["name"] += sum(1 for x in names if not x)
    acc["addr"] += sum(1 for x in addrs if not x)
    acc["country"] += sum(1 for x in countries if not x)


def _write_s1(path_in, out_path):
    """Stream S1 into a .tmp records parquet; returns (tmp path, nums uint32, stats)."""
    tmp = out_path.with_name(out_path.name + ".tmp")
    empty = {"name": 0, "addr": 0, "country": 0}
    all_nums, n = [], 0
    with pq.ParquetWriter(tmp, S1_SCHEMA, compression="zstd") as w:
        for ids, nums, names, addrs, countries in _record_chunks(path_in, 1):
            idx = np.arange(n, n + len(ids), dtype=np.int32)
            w.write_table(pa.table([idx, ids, names, addrs, countries], schema=S1_SCHEMA))
            _empties(names, addrs, countries, empty)
            all_nums.append(nums)
            n += len(ids)
    nums = np.concatenate(all_nums) if all_nums else np.zeros(0, np.uint32)
    _check(np.unique(nums).size == n, f"{path_in}: duplicate S1 ids")
    return tmp, nums, {"rows": n, "empty": empty}


def _write_pool(paths_in, out_path):
    """Stream S2 then S3 into one .tmp pool parquet; returns (tmp path, (src<<32|num) keys, stats)."""
    tmp = out_path.with_name(out_path.name + ".tmp")
    keys, stats, n = [], {}, 0
    with pq.ParquetWriter(tmp, POOL_SCHEMA, compression="zstd") as w:
        for src, path_in in paths_in:
            empty = {"name": 0, "addr": 0, "country": 0}
            file_nums, start = [], n
            for ids, nums, names, addrs, countries in _record_chunks(path_in, src):
                idx = np.arange(n, n + len(ids), dtype=np.uint32)
                srcs = np.full(len(ids), src, dtype=np.int8)
                w.write_table(pa.table([idx, srcs, nums, names, addrs, countries],
                                       schema=POOL_SCHEMA))
                _empties(names, addrs, countries, empty)
                file_nums.append(nums)
                n += len(ids)
            nums = np.concatenate(file_nums) if file_nums else np.zeros(0, np.uint32)
            _check(np.unique(nums).size == nums.size, f"{path_in}: duplicate (src, num)")
            keys.append((np.uint64(src) << np.uint64(32)) | nums.astype(np.uint64))
            stats[f"s{src}"] = {"rows": n - start, "empty": empty}
    keys = np.concatenate(keys) if keys else np.zeros(0, np.uint64)
    return tmp, keys, stats


def _lookup(sorted_keys, order, query, what):
    """Positions of `query` in the key set (via its sort order); raises on unknown keys."""
    pos = np.searchsorted(sorted_keys, query)
    ok = pos < sorted_keys.size
    ok[ok] = sorted_keys[pos[ok]] == query[ok]
    _check(bool(ok.all()), f"ground truth references {int((~ok).sum())} unknown {what} ids")
    return order[pos]


def _ingest_gt(path_in, s1_nums, pool_keys):
    """Parse the train GT -> (s1_idx int32, pool_idx uint32) links sorted, and nmatch int16."""
    s1_order = np.argsort(s1_nums, kind="stable")
    s1_sorted = s1_nums[s1_order]
    pool_order = np.argsort(pool_keys, kind="stable").astype(np.uint32)
    pool_sorted = pool_keys[pool_order]
    gt_s1, link_s1, link_pool = [], [], []
    rows_s1, l_s1, l_key = [], [], []

    def flush():
        q = np.array(rows_s1, dtype=np.uint32)
        gt_s1.append(_lookup(s1_sorted, s1_order, q, "S1").astype(np.int32))
        ls1 = _lookup(s1_sorted, s1_order, np.array(l_s1, dtype=np.uint32), "S1")
        link_s1.append(ls1.astype(np.int32))
        link_pool.append(_lookup(pool_sorted, pool_order, np.array(l_key, dtype=np.uint64), "pool"))
        rows_s1.clear(), l_s1.clear(), l_key.clear()

    for i, fields in _rows(path_in, GT_HEADER):
        where = f"{path_in}:{i}"
        _check(len(fields) == 2, f"{where}: {len(fields)} fields")
        num = _parse_id(fields[0], (1,), where)[1]
        rows_s1.append(num)
        for eid in (fields[1].split(",") if fields[1] else []):
            src, pnum = _parse_id(eid, (2, 3), where)
            l_s1.append(num)
            l_key.append((src << 32) | pnum)
        if len(rows_s1) == CHUNK:
            flush()
    if rows_s1:
        flush()
    cat = (lambda parts, dt: np.concatenate(parts) if parts else np.zeros(0, dt))
    gt_s1 = cat(gt_s1, np.int32)
    link_s1, link_pool = cat(link_s1, np.int32), cat(link_pool, np.uint32)
    _check(gt_s1.size == s1_nums.size and np.unique(gt_s1).size == s1_nums.size,
           "ground truth must have exactly one row per S1")
    _check(np.unique(link_pool).size == link_pool.size, "a pool record has 2+ owners")
    order = np.lexsort((link_pool, link_s1))
    link_s1, link_pool = link_s1[order], link_pool[order]
    nmatch = np.bincount(link_s1, minlength=s1_nums.size).astype(np.int16)
    return link_s1, link_pool, nmatch


def _check_expected(stats, expected):
    for key, want in expected.items():
        got = stats[key]
        _check(got == want, f"{key}: {got} != expected {want}")


def ingest_split(data_dir, split, out_dir, expected_counts=None):
    """Build records/{split}_s1|pool.parquet (+ train GT) under out_dir; return stats.

    expected_counts: {key: count} asserted against the stats (None skips; tests only).
    """
    rec = Path(out_dir) / "records"
    rec.mkdir(parents=True, exist_ok=True)
    s1_out, pool_out = rec / f"{split}_s1.parquet", rec / f"{split}_pool.parquet"
    tmps = []
    try:
        s1_tmp, s1_nums, s1_stats = _write_s1(_source_path(data_dir, split, 1), s1_out)
        tmps.append(s1_tmp)
        pool_tmp, pool_keys, pool_stats = _write_pool(
            [(s, _source_path(data_dir, split, s)) for s in (2, 3)], pool_out)
        tmps.append(pool_tmp)
        stats = {"s1": s1_stats["rows"], "s2": pool_stats["s2"]["rows"],
                 "s3": pool_stats["s3"]["rows"],
                 "files": {"s1": s1_stats, **pool_stats}}
        stats["pool"] = stats["s2"] + stats["s3"]
        if split == "train":
            link_s1, link_pool, nmatch = _ingest_gt(_source_path(data_dir, split, "gt"),
                                                    s1_nums, pool_keys)
            stats.update(gt_links=int(link_pool.size),
                         gt_singletons=int((nmatch == 0).sum()),
                         gt_max_nmatch=int(nmatch.max()) if nmatch.size else 0,
                         gt_orphan_pool=int(pool_keys.size - link_pool.size))
        if expected_counts is not None:
            _check_expected(stats, expected_counts)
    except BaseException:
        for t in tmps:
            t.unlink(missing_ok=True)
        raise
    os.replace(s1_tmp, s1_out)
    os.replace(pool_tmp, pool_out)
    if split == "train":
        gt = pa.table([link_s1, link_pool], names=["s1_idx", "pool_idx"])
        gt_tmp = rec / "train_gt.parquet.tmp"
        pq.write_table(gt, gt_tmp, compression="zstd")
        os.replace(gt_tmp, rec / "train_gt.parquet")
        io_utils.atomic_save_npy(rec / "train_s1_nmatch.npy", nmatch)
    return stats


def run_ingest(args):
    """CLI `ingest --split {train,test}` (plan §5.1, §7)."""
    log = io_utils.get_logger(f"ingest_{args.split}")
    t0 = time.time()
    data_dir = io_utils.paths()["data_root"]
    if args.split == "test":
        for s in (1, 2, 3):
            io_utils.log_test_usage("read raw records", "build test records parquet (M1 ingest)",
                                    _source_path(data_dir, "test", s))
    stats = ingest_split(data_dir, args.split, io_utils.paths()["work_root"],
                         EXPECTED_COUNTS[args.split])
    stats.update(wall_time_s=round(time.time() - t0, 1), peak_rss_mb=round(io_utils.peak_rss_mb(), 1))
    path = io_utils.write_metrics(f"ingest_{args.split}", stats, n_queries=stats["s1"],
                                  n_true_pairs=stats.get("gt_links", 0))
    log.info("ingest %s done: %s", args.split, path)

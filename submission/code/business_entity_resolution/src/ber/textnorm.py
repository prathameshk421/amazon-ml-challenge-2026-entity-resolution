"""Text normalization v2.1 (plan §5.2, §5.5.1 skeleton): verbatim port of plan/reference/translit_ref.py
plus step 11b (mined per-country address abbreviations, §5.2.2); `run_normalize` writes the §4 norm files."""

import collections
import json
import os
import re
import sys
import time
import unicodedata

try:
    import resource  # POSIX only
except ImportError:
    resource = None

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber import io_utils
from ber.translit import romanize


FOLD = {"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "ø": "o", "Ø": "o", "ł": "l",
        "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "þ": "th", "ı": "i", "ħ": "h"}
NAME_ABBREV = {"corp": "corporation", "inc": "incorporated", "ltd": "limited", "pvt": "private",
               "co": "company", "svc": "services", "assoc": "associates", "bros": "brothers",
               "shri": "sri", "shree": "sri"}
# v2.1: "st"/"ste" are ambiguous across languages (street/saint, suite/sainte), so they and their
# expansions map to one canonical class token instead of being expanded.
ADDR_ABBREV = {"rd": "road", "st": "st", "street": "st", "saint": "st", "ave": "avenue",
               "avenu": "avenue", "dr": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard",
               "ste": "ste", "suite": "ste", "sainte": "ste", "apt": "apartment", "hwy": "highway",
               "hway": "highway", "pkwy": "parkway"}


def collapse_single_letters(toks):
    """Join runs of >= 2 consecutive single-letter alphabetic tokens: s a s -> sas, i b m -> ibm."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if t:
            out.append(t)
    return out


_NULL = re.compile(r"<NULL>")
_PH = re.compile(r"<[A-Z_]+>")
_APOS = re.compile("['’‘ʼ`]")
_LD = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")


def normalize_field(text, kind, tdict=None, abbrev_map=None):
    """Plan §5.2 steps 1-14 -> (norm, had_indic). kind: "name" | "addr".

    tdict: learned per-token dictionary applied when had_indic (step 13).
    abbrev_map: mined {short: long} of the record's country (step 11b, addr only; tokens that are keys
    of ADDR_ABBREV are mapped by step 11 instead).
    """
    t = unicodedata.normalize("NFC", text)
    t = _NULL.sub(" ", t)
    t = _PH.sub(" cityname ", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Cf")
    t, had = romanize(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = "".join(FOLD.get(c, c) for c in t)
    t = t.casefold()
    t = t.replace("&", " and ")
    t = _APOS.sub("", t)
    t = "".join(c if c.isalnum() else " " for c in t)
    if kind == "addr":
        t = _LD.sub(" ", t)
    if kind == "addr" and abbrev_map:
        toks = [ADDR_ABBREV[w] if w in ADDR_ABBREV else abbrev_map.get(w, w) for w in t.split()]
    else:
        table = NAME_ABBREV if kind == "name" else ADDR_ABBREV
        toks = [table.get(w, w) for w in t.split()]
    if kind == "name":
        toks = collapse_single_letters(toks)
    if had and tdict:
        toks = [tdict.get(w, w) for w in toks]
    return " ".join(toks), had


_SK_MAP = {**{c: "P" for c in "pbfvw"}, **{c: "T" for c in "td"}, **{c: "K" for c in "kgc"},
           **{c: "S" for c in "szj"}, "l": "L", "r": "R", "m": "M", "n": "N"}


def skeleton(tok):
    """Consonant skeleton of an ASCII token (plan §5.5.1); "" for non-ASCII."""
    if not tok.isascii():
        return ""
    t = tok.replace("x", "ks").replace("ph", "f").replace("ch", "s").replace("sh", "s").replace("q", "k")
    out = []
    for c in t:
        if c.isdigit():
            s = c
        else:
            s = _SK_MAP.get(c, "")
        if s and (not out or out[-1] != s):
            out.append(s)
    return "".join(out)


# ---------------------------------------------------------------- `normalize` command (plan §5.2, §7)

CHUNK = 100_000
FIELDS = ("name", "addr")


def _norm_schema(idx_col, idx_type):
    return pa.schema([(idx_col, idx_type), ("name_norm", pa.string()), ("addr_norm", pa.string()),
                      ("name_indic", pa.bool_()), ("addr_indic", pa.bool_())])


S1_SCHEMA = _norm_schema("s1_idx", pa.int32())
POOL_SCHEMA = _norm_schema("pool_idx", pa.uint32())


def normalize_records(names, addrs, countries, tdict, abbrev_maps):
    """Normalize one chunk of records -> (name_norm, addr_norm, name_indic, addr_indic) lists.

    tdict: {"name": {r: l}, "addr": {r: l}} or {} (no dictionary, step 13 off).
    abbrev_maps: {country_label: {short: long}} (step 11b, per record country) or None (off).
    """
    tn, ta = tdict.get("name"), tdict.get("addr")
    nn, an, ni, ai = [], [], [], []
    for name, addr, country in zip(names, addrs, countries):
        n, hn = normalize_field(name, "name", tdict=tn)
        amap = abbrev_maps.get(country) if abbrev_maps else None
        a, ha = normalize_field(addr, "addr", tdict=ta, abbrev_map=amap)
        nn.append(n)
        an.append(a)
        ni.append(hn)
        ai.append(ha)
    return nn, an, ni, ai


_WORKER = {}


def _init_worker(tdict, abbrev_maps):
    _WORKER["tdict"], _WORKER["abbrev_maps"] = tdict, abbrev_maps


def _normalize_batch(batch, schema):
    """Worker task: raw RecordBatch (idx, name_raw, addr_raw, country_raw) -> norm RecordBatch."""
    cols = normalize_records(batch.column(1).to_pylist(), batch.column(2).to_pylist(),
                             batch.column(3).to_pylist(), _WORKER["tdict"], _WORKER["abbrev_maps"])
    return pa.record_batch([batch.column(0)] + [pa.array(c, type=t.type) for c, t in
                                                 zip(cols, list(schema)[1:])], schema=schema)


def _string_parts(arr):
    """(utf-8 bytes view, int32 offsets) of a pyarrow StringArray without Python objects."""
    off = np.frombuffer(arr.buffers()[1], dtype=np.int32)[arr.offset:arr.offset + len(arr) + 1]
    data = arr.buffers()[2]
    view = memoryview(data)[off[0]:off[-1]] if data is not None else memoryview(b"")
    return view, off


class _PoolStore:
    """Streams name/addr_norm into norm/{split}_pool_store/{field}.bin with int64 offsets (§4)."""

    def __init__(self, store_dir):
        self.dir = store_dir
        store_dir.mkdir(parents=True, exist_ok=True)
        self.fh = {f: open(store_dir / f"{f}.bin.tmp", "wb") for f in FIELDS}
        self.lens = {f: [] for f in FIELDS}
        self.indic = []

    def add(self, batch):
        for f in FIELDS:
            view, off = _string_parts(batch.column(f"{f}_norm"))
            self.fh[f].write(view)
            self.lens[f].append(np.diff(off).astype(np.int64))
        self.indic.append(batch.column("name_indic").to_numpy(zero_copy_only=False))

    def close(self):
        for f in FIELDS:
            self.fh[f].close()
            lens = np.concatenate(self.lens[f])
            off = np.zeros(lens.size + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            tmp = self.dir / f"{f}.bin.tmp"
            if os.path.getsize(tmp) != off[-1]:
                raise RuntimeError(f"{tmp}: size {os.path.getsize(tmp)} != offsets end {off[-1]}")
            os.replace(tmp, self.dir / f"{f}.bin")
            io_utils.atomic_save_npy(self.dir / f"{f}.off.npy", off)
        io_utils.atomic_save_npy(self.dir / "indic.npy", np.concatenate(self.indic))


class _Stats:
    """Row, Indic and empty-norm counts per field."""

    def __init__(self):
        self.d = {"rows": 0, **{f"{f}_indic": 0 for f in FIELDS}, **{f"{f}_empty": 0 for f in FIELDS}}

    def add(self, batch):
        self.d["rows"] += batch.num_rows
        for f in FIELDS:
            self.d[f"{f}_indic"] += int(pc.sum(batch.column(f"{f}_indic")).as_py() or 0)
            empty = pc.equal(pc.binary_length(batch.column(f"{f}_norm")), 0)
            self.d[f"{f}_empty"] += int(pc.sum(empty).as_py() or 0)


def _normalize_file(pool, n_workers, src, out, schema, on_batch):
    """Normalize records parquet `src` over the worker pool into `out` (input order kept)."""
    idx_col, idx_type = schema[0].name, schema[0].type
    cols = [idx_col, "name_raw", "addr_raw", "country_raw"]
    pending, n = collections.deque(), 0
    tmp = out.with_name(out.name + ".tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        def drain(keep):
            nonlocal n
            while len(pending) > keep:
                b = pending.popleft().get()
                idx = b.column(0).to_numpy()
                if not np.array_equal(idx, np.arange(n, n + b.num_rows, dtype=idx.dtype)):
                    raise RuntimeError(f"{src}: {idx_col} is not 0..n-1 in file order")
                n += b.num_rows
                w.write_batch(b)
                on_batch(b)

        for batch in pq.ParquetFile(src).iter_batches(batch_size=CHUNK, columns=cols):
            if batch.schema.field(0).type != idx_type:
                raise RuntimeError(f"{src}: {idx_col} has type {batch.schema.field(0).type}")
            pending.append(pool.apply_async(_normalize_batch, (batch, schema)))
            drain(2 * n_workers)
        drain(0)
    os.replace(tmp, out)
    return n


def s1_chain(name_norm):
    """int32[n]: number of records with the same name_norm, including itself (§4 chain)."""
    enc = pc.dictionary_encode(name_norm.combine_chunks())
    codes = enc.indices.to_numpy()
    return np.bincount(codes, minlength=len(enc.dictionary))[codes].astype(np.int32)


def _load_tdict(no_dict):
    if no_dict:
        return {}
    path = io_utils.work_path("translit", "dict_v1.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run learn-dict first or pass --no-dict")
    doc = io_utils.read_json(path)
    if doc["meta"].get("gate_pass") is not True:
        raise RuntimeError(f"{path}: the §5.3 dictionary gate did not pass; refusing to apply it")
    return {"name": doc["name"], "addr": doc["addr"]}


def _load_abbrev(split, cfg, no_abbrev):
    if no_abbrev or not cfg["abbrev_mining"]["enabled"]:
        return None
    path = io_utils.work_path("norm", f"abbrev_{split}.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run mine-abbrev --split {split} or pass --no-abbrev")
    return io_utils.read_json(path)["maps"]


def _children_peak_rss_mb():
    if resource is None:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def run_normalize(args):
    """normalize --split [--no-dict] [--no-abbrev]: norm parquet, S1 chain sizes, pool store, meta."""
    t0 = time.time()
    cfg = io_utils.load_config()
    log = io_utils.get_logger(f"normalize_{args.split}")
    split = args.split
    tdict = _load_tdict(args.no_dict)
    abbrev_maps = _load_abbrev(split, cfg, args.no_abbrev)
    W = io_utils.n_workers(cfg)
    rec = {k: io_utils.work_path("records", f"{split}_{k}.parquet") for k in ("s1", "pool")}
    out = {k: io_utils.work_path("norm", f"{split}_{k}.parquet") for k in ("s1", "pool")}
    store = _PoolStore(io_utils.paths()["work_root"] / "norm" / f"{split}_pool_store")
    stats = {"s1": _Stats(), "pool": _Stats()}
    s1_names = []

    def on_s1(b):
        stats["s1"].add(b)
        s1_names.append(b.column("name_norm"))

    def on_pool(b):
        stats["pool"].add(b)
        store.add(b)

    log.info("normalize %s: dict=%s abbrev=%s workers=%d", split, bool(tdict), abbrev_maps is not None, W)
    pool = io_utils.spawn_pool(W, _init_worker, (tdict, abbrev_maps))
    try:
        n_s1 = _normalize_file(pool, W, rec["s1"], out["s1"], S1_SCHEMA, on_s1)
        log.info("S1 done: %d rows", n_s1)
        n_pool = _normalize_file(pool, W, rec["pool"], out["pool"], POOL_SCHEMA, on_pool)
        log.info("pool done: %d rows", n_pool)
        pool.close()
        pool.join()
    finally:
        pool.terminate()
    store.close()
    io_utils.atomic_save_npy(io_utils.work_path("norm", f"{split}_s1_chain.npy"),
                             s1_chain(pa.chunked_array(s1_names, type=pa.string())))
    del s1_names

    meta_cols = pq.read_table(rec["pool"], columns=["src", "num", "country_raw"])
    countries = sorted(pc.unique(meta_cols["country_raw"]).to_pylist())
    if len(countries) > np.iinfo(np.int16).max:
        raise RuntimeError(f"{len(countries)} country labels do not fit int16")
    codes = pc.index_in(meta_cols["country_raw"], value_set=pa.array(countries, pa.string()))
    io_utils.atomic_save_npy(store.dir / "country.npy", codes.to_numpy().astype(np.int16))
    io_utils.atomic_write_json(store.dir / "countries.json", countries)
    io_utils.atomic_save_npy(store.dir / "src.npy", meta_cols["src"].to_numpy().astype(np.int8))
    io_utils.atomic_save_npy(store.dir / "num.npy", meta_cols["num"].to_numpy().astype(np.uint32))
    del meta_cols, codes

    meta = {"dict": bool(tdict), "abbrev": abbrev_maps is not None, "config_hash": io_utils.config_hash(cfg),
            "n_s1": n_s1, "n_pool": n_pool}
    io_utils.atomic_write_json(io_utils.work_path("norm", f"{split}_meta.json"), meta)
    payload = {"split": split, **meta, "s1": stats["s1"].d, "pool": stats["pool"].d, "n_workers": W,
               "n_countries": len(countries), "wall_time_s": round(time.time() - t0, 1),
               "peak_rss_mb": round(io_utils.peak_rss_mb(), 1),
               "peak_rss_mb_worker": round(_children_peak_rss_mb(), 1)}
    io_utils.write_metrics(f"normalize_{split}", payload, n_s1, 0, cfg)
    if split == "test":
        io_utils.log_test_usage("normalized test S1 and pool text, S1 name chain sizes, pool string store",
                                "plan §5.2 / §4 inputs for index build and inference (R4)",
                                f"{rec['s1']}, {rec['pool']}")
    log.info("normalize %s done in %.1fs: %s", split, time.time() - t0, payload)

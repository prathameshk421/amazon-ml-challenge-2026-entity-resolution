"""Reverse name retrieval for name-only (empty-address) pool candidates: among ALL S1 records of the split (same country),
where does this candidate's S1 rank by name similarity to the pool record's name? (Laptop, e5-small on MPS.)

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/rev_name.py embed_s1 train|test
  ... rev_name.py feats <SET> [<SET> ...]      # MATCH_TRAIN, MATCH_VALID, DEV_REST_REST, TEST

Why: 77% of in-list true pairs the matcher rejects have an empty pool address, and 97% of those S1 already have an
accepted true match (logs/fnlist_examples.log). For a name-only pool record the open question is which S1 owns it;
exact-name counts (nfreq) miss the altered names these copies carry. Inputs only (no labels).

Features per candidate row (NaN when the pool address is not empty):
  rv_cos      cos(pool name, this S1 name)
  rv_rank     rank of this S1 among all same-country S1 by cos to the pool name (1 = nearest; 21 if outside top 20)
  rv_top      best cos of any S1
  rv_gap      rv_top - rv_cos
  rv_n_close  number of S1 in the top 20 with cos >= rv_cos - 0.02 (competing owners)
  rv_margin   rv_cos - best cos of any OTHER S1
Writes work/retrieval/rev_<SET>.parquet (s1_idx, pool_idx, features) in feature-row order.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ber import io_utils, match

MODEL = "intfloat/multilingual-e5-small"
R = Path("work/retrieval")
TOPK = 20
_M = {}


def model():
    if not _M:
        from transformers import AutoModel, AutoTokenizer
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _M.update(dev=dev, tok=AutoTokenizer.from_pretrained(MODEL),
                  m=AutoModel.from_pretrained(MODEL).to(dev).eval())
    return _M


@torch.inference_mode()
def embed(texts, tag, batch=512):
    M = model()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((len(texts), 384), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        b = M["tok"]([texts[j] for j in idx], padding=True, truncation=True, max_length=32, return_tensors="pt").to(M["dev"])
        h = M["m"](**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        out[idx] = e.cpu().numpy().astype(np.float16)
        if (i // batch) % 400 == 0:
            print(f"[{tag}] {min(i + batch, len(texts)):,}/{len(texts):,} ({(i + batch) / (time.time() - t0):.0f}/s)", flush=True)
    return out


def names(tbl):
    return ("query: " + tbl.name_raw.fillna("").astype(str).str.strip()).tolist()


def embed_s1(split):
    t = pq.read_table(f"work/records/{split}_s1.parquet", columns=["s1_idx", "name_raw", "country_raw"]).to_pandas()
    assert np.array_equal(t.s1_idx.to_numpy(), np.arange(len(t)))
    np.save(R / f"rev_{split}_s1_emb.npy", embed(names(t), f"{split} S1"))
    t[["s1_idx", "country_raw"]].to_parquet(R / f"rev_{split}_s1_country.parquet")


def feats(set_name):
    split = "test" if set_name == "TEST" else "train"
    cfg = io_utils.load_config()
    X, s1, pool, _ = match.load_features(set_name, match.bucketing_fingerprint(cfg))
    pt = pq.read_table(f"work/records/{split}_pool.parquet", columns=["pool_idx", "name_raw", "addr_raw", "country_raw"]).to_pandas()
    empty = pt.addr_raw.fillna("").astype(str).str.strip().eq("").to_numpy()
    rows = np.flatnonzero(empty[pool])
    up = np.unique(pool[rows])
    pe = embed(names(pt.iloc[up]), f"{set_name} name-only pool")
    S = np.load(R / f"rev_{split}_s1_emb.npy")
    sc = pd.read_parquet(R / f"rev_{split}_s1_country.parquet").country_raw.fillna("NA").astype(str).str.strip().to_numpy()
    pc = pt.country_raw.fillna("NA").astype(str).str.strip().to_numpy()[up]
    dev = model()["dev"]
    top_i = np.full((up.size, TOPK), -1, dtype=np.int64)
    top_v = np.full((up.size, TOPK), -np.inf, dtype=np.float32)
    for c in np.unique(pc):
        pi, si = np.flatnonzero(pc == c), np.flatnonzero(sc == c)
        if si.size == 0:
            si = np.arange(S.shape[0])
        St = torch.from_numpy(S[si].astype(np.float32)).to(dev)
        for a in range(0, pi.size, 2048):
            q = torch.from_numpy(pe[pi[a:a + 2048]].astype(np.float32)).to(dev)
            v, j = torch.topk(q @ St.T, min(TOPK, si.size), dim=1)
            top_v[pi[a:a + 2048], :v.shape[1]] = v.cpu().numpy()
            top_i[pi[a:a + 2048], :v.shape[1]] = si[j.cpu().numpy()]
        print(f"{set_name} country {c}: {pi.size:,} pool x {si.size:,} S1", flush=True)
    pos = np.searchsorted(up, pool[rows])
    r_s1 = s1[rows].astype(np.int64)
    cos = (pe[pos].astype(np.float32) * S[r_s1].astype(np.float32)).sum(1)
    tv, ti = top_v[pos], top_i[pos]
    hit = ti == r_s1[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, TOPK + 1).astype(np.float32)
    other = np.where(hit, -np.inf, tv).max(1)
    F = {c: np.full(s1.size, np.nan, dtype=np.float32) for c in ("rv_cos", "rv_rank", "rv_top", "rv_gap", "rv_n_close", "rv_margin")}
    F["rv_cos"][rows], F["rv_rank"][rows], F["rv_top"][rows] = cos, rank, tv[:, 0]
    F["rv_gap"][rows] = tv[:, 0] - cos
    F["rv_n_close"][rows] = (tv >= cos[:, None] - 0.02).sum(1)
    F["rv_margin"][rows] = cos - other
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool), **{k: pa.array(v) for k, v in F.items()}}),
                   R / f"rev_{set_name}.parquet", compression="zstd")
    print(f"{set_name}: {rows.size:,} of {s1.size:,} rows name-only; rank==1 share {np.mean(rank == 1):.3f}", flush=True)


if __name__ == "__main__":
    R.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "embed_s1":
        embed_s1(sys.argv[2])
    else:
        for s in sys.argv[2:]:
            feats(s)

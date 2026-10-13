"""DRACE (Dense Retrieval And Copy Expansion) recall rescue on TEST, on top of a base submission (default S9 = full_c0).
Inputs only (no test labels exist). Rule fixed on train (gate 2, artifacts/metrics/gate2_c0.json + logs/gate2_candsize.log):
S1 query top-K1 + copy expansion (seeds = pool records the base accepts) top-K2; drop pairs already in the candidate
list or whose pool record the base already assigns; add a pair when c0 p_ce >= T; one-owner among added pairs.

  PYTHONPATH=submission/code/business_entity_resolution/src:ce:plan/checks .venv/bin/python plan/checks/drace_test.py prep
  (HF) retrieve_job.py RETR_TAG=drace_test RETR_K=5          -> work/retrieval/drace_test_topk.parquet
  ... drace_test.py pairs [--k1 5 --k2 2]                     -> work/ce/pairs_DRACE_TEST.parquet
  (HF) score_job.py CE_SETS=MATCH_VALID,DRACE_TEST, collect   -> work/ce/ce_rec_c0/DRACE_TEST.parquet
  ... drace_test.py merge [--t 0.995]                         -> submission/output_v10/{matching_results,candidate_pairs}.tsv
  ... drace_test.py s1_queries                                -> work/retrieval/dracd_test_s1_queries.parquet
  (GPU) retrieve_job.py RETR_TAG=drace_test RETR_QUERY_TAG=dracd_test_s1 RETR_MODEL_DIR=<e5_distil> RETR_K=10
        RETR_OUT_TAG=dracd10_test; then pairs_extra per rank band (DRACD k1 2, DRACD5 5, DRACD7 7, DRACD10 10)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (pa.compute)
import pyarrow.parquet as pq

from ber import io_utils, match

R = Path("work/retrieval")
SEED_BASE = 10 ** 9
BASE_OUT = Path("work/outputs/full_c0/matching_results.tsv")
BASE_CAND = Path("submission/output_v9/candidate_pairs.tsv")
OUT = Path("submission/output_v10")
SKIP_EXTRA = []  # extra scored sets whose pairs pairs_extra must not rescore (e.g. DRACD_TEST)


def key(a, b):
    return np.asarray(a, dtype=np.int64) << 32 | np.asarray(b, dtype=np.int64)


def text(names, addrs):
    n = pd.Series(names).fillna("").astype(str).str.strip()
    a = pd.Series(addrs).fillna("").astype(str).str.strip()
    return ("query: " + n + " | " + a).tolist()


def cty(s):
    return pd.Series(s).fillna("NA").astype(str).str.strip().tolist()


def test_candidates():
    parts = sorted(Path("work/candidates/TEST").glob("part-*.parquet"))
    return pd.concat([pq.read_table(p, columns=["s1_idx", "pool_idx"]).to_pandas() for p in parts], ignore_index=True)


def pool_id_map():
    p = pq.read_table("work/records/test_pool.parquet", columns=["pool_idx", "src", "num"]).to_pandas()
    return p, pd.Series(p.pool_idx.to_numpy(), index="S" + p.src.astype(str) + "-" + p.num.astype(str))


def read_results(path):
    """{s1 entity_id: [ids]} file -> long frame (s1_idx, pool_idx)."""
    s1 = pq.read_table("work/records/test_s1.parquet", columns=["s1_idx", "entity_id"]).to_pandas()
    s1map = pd.Series(s1.s1_idx.to_numpy(), index=s1.entity_id)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    col = df.columns[1]
    df = df[df[col] != ""].assign(pid=lambda d: d[col].str.split(",")).explode("pid")
    _, pmap = pool_id_map()
    return pd.DataFrame({"s1_idx": s1map.reindex(df.iloc[:, 0]).to_numpy(),
                         "pool_idx": pmap.reindex(df.pid).to_numpy()}), s1


def prep():
    R.mkdir(parents=True, exist_ok=True)
    pool = pq.read_table("work/records/test_pool.parquet", columns=["pool_idx", "name_raw", "addr_raw", "country_raw"]).to_pandas()
    s1 = pq.read_table("work/records/test_s1.parquet", columns=["s1_idx", "name_raw", "addr_raw", "country_raw"]).to_pandas()
    base, _ = read_results(BASE_OUT)
    if base.isna().any().any():
        raise RuntimeError("base matches with unknown ids")
    seeds = base.astype(np.int64).reset_index(drop=True)
    pq.write_table(pa.table({"pool_idx": pa.array(pool.pool_idx.to_numpy().astype(np.uint32)),
                             "country": cty(pool.country_raw), "text": text(pool.name_raw, pool.addr_raw)}),
                   R / "drace_test_pool.parquet", compression="zstd")
    pp = pool.set_index("pool_idx").loc[seeds.pool_idx]
    pq.write_table(pa.table({
        "qid": pa.array(np.concatenate([s1.s1_idx.to_numpy().astype(np.int64), SEED_BASE + np.arange(len(seeds))])),
        "kind": ["s1"] * len(s1) + ["seed"] * len(seeds),
        "country": cty(s1.country_raw) + cty(pp.country_raw),
        "text": text(s1.name_raw, s1.addr_raw) + text(pp.name_raw, pp.addr_raw),
        "self_pool": pa.array(np.concatenate([np.full(len(s1), -1), seeds.pool_idx.to_numpy()]).astype(np.int64))}),
        R / "drace_test_queries.parquet", compression="zstd")
    seeds.assign(qid=SEED_BASE + np.arange(len(seeds))).to_parquet(R / "drace_test_seeds.parquet")
    print(f"prep: pool {len(pool):,}, S1 queries {len(s1):,}, seed queries {len(seeds):,}; "
          f"countries S1 {pd.Series(cty(s1.country_raw)).value_counts().to_dict()}")


def s1_queries():
    """S1 rows of drace_test_queries (no seed queries): input of the distilled-e5 S1 retrieval (RETR_QUERY_TAG=dracd_test_s1)."""
    q = pq.read_table(R / "drace_test_queries.parquet")
    q = q.filter(pa.compute.equal(q.column("kind"), "s1"))
    pq.write_table(q, R / "dracd_test_s1_queries.parquet", compression="zstd")
    print(f"s1 queries: {q.num_rows:,}")


def pairs(k1, k2):
    import sys
    sys.path.insert(0, "ce")
    from prepare_pairs import record_text
    top = pq.read_table(R / "drace_test_topk.parquet").to_pandas()
    seeds = pd.read_parquet(R / "drace_test_seeds.parquet")
    a = top[(top.qid < SEED_BASE) & (top["rank"] <= k1)].rename(columns={"qid": "s1_idx"})
    b = top[(top.qid >= SEED_BASE) & (top["rank"] <= k2)].merge(seeds[["qid", "s1_idx"]], on="qid")
    m = pd.concat([a[["s1_idx", "pool_idx", "cos"]], b[["s1_idx", "pool_idx", "cos"]]])
    m = m.groupby(["s1_idx", "pool_idx"], as_index=False)["cos"].max()
    cand = test_candidates()
    base, _ = read_results(BASE_OUT)
    n0 = len(m)
    m = m[~np.isin(key(m.s1_idx, m.pool_idx), key(cand.s1_idx, cand.pool_idx))]
    m = m[~m.pool_idx.isin(base.pool_idx)].sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    s1_txt, pool_txt = record_text("test", "s1"), record_text("test", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool),
                             "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1_txt[s1].tolist(), pa.string()),
                             "text_b": pa.array(pool_txt[pool].tolist(), pa.string())}),
                   "work/ce/pairs_DRACE_TEST.parquet", compression="zstd")
    m.to_parquet(R / "drace_test_meta.parquet")
    n_s1 = int(pq.ParquetFile("work/records/test_s1.parquet").metadata.num_rows)
    print(f"pairs k1={k1} k2={k2}: {n0:,} retrieved -> {len(m):,} new pairs to score "
          f"({len(m) / n_s1:.2f} per S1; existing lists {len(cand) / n_s1:.2f} per S1)")


def pairs_extra(topk_tag, set_name, k1):
    """New S1-query pairs from another retrieval (e.g. distilled e5): not in the candidate lists, not already in
    drace_test_meta (scored), not owned by S9. Writes work/ce/pairs_<SET>.parquet + work/retrieval/<set>_meta.parquet."""
    import sys
    sys.path.insert(0, "ce")
    from prepare_pairs import record_text
    top = pq.read_table(R / f"{topk_tag}_topk.parquet").to_pandas()
    m = top[(top.qid < SEED_BASE) & (top["rank"] <= k1)].rename(columns={"qid": "s1_idx"})[["s1_idx", "pool_idx", "cos"]]
    cand = test_candidates()
    old = pd.concat([pd.read_parquet(R / f)[["s1_idx", "pool_idx"]] for f in ["drace_test_meta.parquet"] +
                     [f"{x.lower()}_meta.parquet" for x in SKIP_EXTRA]])
    base, _ = read_results(BASE_OUT)
    k = lambda a, b: key(a, b)
    m = m[~np.isin(k(m.s1_idx, m.pool_idx), k(cand.s1_idx, cand.pool_idx))]
    m = m[~np.isin(k(m.s1_idx, m.pool_idx), k(old.s1_idx, old.pool_idx)) & ~m.pool_idx.isin(base.pool_idx)]
    m = m.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    s1_txt, pool_txt = record_text("test", "s1"), record_text("test", "pool")
    s1, pool = m.s1_idx.to_numpy().astype(np.int32), m.pool_idx.to_numpy().astype(np.uint32)
    pq.write_table(pa.table({"s1_idx": pa.array(s1), "pool_idx": pa.array(pool),
                             "rank": pa.array(np.zeros(len(m), dtype=np.int16)),
                             "text_a": pa.array(s1_txt[s1].tolist(), pa.string()),
                             "text_b": pa.array(pool_txt[pool].tolist(), pa.string())}),
                   f"work/ce/pairs_{set_name}.parquet", compression="zstd")
    m.to_parquet(R / f"{set_name.lower()}_meta.parquet")
    print(f"extra pairs {topk_tag} S1@{k1}: {len(m):,} new pairs to score ({len(m) / len(read_results(BASE_OUT)[1]):.3f} per S1)")


def scored(set_name, meta_name):
    m = pd.read_parquet(R / meta_name)
    sc = pq.read_table(f"work/ce/ce_rec_c0/{set_name}.parquet").to_pandas()
    if not (np.array_equal(sc.s1_idx.to_numpy(), m.s1_idx.to_numpy())
            and np.array_equal(sc.pool_idx.to_numpy(), m.pool_idx.to_numpy())):
        raise RuntimeError(f"{set_name} scores do not match {meta_name} order")
    return m.assign(p=sc.p_ce.to_numpy())[["s1_idx", "pool_idx", "p"]]


def merge(t, base_out=BASE_OUT, out=OUT, tag="drace_test", extra=()):
    m = pd.concat([scored("DRACE_TEST", "drace_test_meta.parquet")] +
                  [scored(x, f"{x.lower()}_meta.parquet") for x in extra], ignore_index=True)
    base, s1 = read_results(base_out)
    add = m[(m.p >= t) & ~m.pool_idx.isin(base.pool_idx)].sort_values("p", ascending=False).drop_duplicates("pool_idx")
    allm = pd.concat([base.assign(p=2.0), add[["s1_idx", "pool_idx", "p"]]])
    pool, _ = pool_id_map()
    pid = "S" + pool.src.astype(str) + "-" + pool.num.astype(str)
    ids = pid.to_numpy()[np.argsort(pool.pool_idx.to_numpy())]
    out.mkdir(parents=True, exist_ok=True)

    def write(frame, path, header):
        g = frame.sort_values(["s1_idx", "p"], ascending=[True, False]).groupby("s1_idx")["pool_idx"] \
            .apply(lambda x: ",".join(ids[x.to_numpy().astype(np.int64)]))
        col = s1.entity_id.to_numpy()
        vals = g.reindex(s1.s1_idx.to_numpy()).fillna("").to_numpy()
        pd.DataFrame({"source1_entity_id": col, header: vals}).to_csv(path, sep="\t", index=False)
    write(allm, out / "matching_results.tsv", "matched_entity_ids")
    cand = test_candidates().assign(p=1.0)
    write(pd.concat([cand, m[["s1_idx", "pool_idx"]].assign(p=0.0)]).drop_duplicates(["s1_idx", "pool_idx"]),
          out / "candidate_pairs.tsv", "candidate_entity_ids")
    n_s1 = len(s1)
    doc = {"base": str(base_out), "threshold": t, "new_pairs_scored": int(len(m)), "added": int(len(add)),
           "added_per_s1": float(len(add) / n_s1), "base_pairs": int(len(base)), "final_pairs": int(len(allm)),
           "candidates_per_s1": float((len(cand) + len(m)) / n_s1),
           "p_ce_quantiles_new": {str(q): float(m.p.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)}}
    io_utils.write_metrics(tag, doc, n_s1, 0)
    print(doc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "s1_queries", "pairs", "merge", "pairs_extra"])
    ap.add_argument("--topk-tag", default="dracd_test")
    ap.add_argument("--set", default="DRACD_TEST")
    ap.add_argument("--skip", default="", help="comma list of already scored extra sets to exclude in pairs_extra")
    ap.add_argument("--extra", default="", help="comma list of extra scored sets to merge (e.g. DRACD_TEST)")
    ap.add_argument("--k1", type=int, default=5)
    ap.add_argument("--k2", type=int, default=2)
    ap.add_argument("--t", type=float, default=0.995)
    ap.add_argument("--base", default=str(BASE_OUT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="drace_test")
    a = ap.parse_args()
    SKIP_EXTRA[:] = [x for x in a.skip.split(",") if x]
    {"prep": prep, "s1_queries": s1_queries, "pairs": lambda: pairs(a.k1, a.k2), "merge": lambda: merge(a.t, Path(a.base), Path(a.out), a.tag, [x for x in a.extra.split(",") if x]),
     "pairs_extra": lambda: pairs_extra(a.topk_tag, a.set, a.k1)}[a.step]()

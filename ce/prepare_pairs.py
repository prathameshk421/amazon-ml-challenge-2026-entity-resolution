"""Cross-encoder pair files: one row per stage-3 feature row, in feature-row order, with raw text.

  .venv/bin/python ce/prepare_pairs.py [--sets DEV_REST_180K MATCH_TRAIN MATCH_VALID TEST DEV_REST_REST]

-> work/ce/pairs_<SET>.parquet: s1_idx:int32, pool_idx:uint32, rank:int16, text_a, text_b, label:int8 (train
sets only), and artifacts/metrics/ce_pairs_manifest.json (rows, S1, positive rate, overlap checks).

Rows come from work/match/features/<SET> (the exact rows and order the matcher reads), text from
work/records/{split}_{s1,pool}.parquet (name_raw, addr_raw). Serialization matches plan/FRIEND_B_HANDOFF.md §3:
"{name} | {addr}" on both sides; empty address -> "{name} | ".

The cross-encoder trains on DEV_REST_180K only (disjoint from MATCH_TRAIN, MATCH_VALID and val S1), because its
score becomes a LightGBM feature trained on MATCH_TRAIN (stacking leakage otherwise, VALIDATION_PLAN L2).
TEST gets no label column; the read is logged in reports/test_usage_log.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission/code/business_entity_resolution/src"))
from ber import io_utils, match, splits  # noqa: E402

SETS = ("DEV_REST_180K", "MATCH_TRAIN", "MATCH_VALID", "TEST", "DEV_REST_REST")
TRAIN_SET = "DEV_REST_180K"


def feature_rows(set_name):
    """(s1_idx, pool_idx, rank) of the set's feature table, same order as match.load_features."""
    d = match.features_dir(set_name)
    parts = sorted(d.glob("part-*.parquet"))
    if not parts:
        raise RuntimeError(f"no feature parts in {d}; run match-features --set {set_name}")
    t = pa.concat_tables([pq.read_table(p, columns=["s1_idx", "pool_idx", "rank"]) for p in parts])
    return (t.column("s1_idx").to_numpy().astype(np.int32), t.column("pool_idx").to_numpy().astype(np.uint32),
            t.column("rank").to_numpy().astype(np.int16))


def record_text(split, kind):
    """name | addr strings indexed by record position (s1_idx / pool_idx == row number, asserted)."""
    t = pq.read_table(io_utils.work_path("records", f"{split}_{kind}.parquet"),
                      columns=[f"{kind}_idx", "name_raw", "addr_raw"])
    idx = t.column(f"{kind}_idx").to_numpy()
    if not np.array_equal(idx, np.arange(idx.size)):
        raise RuntimeError(f"{split}_{kind}: {kind}_idx is not the row number")
    name = np.array(t.column("name_raw").fill_null("").to_pylist(), dtype=object)
    addr = np.array(t.column("addr_raw").fill_null("").to_pylist(), dtype=object)
    name = np.char.strip(name.astype(str)).astype(object)
    addr = np.char.strip(addr.astype(str)).astype(object)
    return name + " | " + addr


def s1_of(set_name):
    return np.unique(feature_rows(set_name)[0]) if set_name.startswith("DEV_REST_") else match.set_queries(set_name)


def overlap_checks():
    """S1 overlap of the CE training set with every set it must not see. All must be 0."""
    train = set(s1_of(TRAIN_SET).tolist())
    out = {}
    for other in ("MATCH_TRAIN", "MATCH_VALID", "DEV_REST_REST"):
        out[f"{TRAIN_SET}&{other}"] = len(train & set(s1_of(other).tolist()))
    out[f"{TRAIN_SET}&HOLDOUT_TEST"] = len(train & set(splits.load_query_sets()["HOLDOUT_TEST"].tolist()))
    bad = {k: v for k, v in out.items() if v}
    if bad:
        raise RuntimeError(f"S1 overlap with the cross-encoder training set: {bad}")
    return out


def build(set_name, texts):
    t0 = time.time()
    split = "test" if set_name == "TEST" else "train"
    s1, pool, rank = feature_rows(set_name)
    s1_txt, pool_txt = texts[split]
    cols = {"s1_idx": pa.array(s1), "pool_idx": pa.array(pool), "rank": pa.array(rank),
            "text_a": pa.array(s1_txt[s1].tolist(), pa.string()),
            "text_b": pa.array(pool_txt[pool].tolist(), pa.string())}
    info = {"rows": int(s1.size), "s1": int(np.unique(s1).size)}
    if split == "train":
        y = match.pair_labels(s1, pool).astype(np.int8)
        cols["label"] = pa.array(y)
        info["pos_rate"] = round(float(y.mean()), 4)
    out = io_utils.work_path("ce", f"pairs_{set_name}.parquet")
    tmp = out.with_name(out.name + ".tmp")
    pq.write_table(pa.table(cols), tmp, compression="zstd", row_group_size=500_000)
    tmp.replace(out)
    info["mb"] = round(out.stat().st_size / 2**20, 1)
    info["sec"] = round(time.time() - t0, 1)
    if set_name == "TEST":
        io_utils.log_test_usage("raw name/address text of every test candidate row (inputs only, no labels)",
                                "cross-encoder pair file for scoring (plan: Kaggle ce_rec_base)", str(out))
    print(set_name, info, flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    args = ap.parse_args()
    overlap = overlap_checks() if set(args.sets) - {"TEST"} else "skipped (TEST only)"
    print("overlap", overlap, flush=True)
    splits_needed = {"test" if s == "TEST" else "train" for s in args.sets}
    texts = {sp: (record_text(sp, "s1"), record_text(sp, "pool")) for sp in sorted(splits_needed)}
    info = {s: build(s, texts) for s in args.sets}
    n_true = sum(round(v["pos_rate"] * v["rows"]) for v in info.values() if "pos_rate" in v)
    path = io_utils.write_metrics("ce_pairs_manifest", {
        "format": "text = '{name_raw} | {addr_raw}' (stripped; empty addr -> '{name} | ')",
        "train_set": TRAIN_SET, "overlap": overlap, "sets": info},
        n_queries=sum(v["s1"] for v in info.values()), n_true_pairs=n_true)
    print("wrote", path)


if __name__ == "__main__":
    main()

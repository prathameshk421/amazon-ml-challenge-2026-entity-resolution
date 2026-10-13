"""Standalone check of cross-encoder scores on MATCH_VALID pairs (run locally, once per model).

  .venv/bin/python ce/evaluate.py --ce ce_rec_base untrained_base [ce_rec] [--name ce_rec_base_eval]

Reads work/ce/<name>/MATCH_VALID.parquet for each --ce name, labels from work/ce/pairs_MATCH_VALID.parquet and
string columns from work/match/features/MATCH_VALID. Reports ROC-AUC / PR-AUC / logloss for every scorer,
overall and on the hard subsets (pool address empty; name token-set >= 0.95; alias = name token-set < 0.5 and
address token-set >= 0.8), plus positive/negative score histograms. Writes artifacts/metrics/<name>.json.
A standalone win is necessary, not sufficient: adoption is decided by the paired matcher experiment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission/code/business_entity_resolution/src"))
from ber import io_utils  # noqa: E402

STRING_COLS = ["n_tfidf_cos", "a_tfidf_cos", "n_tset", "a_tset", "p_rerank"]


def metrics(y, s, prob):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or y.min() == y.max():
        return {"n": int(y.size)}
    out = {"n": int(y.size), "pos_rate": round(float(y.mean()), 4), "roc_auc": round(float(roc_auc_score(y, s)), 5),
           "pr_auc": round(float(average_precision_score(y, s)), 5)}
    if prob:
        out["logloss"] = round(float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6), labels=[0, 1])), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", nargs="+", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    work = io_utils.paths()["work_root"]
    pairs = pq.read_table(work / "ce" / "pairs_MATCH_VALID.parquet", columns=["s1_idx", "pool_idx", "text_b", "label"])
    fd = work / "match" / "features" / "MATCH_VALID"
    feats = pa.concat_tables([pq.read_table(p, columns=["s1_idx", "pool_idx"] + STRING_COLS)
                              for p in sorted(fd.glob("part-*.parquet"))])
    for c in ("s1_idx", "pool_idx"):
        if not np.array_equal(pairs.column(c).to_numpy(), feats.column(c).to_numpy()):
            raise RuntimeError(f"pair file and feature table differ in {c} order")
    y = pairs.column("label").to_numpy().astype(int)
    scores = {c: feats.column(c).to_numpy().astype(np.float64) for c in STRING_COLS}
    for name in args.ce:
        t = pq.read_table(work / "ce" / name / "MATCH_VALID.parquet")
        if not np.array_equal(t.column("pool_idx").to_numpy(), pairs.column("pool_idx").to_numpy()):
            raise RuntimeError(f"{name}: row order differs from the pair file")
        scores[name] = t.column("p_ce").to_numpy().astype(np.float64)

    text_b = pairs.column("text_b").to_pylist()
    n_tset, a_tset = scores["n_tset"], scores["a_tset"]
    subsets = {"all": np.ones(y.size, bool),
               "pool_addr_empty": np.array([s.endswith(" | ") for s in text_b]),
               "name_tset_ge_0.95": n_tset >= 0.95,
               "alias": (n_tset < 0.5) & (a_tset >= 0.8)}
    res = {sub: {k: metrics(y[m], v[m], prob=k in args.ce or k == "p_rerank") for k, v in scores.items()}
           for sub, m in subsets.items()}
    bins = np.linspace(0, 1, 11)
    hist = {k: {"pos": np.histogram(scores[k][y == 1], bins)[0].tolist(),
                "neg": np.histogram(scores[k][y == 0], bins)[0].tolist()} for k in args.ce}
    name = args.name or f"{args.ce[0]}_eval"
    path = io_utils.write_metrics(name, {"set": "MATCH_VALID", "scorers": list(scores), "subsets": res,
                                         "hist_bins": bins.tolist(), "hist": hist},
                                  n_queries=np.unique(pairs.column("s1_idx").to_numpy()).size,
                                  n_true_pairs=int(y.sum()))
    print(json.dumps(res["all"], indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()

"""FP / FN report with raw text for the live matcher (full_single, submission 2) and full_comp (submission 3 input).

Rows: the MATCH_ALL calibration slice (165,545 train S1 that neither model fitted or early-stopped on; same seeded
split as train_final_single.py / train_final_comp.py). Train labels only; no test rows.

Writes work/reports/errors/errors_<model>.parquet, one row per wrong pair:
  kind = FP (predicted, not a GT link) | FN (GT link in the candidate list, not predicted)
       | FN_MISSED (GT link that blocking/re-ranking never put in the candidate list)
  s1_idx, pool_idx, p_cal, rank, p_rerank, the matcher features, S1 n_true / n_pred / n_tp / per-entity F0.5,
  and raw name/address/country of both sides (records/train_s1, records/train_pool) plus pool src/num.
Summary counts -> artifacts/metrics/error_report.json.

Run: PYTHONPATH=submission/code/business_entity_resolution/src:plan/checks .venv/bin/python plan/checks/error_report.py
"""
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import exp, io_utils, match
from final_hybrid import SETS, models, score_rows
from test_shift_diag import cal_queries

KEEP_FEATS = ["rank", "p_rerank", "n_tset", "n_ratio", "n_jw_sorted", "n_idf_jacc", "n_eq_core", "a_tset", "a_idf_jacc",
              "a_tfidf_cos", "a_num_jacc", "a_house", "a_empty", "l_n", "l_p_max", "c_log_n_own"]


def attach_text(df):
    s1_ids = df["s1_idx"].to_numpy()
    pool_ids = df["pool_idx"].to_numpy()
    s1 = pq.read_table(io_utils.work_path("records", "train_s1.parquet")).to_pandas().set_index("s1_idx")
    s1 = s1.loc[s1_ids].reset_index(drop=True)
    pool_tab = pq.read_table(io_utils.work_path("records", "train_pool.parquet"))
    pidx = pool_tab.column("pool_idx").to_numpy()
    if not np.array_equal(pidx, np.arange(pidx.size)):
        raise RuntimeError("train_pool.parquet: pool_idx is not 0..n-1")
    pool = pool_tab.take(pool_ids.astype(np.int64)).to_pandas()
    out = df.reset_index(drop=True)
    out["s1_entity_id"] = s1["entity_id"].to_numpy()
    out["s1_name"], out["s1_addr"], out["s1_country"] = s1["name_raw"].to_numpy(), s1["addr_raw"].to_numpy(), s1["country_raw"].to_numpy()
    out["pool_src"], out["pool_num"] = pool["src"].to_numpy(), pool["num"].to_numpy()
    out["pool_name"], out["pool_addr"], out["pool_country"] = pool["name_raw"].to_numpy(), pool["addr_raw"].to_numpy(), pool["country_raw"].to_numpy()
    return out


def main():
    t0 = time.time()
    cfg = io_utils.load_config()
    fp = match.bucketing_fingerprint(cfg)
    cd, cdoc, cb, sdoc, sb = models()
    q = cal_queries()
    parts = []
    for name in SETS:
        Xf, names, s1, pool, _ = exp.full_matrix(name, cdoc["extra_groups"], fp)
        m = np.isin(s1, q)
        pc, ps, _ = score_rows(Xf[m], names, cdoc, cb, sdoc, sb)
        ci = {n: i for i, n in enumerate(names)}
        F = pd.DataFrame(Xf[m][:, [ci[f] for f in KEEP_FEATS]], columns=KEEP_FEATS)
        F["s1_idx"], F["pool_idx"], F["p_single"], F["p_comp"] = s1[m], pool[m], ps, pc
        parts.append(F)
        del Xf
        print(f">>> {name}: {int(m.sum())} rows ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True).sort_values(["s1_idx", "pool_idx"], kind="stable").reset_index(drop=True)
    s1, pool = df["s1_idx"].to_numpy(), df["pool_idx"].to_numpy()
    df["label"] = match.pair_labels(s1, pool)
    n_true_q = match.true_counts(q)
    sc = match.Scorer(q, s1, df["label"].to_numpy(), n_true_q)

    gt = pq.read_table(io_utils.work_path("records", "train_gt.parquet")).to_pandas()
    gt = gt[gt["s1_idx"].isin(q)]
    key_c = s1.astype(np.int64) << 32 | pool.astype(np.int64)
    key_g = gt["s1_idx"].to_numpy().astype(np.int64) << 32 | gt["pool_idx"].to_numpy().astype(np.int64)
    missed = gt[~np.isin(key_g, key_c)]
    ntrue = pd.Series(n_true_q, index=q)

    summary = {}
    for model, doc, pcol in (("full_single", sdoc, "p_single"), ("full_comp", cdoc, "p_comp")):
        pred = match.apply_decision(doc["decision"], s1, pool, df[pcol].to_numpy())
        f_ent = sc.per_entity(pred)[0]
        per = pd.DataFrame({"s1_idx": q, "n_true": n_true_q, "f05_entity": f_ent})
        g = pd.DataFrame({"s1_idx": s1, "pred": pred, "tp": pred & df["label"].to_numpy()}).groupby("s1_idx").sum()
        per["n_pred"] = g["pred"].reindex(q).fillna(0).astype(int).to_numpy()
        per["n_tp"] = g["tp"].reindex(q).fillna(0).astype(int).to_numpy()
        lab = df["label"].to_numpy()
        err = df[(pred & ~lab) | (~pred & lab)].copy()
        err["kind"] = np.where(pred[(pred & ~lab) | (~pred & lab)], "FP", "FN")
        err["p_cal"] = err[pcol]
        m = missed.copy()
        m["kind"] = "FN_MISSED"
        err = pd.concat([err, m], ignore_index=True)
        err = err.merge(per, on="s1_idx", how="left")
        err = attach_text(err)
        err["model"] = model
        out = io_utils.work_path("reports", "errors", f"errors_{model}.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        err.to_parquet(out, index=False)
        lost = 1.0 - f_ent
        wrong = per["f05_entity"] < 1.0
        summary[model] = {
            "file": str(out), "decision": doc["decision"], "f05": float(f_ent.mean()),
            "n_FP": int((err["kind"] == "FP").sum()), "n_FN": int((err["kind"] == "FN").sum()),
            "n_FN_MISSED": int((err["kind"] == "FN_MISSED").sum()), "n_s1_imperfect": int(wrong.sum()),
            "loss_total": float(lost.sum()),
            "loss_by_country": {str(c): float(lost[(per["s1_idx"].map(dict(zip(err["s1_idx"], err["s1_country"])))
                                                    == c).to_numpy()].sum()) for c in sorted(err["s1_country"].dropna().unique())},
            "loss_singleton_fp": float(lost[(per["n_true"] == 0).to_numpy()].sum()),
        }
        print(f">>> {model}: F0.5 {f_ent.mean():.5f}  FP {summary[model]['n_FP']}  FN {summary[model]['n_FN']}  "
              f"FN_MISSED {summary[model]['n_FN_MISSED']}  -> {out} ({time.time() - t0:.0f}s)", flush=True)
    io_utils.write_metrics("error_report", {"slice": "MATCH_ALL calibration slice (train S1)", "models": summary,
                                            "wall_s": round(time.time() - t0, 1)}, int(q.size), int(len(gt)), cfg)


if __name__ == "__main__":
    main()

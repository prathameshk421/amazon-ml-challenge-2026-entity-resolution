"""CE-only submission: the cross-encoder's p_ce alone decides test matches (no LightGBM matcher).

Why a separate path: the LightGBM matcher can only use p_ce as a feature after MATCH_TRAIN is scored and the matcher
retrained. This script skips that and measures the cross-encoder by itself. The decision (threshold + one-owner)
is fitted on a slice of *train* S1 (labels allowed for thresholds, AGENTS.md rule 5), never on test or the
leaderboard. MATCH_VALID, if scored, is reported for scoring only.

  # 1. write a small labelled slice to score (train S1 the cross-encoder did not train on; default 20,000 S1)
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python ce/ce_submit.py slice --n-s1 20000
  # 2. on the GPU: score work/ce/pairs_MATCH_TRAIN_S20000.parquet (+ optionally MATCH_VALID) and TEST with
  #    ce/inference.py, --merge each, copy <SET>.parquet files into work/ce/<ce>/
  # 3. fit the decision on the slice, write work/outputs/<ce>_only/matching_results.tsv
  PYTHONPATH=... .venv/bin/python ce/ce_submit.py submit --ce ce_rec --fit-set MATCH_TRAIN_S20000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission/code/business_entity_resolution/src"))
from ber import io_utils, match  # noqa: E402

# train S1 sets the cross-encoder did not train on (DEV_REST_180K is its training set)
FIT_BASES = ("MATCH_TRAIN", "DEV_REST_REST")


def ce_dir():
    return io_utils.paths()["work_root"] / "ce"


def slice_queries(name):
    """(base set, S1 array) for a slice name like MATCH_TRAIN_S20000; the S1 list is saved next to the pairs."""
    base = name.rsplit("_S", 1)[0]
    if base not in FIT_BASES:
        raise ValueError(f"fit set must be a slice of {FIT_BASES}, got {name!r}")
    return base, np.load(ce_dir() / f"queries_{name}.npy")


def run_slice(args):
    if args.base not in FIT_BASES:
        raise ValueError(f"--base must be one of {FIT_BASES}")
    q_all = match.set_queries(args.base)
    rng = np.random.default_rng(args.seed)
    q = np.sort(rng.choice(q_all, size=min(args.n_s1, q_all.size), replace=False)).astype(np.int32)
    name = f"{args.base}_S{q.size}"
    t = pq.read_table(ce_dir() / f"pairs_{args.base}.parquet")
    m = np.isin(t.column("s1_idx").to_numpy(), q)
    out = ce_dir() / f"pairs_{name}.parquet"
    pq.write_table(t.filter(m), out, compression="zstd")
    np.save(ce_dir() / f"queries_{name}.npy", q)
    print(f"{out}: {int(m.sum()):,} rows, {q.size:,} S1 ({np.unique(t.column('s1_idx').to_numpy()[m]).size:,} "
          f"with >= 1 candidate), seed {args.seed}")


def read_scores(ce, set_name, pairs_name=None):
    """p_ce aligned to pairs_<SET>.parquet row order; raises on any row-count or key mismatch."""
    pairs = pq.read_table(ce_dir() / f"pairs_{pairs_name or set_name}.parquet", columns=["s1_idx", "pool_idx"])
    t = pq.read_table(ce_dir() / ce / f"{set_name}.parquet")
    for c in ("s1_idx", "pool_idx"):
        if t.num_rows != pairs.num_rows or not np.array_equal(t.column(c).to_numpy(), pairs.column(c).to_numpy()):
            raise RuntimeError(f"{ce}/{set_name}.parquet: {c} differs from pairs_{pairs_name or set_name}.parquet")
    p = t.column("p_ce").to_numpy().astype(np.float32)
    if not np.isfinite(p).all():
        raise RuntimeError(f"{ce}/{set_name}: {int((~np.isfinite(p)).sum())} non-finite p_ce")
    return t.column("s1_idx").to_numpy(), t.column("pool_idx").to_numpy(), p


def fit_decision(s1, pool, p, q, mcfg):
    """Best threshold per one-owner setting on the slice; one-owner kept only if it wins by the config's margin."""
    o = np.lexsort((pool, s1))
    s1, pool, p = s1[o], pool[o], p[o]
    sc = match.Scorer(q, s1, match.pair_labels(s1, pool), match.true_counts(q))
    grid = match.threshold_grid(mcfg["decision"])
    cands, sweeps = {}, {}
    for owner in (False, True):
        pp = match.one_owner(pool, p) if owner else p
        sw = {float(t): sc.f05(pp >= t) for t in grid}
        tb = max(sw, key=sw.get)
        dec = {"rule": "threshold", "threshold": tb, "one_owner": owner}
        cands[owner] = (dec, sc.per_entity(match.apply_decision(dec, s1, pool, p))[0])
        sweeps[str(owner)] = {f"{t:.2f}": round(v, 5) for t, v in sw.items()}
    oo = match.paired_delta(cands[True][1], cands[False][1])
    dec, f = cands[match.adopt(oo, mcfg["decision"]["one_owner_min_gain"])]
    return dec, {"f05": float(f.mean()), "one_owner_vs_not": oo, "sweep_f05": sweeps,
                 "report": sc.report(match.apply_decision(dec, s1, pool, p))[0]}


def score_valid(s1, pool, p, dec):
    q = match.set_queries("MATCH_VALID")
    o = np.lexsort((pool, s1))
    s1, pool, p = s1[o], pool[o], p[o]
    sc = match.Scorer(q, s1, match.pair_labels(s1, pool), match.true_counts(q))
    return sc.report(match.apply_decision(dec, s1, pool, p))[0]


def write_test(out_dir, s1, pool, p, keep):
    rec = pd.read_parquet(io_utils.work_path("records", "test_s1.parquet"), columns=["s1_idx", "entity_id", "country_raw"])
    if not np.array_equal(rec["s1_idx"].to_numpy(), np.arange(len(rec))):
        raise RuntimeError("records/test_s1.parquet: s1_idx is not 0..n-1")
    s1_ids = rec["entity_id"].tolist()
    store = io_utils.paths()["work_root"] / "norm" / "test_pool_store"
    src, num = np.load(store / "src.npy", mmap_mode="r"), np.load(store / "num.npy", mmap_mode="r")
    out = io_utils.work_path("outputs", out_dir, "matching_results.tsv")
    k = np.flatnonzero(keep)
    k = k[np.lexsort((-p[k], s1[k]))]
    match.write_matching_file(out, s1_ids, s1[k], pool[k], src, num)
    issues = match.check_matching_file(out, s1_ids, np.sort(match.pair_keys(s1, np.asarray(src)[pool], np.asarray(num)[pool])))
    if issues:
        raise RuntimeError(f"{out} failed checks: {issues[:5]}")
    country = rec["country_raw"].to_numpy()
    n_pred = np.bincount(s1, weights=keep, minlength=len(s1_ids))
    per = {c: {"n_s1": int((country == c).sum()), "mean_pred_size": float(n_pred[country == c].mean()),
               "share_empty": float((n_pred[country == c] == 0).mean())} for c in sorted(set(country.tolist()))}
    return {"file": str(out), "file_bytes": out.stat().st_size, "pairs": int(keep.sum()), "per_country": per,
            "check_passed": True}


def run_submit(args):
    t0 = time.time()
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    _, q = slice_queries(args.fit_set)
    s1, pool, p = read_scores(args.ce, args.fit_set)
    if not np.isin(s1, q).all():
        raise RuntimeError(f"{args.fit_set} scores contain S1 outside the slice")
    dec, fit = fit_decision(s1, pool, p, q, mcfg)
    out = {"script": "ce/ce_submit.py", "ce": args.ce, "fit_set": args.fit_set, "fit_s1": int(q.size),
           "decision": dec, "fit": fit}
    print(f"fit on {args.fit_set} ({q.size:,} train S1): {dec}  F0.5 {fit['f05']:.5f}", flush=True)
    if (ce_dir() / args.ce / "MATCH_VALID.parquet").exists():
        out["match_valid"] = score_valid(*read_scores(args.ce, "MATCH_VALID"), dec)
        print(f"MATCH_VALID (scoring only): F0.5 {out['match_valid']['f05']:.5f}", flush=True)
    else:
        out["match_valid"] = "skipped: no MATCH_VALID scores"

    io_utils.log_test_usage(f"{args.ce} p_ce over all test candidates, CE-only threshold decision (inputs only)",
                            f"CE-only submission (ce/ce_submit.py, {args.ce})", str(ce_dir() / args.ce / "TEST.parquet"))
    s1t, poolt, pt = read_scores(args.ce, "TEST")
    if np.unique(s1t).size > match.set_queries("TEST").size:
        raise RuntimeError("TEST scores reference unknown S1")
    keep = match.apply_decision(dec, s1t, poolt, pt)
    out["test"] = write_test(f"{args.ce}_only", s1t, poolt, pt, keep)
    out["wall_s"] = round(time.time() - t0, 1)
    path = io_utils.write_metrics(f"ce_submit_{args.ce}", out, int(q.size), int(match.true_counts(q).sum()), cfg)
    print(out["test"], flush=True)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("slice")
    a.add_argument("--base", default="MATCH_TRAIN")
    a.add_argument("--n-s1", type=int, default=20_000)
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("submit")
    b.add_argument("--ce", required=True)
    b.add_argument("--fit-set", required=True)
    args = ap.parse_args()
    run_slice(args) if args.cmd == "slice" else run_submit(args)


if __name__ == "__main__":
    main()

"""R1: distil c0 (cross-encoder) into multilingual-e5-small for DRACE retrieval (laptop MPS).

  PYTHONPATH=submission/code/business_entity_resolution/src:ce .venv/bin/python plan/checks/distill_e5.py [--n-s1 150000]

Training data: DEV_REST_REST fold-1 S1 only (disjoint from MATCH_TRAIN, which gate 1/2 use for evaluation). c0 never
trained on fold 1, so its scores there are out-of-sample teacher signal (work/ce/ce_rec_c0/DEV_REST_REST.parquet).
Per S1: positives = true pool records (in the candidate list: teacher p = c0; missed by blocking: teacher p = 0.999);
hard negatives = the 2 false in-list candidates with the highest c0 p. Triplets (q, pos, neg).
Loss (same pooling as ce/hf_job/retrieve_job.py: mean over tokens, L2 norm, text "query: name | address"):
  InfoNCE over the batch (q vs every pos and neg in the batch, temperature 0.05)
  + MarginMSE: (cos(q,pos) - cos(q,neg)) regressed to (p_teacher(pos) - p_teacher(neg)).
Saves the model to work/retrieval/e5_distil/ (HF format, loadable by AutoModel) + train_meta.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from prepare_pairs import record_text

MODEL = "intfloat/multilingual-e5-small"
OUT = Path("work/retrieval/e5_distil")


def fold_of(s1):
    return ((s1.astype(np.uint64) * np.uint64(0x9E3779B1)) % np.uint64(2 ** 32)) >> np.uint64(31)


def build(n_s1, seed=0):
    t = pq.read_table("work/ce/pairs_DEV_REST_REST_F1.parquet", columns=["s1_idx", "pool_idx", "label"]).to_pandas()
    sc = pq.read_table("work/ce/ce_rec_c0/DEV_REST_REST_F1.parquet").to_pandas()
    assert np.array_equal(sc.pool_idx.to_numpy(), t.pool_idx.to_numpy())
    t["p"] = sc.p_ce.to_numpy()
    rng = np.random.default_rng(seed)
    s1s = rng.choice(t.s1_idx.unique(), size=n_s1, replace=False)
    t = t[t.s1_idx.isin(s1s)]
    gt = pq.read_table("work/records/train_gt.parquet").to_pandas()
    gt = gt[gt.s1_idx.isin(s1s)]
    k = lambda a, b: np.asarray(a, np.int64) << 32 | np.asarray(b, np.int64)
    missed = gt[~np.isin(k(gt.s1_idx, gt.pool_idx), k(t.s1_idx, t.pool_idx))].assign(p=0.999, label=1)
    pos = pd.concat([t[t.label == 1][["s1_idx", "pool_idx", "p"]], missed[["s1_idx", "pool_idx", "p"]]])
    neg = t[t.label == 0].sort_values("p", ascending=False).groupby("s1_idx").head(2)[["s1_idx", "pool_idx", "p"]]
    trip = pos.merge(neg, on="s1_idx", suffixes=("_pos", "_neg"))
    trip = trip.sample(frac=1.0, random_state=seed).groupby(["s1_idx", "pool_idx_pos"]).head(1)  # one neg per positive
    print(f"S1 {n_s1:,}: positives {len(pos):,} ({len(missed):,} missed by blocking), triplets {len(trip):,}", flush=True)
    return trip.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--dump", action="store_true", help="only write work/retrieval/e5_distil_triplets.parquet (HF job input)")
    a = ap.parse_args()
    if a.dump:
        import pyarrow as pa
        trip = build(a.n_s1)
        s1_txt, pool_txt = record_text("train", "s1"), record_text("train", "pool")
        pq.write_table(pa.table({"q": ("query: " + s1_txt[trip.s1_idx.to_numpy()]).tolist(),
                                 "p": ("query: " + pool_txt[trip.pool_idx_pos.to_numpy()]).tolist(),
                                 "n": ("query: " + pool_txt[trip.pool_idx_neg.to_numpy()]).tolist(),
                                 "tp": trip.p_pos.to_numpy(np.float32), "tn": trip.p_neg.to_numpy(np.float32)}),
                       "work/retrieval/e5_distil_triplets.parquet", compression="zstd")
        print("wrote work/retrieval/e5_distil_triplets.parquet")
        return
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev)
    trip = build(a.n_s1)
    s1_txt, pool_txt = record_text("train", "s1"), record_text("train", "pool")
    q = ("query: " + s1_txt[trip.s1_idx.to_numpy()]).tolist()
    p = ("query: " + pool_txt[trip.pool_idx_pos.to_numpy()]).tolist()
    n = ("query: " + pool_txt[trip.pool_idx_neg.to_numpy()]).tolist()
    tp, tn = trip.p_pos.to_numpy(np.float32), trip.p_neg.to_numpy(np.float32)

    def enc(texts):
        b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = int(len(trip) * a.epochs) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06, anneal_strategy="linear")
    model.train()
    t0, log = time.time(), []
    order = np.random.default_rng(1).permutation(len(trip))
    for st in range(steps):
        idx = order[(st * a.batch) % len(trip):][:a.batch]
        eq, ep, en = enc([q[i] for i in idx]), enc([p[i] for i in idx]), enc([n[i] for i in idx])
        cand = torch.cat([ep, en])
        logits = eq @ cand.T / 0.05
        lab = torch.arange(len(idx), device=dev)
        nce = torch.nn.functional.cross_entropy(logits, lab)
        margin_s = (eq * ep).sum(1) - (eq * en).sum(1)
        margin_t = torch.from_numpy(tp[idx] - tn[idx]).to(dev)
        mse = torch.nn.functional.mse_loss(margin_s, margin_t)
        loss = nce + a.lam * mse
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 50 == 0 or st == steps - 1:
            log.append({"step": st, "nce": float(nce), "mse": float(mse), "s": round(time.time() - t0, 1)})
            print(f"step {st}/{steps} nce {float(nce):.4f} marginmse {float(mse):.4f} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    json.dump({"base": MODEL, "train_set": "DEV_REST_REST fold 1 (disjoint from MATCH_TRAIN)", "n_s1": a.n_s1,
               "triplets": int(len(trip)), "batch": a.batch, "lr": a.lr, "epochs": a.epochs, "lam": a.lam,
               "loss": "InfoNCE(t=0.05, in-batch pos+neg) + lam * MarginMSE(cos margin vs c0 p margin)",
               "teacher": "ce_rec_c0 p_ce (out-of-sample on fold 1)", "log": log}, open(OUT / "train_meta.json", "w"), indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

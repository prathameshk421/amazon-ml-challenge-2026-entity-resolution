"""M029: MLP in place of / blended with LightGBM on M027's setup (68 live features + ce_rec_c0 p_ce, MATCH_TRAIN
180K S1, M027's 5 GroupKFold folds and early-stop split). Laptop (MPS/CPU), no HF.

  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/nn_exp.py

Preprocessing per fold (fit rows only): NaN indicator per column that has NaN, QuantileTransformer (normal output,
1000 quantiles, 200K-row subsample), NaN -> 0. MLP d->256->256->128->1, SiLU, dropout 0.1, AdamW 2e-3, batch 2048,
BCE; early stop on the fold's early-stop S1 (patience 2, max 20 epochs).
Runs written like exp-train (work/match/runs/<id>/: models/, oof.parquet, OOF_per_entity.npy, queries.npy;
artifacts/metrics/exp_<id>.json), each with isotonic calibration + choose_decision + paired delta vs M027:
  M029   MLP alone
  M029b  0.5 * LightGBM (M027 OOF p_raw) + 0.5 * MLP, raw-probability average
"""
import json
import pickle
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer

from ber import exp, io_utils, match

PARENT = "M027"
GROUPS = ["cex_p:ce_rec_c0"]


class Prep:
    def fit(self, X):
        self.nan_cols = np.flatnonzero(np.isnan(X).any(axis=0))
        rng = np.random.default_rng(0)
        sub = X[rng.choice(X.shape[0], size=min(200_000, X.shape[0]), replace=False)]
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10 ** 9,
                                      random_state=0).fit(sub)
        return self

    def transform(self, X):
        Z = np.nan_to_num(self.qt.transform(X), nan=0.0)
        return np.hstack([Z, np.isnan(X[:, self.nan_cols]).astype(np.float64)]).astype(np.float32)


class PLR(torch.nn.Module):
    """Periodic numeric embeddings (Gorishniy et al. 2022, MLP-PLR): x_j -> ReLU(W_j [sin(2 pi c_j x_j), cos(...)])."""
    def __init__(self, d, k=16, out=16, sigma=0.1):
        super().__init__()
        self.c = torch.nn.Parameter(torch.randn(d, k) * sigma)
        self.w = torch.nn.Parameter(torch.randn(d, 2 * k, out) / (2 * k) ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(d, out))

    def forward(self, x):
        v = 2 * torch.pi * x.unsqueeze(-1) * self.c
        e = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return torch.relu(torch.einsum("bdk,dko->bdo", e, self.w) + self.b).flatten(1)


def mlp(d, arch="mlp"):
    if arch == "plr":
        return torch.nn.Sequential(PLR(d), torch.nn.Linear(d * 16, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                                   torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                                   torch.nn.Linear(256, 1))
    return torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Linear(128, 1))


def predict(model, Z, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, Z.shape[0], 65536):
            out.append(torch.sigmoid(model(torch.from_numpy(Z[i:i + 65536]).to(dev))).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_fold(Zf, yf, Ze, ye, dev, lg, k, H):
    torch.manual_seed(42 + k)
    model = mlp(Zf.shape[1], H["arch"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=H["lr"], weight_decay=H["wd"])
    steps = H["max_epochs"] * ((Zf.shape[0] + H["batch"] - 1) // H["batch"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if H["cosine"] else None
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(Zf), torch.from_numpy(yf.astype(np.float32))
    best, best_state, bad, epochs = np.inf, None, 0, []
    for ep in range(H["max_epochs"]):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], H["batch"]):
            idx = perm[i:i + H["batch"]]
            opt.zero_grad()
            loss = lossf(model(Xt[idx].to(dev)).squeeze(1), yt[idx].to(dev))
            loss.backward()
            opt.step()
            if sched:
                sched.step()
        p = np.clip(predict(model, Ze, dev), 1e-7, 1 - 1e-7)
        ll = float(-np.mean(ye * np.log(p) + (1 - ye) * np.log(1 - p)))
        epochs.append(ll)
        lg.info("fold %d epoch %d early-stop logloss %.5f", k, ep, ll)
        if ll < best - 1e-5:
            best, bad = ll, 0
            best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= H["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, epochs


def score_run(run, p_raw, s1, pool, y, queries, mcfg, extra, lg):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_raw, y)
    calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    p_cal = match.calibrate(p_raw, calib)
    decision, per_entity, report, rule_scores, sweep = exp.choose_decision(queries, s1, pool, y, p_cal, mcfg)
    rd = match.runs_dir(run)
    (rd / "models").mkdir(parents=True, exist_ok=True)
    np.save(rd / "queries.npy", queries)
    np.save(rd / "OOF_per_entity.npy", per_entity)
    pq.write_table(pa.table({"s1_idx": s1, "pool_idx": pool, "y": y, "p_raw": p_raw.astype(np.float32),
                             "p_cal": p_cal}), rd / "oof.parquet", compression="zstd")
    vp = exp.compare_to_parent(PARENT, queries, per_entity)
    io_utils.atomic_write_json(rd / "models" / "nn_calibration.json", {"calibration": calib, "decision": decision, **extra})
    io_utils.write_metrics(f"exp_{run}", {"run": run, "decision": decision, "oof": report, "vs_parent": vp,
                                          "oof_rule_scores": rule_scores, **extra}, queries.size, int(y.sum()))
    lg.info("%s: OOF F0.5 %.5f; vs %s %s; band %s", run, report["f05"], PARENT, vp["all"], vp["band_s1"])
    return report["f05"], vp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="M029")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "plr"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--cosine", action="store_true")
    a = ap.parse_args()
    H = {"arch": a.arch, "lr": a.lr, "wd": a.wd, "batch": a.batch, "max_epochs": a.max_epochs, "patience": a.patience,
         "cosine": a.cosine}
    RUN = a.run
    t0 = time.time()
    lg = io_utils.get_logger(f"nn_exp_{RUN}")
    cfg, mcfg = io_utils.load_config(), match.load_match_config()
    fp = match.bucketing_fingerprint(cfg)
    Xf, names, s1, pool, meta = exp.full_matrix("MATCH_TRAIN", GROUPS, fp, lg)
    X = Xf.astype(np.float64)
    y = match.pair_labels(s1, pool)
    queries = match.set_queries("MATCH_TRAIN")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lg.info(RUN + ": %d rows, %d features, device %s", y.size, X.shape[1], dev)
    rd = match.runs_dir(RUN)
    (rd / "models").mkdir(parents=True, exist_ok=True)
    oof = np.zeros(y.size)
    folds = []
    for k, tr, te, fit_i, es_i in exp._folds(y, s1, mcfg):
        prep = Prep().fit(X[fit_i])
        model, best, epochs = train_fold(prep.transform(X[fit_i]), y[fit_i], prep.transform(X[es_i]),
                                         y[es_i].astype(np.float64), dev, lg, k, H)
        oof[te] = predict(model, prep.transform(X[te]), dev)
        torch.save(model.state_dict(), rd / "models" / f"mlp_fold{k}.pt")
        pickle.dump(prep, open(rd / "models" / f"prep_fold{k}.pkl", "wb"))
        folds.append({"fold": k, "epochs": len(epochs), "best_es_logloss": best, "es_curve": epochs,
                      "hit_max_epochs": len(epochs) == H["max_epochs"]})
        lg.info("fold %d done: %d epochs, best early-stop logloss %.5f (%.0fs)", k, len(epochs), best, time.time() - t0)
    json.dump({"features": names, "groups": GROUPS, "hparams": H, "folds": folds},
              open(rd / "models" / "nn_meta.json", "w"), indent=1)
    extra = {"folds": folds, "groups": GROUPS, "features_used": names, "hparams": H}
    score_run(RUN, oof, s1, pool, y, queries, mcfg, extra, lg)
    par = pq.read_table(match.runs_dir(PARENT) / "oof.parquet").to_pandas()
    if not (np.array_equal(par.s1_idx.to_numpy(), s1) and np.array_equal(par.pool_idx.to_numpy(), pool)):
        raise RuntimeError("M027 OOF rows differ")
    lgb_raw = par.p_raw.to_numpy().astype(np.float64)
    lg.info("raw-probability correlation LightGBM vs MLP: %.5f", np.corrcoef(lgb_raw, oof)[0, 1])
    for w, suf in ((0.5, "b"), (0.25, "b25"), (0.75, "b75")):  # w = MLP weight; 0.5 fixed before the run
        score_run(RUN + suf, (1 - w) * lgb_raw + w * oof, s1, pool, y, queries, mcfg,
                  dict(extra, blend={"M027": 1 - w, RUN: w}, corr_lgb_mlp=float(np.corrcoef(lgb_raw, oof)[0, 1])), lg)
    lg.info("total %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()

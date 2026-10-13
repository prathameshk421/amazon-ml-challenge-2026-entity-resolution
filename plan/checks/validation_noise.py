"""Noise floor for the validation plan (plan/VALIDATION_PLAN.md §6).

Scores the frozen matcher on MATCH_VALID (train-S1 slice; labels allowed) and writes
artifacts/metrics/validation_noise.json with:
  - per-entity F0.5 mean / std, standard error at several set sizes (the minimum detectable delta),
  - bootstrap 95% CI of the mean,
  - paired bootstrap of one_owner on vs off (example of a paired comparison),
  - singleton vs matched components for the singleton-rate reweighting.

Run from the repo root:
  PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/validation_noise.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from ber import io_utils, match

SET = "MATCH_VALID"
N_BOOT = 2000
SEED = 0


def boot_ci(x, rng, n_boot=N_BOOT):
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    cfg = io_utils.load_config()
    doc, models_dir = match.load_matcher(cfg)
    X, s1, pool, meta, p_cal, keep = match.score_set(SET, doc, models_dir)
    queries = match.set_queries(SET)
    y = match.pair_labels(s1, pool)
    n_true = match.true_counts(queries)
    scorer = match.Scorer(queries, s1, y, n_true)
    f, _, _ = scorer.per_entity(keep)

    dec_off = dict(doc["decision"], one_owner=False)
    f_off, _, _ = scorer.per_entity(match.apply_decision(dec_off, s1, pool, p_cal))
    diff = f - f_off

    rng = np.random.default_rng(SEED)
    sd = float(f.std(ddof=1))
    single = n_true == 0
    out = {
        "set": SET,
        "decision": doc["decision"],
        "f05": float(f.mean()),
        "per_entity_std": sd,
        "ci95_bootstrap": boot_ci(f, rng),
        "se_at_n": {str(n): sd / np.sqrt(n) for n in (20_000, 200_000, 441_363, 1_655_458)},
        "share_entities_f_eq_1": float((f == 1.0).mean()),
        "components": {"singleton_f05": float(f[single].mean()), "matched_f05": float(f[~single].mean()),
                       "singleton_rate": float(single.mean())},
        "paired_one_owner_on_minus_off": {
            "delta": float(diff.mean()),
            "ci95_paired_bootstrap": boot_ci(diff, rng),
            "n_entities_changed": int((diff != 0).sum()),
            "diff_std": float(diff.std(ddof=1)),
            "unpaired_se_would_be": float(np.sqrt(2) * sd / np.sqrt(f.size)),
        },
    }
    io_utils.write_metrics("validation_noise", out, n_queries=int(queries.size), n_true_pairs=int(n_true.sum()))
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

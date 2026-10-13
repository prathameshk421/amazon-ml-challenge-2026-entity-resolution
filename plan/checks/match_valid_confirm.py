"""Mini-lockbox confirmation (EXPERIMENT_ROUTINE §6): score an adopted exp run once on MATCH_VALID (20,000 S1),
paired against x1_strsim's MATCH_VALID per-entity scores. Nothing is tuned here.
Run: PYTHONPATH=submission/code/business_entity_resolution/src .venv/bin/python plan/checks/match_valid_confirm.py RUN"""
import sys

import numpy as np

from ber import exp, io_utils, match


def main():
    run = sys.argv[1]
    cfg = io_utils.load_config()
    md = match.runs_dir(run) / "models"
    doc = io_utils.read_json(md / "matcher.json")
    _, _, s1, pool, _, p_cal, _ = exp.predict_run("MATCH_VALID", doc, md)
    q = match.set_queries("MATCH_VALID")
    y = match.pair_labels(s1, pool)  # scoring only
    f = match.Scorer(q, s1, y, match.true_counts(q)).per_entity(match.apply_decision(doc["decision"], s1, pool, p_cal))[0]
    base = np.load(match.runs_dir("x1_strsim") / "MATCH_VALID_per_entity.npy")
    np.save(match.runs_dir(run) / "MATCH_VALID_per_entity.npy", f)
    out = {"run": run, "f05": float(f.mean()), "f05_x1_strsim": float(base.mean()),
           "vs_x1_strsim": match.paired_delta(f, base, n_boot=2000)}
    io_utils.write_metrics(f"match_valid_confirm_{run}", out, q.size, int(match.true_counts(q).sum()), cfg)
    print(out)


if __name__ == "__main__":
    main()

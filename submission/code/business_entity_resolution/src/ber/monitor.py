"""CLI `monitor` (plan §5.10 M9): per-country-label diagnostics of the TEST candidate set, unsupervised
over `work/candidates/TEST/query_stats.parquet` (candidate-list size, p_rerank at rank 1-5, max p_rerank,
share of empty lists, share of queries that hit `max_postings_per_query`); test labels are never read.
Alarm rule: a country's mean max-p < 0.5x the train-covered-country (US/India) mean, or its empty-list
share > 3x theirs.

Plan text says an alarm is "written to PROGRESS.md"; this script cannot safely edit a human-maintained
markdown document from an automated run, so alarms are written into `artifacts/metrics/monitor_test.json`
instead (every other number in this pipeline already goes there first, plan R5/R6) and logged at WARNING
level -- the operator transcribes any alarm into PROGRESS.md. An alarm never changes the pipeline
automatically (plan §5.10).
"""

import numpy as np
import pyarrow.parquet as pq

from ber import io_utils

TRAIN_COUNTRIES = ("US", "India")
RANKS = range(1, 6)


def _summary(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90))}


def per_country_stats(country, size, max_p, rank_p, n_dropped):
    """{country_label: {...}} for the §5.10 metric set. rank_p: {rank: array}."""
    country = np.asarray(country)
    out = {}
    for c in sorted(set(country.tolist())):
        m = country == c
        out[c] = {
            "n_queries": int(m.sum()),
            "size": _summary(size[m]),
            "max_p": _summary(max_p[m]),
            "p_at_rank": {r: _summary(rank_p[r][m]) for r in RANKS},
            "share_empty": float((size[m] == 0).mean()),
            "share_hit_postings_limit": float((n_dropped[m] > 0).mean()),
        }
    return out


def alarms_for(per_country, train_covered):
    """(ref_mean_max_p, ref_empty_share, [alarm strings]) per the §5.10 rule, using the mean over the
    train-covered countries (US/India) present in this run as the reference."""
    ref_max_p = [per_country[c]["max_p"]["mean"] for c in train_covered
                if per_country[c]["max_p"]["mean"] is not None]
    ref_empty = [per_country[c]["share_empty"] for c in train_covered]
    ref_mean_max_p = float(np.mean(ref_max_p)) if ref_max_p else None
    ref_empty_share = float(np.mean(ref_empty)) if ref_empty else None
    alarms = []
    for c, d in per_country.items():
        mp = d["max_p"]["mean"]
        if ref_mean_max_p is not None and mp is not None and mp < 0.5 * ref_mean_max_p:
            alarms.append(f"{c}: mean max-p {mp:.4f} < 0.5x train-covered mean {ref_mean_max_p:.4f}")
        if ref_empty_share is not None and d["share_empty"] > 3 * ref_empty_share:
            alarms.append(f"{c}: empty-list share {d['share_empty']:.4f} > 3x train-covered mean "
                          f"{ref_empty_share:.4f}")
    return ref_mean_max_p, ref_empty_share, alarms


def run_monitor(args):
    lg = io_utils.get_logger("monitor")
    cfg = io_utils.load_config()
    stats_path = io_utils.work_path("candidates", "TEST", "query_stats.parquet")
    if not stats_path.exists():
        raise RuntimeError(f"monitor requires {stats_path} -- run `candidates --set TEST` first "
                          "(it writes query_stats.parquet for the TEST split only)")
    t = pq.read_table(stats_path)
    country = t.column("country_raw").to_numpy()
    size = t.column("size").to_numpy()
    max_p = t.column("max_p").to_numpy()
    n_dropped = t.column("n_dropped").to_numpy()
    rank_p = {r: t.column(f"p_rank{r}").to_numpy() for r in RANKS}

    per_country = per_country_stats(country, size, max_p, rank_p, n_dropped)
    train_covered = [c for c in TRAIN_COUNTRIES if c in per_country]
    ref_mean_max_p, ref_empty_share, alarms = alarms_for(per_country, train_covered)

    payload = {"train_covered_countries": train_covered, "ref_mean_max_p": ref_mean_max_p,
               "ref_empty_share": ref_empty_share, "per_country": per_country, "alarms": alarms,
               "max_postings_per_query": cfg["max_postings_per_query"]}
    path = io_utils.write_metrics("monitor_test", payload, int(size.size), 0, cfg)
    if alarms:
        lg.warning("ALARM (plan §5.10, needs human review, not auto-actioned): %s", "; ".join(alarms))
    lg.info("monitor done: %s -> %s", sorted(per_country), path)

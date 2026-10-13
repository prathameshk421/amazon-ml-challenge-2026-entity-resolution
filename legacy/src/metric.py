"""Per-S1-entity F0.5 metric (spec: notes/problem_statement.md + prompt section 4).

F0.5 = 1.25*P*R / (0.25*P + R), computed PER S1 ENTITY,
then macro-averaged over ALL S1 entities in the split (singletons included).
Entities with no candidates count as empty predictions.
"""

from __future__ import annotations


def f05_single(pred: set, truth: set) -> float:
    """F0.5 for one S1 entity given pred/truth sets of matched IDs."""
    pred = set(pred)
    truth = set(truth)
    if not pred and not truth:
        return 1.0
    if not truth and pred:
        return 0.0
    if not pred and truth:
        return 0.0
    inter = len(pred & truth)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(truth) if truth else 0.0
    if p * r == 0.0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def pr_single(pred: set, truth: set):
    """Return (P, R, F0.5) for one S1 entity."""
    pred = set(pred)
    truth = set(truth)
    if not pred and not truth:
        return 1.0, 1.0, 1.0
    if not truth and pred:
        return 0.0, 1.0, 0.0
    if not pred and truth:
        return 1.0, 0.0, 0.0
    inter = len(pred & truth)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(truth) if truth else 0.0
    if p * r == 0.0:
        return p, r, 0.0
    return p, r, 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_dict: dict, truth_dict: dict, all_entities=None) -> dict:
    """Macro-average P/R/F0.5 over ALL S1 entities in the split.

    pred_dict/truth_dict: {s1_id: set-of-ids}. Missing key = empty set
    (covers entities with zero candidates). all_entities optionally gives
    the full entity universe; defaults to union of both dicts' keys.
    Returns dict with F0.5, P, R, n_entities, singleton_acc.
    singleton_acc = fraction of true-singleton entities predicted empty.
    """
    if all_entities is None:
        all_entities = set(truth_dict) | set(pred_dict)
    else:
        all_entities = list(all_entities)
    n = len(all_entities)
    sp = sr = sf = 0.0
    sing_ok = sing_n = 0
    nonsing_scores = []
    sing_scores = []
    for e in all_entities:
        t = set(truth_dict.get(e, ()))
        p_ = set(pred_dict.get(e, ()))
        p, r, f = pr_single(p_, t)
        sp += p
        sr += r
        sf += f
        if len(t) == 0:
            sing_n += 1
            sing_scores.append(f)
            if len(p_) == 0:
                sing_ok += 1
        else:
            nonsing_scores.append(f)
    import statistics

    def _mean(x):
        return float(sum(x) / len(x)) if x else 0.0

    return {
        "F0.5": sf / n if n else 0.0,
        "P": sp / n if n else 0.0,
        "R": sr / n if n else 0.0,
        "n_entities": n,
        "singleton_acc": sing_ok / sing_n if sing_n else 1.0,
        "n_singleton": sing_n,
        "n_nonsingleton": n - sing_n,
        "F0.5_singleton": _mean(sing_scores),
        "F0.5_nonsingleton": _mean(nonsing_scores),
    }


def candidate_recall(cand_dict: dict, truth_dict: dict, all_entities=None) -> dict:
    """Pair-level recall + entity-level recall of a candidate set."""
    if all_entities is None:
        all_entities = list(truth_dict.keys())
    tp = tot = 0
    ent_hit = ent_n = 0
    for e in all_entities:
        t = set(truth_dict.get(e, ()))
        c = set(cand_dict.get(e, ()))
        if t:
            ent_n += 1
            hit = len(t & c)
            tp += hit
            tot += len(t)
            if hit > 0:
                ent_hit += 1
    return {
        "pair_recall": tp / tot if tot else 1.0,
        "pair_hit": tp,
        "pair_total": tot,
        "entity_recall": ent_hit / ent_n if ent_n else 1.0,
        "entity_hit": ent_hit,
        "entity_total": ent_n,
    }

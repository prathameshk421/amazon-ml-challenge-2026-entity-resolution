# Experiment routine — stage-3 features and matcher, ≤ 1 h per run

Status: PLAN (2026-09-26). Nothing here is implemented unless marked **exists**.
Priority: `AGENTS.md` > `split.md` > `plan/VALIDATION_PLAN.md` > this file. This file is the day-to-day
loop; VALIDATION_PLAN §5 tier T1 is what it runs, T2/T3 are the end-of-competition steps (§7 here).

---

## 1. Where submission 1 spent its time (measured)

| Step | Wall | Source |
|---|---|---|
| Candidates, TEST (1.73M S1) | 8,463 s | `candidates_TEST.json` |
| Match predict + write, TEST | 1,806 s | `match_predict_TEST.json` |
| Candidates, MATCH_TRAIN (180K) | 938 s | `candidates_MATCH_TRAIN.json` |
| Stage-3 features, MATCH_TRAIN (975,699 pairs) | 83 s | `match_features_MATCH_TRAIN.json` |
| Stage-3 train, 5-fold OOF + sweeps + error export | 988 s | `match_train.json` |

Stage 1/2 and test inference are ~85% of the wall time. A stage-3 experiment on frozen candidates
is already ~18 min. So the routine is: **freeze everything upstream, touch only stage-3 features,
model, and decision rule, and never run test inside the loop.**

## 2. What the matcher is actually losing (measured, `match_train.json` OOF, 180K queries)

Stage 2 hands stage 3 **5.42 candidates per S1, 3.45 true** (positive rate 0.62), not "15 for 5".

OOF entity F0.5 0.97412, ceiling 0.99298, total loss 0.02588:

| Loss bucket | Entities | Loss | Share |
|---|---|---|---|
| matched, missed trues only (FN) | 16,966 | 0.01219 | 47% |
| blocking miss only (stage 3 cannot fix) | — | 0.00625 | 24% |
| matched, FP only | 3,895 | 0.00513 | 20% |
| singleton with any FP (entity scores 0) | 248 | 0.00138 | 5% |
| FP and missed | 505 | 0.00093 | 4% |

Per event, F0.5 does punish FPs more: on an entity with 4 trues, one extra FP costs 0.167, one missed
true costs 0.0625 (2.7×). That asymmetry is already applied by the decision threshold (0.70, chosen by
entity-F0.5 sweep; pairwise precision 0.9916 vs recall-in-candidates 0.968). What remains is dominated
by FNs: 19,275 missed pairs, median p_matcher 0.33 — far below the threshold, so no threshold move
recovers them without adding FPs. They need features that separate them.

**Consequence for the model:** keep training on unweighted binary logloss (calibrated p; reliability
bins match within 0.001 on OOF) and put the F0.5 asymmetry in the decision rule (per-entity
expected-F0.5, **exists**, 0.97433 vs 0.97412 threshold, both with one-owner). Down-weighting positives
in training only shifts where the threshold lands (c) and breaks calibration that expected-F needs.

## 3. Frozen inputs

- Stage 1+2 config hash `fc7811e…`; candidates in `work/candidates/MATCH_TRAIN|MATCH_VALID` (**exist**).
- **EXP set = MATCH_TRAIN, 180,000 S1, 975,699 pairs, 5-fold GroupKFold by S1, `fold_seed` 42**
  (**exists**; baseline OOF 0.97412 reproduces with no new run). Folds never change, so every run is
  scored on the same entities and comparisons are paired.
- **Mini-lockbox = MATCH_VALID, 20,000 S1**: scored only when a change is adopted (48 s,
  `match-eval`), never used to choose. Val S1 stays untouched until §7.
- Upstream changes (normalization, keys, quotas, stage 2) are **out of this loop**; they need new
  candidates and follow VALIDATION_PLAN §5.1.

## 4. One experiment = one run directory

```
exps/M014.json          # parent run, feature groups, lgb params, decision rule, one-line hypothesis
work/exp/runs/M014/     # oof.parquet, per_entity.parquet, models/
artifacts/metrics/exp_M014.json
reports/exp/M014/       # error files + plots below
```

Feature groups are modules `ber/feats/<group>.py` with `compute(cand, ctx) -> DataFrame` (one row per
candidate pair, same row order as the frozen candidate table). Each group is cached to
`work/exp/features/<group>@<code_sha1>.parquet`; a run concatenates the base table (existing 50
columns) with the cached groups it lists. Only new or edited groups are recomputed.

Command: `python -m ber exp run exps/M014.json [--quick]`.

| Step | Budget |
|---|---|
| Compute new feature group(s) over 975,699 pairs | ≤ 15 min (split the group if slower) |
| 5-fold LightGBM, lr 0.05, early stop 100 | ~16 min (a: 988 s incl. sweeps) |
| Isotonic calibration, threshold + expected-F sweep, one-owner | ~1 min |
| Reports (§5) | ≤ 5 min |
| **Total** | **~35–40 min** |

`--quick`: first 30,000 S1 of EXP, 3 folds, lr 0.1. ~3 min. For debugging a feature, not for decisions
(unpaired SE ≈ 0.0006 at 30K, too wide for typical gains).

## 5. What every run writes

Scorecard `exp_<id>.json` (via `io_utils.write_metrics`, so it carries config_hash + git_commit):

1. Entity F0.5 (headline) with bootstrap 95% CI; ceiling; the §2 loss table; P / R in candidates.
2. **Paired Δ vs parent**: mean of per-entity (F_new − F_parent), bootstrap CI over S1 rows (2,000
   resamples), entities improved / worsened, Δ per segment (country, chain size, match count,
   singleton, name_unseen, empty pool address, Indic pool name — segments **exist** in `match_train`).
3. **Threshold curve on the metric we are scored on**: entity F0.5 vs threshold (0.05–0.95, **exists**
   as `oof_threshold_sweep`) plotted next to the pair-level PR curve. Threshold is chosen from the
   entity curve, not the PR curve: PR optimizes pairwise precision/recall, the contest averages per S1.
   The width of the plateau (thresholds within 0.0002 of best) is reported: a narrow peak means the
   chosen threshold will not transfer to test.
4. Calibration reliability (**exists**) and gain importance (**exists**) plus mean |SHAP| per feature
   (LightGBM `pred_contrib=True`, no new library).

Error files in `reports/exp/<id>/` (extend `export_errors`, **exists** for FP/FN/blocking CSVs):

- `fp.csv`, `fn.csv`: both records, p_matcher, p_rerank, rank, **all feature values**, and the top-5
  SHAP contributions as text (`a_num_jacc −1.8; n_tset +0.9 …`) so a reader sees why the model decided.
  Sorted by confidence (FP by p desc, FN by p asc). Capped at 20,000 rows each.
- `flips.csv`: pairs whose decision changed vs parent, tagged fixed / broken. This is the file to read
  after every run: it shows what the new feature actually did.
- `error_tags.json`: counts of FP and FN per rule-based tag, so a hypothesis has a before/after number.
  Tags (regex/normalizer based, reuse `textnorm`): `legal_form_conflict` (Inc vs LLC, Pvt Ltd vs Ltd),
  `house_num_mismatch`, `house_num_digit_drop` (106 vs 06), `pool_addr_empty`, `name_unrelated_same_addr`
  (name token-set < 0.2, address token-set > 0.8), `script_mismatch` (Latin vs Devanagari/Bengali),
  `name_subset` (Bay Smart ⊂ Bay Smart Biologics), `singleton_fp`.

`reports/experiments.md` gets one generated row per run: id, parent, hypothesis, F0.5, Δ vs parent
[CI], fixed/broken, threshold, wall time, verdict.

After reading `flips.csv` and `error_tags.json`, write an entry in `RESEARCH_LOG.md` (repo root) using
its template: hypothesis, change, result with the metrics file cited, what was learned, next step. This
applies to dropped, parked, and failed runs too. The generated table holds the numbers; the log holds
the reasoning.

## 6. Adoption rule (per run, on EXP OOF)

- **Adopt** (becomes the new parent): paired Δ > 0 with CI lower bound > 0, and no segment among
  country_US, country_India, singletons worse by > 0.002. Then score the mini-lockbox once and log it.
- **Park**: CI straddles 0. Keep the cached group; it can be retried in a later combination.
- **Drop**: CI upper bound < 0.
- Paired SE for a feature change is not measured yet (one-owner on/off had per-entity diff std 0.0016,
  but it touched 2 entities). The first adopted run measures it; if paired SE > 0.0003 at 180K, move the
  EXP set to FAST (200K) plus DEV_REST slices per VALIDATION_PLAN.
- **Label-shuffle canary** (VALIDATION_PLAN L8) once for every group that uses other candidates of the
  same S1 or a first-model score (list and stacked features): permute labels within each S1, retrain,
  F0.5 must fall to the p_rerank floor (0.9466 ± 0.002). Costs one run.
- Mini-lockbox drift: if MATCH_VALID gain over the original 0.97449 trails the cumulative OOF gain by
  > 0.002, the loop is overfitting its own OOF — stop adopting and review the last few adoptions.

## 7. From subset to full dataset (how the estimate extrapolates)

Two different things change when going "full size", and they are estimated separately:

1. **Scoring set size.** The metric is a mean over S1 rows. For a fixed model, the OOF mean on 180K
   random S1 is an unbiased estimate of the mean over all train S1, with unpaired SE 0.00024 (a:
   `validation_noise.json`, per-entity std 0.1006). No correction needed. One exception: one-owner is a
   set-level rule and behaves differently at 100% coverage (VALIDATION_PLAN P3/§5.3 global pass).
2. **Training set size.** Measured once with a learning curve on the current parent: train on
   22.5K / 45K / 90K / 144K S1 from the 4 fit folds, score all on the same held-out fold; fit
   F(n) = F∞ − a·n^(−b). Gives the expected gain from training on MATCH_ALL (1,655,458 S1). Re-measure
   only if the model family changes. Feature adoptions are assumed to keep their ranking at larger n
   (c); model-capacity changes (num_leaves, more rounds, a second-stage model) are not — gains from
   capacity grow with data, so those are re-checked at T2.
3. **Test shift that the subset cannot estimate**: France (unseen country), test pool density 5.75 vs
   4.68 pool records per S1. Proxies per VALIDATION_PLAN §5.4 (LOCO, list-size buckets), run for
   finalists only.
4. **Leaderboard offset**: log submission 1's public score against its OOF 0.97412 in
   `reports/test_usage_log.md`; the difference is the empirical OOF → test offset for this pipeline.

**End of competition (fixed sequence, one heavy job at a time):**

1. Generate DEV_REST candidates (1,455,458 S1, ~2.2 h, frozen stage 1+2). Do this **once, early**, in a
   background slot between experiments; nothing in the loop depends on it.
2. Take the best 2 parents from `reports/experiments.md`. Retrain each on MATCH_ALL 5-fold OOF (T2,
   ~2.5 h each). Pick by paired Δ on MATCH_ALL OOF; ties go to fewer features.
3. Score the winner once on val S1 via the file path (VALIDATION_PLAN §8 ledger). Gap alarm at 0.003.
4. Test inference and submission.

## 8. Feature backlog (from the error files, first pass)

Each entry names the pattern seen in `reports/match_errors/*.csv`, the feature, and evidence class.
All are (d) until a run measures them; the tag counts in §5 turn them into (a).

| # | Pattern (example from the CSVs) | Feature | Class |
|---|---|---|---|
| F1 | FP with near-identical name, house number differs by edit: 2565 vs 2578, 313 vs 31, Q.No 307 vs 310 | house-number relation: equal / digit dropped / substituted / disjoint; abs numeric diff; Levenshtein on the first number token | (d) |
| F2 | FP Inc vs LLC, Private Limited vs Limited | legal-form class per side (inc, llc, corp, pllc, ltd, pvt ltd, llp, none) and `legal_conflict` = both present and different | (d) |
| F3 | FN when pool address is empty, name equal up to suffix (Metropolitan Academy Inc / Incorporated) | name equality after legal-suffix canonicalisation; interaction flag `a_empty × n_eq_core` given as one column | (d) |
| F4 | Same address, unrelated coined name, sometimes true (Belodelta, Yumalyra), sometimes false (Vantagezeph, Jaxjax) | name "coinedness": in-vocabulary token share against the pool token counts (IDF table **exists**); chars-per-token; whether the pool name occurs elsewhere in the pool | (d) |
| F5 | Name subset: Bay Smart vs Bay Smart Biologics (FN), Saria Co Therapeutics (FP) | extra-token IDF mass on each side (not only the max): sum IDF of q-only and p-only tokens, split into legal / generic / distinctive | (d) |
| F6 | True matches of one S1 are duplicates of each other; the model scores each pair alone | **candidate-to-candidate**: similarity of this candidate to the highest-p_rerank candidate in the list (name tset, address tset, house number equal); number of other candidates it matches ≥ 0.8. ~30 pairs per S1, cheap | (c) |
| F7 | List context is already the 2nd–4th strongest signal (`l_p_ratio_max`, `l_p_gap_max`, `l_p_max` gain) | **stacked pass**: first matcher's OOF p for all candidates of the S1 → rank by p, gap to top, count of p > 0.5 among others; second LightGBM on base + these. Needs the canary | (c) |
| F8 | Metric is a macro mean over S1; LightGBM loss is a mean over pairs, so S1 with long lists weigh more | sample weight 1 / n_candidates(S1) | (c) |
| F9 | Indic script on one side (फ्यूचर कंस्ट्रक्शंस, লোটাস পাওয়ার) in both FN and FP | per-pair script pair (Latin–Latin, Latin–Devanagari, …) as a categorical; name features computed on the transliterated form reported separately from the raw form | (d) |

Order: F1, F2, F5 first (single-pair string features, one run each, direct tag counts), then F6, F8,
then F7 (costs the most, needs the canary). F9 depends on what `translit` already covers; check the
FN tag count first.

## 9. Implementation order

| Step | Work | Est. |
|---|---|---|
| 1 | `ber/exp.py`: read `exps/<id>.json`, load base features + cached groups, reuse `run_match_train` internals, write run dir | 2 h |
| 2 | Per-entity parquet + paired bootstrap + `experiments.md` generator (overlaps VALIDATION_PLAN step 4) | 1.5 h |
| 3 | Error files with SHAP text, `flips.csv`, `error_tags.json`, threshold/PR plot | 2 h |
| 4 | M000 = baseline through the new path; must reproduce OOF 0.97412 to 1e-6 | 20 min run |
| 5 | Learning curve (§7.2) | ~1 h run |
| 6 | Start DEV_REST candidates in a free slot (§7 step 1) | 2.2 h background |
| 7 | Backlog F1 → F9, one run each | ~40 min per run |

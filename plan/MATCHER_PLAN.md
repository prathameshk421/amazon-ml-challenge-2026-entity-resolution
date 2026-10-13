# Stage-3 Matcher Plan (candidate pairs → matching_results.tsv)

Status: plan v1, 2026-09-25. Supersedes the "Baseline Model Build Plan" draft (5 features, lowercase only).
Scope: everything after bucketing. Bucketing (stage 1 retrieval + stage-2 re-ranker + cut) is done and frozen;
see `plan/BUCKETING_PLAN.md`. Rules: `AGENTS.md` > `data/student_resource/dataset/split.md` > this file.

Every number below cites a metrics file. `artifacts/metrics/matcher_preflight.json` is written by
`plan/checks/matcher_preflight.py` (run it from repo root with `.venv/bin/python`; 8 s).

---

## 0. What the matcher receives (read this first)

```
BUCKETING (done)
  stage 1  normalize → key families → inverted index → ~264 candidates / S1
  stage 2  LightGBM re-ranker (35 cheap features) → p_rerank → cut (K_min=5, K_max=60, p_min=0.05)
           → ~5.42 candidates / S1  ==  candidate_pairs.tsv
MATCHER (this plan)
  stage 3  richer pair + list features → LightGBM → calibrated p → per-S1 decision → matching_results.tsv
```

| Fact | Value | Source |
|---|---|---|
| Candidates per S1 | mean 5.42; p50 5, p99 9, max 41. At least 5 per S1 whenever stage 1 found 5, because K_min=5 | `candidates_MATCH_TRAIN.json`, `rerank_eval_HOLDOUT_TEST.json` |
| Positive rate inside candidate lists | **0.6215** (606,417 / 975,699 rows, MATCH_TRAIN) | `matcher_preflight.json` |
| F0.5 ceiling (perfect matcher on these candidates) | MATCH_VALID 0.9928; val-S1 0.9928 | `matcher_preflight.json`, `rerank_eval_HOLDOUT_TEST.json` |
| Pair recall of the candidates | 0.9751 (MATCH_VALID) | `matcher_preflight.json` |
| **Re-ranker-score floor**: global threshold on `p_rerank`, no training | **0.9461** at t = 0.72 (MATCH_VALID, grid 0.05–0.95 step 0.01) | `matcher_preflight.json` |
| Predict-empty-for-everyone | 0.0576 (= singleton share) | `matcher_preflight.json` |

**The matcher must beat 0.9461 on MATCH_VALID.** The recoverable gap to the ceiling is 0.0467.

### 0.1 Where the floor loses F0.5 (MATCH_VALID, t = 0.72; loss = 1 − F0.5, averaged over 20,000 S1)

| Bucket | Entities | Loss |
|---|---|---|
| Matched S1, true match in candidates but p < t, no FP (**missed matches**) | 4,424 | **0.0308** |
| Matched S1, FP only | 797 | 0.0090 |
| Blocking miss only (true match not in candidates; the matcher cannot fix this) | 953 | 0.0053 |
| Matched S1, FP and FN | 250 | 0.0045 |
| Singleton predicted non-empty | 87 | 0.0044 |
| **Total** | | **0.0539** |

By S1 chain size: chain 1: 0.0284, chain 2–9: 0.0134, chain ≥ 10: 0.0120.

Consequence: F0.5 is precision-weighted, but on these candidates **57% of the loss is missed matches**.
The matcher's job is to raise recall on candidates the re-ranker under-scored without adding false merges.
Rebalancing toward positives (`is_unbalance`) is the wrong tool: positives are already the majority class (62%).

---

## 1. Data: which S1 goes where (do not deviate)

Slices are in `work/splits/query_sets.json` (`sets` key). All are drawn from train S1 only, except val-S1.

| Slice | S1 count | Matcher use | Why |
|---|---|---|---|
| MATCH_TRAIN | 180,000 | fit (5-fold OOF, grouped by S1), isotonic calibration on OOF, decision-rule tuning on OOF, error export | never seen by stage 1/2 training → `p_rerank` is out-of-sample, as on test |
| MATCH_VALID | 20,000 | one confirmation of the tuned pipeline (like DEV_B for bucketing); floor comparison | untouched by fitting and tuning |
| val-S1 (HOLDOUT_TEST) | 441,363 | **headline score, once, at the end. Labels for scoring only** | split.md rule 4 |
| DEV_REST | 1,455,458 | optional extra training data (experiment M4) | out-of-sample for stage 1/2 |
| RR_TRAIN, RR_CAL, S1R_TRAIN, DEV_A, DEV_B | — | **never use for matcher training** | stage 1/2 were trained or tuned on them → `p_rerank` in-sample, optimistic |
| TEST | 1,732,544 | inference only | no labels exist |

Forbidden (all from AGENTS.md / split.md):
- val-S1 for early stopping, threshold choice, calibration, feature selection, or error inspection.
- Tuning or inspecting test.
- Country one-hot or country filtering. Country appears only as the 3-state feature `same / different / unknown`.
- External data, APIs, gazetteers, hosted LLMs.

Candidate files (parquet, columns `s1_idx:int32, pool_idx:uint32, rank:int16, p_rerank:float32`):
`work/candidates/{MATCH_TRAIN, MATCH_VALID, HOLDOUT_TEST, TEST}/part-*.parquet`. MATCH_TRAIN and
MATCH_VALID exist (21:06, 20:51). HOLDOUT_TEST exists. TEST is being generated (`candidates --set TEST`).

Labels: `work/records/train_gt.parquet` (`s1_idx, pool_idx`). Label = 1 if `(s1_idx, pool_idx)` is a row there.
Text:
- S1: `work/norm/{split}_s1.parquet` (`name_norm, addr_norm, name_indic, addr_indic`) and
  `work/records/{split}_s1.parquet` (`name_raw, addr_raw, country_raw`).
- Pool: `ber.rerank.PoolTextReader(split)` (`.name(idx)`, `.addr(idx)`, `.country(idx)`, `.src(idx)`,
  `.indic(idx)`) and `work/records/{split}_pool.parquet` for raw text.
- S1 chain size: `work/norm/{split}_s1_chain.npy`.

**Use `*_norm` text, not lowercased raw text.** textnorm v2.1 romanizes Indic script (752,869 train-pool
names are Indic, S1 is Latin only), folds accents, splits letter|digit runs, and expands abbreviations.
Lowercased raw text gives about 0 token overlap on every Indic pair.

---

## 2. Features (`ber/match.py: match_features`)

Missing values are NaN, not 0; LightGBM routes NaN on its own. Every definition applies identically to
train and test, and none mentions a country name.

### 2A. From bucketing (free)
`p_rerank`, `logit_p_rerank`, `rank`.

### 2B. Name (on `name_norm`)
- rapidfuzz: `token_set_ratio`, `token_sort_ratio`, `ratio`, `partial_ratio` (÷100).
- `jw_sorted`: Jaro-Winkler on the sorted token concatenation (handles word-order transposition).
- `tok_jacc`: token-set Jaccard. `tok_jacc_idf`: IDF-weighted Jaccard, Σ idf(shared) / Σ idf(union),
  where idf = ln(N_pool / df) is computed on the **same split's pool** (unsupervised, split.md rule 5).
- `name_eq`: exact equality. `name_eq_core`: equality after dropping edge tokens. Use the data-mined
  edge-token list `ber.keys.edge_token_set` (config `edge_tokens`), not a hand-written list of legal
  suffixes, so French `sarl`/`sas` are handled the same way as `llc`/`pvt`.
- `acronym`: 1 if the initials of one name (≥ 2 tokens) equal a token of the other.
- `first_tok_eq`, `longest_shared_tok_len`, `q_ntok`, `p_ntok`.
- `sk_jacc`: consonant-skeleton Jaccard (reuse `ber.textnorm.skeleton`), for transliteration variants.
- `pool_indic`: pool name was romanized from Indic script.

### 2C. Address (on `addr_norm`; `<NULL>` is already replaced by a space in `textnorm`)
- `a_token_set`, `a_token_sort` (÷100), `a_tok_jacc_idf`.
- Numbers: `num_jacc` = Jaccard of digit tokens (NaN if both sides have none). `num_only_q`, `num_only_p` (counts).
- `a_longnum`: digit tokens with ≥ 3 digits → 2 = shared, 1 = both present with none shared
  (**conflict**), 0 = one side missing, NaN = both missing. (Originally a 5–6-digit "postal" feature.
  A sample of 20,000 train-S1 addresses has almost no ZIP/PIN codes: 5-digit tokens are mostly US house
  numbers such as "25622". So the feature is named after what it measures.)
- `house_no`: first digit token in each address → 1 equal, 0 different, NaN if either is missing.
- `addr_empty`: 0 both present, 1 pool empty, 2 S1 empty, 3 both empty.
- Rationale: chain ≥ 10 S1 are same-name branches. There the name features are about 1.0 for true and
  false candidates alike, and only number conflicts separate them. Evidence (c): should work from the mechanism.

### 2D. Record meta
`country3` (1 same, 0 different, 2 either empty), `src` (2/3), `log_chain` = ln(1 + chain_S1).

### 2E. List context (per S1; computed after 2A–2D)
- `n_cand`, `p_max`, `p_minus_max`, `p_over_max`, `gap_next` (p minus the next-lower p), `sum_p`
  (≈ expected number of matches).
- `n_name_close` (candidates with `token_set_ratio` ≥ 0.9), `n_longnum_shared`, `share_same_src`.
- Rank of this candidate by `token_set_ratio` and by `a_token_set` inside the list.
- Why: the metric is averaged per S1, so a candidate should be judged against its neighbours in the same
  list. A 0.6 in a list whose top is 0.99 is different from a 0.6 that is the top.

### 2F. Pool-side competition (experiment M5 only; needs candidates for every S1 of the split)
`n_owners` = number of S1 lists containing this `pool_idx`; `is_best_owner`; `margin_to_next_owner`
(difference in `p_rerank`).
- Ground truth has **no pool record with 2+ owners** (`PLAN_PROGRESS.md`, verified).
- Measured lower bound inside MATCH_TRAIN ∪ MATCH_VALID (200,000 of 2,206,821 S1): 9.8% of candidate rows
  are in a collision. Keeping only the highest-`p_rerank` owner removes 59,450 negatives and 1,234 positives,
  but moves MATCH_VALID F0.5 only 0.9461 → 0.9464 at t = 0.72 (`matcher_preflight.json` `one_owner`).
- Limitation: the counts must be computed over the full S1 file on both train and test, or train/test
  feature distributions differ. Train needs candidates for all 2.2M train S1: about 3.2 h, extrapolated
  from 938 s per 180,000 queries.

### 2G. Stage-2 feature re-use (experiment M3)
The 35 re-ranker features (family scores `s_*`, `cov_*`, `learn`, `rk_*`, `log_union`, `log_dEN`) are not
saved in the candidate parquet, and 17 of them need stage-1 state. Add a `--dump-features` flag to
`rerank.generate_candidates` that writes `X` for the kept rows. Rerun MATCH_TRAIN + MATCH_VALID (~19 min).
Keep it only if it wins (see M3). Winning means rerunning TEST too (~2 h).

---

## 3. Model

- LightGBM `binary` objective, metric `binary_logloss` (plus AUC for logs).
- **No `is_unbalance`, no `scale_pos_weight`.** Positive rate is 0.62. Reweighting shifts probabilities,
  and the decision rule needs calibrated probabilities.
- Start params: `learning_rate 0.05, num_leaves 63, min_data_in_leaf 100, feature_fraction 0.8,
  bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, max_bin 255, seed 42, deterministic true`;
  `num_boost_round 3000`, `early_stopping_rounds 100`. Stage 2 hit its 400-round cap without early stopping
  (`rerank_train.json`), so do not cap low.
- **5-fold GroupKFold by `s1_idx` on MATCH_TRAIN.** Early stopping uses 10% of each fold's *training*
  S1 (GroupShuffleSplit), so the out-of-fold predictions never influence the stopping round.
  Output: OOF p for all 975,699 rows, plus the per-fold best iteration.
- Inference model: the mean of the 5 fold models. There is no refit, so the isotonic map fitted on OOF
  predictions applies to models trained the same way.
- **Calibration:** isotonic regression (`sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`)
  fit on OOF p versus label. It is applied to the fold-average outputs.
- Saved as `work/match/models/model_fold{0..4}.txt` plus `matcher.json`, which holds the features,
  calibration table, decision rule and bucketing fingerprint. The fingerprint is the sha1 of the stage-2
  model file, the stage-1 hashes and the cut. Refuse to predict on candidates
  with a different `config_hash`, the same staleness guard as stage 2.

## 4. Decision rule (per S1), tuned on MATCH_TRAIN OOF, confirmed once on MATCH_VALID

With calibrated p, compare:
- **D1 global threshold:** keep p ≥ t, with t on a 0.01 grid.
- **D2 expected-F0.5 top-k:** sort a list by p. For k = 0..n, estimate E[F0.5(top-k)] by Monte Carlo
  over independent Bernoulli(p_i) labels (M = 1000 draws, seed 0). F0.5 = 1.25·TP / (k + 0.25·T), and
  for k = 0 the score is 1 if T = 0. Pick the k with the highest estimate. This is the plug-in decision
  rule for F-measures (Jansche 2007; Nan Ye et al. 2012), evidence level (b).
  - Measured with uncalibrated `p_rerank` on DEV_B (scratch run, not in a metrics file): 0.9466 versus
    0.9476 for the best global threshold, so no gain without calibration. Evidence (c) that calibration
    changes this; M2 measures it.
- **D3:** D2 plus the one-owner filter over the full S1 set being predicted. The filter drops a pool
  record from every list except the one with the highest calibrated p.

Pick the rule with the highest OOF entity F0.5. MATCH_TRAIN has 180,000 S1, so the standard error is
about 0.0005. MATCH_VALID has 20,000 S1, so its standard error is about 0.0015 and it only confirms. If
MATCH_VALID is more than 0.003 below OOF, stop and investigate. Do not re-tune on MATCH_VALID.

---

## 5. Metrics (script-written only, `artifacts/metrics/match_*.json`)

Per set (`match_oof_MATCH_TRAIN.json`, `match_eval_MATCH_VALID.json`, `match_eval_HOLDOUT_TEST.json`):
- **Headline:** entity F0.5, macro over **all** S1 in the set. S1 with no candidates or no surviving
  pairs still count: 1.0 if singleton and predicted empty, else 0. Use the same `f05` as
  `matcher_preflight.py` / `ber.evaluate`.
- Floor (`p_rerank` at the frozen t) and ceiling on the same set, side by side.
- The loss breakdown of §0.1 (missed-only / FP-only / FP+FN / singleton FP / blocking miss).
- Segments: chain {1, 2–9, ≥ 10}, country, singleton vs matched, name-unseen (reuse `ber.evaluate.segment_data`).
- Singleton accuracy (true singletons predicted empty ÷ true singletons).
- Candidate-recall retained (true pairs predicted ÷ true pairs in candidates).
- Diagnostics only (never headline): pairwise precision/recall/F1 at the chosen rule, PR-AUC, ROC-AUC, logloss.
- Test: `match_test_monitor.json` with per-country predicted-list size distribution, share of empty
  lists, and mean max p. Alarm rule as in `monitor.py`. France has no labels, so only its distributions
  can be compared.

## 6. Error export (train only)

From **MATCH_TRAIN OOF** predictions at the chosen rule (in-sample predictions understate errors):
- `reports/match_errors/train_false_positives.csv`, `train_false_negatives.csv`, columns
  `Name_S1, Address_S1, Name_Other, Address_Other` (raw text) plus `source1_entity_id, other_entity_id,
  p_cal, rank, chain_S1, country`. Sort by p (FP descending, FN ascending).
- `train_blocking_misses.csv`: true pairs of MATCH_TRAIN S1 that are not in the candidates. These go to
  the bucketing side, not to feature work.
- Never export val-S1 or test errors.

---

## 7. Experiments, in order (each writes a metrics json; decide by the rule stated)

| ID | What | Decision rule |
|---|---|---|
| M0 | Feature build for MATCH_TRAIN / MATCH_VALID, with unit tests on hand-made pairs (NaN handling, postal conflict, Indic pair) | tests pass; no NaN in features that are defined as never-NaN |
| M1 | Model on 2A–2E, 5-fold OOF, D1 | OOF entity F0.5 must beat the floor measured the same way on MATCH_TRAIN (`p_rerank` + D1); else stop and inspect features |
| M2 | Calibration + D2 vs D1 | keep D2 if OOF F0.5 ≥ D1 + 0.001 |
| M3 | + 35 stage-2 features (§2G) | keep if OOF F0.5 ≥ M2 + 0.001 (it costs a ~2 h TEST rerun) |
| M4 | Learning curve: 45k / 90k / 180k MATCH_TRAIN S1; if still rising, add DEV_REST (candidates cost ~938 s per 180k S1) | add data while OOF F0.5 rises ≥ 0.001 per doubling |
| M5 | Pool-side features 2F + D3 (needs candidates for all train S1, ~3.2 h, and all test S1, already running) | keep if OOF F0.5 ≥ best + 0.001 |
| M6 | Entity-level singleton model: LightGBM on list features → P(S1 has 0 matches), used as the k = 0 term in D2 | keep if OOF F0.5 ≥ best + 0.0005; singleton FP loss is 0.0044 total, so the gain is capped there |
| C | **Confirm once** on MATCH_VALID with everything frozen | within 0.003 of OOF |
| H | **Headline once** on val-S1 (HOLDOUT_TEST candidates exist) | report; no changes afterwards |

Not in this plan (would need your approval per AGENTS.md, since it changes the model family): a
transformer cross-encoder. The bucketing plan notes the allowlisted embedding models are English-only.

## 8. Test inference and deliverables

1. `match-predict --set TEST`: features, model, calibration, chosen rule → `output/matching_results.tsv`.
   Matched IDs are a subset of `candidate_pairs.tsv` by construction. One row per test S1, in file order.
   Write with `newline="\n"`.
2. `ber.validate_stream` on both files, then the official `utils/validate_submission.py`.
3. Deliverables:
   - `work/match/model_v1.txt`, `calib_v1.pkl`, `meta_v1.json`
   - `work/match/features_{MATCH_TRAIN,MATCH_VALID,HOLDOUT_TEST,TEST}.parquet`
   - `artifacts/metrics/match_*.json`
   - `reports/match_errors/*.csv`
   - `output/matching_results.tsv`
   - A matcher section in `Documentation_template.md`, including the one-owner rule if M5 is kept. That
     rule uses test inputs only, never test labels.

## 9. Code layout

- `submission/code/business_entity_resolution/src/ber/match.py`
- CLI commands `match-build`, `match-train`, `match-decide`, `match-eval --set`, `match-predict --set TEST`
- Tests in `tests/test_match.py`
- Packages: already pinned (`lightgbm` MIT, `rapidfuzz` MIT, `jellyfish` MIT, `scikit-learn` BSD-3,
  `pyarrow`, `numpy`, `pandas`). No new libraries.
- Performance: rapidfuzz on about 1.08M MATCH rows plus 9.4M TEST rows. Chunk by S1, 6 workers, write
  parquet parts. Evidence (d) for speed: the old Python featurizer ran about 8.3K pairs/s on 6 workers,
  so TEST takes roughly 20 min.

## 10. Changes from the original draft, and why

| Draft | Change | Reason |
|---|---|---|
| Early stopping + threshold on "val split" | GroupKFold OOF on MATCH_TRAIN; val-S1 scored once | split.md rule 4 bans tuning on validation labels |
| `is_unbalance=True` "labels skewed to 0" | no reweighting + isotonic calibration | positive rate is 0.6215, not skewed to 0 |
| lowercase-only raw text | textnorm v2.1 `*_norm` text | lowercase erases Indic pool names (752,869) |
| 5 features | 2A–2E, about 45 features | the 5 add nothing over stage 2's 35; floor is already 0.9461 |
| numeric Jaccard = 0 when both empty; char-diff ratio divides by 0 | NaN | "no info" is not "conflict"; about 345K pool addresses are empty or `<NULL>` |
| train on all train S1 | MATCH_TRAIN (+ DEV_REST in M4) | RR_TRAIN/RR_CAL etc. have in-sample `p_rerank` |
| FP/FN from in-sample train predictions | from OOF | in-sample errors understate real errors |
| threshold grid 0.05 | 0.01 grid plus per-S1 expected-F0.5 | the curve moves about 0.005 per 0.1 near the peak |
| lists rebuilt "from surviving pairs" | score every S1 in the set | S1 with nothing surviving must count (0, or 1 if singleton) |

## 11. Limits (flat)

- The ceiling is 0.9928 on val-S1. The 0.0053 blocking-miss loss cannot be recovered here.
- France has no labels, so France quality is unmeasured before submission.
- The one-owner gain is measured only as a lower bound (+0.0003 with `p_rerank` on 9% of S1 coverage).
- No guarantee of any leaderboard rank; other teams' scores are unknown.

---

## 12. Implementation status (2026-09-25)

Code: `submission/code/business_entity_resolution/src/ber/match.py`, CLI `match-features`, `match-train`,
`match-eval`, `match-predict`, one-shot script `run_matcher.sh`, tests `tests/test_match.py` (20 tests,
including exact expected-F0.5 vs brute force and the official validator on a written file). Reproduction:
the package `README.md`.

| Item | Status |
|---|---|
| Features 2A–2E (50 columns) | implemented |
| Model, OOF, calibration, D1 / D2 / D3 selection on OOF | implemented; choice rule: D2 if ≥ D1 + 0.001; one-owner if it raises OOF F0.5 at all |
| Error export (OOF FP / FN, blocking misses) | implemented, `reports/match_errors/` |
| M3 (stage-2 features), M4 (DEV_REST), M5 (pool-side features), M6 (singleton model) | not implemented; run the full matcher first, then decide by §7 rules |
| Full MATCH_TRAIN run, MATCH_VALID confirmation, val-S1 headline, TEST prediction | not run yet |

A smoke run (20,000 MATCH_TRAIN queries, 2,000 MATCH_VALID) showed that the code path works end to end:
OOF F0.5 0.9702 vs floor 0.9470; MATCH_VALID 0.9672 vs floor 0.9450. These are smoke numbers on
subsets, not results. Their metrics files were deleted so they cannot be mistaken for the real run.

# Validation plan — benchmarking experiments without leakage

Status: PLAN (2026-09-26). Nothing below is implemented unless marked **exists**.
Priority: `AGENTS.md` > `data/student_resource/dataset/split.md` > this file.
Code root: `submission/code/business_entity_resolution/src/ber/` (called `ber/` below).

Goal: one protocol that (1) scores every experiment with the contest metric under test-like conditions,
(2) says with a number whether approach A beats approach B, (3) uses all 1,765,458 train-S1 rows,
and (4) cannot leak validation-S1 or test labels by construction.

---

## 1. What is wrong with the current setup (measured)

| # | Problem | Number | Source |
|---|---|---|---|
| P1 | The matcher trains on 180,000 train-S1 queries. 1,455,458 train-S1 queries (DEV_REST) are unused. | 10.9% of train S1 used by stage 3 | `artifacts/metrics/splits.json` |
| P2 | Model comparisons run on MATCH_VALID (20,000 queries). Standard error of the mean F0.5 there is 0.00071, so an unpaired A-vs-B difference below ~0.002 is noise. | per-entity F0.5 std 0.1006 | `artifacts/metrics/validation_noise.json` |
| P3 | The one-owner rule is transductive: it resolves a pool record claimed by several S1 **within the scored set**. On test it runs over all 1,732,544 test S1; on MATCH_VALID over 20,000 of 2,206,821 train S1 (0.9%). It changed 2 of 20,000 entities on MATCH_VALID, so its value on test is unmeasured. | delta +0.000007, CI [−0.000009, +0.000032] | `validation_noise.json` |
| P4 | The Indic token dictionary (`translit_dict`) is mined from true pairs of DICT_SRC (1,745,458 train S1, includes MATCH_TRAIN/MATCH_VALID). OOF scores on those queries use a dictionary built partly from their own labels; val S1 and test do not. Size of the inflation: unmeasured. | — | `ber/splits.py` (DICT_SRC) |
| P5 | Val S1 (441,363) has been scored once, for stage 1+2 only (F0.5 ceiling 0.99276). The stage-3 matcher has never been scored on it. | — | `rerank_eval_HOLDOUT_TEST.json` |

What already works and is kept: the 80/20 stratified split (split.md), disjoint work slices (`ber/splits.py`,
disjointness asserted in `build_sets`), GroupKFold by S1 with a separate early-stopping slice (`ber/match.py`),
the `match-eval --set HOLDOUT_TEST` refuse-to-rescore guard, `reports/test_usage_log.md`, and metric files
carrying `config_hash` + `git_commit`.

---

## 2. Data partition

Train S1 = 1,765,458 rows (`train_source1_train.tsv`). Val S1 = 441,363 (`train_source1_validation.tsv`).
S2 and S3 are always the full files: every query set retrieves from the full 10.32M-record train pool.

| Set | Queries | Labels used for | Status |
|---|---|---|---|
| UPSTREAM = DEV_A 10K, DEV_B 10K, S1R_TRAIN 20K, RR_TRAIN 50K, RR_CAL 20K | 110,000 | stage-1 tuning (DEV_A), stage-1/2 confirmation (DEV_B), stage-1 logistic fit, stage-2 LightGBM fit, cut calibration | exists, frozen |
| **MATCH_ALL** = MATCH_TRAIN + MATCH_VALID + DEV_REST | **1,655,458** | stage-3 training, K-fold OOF scoring, thresholds, decision rule, all A-vs-B comparisons | new (union of existing slices) |
| FAST ⊂ MATCH_ALL = MATCH_TRAIN + MATCH_VALID | 200,000 | quick screening (tier 1) | candidates exist |
| VAL_S1 (= HOLDOUT_TEST) | 441,363 | scoring only, once per frozen pipeline | exists |
| TEST | 1,732,544 | no labels exist; inference only | exists |

Why MATCH_ALL excludes UPSTREAM: the stage-3 feature `p_rerank` comes from the stage-2 model. On
RR_TRAIN/RR_CAL queries that model has seen the labels, so `p_rerank` is over-confident there; training
or scoring stage 3 on them would inflate OOF. This is the standard stacking rule: a downstream model
only sees upstream outputs produced out-of-sample.

**Fold file (new):** `work/splits/match_folds.json` assigns each MATCH_ALL query to one of 5 folds,
stratified by country × singleton × match-count bucket {0, 1, 2–3, 4–5, 6+}, seed 42, written once and
never regenerated. Every experiment uses the same folds, so per-entity scores of two experiments are
paired. Folds are random at the S1 level, not grouped by name: `plan/checks/split_design.json` measured
that a random 20% train holdout has 0.378 of its names in the other 80%, against 0.3526 for official
test S1 in train S1, so a random split already reproduces test's name overlap for US/India.

---

## 3. Reproducing the contest metric

Contest rule (student_resource README): per-S1 F0.5, macro-averaged over **all** S1 rows in the set;
singleton + empty prediction = 1.0, singleton + any prediction = 0.0; a missing S1 row causes rejection
(we score it as an empty prediction).

1. **In-memory scorer — exists:** `match.entity_f05` / `match.Scorer`. Used for OOF and sweeps.
2. **File-level scorer — new `ber/score_tsv.py`:** reads a `matching_results.tsv` and a ground-truth TSV,
   nothing else (no work/ indices). Applies the validator rules first: header, tab separation, no
   duplicate S1 rows, no duplicate IDs in a list, only S2-/S3- IDs, every S1 of the set present.
   Unit tests: the README example (0.714), all singleton cases, missing row, `\r\n` line ends.
   Cross-check: on MATCH_VALID its F0.5 must equal `Scorer` to 1e-12.
3. **Val-S1 and final numbers are always produced through the file path:** write val S1 predictions as a
   real `matching_results.tsv` (same writer as test, `match.write_matching_file`), run the validator,
   score with `score_tsv.py` against `train_ground_truth_validation.tsv`. This tests the whole
   submission path, not only the model.

---

## 4. Leakage register

Each entry: the mechanism, how the leak would inflate the score, and the guard.

| ID | Mechanism | Guard |
|---|---|---|
| L1 | **Pool ownership through labels.** No pool record has two owners (`plan/PLAN_PROGRESS.md`, verified). If train-S1 labels are used to remove or down-weight pool records already owned by train S1, val-S1 precision rises because every remaining candidate is more likely to belong to a val query. Test has no labels, so this cannot happen there. | Forbidden: any feature, filter or post-process that reads `train_gt.parquet` for queries other than the one being trained on. Label reads go through one function (see L10). |
| L2 | **Stacked-model leakage.** Stage-2 scores on its own training queries. | MATCH_ALL excludes UPSTREAM (§2). `build_sets` disjointness assertion stays. |
| L3 | **Label-derived dictionary** (P4). | (a) VAL_S1 is outside DICT_SRC — **exists**. (b) Report the OOF→VAL_S1 gap for every val run; a gap > 0.003 blocks submission until explained. (c) Diagnostic: score DEV_B (10,000 queries, outside DICT_SRC, candidates exist) with the fold ensemble and compare to MATCH_ALL OOF; detects a gap ≥ ~0.002 (SE 0.001). |
| L4 | **Selection on the scoring set.** Picking thresholds, the decision rule (threshold vs expected-F), one-owner on/off, or features by their val-S1 score. | All selection on MATCH_ALL OOF. VAL_S1 scored once per frozen pipeline (§8 ledger). split.md rule 4. |
| L5 | **In-sample calibration.** Isotonic fit on OOF, then scored on the same OOF: the step function memorizes OOF noise. | Cross-fit the calibrator: calibrator for fold k is fitted on OOF of the other 4 folds. Measure in-sample minus cross-fit F0.5 once; if < 0.0002, keep the in-sample fit for simplicity. |
| L6 | **Early stopping on the scored fold.** | **Exists:** `match-train` takes the early-stopping slice from the fit folds (`n_early_stop_rows` in `match_train.json`). Keep. |
| L7 | **Transductive coverage mismatch** (P3). A set-level rule behaves differently at 0.9% coverage than at 100%. Not a label leak, but it makes validation unrepresentative. | Global pass (§5.3): predict all 2,206,821 train S1, apply set-level rules over all of them, then score only the target subset. |
| L8 | **Label shuffle canary.** A feature that encodes the label (join bug, sorted order, ID arithmetic) is invisible in normal metrics. | Once per feature-set change: train stage 3 on FAST with labels permuted within each S1's candidate list. OOF F0.5 must fall to the `p_rerank`-threshold floor (±0.002). If it stays above, a feature carries label information. |
| L9 | **Test peeking.** Tuning on test prediction statistics, hand-labeling test rows, or choosing between variants by leaderboard score. | Test is used only for inference and format validation, logged in `reports/test_usage_log.md` (**exists**). Leaderboard policy in §9. |
| L10 | **Unlogged label access.** | New `ber/labels.py`: the only reader of `records/train_gt.parquet`. Signature `load_labels(s1_idx, purpose)`; purpose ∈ {`fit`, `oof_score`, `val_score`}. `val_score` requires the caller to be `score_tsv`/`match-eval --set HOLDOUT_TEST` and appends to the ledger; any other request that touches a VAL_S1 index raises. A test (`tests/test_labels.py`) greps `ber/` for other `train_gt` reads. |

Allowed (not leaks): IDF, token counts, abbreviation mining, chain sizes on S2+S3 and S1 text of the same
split (split.md rule 5); country as a feature value (never a filter; France is unseen).

---

## 5. Experiment protocol

### 5.1 Tiers

| Tier | Queries | Scored by | Paired SE (see §6) | Wall time (est.) | Use |
|---|---|---|---|---|---|
| T0 smoke | 5,000 of FAST | OOF | — | minutes | code runs, no NaN, format OK |
| T1 screen | FAST 200,000 | 5-fold OOF | ~0.00005–0.0002 | ~20 min stage 3 (c: `match_train.json` wall 988 s at 180K) | discard clear losers |
| T2 decide | MATCH_ALL 1,655,458 | 5-fold OOF + global pass | ~0.00002–0.00008 | ~2.5 h stage 3 (c: 9× T1 rows); +2 h candidates if stage 1/2 changed (a: `candidates_TEST.json` 8,463 s for 1.73M) | adopt / reject |
| T3 confirm | VAL_S1 441,363 | fold ensemble, file-level scorer | 0.00015 unpaired | ~40 min | one run per frozen pipeline, reported, not used to choose |

Upstream changes (normalization, stage-1 keys/quotas, stage-2) change the candidates, so both arms need
fresh candidates. Order: stage-level metric on DEV_A/DEV_B first (existing E1–E6 / `rerank-eval DEV_B`
protocol: F0.5 ceiling, pair recall, mean candidate size) → T1 end-to-end → T2 end-to-end.

### 5.2 Per-stage metrics (all queries in the set, singletons included)

- **Stage 1 (blocking):** pair recall, entity recall, F0.5 ceiling (recall ceiling expressed as F0.5), mean/p99 union size. **exists** (`ber/evaluate.py`).
- **Stage 2 (candidate cut):** same, plus mean candidate size (drives `candidate_pairs.tsv` size). **exists**.
- **Stage 3 (matcher):** entity F0.5 (headline), then ceiling, pairwise precision, recall inside candidates, singleton accuracy, loss breakdown {singleton_with_fp, matched_fp_only, matched_missed_only, matched_fp_and_missed, blocking_miss_only}. **exists** (`Scorer.report`).

### 5.3 Global pass (test-like set-level conditions)

For any pipeline with a set-level rule (one-owner now; any future cross-S1 feature):
1. p for MATCH_ALL = its OOF p. p for UPSTREAM and VAL_S1 = mean of the 5 fold models (same as test).
2. Apply the set-level rule over all 2,206,821 train S1 at once (the pool is shared by all of them, exactly as all test S1 share the test pool).
3. Score the subset: MATCH_ALL for selection, VAL_S1 for the one-shot confirmation.

UPSTREAM predictions are over-confident (L2); they only compete for pool records here and are never scored.
Needs candidates for DEV_A, S1R_TRAIN, RR_TRAIN, RR_CAL (100,000 queries, ~8 min).

### 5.4 Test-shift proxies (reported for every T2 run)

- **Unseen country (France):** leave-one-country-out. Stage 3 trained on US-only folds, scored on India MATCH_ALL OOF, and the reverse. Report the drop against the all-country model on the same queries. Stage-2 LOCO **exists** (`rerank_loco.json`: US→India ceiling 0.98492). A change that improves all-country F0.5 but worsens LOCO by > 0.002 is flagged.
- **Singleton rate:** F0.5 = s · F_singleton + (1 − s) · F_matched. Test's singleton rate is unknown. Report the reweighted score at s ∈ {0.03, 0.056, 0.10, 0.20}. On MATCH_VALID: F_singleton 0.9783, F_matched 0.9743, s 0.0576 (`validation_noise.json`).
- **Pool density:** test has 5.754 pool records per S1 against train's 4.677 (`split_design.json`). Report F0.5 by candidate-list-size bucket {1–3, 4–6, 7–10, 11+}; a method that degrades on long lists is at risk on test. Limitation: this does not simulate test density, only exposes sensitivity to it.
- **Segments — exists:** country, chain size {1, 2–9, ≥10}, name unseen, match count, Indic pool name, empty pool address.

---

## 6. Deciding "A beats B"

Measured on MATCH_VALID (`validation_noise.json`): per-entity F0.5 std 0.1006; 82.4% of entities score exactly 1.0.
Unpaired SE of a mean: 0.00071 at 20K, 0.00022 at 200K, 0.00015 at 441K, 0.000078 at 1.655M.

Paired comparison: both arms are scored on the same queries with the same folds; compute per-entity
d_i = F_A,i − F_B,i and bootstrap the mean of d (2,000 resamples of S1 rows). Most entities have d_i = 0,
so the paired SE is far below the unpaired one: the one-owner on/off diff on MATCH_VALID had std 0.0016
against 0.1006 per arm. New module `ber/compare.py` stores per-entity scores per run in
`work/runs/<run_id>/per_entity.parquet` (s1_idx, f05, n_pred, n_tp, n_true) and writes
`artifacts/metrics/compare_<A>_vs_<B>.json` (delta, 95% CI, entities changed, per-segment deltas).

**Adoption rule (T2, MATCH_ALL OOF):**
1. Paired delta ≥ +0.0005 and 95% CI lower bound > 0. (0.0005 ≈ 1 rank on a tight leaderboard; set lower if the team wants.)
2. No guard segment gets worse by more than 0.002 with CI upper bound < 0: country_US, country_India, singletons, LOCO both directions.
3. Runtime and memory stay inside budget (test inference ≤ 4 h on the laptop, peak RSS ≤ 12 GB on 16 GB), and mean candidate size ≤ 100 (plan Q1, zip size).
4. T1 screens only: promote to T2 if the T1 paired delta > 0; drop if its CI upper bound < 0.

Ties go to the simpler pipeline (fewer features, fewer stages).

---

## 7. Scorecard (written by every run)

`artifacts/metrics/run_<run_id>.json`, header from `io_utils.write_metrics` (config_hash, git_commit):
tier, query set, fold file hash; headline F0.5 + bootstrap CI; stage metrics (§5.2); loss breakdown;
segments; singleton-rate reweighting; LOCO; list-size buckets; label-shuffle canary result (if run);
wall time, peak RSS, mean candidate size. Plus `per_entity.parquet` for pairing.
`reports/experiments.md` gets one line per run, generated from these files, never typed by hand
(AGENTS rule 6).

**Research log:** every run, adoption decision, val-S1 score, and leaderboard upload also gets a
narrative entry in `RESEARCH_LOG.md` (repo root; newest first, append-only, every number cites its
metrics file). That includes dropped and failed runs. The §8 val ledger and the §9 upload log stay the
authoritative records; the log entry links to them and records what was concluded.

---

## 8. Val S1 protocol

- Scored only after a pipeline is frozen by §6 on MATCH_ALL OOF. One run per frozen `config_hash`.
- Append-only ledger `artifacts/metrics/val_ledger.jsonl`: timestamp, git commit, config hash, model
  fingerprint, F0.5 + CI, OOF F0.5 on MATCH_ALL, gap. `score_tsv` refuses a second VAL_S1 score for a
  config hash already in the ledger. Seed the ledger with the existing stage-1+2 run
  (`rerank_eval_HOLDOUT_TEST.json`).
- Val S1 is not a tie-breaker between two finalists (that would be choosing a design by val labels,
  split.md rule 4). If two finalists tie on OOF, take the simpler one, then score it.
- Gap alarm: |F0.5(VAL_S1) − F0.5(MATCH_ALL OOF)| > 0.003 → stop and check L1–L3 before submitting.
  Reference: MATCH_VALID vs 180K OOF differed by 0.0004 (`match_eval_MATCH_VALID.json`).

---

## 9. Leaderboard policy

Each portal upload returns a score computed with hidden test labels, so it is a label query on test.
Upload only pipelines already frozen and val-scored; log each upload (run_id, config hash, public score)
in `reports/test_usage_log.md`. Do not choose hyperparameters or thresholds by leaderboard score. Final
rank uses the private part of test, so a pipeline tuned on the public part gains nothing there.

---

## 10. Implementation order

| Step | Work | Output | Est. time |
|---|---|---|---|
| 1 | `ber/labels.py` (single label reader, purpose check) + test; route `match.pair_labels`, `match.true_counts`, `evaluate.truth_lists` through it | L10 guard | 1–2 h code |
| 2 | `ber/score_tsv.py` + tests (README example, singletons, validator rules); cross-check vs `Scorer` on MATCH_VALID | exact metric | 1–2 h code |
| 3 | `match_folds.json` over MATCH_ALL; `match-train --set FAST|MATCH_ALL` using it; cross-fit calibration option | folds | 1 h code |
| 4 | `ber/compare.py`, per-entity run files, scorecard writer | paired stats | 2 h code |
| 5 | Candidates for DEV_REST (1,455,458) and UPSTREAM minus DEV_B (100,000) with the frozen stage 1+2 | full train-S1 candidates | ~2.2 h run |
| 6 | Baseline T1 (FAST) with current matcher config → must reproduce OOF 0.9741 ± 0.001 (`match_train.json`); label-shuffle canary on FAST | baseline + canary | ~40 min run |
| 7 | Baseline T2 (MATCH_ALL) + global pass + LOCO + DEV_B dictionary-gap check | reference run for all comparisons | ~4 h run |
| 8 | Learning curve: stage 3 trained on 50K / 180K / 500K / 1.655M, scored on the same held-out fold | how much "whole dataset" is worth | ~3 h run |
| 9 | One-owner and expected-F decision re-chosen under the global pass (P3) | decision rule at test-like coverage | 20 min |
| 10 | Freeze → VAL_S1 via file path → ledger → only then build test submission | first honest val number for stage 3 | ~1 h |

Steps 5, 7, 8 are heavy jobs; run one at a time (AGENTS). After step 7 every new idea follows §5–§6.

---

## 11. Decisions for the team

1. **Retrain on train S1 + val S1 for the final test submission?** It adds 441,363 labelled queries
   (+27% for stage 3) with the config frozen. It conflicts with split.md rule 3 ("validation labels for
   scoring only") and AGENTS rule 5, so it needs an explicit amendment to split.md before anyone does it.
   Default in this plan: no.
2. **Cross-fit stage 2 over MATCH_ALL** (so stage 2 also trains on ~1.6M queries instead of 50K)? Cost:
   one extra full candidate run per fold (~2 h each). Decide after a stage-2 learning curve on DEV_B
   (RR_TRAIN at 12.5K / 25K / 50K): if F0.5 ceiling gains < 0.0005 from 25K→50K, skip.
3. **Adoption threshold** 0.0005 (§6) — change if the team wants to chase smaller gains; below 0.0002
   the paired CI at T2 is the limit.

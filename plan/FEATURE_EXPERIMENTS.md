# Stage-3 feature experiments: protocol, queue, ledger

Status: active, 2026-09-26. Rules: `AGENTS.md` > `split.md` > `plan/VALIDATION_PLAN.md` > this file.
Idea source: the "contest tips" analysis (section numbers F1–F7, C1–C4, N1, D2 below refer to it).
Blocking and the stage-2 cut are frozen for this round: `K_min` 5, `K_max` 60, `p_min` 0.05
(the K-width question is closed by `ADDRESS_EXPERIMENTS.md` §4 S4).

## 1. Current best (the baseline every experiment is paired against)

| Run | Features | OOF F0.5 (MATCH_TRAIN, 180,000 S1) | MATCH_VALID F0.5 | Metrics |
|---|---|---|---|---|
| submission 1 | 50 | 0.97412 | 0.97449 | `match_train.json` (git history), `match_eval_MATCH_VALID.json` |
| k5_base | 51 (+ zero-strip, `a_digits_state`) | 0.97517 | 0.97594 | `match_train_k5_base.json`, `match_eval_MATCH_VALID_k5_base.json` |
| k5_feat | 62 | 0.97729 | 0.97786 | `match_train_k5_feat.json`, `compare_k5_base_vs_k5_feat_{OOF,MATCH_VALID}.json` |
| **x1_strsim** (best) | 68 | **0.97807** | **0.97848** | `match_train_x1_strsim.json`, `compare_k5_feat_vs_x1_strsim_{OOF,MATCH_VALID}.json` |

k5_feat vs k5_base: OOF +0.00211, 95% CI [0.00190, 0.00235]; MATCH_VALID +0.00192, CI [0.00129, 0.00257].

Remaining OOF loss of k5_feat (0.0227), counted from `work/match/runs/k5_feat/oof.parquet`:
missed true candidates with a non-empty answer 0.0076; blocking miss 0.0064; FP-only 0.0041;
matched S1 answered empty while a true match was in its candidates (634 S1) 0.0035; matched S1
answered empty with no true candidate 0.0014; singleton with FP 0.0011. FN-row calibrated p median 0.31.

## 2. Protocol (every experiment, no exceptions)

Commands run from the repository root with
`export PYTHONPATH=submission/code/business_entity_resolution/src` and `.venv/bin/python -m ber.cli`.

1. **Backup before editing:** `cp src/ber/match.py work/match/snapshots/match_<exp>_before.py` (and any
   other file you edit). Never run `git checkout`, `git reset`, `git stash`, `git clean`, or `rm -rf`
   on anything outside `work/match/features/` and `work/match/runs/<your run>/`. Do not commit.
2. Implement the feature group in `src/ber/match.py` (append new columns to `PAIR_FEATURES` /
   `LIST_FEATURES`; keep existing columns). Missing information is NaN, never 0. No feature reads the
   country label or any label; statistics (IDF, df, counts, n-gram weights) come from the pool of the
   same split only. Anything computed from model predictions uses OOF predictions only.
3. Unit tests for every new column in `tests/test_match.py`; run
   `cd submission/code/business_entity_resolution && PYTHONPATH=src ../../../.venv/bin/python -m unittest discover -s tests`
   (use the absolute venv path if the relative one fails). All tests must pass.
4. Smoke: `match-features --set MATCH_VALID --limit 2000 --workers 2`; check NaN shares in
   `artifacts/metrics/match_features_MATCH_VALID.json` are plausible.
5. Full: `match-features --set MATCH_TRAIN`, `match-features --set MATCH_VALID`,
   `match-train --run <exp>`, `match-eval --set MATCH_VALID`,
   `match-compare --a <current best> --b <exp> --set OOF`, same with `--set MATCH_VALID`.
   One heavy job at a time; do not start a second training while one runs.
6. **Adopt** only if all hold (`VALIDATION_PLAN.md` §6):
   - OOF paired delta >= +0.0005 and its 95% CI lower bound > 0;
   - MATCH_VALID paired delta CI upper bound >= 0 (MATCH_VALID confirms, it never selects);
   - no guard segment (`country_US`, `country_India`, `singleton`) drops by more than 0.002 with CI
     upper bound < 0 (segments are in `compare_*_OOF.json`).
7. If **rejected**: remove the group's code and tests by editing (restore from your snapshot), rebuild
   nothing else. If **adopted**: leave it in; the run becomes the new current best.
8. Append one row to the ledger (§4) with numbers copied from the metrics files, and a 3–6 line note
   under it: what was added, delta + CI, which segments moved, decision. No number without a file.
9. Leave `configs/match.json` with `cand_filter` {K_min 5, p_min 0.05} and `drop_features` [] at the end.

Never read validation-S1 (HOLDOUT_TEST) labels or test labels; never run `match-eval --set HOLDOUT_TEST`
or `match-predict`. Never tune on MATCH_VALID.

## 3. Queue (run in order; each is one subagent, one group, one decision)

| ID | Group | Contents | Targets |
|---|---|---|---|
| X1 | String similarity | F2 TF-IDF char 2–4-gram cosine for name and address (IDF fit on the split pool); F3 Monge-Elkan with Jaro-Winkler (both directions, mean and min); F4 Double Metaphone and Soundex token-code set Jaccard on names | missed true candidates, FP |
| X2 | Heuristic / pattern | F5 name digit and Roman-numeral agreement (3-state, NaN if absent); F6 ordinal words -> numbers and zero-strip inside address token features (feature path only); F7 domain-style pool name flag + segmented-domain vs S1 name token-set ratio; C1 regex pattern features (compound house id, unit/suite, PO box, landmark-cut address similarity, sector/block/phase, postal-like tail number) | FP in chains, India |
| X3 | Sibling (list) | F1: pass-1 OOF p per row; per candidate: best OOF p among other candidates with near-equal name, whether a candidate from the other source with near-equal name is address-confirmed, count of same-name candidates in the list, name similarity to the list's top candidate. Inference uses the fold-average model p for pass 1 | exact name + empty pool address FNs |
| X4 | Entity model | N1: one row per S1 (list features + top p's + counts), target "S1 has >= 1 true match", LightGBM, same folds, OOF, isotonic; use its P(no match) as the k=0 term of the expected-F rule and as a gate for the threshold rule; pick by the paired rule | 0.0035 + 0.0011 empty/singleton loss |
| X5 | Model side | LightGBM `lambdarank` (group = S1) vs binary; monotone constraints on similarity columns; LOCO check (train US-only, score India OOF and reverse) on the final feature set | robustness, France proxy |

## 4. Ledger

| Run | Base | Features | OOF F0.5 | OOF delta [95% CI] | VALID delta [95% CI] | Decision |
|---|---|---|---|---|---|---|
| k5_feat | k5_base | 62 | 0.97729 | +0.00211 [0.00190, 0.00235] | +0.00192 [0.00129, 0.00257] | adopted |
| x1_strsim | k5_feat | 68 | 0.97807 | +0.00078 [0.00060, 0.00097] | +0.00062 [0.00021, 0.00104] | adopted |
| x2_pattern | x1_strsim | 79 | 0.97854 | +0.00047 [0.00031, 0.00065] | +0.00069 [0.00025, 0.00115] | rejected |
| x3_sibling | x1_strsim | 78 (73 stored + 5 stacked) | 0.97881 | +0.00074 [0.00053, 0.00095] | +0.00175 [0.00109, 0.00239] | rejected (guard) |
| x4_entity | x1_strsim | 68 (unchanged) + entity gate (23 entity features, separate LightGBM) | 0.97823 | +0.00016 [0.00004, 0.00029] | +0.00033 [-0.00001, 0.00072] | rejected |
| x5_mono | x1_strsim | 68 (unchanged) + `monotone_constraints` (19 similarity columns, advanced method) | 0.97705 | -0.00102 [-0.00122, -0.00084] | -0.00097 [-0.00150, -0.00046] | rejected |
| x5_rank | x1_strsim | 68 (unchanged) + `lambdarank` objective (ndcg@1/3/5, group=S1) | 0.95824 | -0.01983 [-0.02037, -0.01935] | -0.01892 [-0.02044, -0.01743] | rejected |

X1 note: added name/address TF-IDF char 2–4-gram cosine (IDF on the full train pool, 10,320,219 records),
Monge-Elkan JW (mean, min), Metaphone and Soundex token-code Jaccard. Every OOF segment rose; largest:
match_1 +0.00322 [0.00121, 0.00529], India +0.00109, US +0.00058; singleton +0.00039 (CI spans 0).
Gain rank of 68: n_tfidf_cos 12, n_me_min 23, n_me_mean 28, a_tfidf_cos 29, metaphone 41, soundex 55.
Chosen rule: threshold 0.67 + one-owner. Code snapshot: `work/match/snapshots/match_x1_strsim.py`.

X3 note: two-pass stacking on the frozen x1_strsim matcher (`stack_from`), with pool-to-pool sibling
columns (`sb_best_name_sim`, `sb_best_pos`, `sb_best_other_src`, `sb_n_same_name`, `sb_addr_sim`, 5 stored,
68 -> 73) plus 5 columns stacked at train/score time from pass-1 p (`p1`, `p1_rank`, `sb_p`,
`sb_p_max_samename`, `sb_confirmed`) -- 78 columns total in the LightGBM matrix, none read from labels or
country. OOF (MATCH_TRAIN, 180,000 queries): +0.00074 [0.00053, 0.00095], both the +0.0005 minimum and the
CI-excludes-zero bullet pass. MATCH_VALID confirms: +0.00175 [0.00109, 0.00239] (20,000 queries), CI lower
bound > 0 as X3's extra adoption condition requires. Gain rank of 78 (feature_gain, `match_train_x3_sibling.json`):
`p1` 1 (the pass-1 score itself, as expected for stacking), `sb_addr_sim` 9, `sb_p` 11, `p1_rank` 17,
`sb_p_max_samename` 22, `sb_best_name_sim` 34, `sb_best_other_src` 35, `sb_best_pos` 47, `sb_n_same_name`
57, `sb_confirmed` 69 (near-bottom -- the address-confirmed-twin flag adds little once `sb_p` and
`sb_addr_sim` are already in the model). `matched_missed_only` OOF loss (`match_train_{x1_strsim,x3_sibling}.json`):
0.01005 -> 0.00943 (loss down, fewer S1 answered with a missed true candidate), consistent with the sibling
signal recovering some of the empty-pool-address false negatives it targeted.

**Rejected mechanically** despite passing both gain rules: the `singleton` guard segment breaks the
guard-rail bullet (segment drop > 0.002 with CI upper bound < 0) on both sets --
OOF -0.01088 [-0.01343, -0.00892] and MATCH_VALID -0.00868 [-0.01476, -0.00347]
(`compare_x1_strsim_vs_x3_sibling_{OOF,MATCH_VALID}.json`). Mechanism: OOF `singleton_accuracy` (S1 with no
true match, correctly predicted empty) fell from 0.98137 (x1_strsim) to 0.97049 (x3_sibling); `singleton_with_fp`
entities rose 190 -> 301 (loss 0.00106 -> 0.00167, `match_train_{x1_strsim,x3_sibling}.json`
`oof.loss_breakdown`). A singleton S1's candidate list can still contain an address-confirmed sibling pair
among the wrong pool records (siblings measure candidate-to-candidate similarity, not candidate-to-S1
correctness), and `sb_p`/`sb_confirmed` push p up for candidates that borrow a confident neighbor's score
even when no candidate is actually this S1's match -- costly under F0.5's singleton rule (any non-empty
prediction on a true singleton scores 0). No country/chain guard segment breached. Wall times: feature
build 28s (MATCH_TRAIN, 975,699 rows) + 6s (MATCH_VALID, 108,493 rows); training 93s (5 folds, stacked);
match-eval 18s. Code + test snapshot before/after: `work/match/snapshots/match_x3_sibling_before.py`,
`test_match_x3_sibling_before.py` (pre-edit; matches x1_strsim) and `work/match/runs/x3_sibling/` (trained,
not adopted). `src/ber/match.py` / `tests/test_match.py` were restored to the x1_strsim snapshot;
`configs/match.json` had `stack_from` removed; `match-features`/`match-train` were re-run on MATCH_TRAIN,
MATCH_VALID and the live model to leave the repo's feature files and matcher.json back at the x1_strsim
68-feature schema (OOF F0.5 0.97807, exact match, confirming the restoration).

X2 note: added F5 name digit/Roman-numeral 3-state agreement (`n_num_state`), F6 ordinal-word +
zero-strip canonical address tokens (`a_tset_canon`, `a_idf_jacc_canon`), F7 domain-style name flags +
Viterbi-segmented-domain token-set ratio (`p_is_domain`, `q_is_domain`, `n_domain_tset`), C1 RAW-address
regex patterns (`a_hid_state`, `a_unit_state`, `a_pobox_state`, `a_sector_state`, `a_nolandmark_tset`) --
11 columns, 68 -> 79 features. OOF delta +0.00047 [0.00031, 0.00065] on 180,000 MATCH_TRAIN queries: CI
excludes zero but the point estimate is below the +0.0005 minimum gain (rule §2.6, first bullet), so the
run is **rejected mechanically** despite MATCH_VALID also confirming (+0.00069 [0.00025, 0.00115], 20,000
queries) and no guard segment dropping (match_1 +0.00128 [-0.00072, 0.00359], singleton +0.00127
[-0.00020, 0.00294], country_US/India deltas smaller, no segment breached -0.002). Gain rank of 79:
`a_idf_jacc_canon` 24, `a_nolandmark_tset` 30, `n_domain_tset` 40, `a_hid_state` 53, `a_tset_canon` 56,
`n_num_state` 64, `a_pobox_state` 66, `a_unit_state` 68, `p_is_domain` 71, `a_sector_state` 73,
`q_is_domain` 79 (zero gain -- S1 is rarely itself a raw domain string in this pool). NaN share: sparse as
expected for pattern hits (`a_pobox_state` 0.993, `a_sector_state` 0.961, `n_domain_tset` 0.963,
`n_num_state` 0.965, `a_hid_state` 0.904, `a_unit_state` 0.857) vs. address-emptiness-only for the F6
columns (`a_tset_canon`/`a_idf_jacc_canon`/`a_nolandmark_tset` 0.124, matching `a_tset`'s empty-address
rate). Code + test snapshot before/after: `work/match/snapshots/match_x2_pattern_before.py`,
`test_match_x2_pattern_before.py` (pre-edit) and `work/match/runs/x2_pattern/` (trained, not adopted;
`src/ber/match.py` / `tests/test_match.py` were restored to the x1_strsim snapshot). Feature build 37s
(MATCH_TRAIN, 180,000 queries / 975,699 rows) + 10s (MATCH_VALID); training 335s (5 folds).

X4 note: entity-level gate on top of the frozen x1_strsim pair model (pair model not retrained), per
plan §3 X4. Implemented as a separate command, `match-gate-train` (`ber/match.py`: `entity_feature_table`,
`apply_entity_gate`, `_gate_grid_search`, `run_match_gate_train`; `cli.py` command; `configs/match.json`
`entity_gate.from`), not folded into `match-train`/`score_set`/`match-eval`, per the plan's stated
fallback -- it writes a run dir (`work/match/runs/x4_entity/`: `models/entity_fold{0..4}.txt`,
`matcher.json`, `queries.npy`, `OOF_per_entity.npy`, `MATCH_VALID_per_entity.npy`, `entity_oof.parquet`)
that `match-compare` reads unmodified. 23 entity features (one row per S1 with >= 1 candidate, 180,000 of
180,000 MATCH_TRAIN S1 in this run) built from the candidate list ranked by x1_strsim's calibrated p (OOF
`p_cal` for MATCH_TRAIN, fold-average `score_set` p_cal for MATCH_VALID): candidate count; top-1/2/3 p +
their gap; sum p; count p>=0.5; normalized Shannon entropy of the p distribution; max/mean of `n_tset`,
`n_tfidf_cos`, `a_tset`, `a_digits_state==3`, `n_idf_qcov`; count with `a_empty!=0`; `log_chain`;
`q_name_cnt`/`q_addr_cnt`; max `p_rerank`. Target: `has_true_in_candidates` (positive rate 0.9420), same
5-fold `GroupKFold(shuffle=True, seed=42)` groups=s1_idx as the pair model (fold membership depends only
on the set of S1, verified empirically to line up 1:1 with pass-1's folds), LightGBM binary + isotonic
calibration -> `P_any`, OOF AUC 0.99947 (`match_train_x4_entity.json`). Decision: base threshold `t` and
gate (tau_empty, tau_force) chosen jointly on MATCH_TRAIN OOF by an analytic grid (t in {0.55..0.80} step
~0.05 plus x1's 0.67; tau_empty 0.02..0.60 step 0.02; tau_force 0.50..0.98 step 0.02 + "off") that scores
every combination in closed form from entity-level counts (S1 with 0 candidates are threshold-invariant,
so their contribution is added once, not recomputed per combo) -- chosen: t=0.70, tau_empty=0.40,
tau_force=0.62; the exact row-level OOF F0.5 at that combo (0.97823) matched the analytic grid value to
within 1e-6, confirming the grid math. Gain rank of the 23 entity features (`match_train_x4_entity.json`
`feature_gain`): `e_p_top1` 1, `e_p_sum` 2, `e_n_p_ge_05` 3 (the pass-1 signal itself dominates, as
expected for a model stacked on frozen pass-1 p), `e_p_gap12` 4, `e_p_top2` 5, `e_max_p_rerank` 6,
`e_atset_mean` 7, `e_ntfidf_mean` 8.

**Rejected**, on two independent grounds plus a guard breach (`compare_x1_strsim_vs_x4_entity_{OOF,MATCH_VALID}.json`):
(1) OOF paired delta +0.00016 [0.00004, 0.00029] -- CI excludes zero but the point estimate is below the
+0.0005 minimum gain (rule §2.6, first bullet), same failure mode as X2; (2) MATCH_VALID paired delta
+0.00033 [-0.00001, 0.00072] -- CI lower bound is negative, failing X4's extra adoption condition (X3
precedent: any gate/feature stacked on pass-1 p needs the MATCH_VALID CI lower bound > 0, because pass-1 p
already carries most of the signal and small OOF gains do not reliably replicate); (3) the `singleton`
guard segment drops OOF -0.00206 [-0.00333, -0.00069] (CI upper bound < 0, breaching the guard-rail bullet
even though the drop itself is under the 0.002 threshold... loss magnitude 0.00206 is just over it) --
mechanism: `singleton_with_prediction` entities (true singleton, matcher predicts something -> F0.5 0.0)
rose 190 -> 211 under the gate (`counts_before_after` in `match_train_x4_entity.json`), while
`matched_empty_true_in_candidates` (a true match was in the candidate list but the matcher predicted
nothing) fell 599 -> 551 as intended by tau_force. Net effect: the gate trades genuine empty->recovered
matches for new false positives on true singletons, at a ratio that does not clear F0.5's precision-weighted
cost of a singleton false positive (any non-empty prediction on a true singleton scores 0, vs. a partial
recovery scoring < 1.0). country_India improved most (OOF +0.00032 [0.00013, 0.00052]), consistent with
X1/X3's pattern of India carrying more of the gate's available gain; country_US was flat (+0.00006
[-0.00009, 0.00021]). Wall time: whole `match-gate-train` run (entity table + 5-fold training + grid
search + MATCH_VALID confirmation) 33s -- reusing the existing 68-feature MATCH_TRAIN/MATCH_VALID feature
files and the frozen x1_strsim pair models, no `match-features` or pair-model retraining needed. Code +
test snapshot before/after: `work/match/snapshots/match_x4_entity_before.py`,
`test_match_x4_entity_before.py`, `cli_x4_entity_before.py`, `match_config_x4_entity_before.json`
(pre-edit) and `work/match/runs/x4_entity/` (trained, not adopted). `src/ber/match.py`, `tests/test_match.py`
and `src/ber/cli.py` were restored to the pre-edit snapshots and `configs/match.json`'s `entity_gate` key
was removed; the full test suite (168 tests) passes on the restored tree; the live `work/match/models/`
was never touched by `match-gate-train` (it only writes to `work/match/runs/x4_entity/`), and
`match-eval --set MATCH_VALID` on the restored tree reproduces F0.5 0.97848 exactly
(`artifacts/metrics/match_eval_MATCH_VALID.json`), confirming x1_strsim remains live and current best.

X5 note (model side, plan §3 X5; both arms independent runs against x1_strsim, neither against the other):
implemented `configs/match.json` keys `monotone_increasing` (optional list; `_monotone_vector` in
`src/ber/match.py` builds the `monotone_constraints` vector aligned 1:1 to `used_features(mcfg)`, i.e.
after `drop_features` shrinks the column list, not just zeroing entries -- covered by
`TestMonotoneConstraints.test_vector_aligned_with_drop_features`) and `objective_mode` (`"lambdarank"`;
`_lgb_params` sets `objective`, `metric: ndcg`, `eval_at: [1,3,5]`, leaves
`lambdarank_truncation_level` at its LightGBM default; `run_match_train`'s fold loop sorts `fit_i`/`es_i`
by `s1_idx` and calls `Dataset.set_group` with run lengths from `group_bounds`, covered by
`TestLambdarankObjective.test_group_sizes_sum_to_row_count`). `load_matcher` refuses to score a matcher
whose stored `objective_mode` differs from the live config's. calibration and the decision rule are
unchanged in both arms (isotonic on OOF scores, threshold + one-owner).

- **x5_mono** (19 monotone-increasing similarity columns: all pair-similarity features in the X5a list
  exist in `FEATURES`, none skipped). OOF **-0.00102** [-0.00122, -0.00084] (180,000 queries,
  `match_train_x5_mono.json`); MATCH_VALID confirms the loss, **-0.00097** [-0.00150, -0.00046] (20,000
  queries). Every guard segment moved negative: `country_India` -0.00141 [-0.00172,-0.00108],
  `country_US` -0.00076 [-0.00101,-0.00054], `singleton` -0.00333 [-0.00505,-0.00152], `match_1`
  -0.00363 [-0.00583,-0.00160] (`compare_x1_strsim_vs_x5_mono_OOF.json`). Mechanism: constraining 19
  columns to be monotone-increasing removes every interaction LightGBM previously used to down-weight a
  high similarity score in combination with a conflicting signal elsewhere in the pair (e.g. high
  `n_tset` but `a_digits_state`==1 conflict); those interactions carried real separating signal on this
  data, so the constrained trees fit worse everywhere, not just on the columns constrained. Rejected
  mechanically (rule §2.6, first bullet: OOF delta must be >= +0.0005 with CI lower bound > 0; this
  delta is negative). Wall time: training 297s (5 folds).
- **x5_rank** (`lambdarank`, `ndcg` early stopping, same 5 folds/groups as the pair model). OOF
  **-0.01983** [-0.02037, -0.01935] (`match_train_x5_rank.json`); MATCH_VALID confirms the loss,
  **-0.01892** [-0.02044, -0.01743]. Early stopping converged in 90-184 boosting rounds (`ndcg@1`
  0.999+) vs. x1_strsim's 1000+ rounds on binary logloss -- lambdarank optimizes within-S1 ordering, which
  saturates fast once the top candidate is usually ranked first, but the isotonic-calibrated score is a
  much worse absolute probability estimate for the threshold+one-owner decision rule (built for a
  probability, not a rank score): OOF `rule_comparisons.expected_f_vs_threshold` delta is -0.02620
  [-0.02706,-0.02538] under x5_rank (`match_train_x5_rank.json`) vs. a small positive delta under
  x1_strsim, showing the score's shape stops matching the decision rule's assumptions, not just a small
  regression. Rejected mechanically (same rule; delta far below +0.0005 and negative).
- Neither arm adopted; no combined/stacked run. `configs/match.json` was restored to the pre-X5 snapshot
  (`work/match/snapshots/match_config_x5_before.json`, diff: `objective_mode`/`monotone_increasing`
  removed) and the live `work/match/models/` was recopied from `work/match/runs/x1_strsim/models/`;
  `match-eval --set MATCH_VALID` on the restored tree reproduces F0.5 0.97848383 exactly (exact match to
  `match_eval_MATCH_VALID.json`'s value before X5). Full test suite (175 tests: 168 pre-X5 + 7 new for
  `_monotone_vector` / lambdarank param building / group-size partitioning) passes on the restored tree.
  Code + config snapshot before edits: `work/match/snapshots/match_x5_before.py`,
  `test_match_x5_before.py`, `match_config_x5_before.json`; both arms' trained folds live at
  `work/match/runs/{x5_mono,x5_rank}/` (not adopted, not deleted).

X5c LOCO (report only, no adoption decision; `plan/checks/loco_matcher.py`, x1_strsim's frozen 68-feature
set and decision rule `threshold=0.67, one_owner=True`; country used only to split MATCH_TRAIN rows into
each direction's 5 training folds, never as a feature; live `work/match/models/` untouched, each
direction's folds written to its own `work/match/runs/loco_{us,india}/`). Results in
`artifacts/metrics/loco_x1_strsim.json`:

- **Train US (107,859 S1) -> score India (72,141 S1):** F0.5 LOCO 0.95387 vs. x1_strsim's own OOF on the
  same India S1 0.97332 -- paired delta **-0.01945** [-0.02018, -0.01875], 8,492 / 72,141 S1 changed.
- **Train India (72,141 S1) -> score US (107,859 S1):** F0.5 LOCO 0.97256 vs. x1_strsim's own OOF on the
  same US S1 0.98125 -- paired delta **-0.00868** [-0.00914, -0.00830], 8,470 / 107,859 S1 changed.

Both directions drop well outside their CIs (no overlap with zero): losing a country from training costs
more OOF F0.5 than any single X-series feature group gained, and the two directions are asymmetric
(dropping US from training costs India roughly 2.2x what dropping India costs US), consistent with the
model's larger installed base of US training rows (1,323,633 raw S1 vs. 883,188 India) giving it less to
fall back on when scoring the country it never saw a single training S1 from. Since France (15% of TEST)
is a third, fully-unseen country in this diagnostic's own train/score split, this LOCO number is a lower
bound on what a genuinely unseen country could cost, not a France-specific estimate; x5_mono was not
adopted so no monotone-constrained LOCO run was made per §5's conditional ("if X5a was adopted"). Action:
given the LOCO drop is large and unaddressed by either X5a or X5b, a country-invariance-focused feature
or training change (e.g. per-country IDF averaging already used, or adversarial-validation-style feature
audit for country-correlated columns) is a candidate for a future round; not implemented here since X5 is
scoped to LOCO measurement plus the two named model-side arms.

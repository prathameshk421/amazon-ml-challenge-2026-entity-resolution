# RCA of the leaderboard gap and research plan (after submission 2)

Status: 2026-09-26. Diagnosis measured; experiments in §4 not started. Evidence labels per `AGENTS.md`:
**(a)** measured on our data (file cited), **(b)** published result on other data, **(c)** follows from the
mechanism or arithmetic, **(d)** guess. Test is used for inputs only (logged in `reports/test_usage_log.md`);
no test labels exist and nothing below is tuned on test.

New measurement scripts: `plan/checks/test_shift_diag.py` → `artifacts/metrics/test_shift_diag.json`,
`plan/checks/dense_stratum_rules.py` → `artifacts/metrics/dense_stratum_rules.json`. The train reference
throughout is full_single's calibration slice (165,545 train S1, never used for fitting or early stopping).

---

## 1. The problem, in numbers

| | Internal | Public LB | Gap |
|---|---|---|---|
| Submission 1 (x1-era, 50 feats, 180k S1) | 0.97449 MATCH_VALID | 0.966109 | −0.0084 |
| Submission 2 (full_single, 68 feats, 1.65M S1) | 0.98161 held-out | 0.966797 | −0.0148 |

(a) `match_train_full_single.json`, `reports/test_usage_log.md`. Internal gain +0.0071, leaderboard gain +0.0007:
about 10% of the matcher's improvement reached test.

## 2. Root cause: what the diagnostics show

### 2.1 Test lists are more ambiguous, and ambiguity is where the loss lives
Uncertain-band candidates = rows with calibrated matcher p in [0.1, 0.9) (`test_shift_diag.json` §A):

| Set | band rows / S1 | S1 with ≥1 band row | stage-2 list size | S1 with list > 5 |
|---|---|---|---|---|
| train ref (US+India) | 0.254 | 20.8% | 5.42 | 23.7% |
| test US | 0.473 | 34.8% | 5.60 | 31.9% |
| test India | 0.337 | 26.1% | 5.63 | 32.0% |
| test France | 0.912 | 57.0% | 8.99 | 64.2% |

On train labels (`dense_stratum_rules.json`), S1 with ≥1 band row score **0.9502** and S1 with none score
**0.9897**. The band stratum is 20.8% of S1 but carries 56% of the loss (0.0104 of 0.0185). (a)

Mix-shift arithmetic: re-weight the train within-stratum scores to each test country's band share
→ US 0.9760, India 0.9794, France 0.9672, test-weighted **0.9763** (−0.0052 vs 0.9815). This assumes the
within-stratum score is unchanged on test, which is optimistic for France. (a) inputs, (c) extrapolation.

**Mechanism:** the test pool has 5.75 records per S1 against 4.68 on train, in every country
(US 5.76, India 5.82, France 5.53; train 4.67/4.68). Predicted matches per S1 on test (3.29–3.43) and the
empty share (5.3–5.9%) look like train (3.46 true links, 5.6% singletons). So the extra records are most
likely distractors, not bigger clusters: about 2.29 non-matching pool records per S1 on test against 1.215
orphans per S1 on train. (a) counts, (c) interpretation. More near-duplicate distractors means more
candidates the matcher cannot separate, and those are exactly the S1 that lose points.

### 2.2 The model knows test is harder, and still overestimates itself
- Self-estimated F0.5 (labels drawn ~ Bernoulli(p_cal), candidates only): train ref 0.9883, actual 0.9887,
  so calibration holds in-distribution. Test: US 0.9812, India 0.9851, **France 0.9637**. (a)
- Performance estimator (LightGBM on list statistics → per-S1 F0.5, trained on the ref, OOF check within
  ±0.004 of actual per country): test US 0.9749, India 0.9775, **France 0.9552**, weighted **0.9732**. (a)
  Valid only under covariate shift. Its blocking-miss detector is weak (AUC 0.63–0.65), so its
  blocking-loss component is poorly identified.
- LB 0.9668 vs estimate 0.9732: **about 0.006 of the gap is invisible to list statistics**. That is confident
  errors under shift: France (unseen country; LOCO costs 0.009–0.019, `loco_x1_strsim.json`) and blocking
  misses that grow with density. (a) numbers, (c) attribution.

Gap budget (−0.0148): about −0.005 from the shift toward ambiguous lists, about −0.003 more that the list
statistics also predict (France plus harder US/India lists), about −0.006 not visible without labels. (c)

### 2.3 Things ruled out
- **IDF scale drift is not a bug.** Adversarial validation of train-ref vs test rows (US+India) reaches
  AUC 0.986, dominated by IDF features (`n_p_miss_maxidf` 37% of gain), because the maximum idf =
  ln(1 + n_pool) differs (16.150 train vs 16.115 test). The live model has **0** of 139,796 IDF splits
  inside either gap (16.115–16.15, 15.42–15.457), so predictions are unaffected. (a) This AUC hides the
  real shift; re-run it without IDF/count columns (D1).
- **The decision rule is not the fix.** On the test-like strata of train, the best threshold equals the
  live 0.69 for band S1. Expected-F vs threshold: +0.00024 [−0.00065, +0.00107] on band S1. Lists > 5 prefer
  0.79, but only +0.00019. (a) `dense_stratum_rules.json`
- **The stage-2 cut does not truncate positives at the list tail.** On train, "last kept slot predicted
  positive" does not predict a missed true match (7.3% vs 7.4%). On test, fewer lists end in a positive
  slot (8.9% vs 11.4%; France 4.0%). Test lists are padded with distractors, not cut short. (a)

## 3. Answers to the three questions

### 3.1 Has LightGBM hit its ceiling?
Not a capacity ceiling. It has hit a **returns ceiling on train-like data**, and that is not what limits the leaderboard.
- Data scaling: 45k→1.65M S1 gives 0.9746 → 0.9822 (5-fold, capped), about +0.0012–0.0021 per doubling
  (`learning_curve.json`). No more train S1 can be added: validation S1 is scoring-only and UPSTREAM is
  excluded (leak L2). (a)
- Feature groups: +0.00211 → +0.00078 → +0.00047 / +0.00074 (rejected) → +0.00016 (rejected).
  Model-side: monotone −0.00102, lambdarank −0.01983 (`plan/FEATURE_EXPERIMENTS.md` §4). (a)
- Transfer: 10% of internal gain reached LB (§1). (a)
- The remaining in-candidate loss is information-limited for pairwise features. 45% of false negatives have
  name token-set ≥ 95 and 41% have an empty pool address (`error_patterns.json`): the pair looks the same as
  a chain sibling. (a) That needs list-level or semantic evidence, not more string metrics. (c)
- Not yet measured: hyperparameters (hparam_search results not back). Expected size is small, (d).

**Conclusion:** stop feature engineering aimed at train-like errors. Feature work is still worth doing
if it targets **ambiguous dense lists** or **country invariance** (§4 tiers 1 and 3), because those
are the two things test has more of.

### 3.2 Is the re-ranker the problem on test? Should it be replaced by bi-encoder + cross-encoder?
- Stage 1+2 cost on train: in-candidate F 0.9887 → full 0.9815 = **0.0072** (US 0.0044, India **0.0113**;
  7.4% of S1 miss ≥1 true match). Split on DEV_B: stage 1 ceiling 0.9955 (E6), after the stage-2 cut
  0.9933, so stage 1 is about 0.0045 and the stage-2 cut about 0.0022. (a) `s1_E6.json`, `rerank_eval_DEV_B.json`
- The stage-2 model is under-built. It trained on 45,000 queries and `best_iteration` 400 = cap 400, so it was
  **truncated** (`rerank_train.json`). US→India LOCO costs 0.004 ceiling (`rerank_loco.json`). (a)
- On test, stage 2 lets in more distractors (list size +3.5% US/India, **+66% France**; France p_rerank
  at rank 5 is 0.41 vs 0.25 on train). There is no sign it drops more positives at the tail (§2.3). (a) Test
  recall itself cannot be measured without labels.
- **A cross-encoder cannot replace stage 2 on this hardware.** Stage 2 scores ~265 stage-1 candidates per
  S1 (E6 mean size), about 460M pairs for test alone. At an assumed 2,000 pairs/s on the M3 GPU (MPS),
  that is ~64 hours. (c) arithmetic, throughput (d) until measured in X0. A cross-encoder fits at stage 3
  on the ~10.6M test candidate rows, or only the ~0.8M band rows.
- **A bi-encoder is a stage-1 recall tool,** a new blocking channel, not a re-ranker. It can only recover
  the stage-1 share (~0.0045 train; unknown on test).

**Conclusion:** the re-ranker is a measured but secondary loss (~0.002 on train). Retrain it properly
(R1). Add the transformer where the loss is: a cross-encoder as a stage-3 feature on ambiguous lists
(M1), and a bi-encoder as an extra stage-1 channel only if its recall check (R2) pays.

### 3.3 Where the next big wins are
1. **Ambiguous-list S1**: 56% of train loss, and 1.3–2.7× more common on test. A gain there has more
   leverage on test than on internal validation, the reverse of what the X-series features did. (a)+(c)
2. **France / unseen-country invariance**: 15% of test, self-estimate 0.955–0.964 vs 0.975–0.985 for
   US/India. Closing half the gap is worth ~+0.002 overall. (a)+(c)
3. **Blocking recall under density**, mostly India (0.0113 on train) and probably larger on test. (a)+(d)

## 4. Experiment plan (ordered)

Every training run saves a submittable model (`AGENTS.md`). Each row states the training data and save path.
Adoption uses the usual paired rule (`plan/FEATURE_EXPERIMENTS.md` §2.6), **plus a new required
report: the paired delta on the band stratum (S1 with ≥1 row p_cal ∈ [0.1, 0.9))**, because that stratum
is the test-relevant one (§2.1).

### Tier 0: diagnostics (no model, < 1 h each)
| ID | What | Why | Cost |
|---|---|---|---|
| D1 | Adversarial validation re-run **without** idf/count columns; per-feature drift, US+India only | find the real train→test shift hidden behind the IDF artifact | 10 min |
| D2 | Stage-1/2 loss decomposition per country on DEV_B: true pair not in union / in union but cut / below p_min | decides R1 vs R2 priority | 30 min |
| D3 | Synthetic densification: into each train-ref list, add candidates from other S1 lists with near-equal names (real records, known non-matches); recompute `list_features`; score F0.5 vs added density 0×/1×/2× | gives a **test-like validation set built from train labels only**; measures density elasticity | 1–2 h code |
| D4 | France input audit (aggregate token stats only, no hand-labelling): legal forms (sarl/sas/eurl), "rue/avenue/bd", CEDEX, arrondissement digits; are they in generic set G and the abbreviation map | cheap French normalization gaps | 30 min |

### Tier 1: ambiguous lists (largest expected transfer)
| ID | Mechanism | Evidence | Train data → save | Cost / time to submission |
|---|---|---|---|---|
| X0 | Throughput check: `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M) cross-encoder forward on MPS, seq len 64 | sizes M1 | none | 15 min |
| M1 | **Cross-encoder stage-3 feature.** Fine-tune a multilingual MiniLM on serialized pairs (Ditto format, `name [SEP] addr` both sides), BCE, 5-fold by S1 on MATCH_TRAIN pairs (975,699 rows) for OOF `p_ce`. Score band rows + top-3 per S1. Add `p_ce`, `p_ce_rank`, `p_ce_gap` to LightGBM | (b) Ditto (Li et al. 2020) large F1 gains on hard EM benchmarks; (c) targets information-limited pairs; France gets multilingual pretraining | MATCH_TRAIN → `work/match/runs/m1_ce/models/` + CE weights under `work/ce/`; OOF + band-stratum + LOCO report | ~1 day. Needs team sign-off on pretrained checkpoints (`CROSS_ENCODER_NOTES.md` §1) |
| M2 | **Density-augmented training**: train the matcher on D3-densified lists (1× extra) so it learns the dense regime; test-time unchanged | (c) matches the §2.1 shift; (a) after D3 | MATCH_TRAIN(+densified) → `work/match/runs/m2_dense/models/`; judged on D3 2× set + band stratum | 3 h |
| M3 | Density-invariant list features: p_rerank and name-sim z-scores within the list, rank of pair among same-name candidates, share of the list above 0.5 | (c) list features currently count absolute numbers that grow with density (`l_n` SMD +0.20 train→test US/India, +1.26 France) | MATCH_TRAIN → `runs/m3_linv/models/` | 2 h |
| M4 | Re-try X3 sibling features with a singleton guard (sibling boost only if S1 has ≥1 candidate with name+address confirmed) | (a) X3 gave +0.00074 OOF, +0.00175 VALID, failed only the singleton guard | MATCH_TRAIN → `runs/m4_sib/models/` | 2 h |

### Tier 2: candidate generation
| ID | Mechanism | Evidence | Cost |
|---|---|---|---|
| R1 | **Retrain stage 2 properly**: 45k → ≥300k queries, round cap by `ROUND_CAP_EVIDENCE.md` (no truncation), then DEV_B ceiling at the same mean size (5.43) and LOCO | (a) truncated at 400/400 on 45k queries | train 1–2 h; then all candidates must be regenerated (TEST ~2.4 h) and the matcher retrained → about 1 day to submission |
| R2 | **Bi-encoder recall check, then channel**: embed with the same multilingual MiniLM (no fine-tune first); per-country exact top-50 cosine search (torch matmul in chunks; FAISS is not installed and would need a license entry). Measure: what share of DEV_B true pairs that stage 1 misses land in the bi-encoder top-50 | (b) dense retrieval recovers lexically dissimilar pairs; (d) here | pool embedding ~10M records on MPS: 1–2 h (d until X0). Ship as a channel only if it recovers ≥ 20% of stage-1 misses at +≤10 candidates |
| R3 | Adaptive stage-2 cut from list statistics (keep more when p_rerank at rank 5 is high) | (a) p_min 0.0005 gives ceiling +0.0017 at 2.1× size (`rerank_cut.json`); test lists already longer | cheap, but judge by matcher F0.5 on train, not ceiling |

### Tier 3: unseen country (France)
| ID | Mechanism | Evidence | Cost |
|---|---|---|---|
| F1 | LOCO-guided feature audit: drop one feature group at a time, re-run US→India LOCO; keep removals that shrink the LOCO drop without in-distribution loss | (a) LOCO drop −0.0195 US→India, −0.0087 India→US | 5 groups × 10 min |
| F2 | LOCO with M1's `p_ce`: does a multilingual scorer shrink the unseen-country drop? | (c) the only source of cross-language knowledge we are allowed | part of M1 |
| F3 | French normalization fixes from D4 (unsupervised, test pool statistics only, same as the existing per-country abbreviation mining) | (d) until D4 | 1 h + test feature rebuild |

### Tier 4: LightGBM itself (low expected transfer; do only as cheap add-ons)
- G1 hparam search results (pending, friend's machine). G2 3-seed bagging, or XGBoost/CatBoost (Apache-2.0,
  not installed). (b) GBDT libraries land within a small margin of each other; (c) the upside is variance
  reduction on the band stratum only.
- **Stop:** new string-similarity feature groups aimed at train-like errors (§3.1).

### Not recommended / needs a ruling first
- Importance-weighting training to match test list statistics, or pseudo-labelling test. Both use test
  inputs to shape training (`VALIDATION_PLAN.md` L9), so they need an explicit organiser/team ruling.
- A cross-encoder as the stage-2 re-ranker (§3.2 arithmetic).
- A local LLM judge (7B) over band rows: ~0.8M test rows at a few pairs/s on the M3 is several days (c).

## 5. Suggested order and decision points
1. D1, D2, D3, X0 (half a day, no training). D3 gives a test-like validation set; X0 decides whether M1
   fits this machine.
2. M1 (with sign-off) and M2 in parallel design, one training at a time. Adopt on OOF + band stratum +
   D3-2× set. Build a submission from each adopted run.
3. R1 only if D2 shows the stage-2 cut or stage-2 ranking holds most of the India loss. Otherwise R2.
4. F1/F3 folded into whichever matcher run comes next.

Expected size (all (d) until measured): M1 is the single largest bet. If it lifts band-stratum F by
+0.01 (of the 0.04 band gap), the test effect is about +0.0034 at the test band share (34%), against +0.0021 internally.

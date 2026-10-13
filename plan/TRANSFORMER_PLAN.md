# Transformer work: where it fits, and the task split (3 machines)

Status: 2026-09-26, plan only. Context and evidence: `plan/RCA_RESEARCH_PLAN.md` (test lists carry 1.25–2.7× more
ambiguous candidates; S1 with ≥1 candidate at p_cal ∈ [0.1, 0.9) score 0.950 vs 0.990 and hold 56% of loss,
`artifacts/metrics/dense_stratum_rules.json`). Labels: (a) measured, (b) published, (c) mechanism/arithmetic, (d) guess.

## 1. Where a transformer fits in this pipeline

| Place | Role | Verdict | Why |
|---|---|---|---|
| Stage 3 feature (cross-encoder → `p_ce` column in LightGBM) | score each candidate pair jointly | **Do first** | ~10.6M test rows, ~5.4 per S1; targets the ambiguous-list stratum that grew on test; stays inside the existing paired-OOF adoption rule (c) |
| Stage 1 extra channel (bi-encoder, e.g. `BAAI/bge-m3` embeddings + exact top-k per country) | recall for pairs with no shared token | **Second, gated on a recall check** | can only recover the stage-1 share of loss (~0.0045 train, `s1_E6.json`) (a) |
| Stage 2 re-ranker replacement (cross-encoder) | rank ~265 stage-1 candidates per S1 | **No** | ~460M test pairs; at an assumed ~1,000 pairs/s on a 4060 for a 568M model, that is ~5 days (c)+(d). Retrain the LightGBM re-ranker instead (truncated at 400/400, `rerank_train.json`) |
| List-level model (a transformer reading the whole candidate list) | resolve siblings jointly | Later / research | direct fit to "same name, which branch" errors, but no prior run; (d) |
| Local 7B LLM judge | decide band pairs | No | several days for ~0.8M band rows on this hardware (c) |

## 2. Corrections to the pasted cross-encoder task (for Friend A's agent)

The task is clear and sound as a standalone model. Four things must change for it to help the submission:

1. **8 GB will not fit full fine-tuning of a 568M model with AdamW.** fp32 master weights 2.3 GB + grads 2.3 GB +
   Adam moments 4.5 GB ≈ 9.1 GB before activations (c). Use one of: LoRA (`peft`, Apache-2.0, r=16 on attention +
   classifier head fully trained), or 8-bit AdamW (`bitsandbytes`, MIT) + gradient checkpointing, batch 16 × accum 4.
   Fallback: `BAAI/bge-reranker-base` (278M) if throughput is too low. Log any new library in `reports/licenses.md`.
2. **Do not train on MATCH_TRAIN if the score becomes a LightGBM feature trained on MATCH_TRAIN.** The cross-encoder
   memorizes its training labels, so `p_ce` would be over-confident exactly on the rows LightGBM learns from, and
   LightGBM would over-trust it (stacking leakage, `VALIDATION_PLAN.md` L2). Train the cross-encoder on
   **DEV_REST_180K** (180,000 train S1, 976,902 candidate rows, disjoint from MATCH_TRAIN and MATCH_VALID), then score
   MATCH_TRAIN, MATCH_VALID and TEST. The standalone check (step 3) uses MATCH_VALID.
3. **Keep empty-address rows in the output, as NaN.** Drop them from cross-encoder *training* only. Every candidate
   row needs a row in the score file, so the matcher can join on (s1_idx, pool_idx); LightGBM handles NaN.
4. **The real bar is not "AUC above the untrained model".** Report standalone ROC-AUC / PR-AUC on MATCH_VALID pairs
   next to our existing address columns on the same rows (`a_tfidf_cos`, `a_tset`, `a_idf_jacc`). Adoption is only
   decided by the matcher experiment (§4): paired OOF delta ≥ +0.0005 with CI > 0, plus the band-stratum delta.

Other details:
- Input text: raw `business_address` (the multilingual model benefits from French accents and native script).
  Ablation later: normalized text from `work/norm/*_pool_store`.
- Negatives: the candidate lists already contain hard negatives (positive rate ~0.62). Use them as-is; no random negatives.
- max_length 128 for address pairs is enough; use dynamic padding (most pairs are far shorter).
- One epoch on ~855k pairs first; a second epoch only if MATCH_VALID logloss still falls.

## 3. Task split

**Friend A (current GPU): address cross-encoder `ce_addr`** — the pasted task with the §2 fixes.
Train data: DEV_REST_180K pairs with a non-empty address on both sides. Save: `artifacts/bge-reranker-address/`
(model + tokenizer + config) and the score files in §5. Time to a submittable matcher once scores arrive:
~1 h of matcher work plus ~3–6 h to score all 10.6M test rows (d, measure throughput on 10k rows first).

**Friend B (RTX 4060): full-record cross-encoder `ce_rec`** — same scripts, different input:
`"{name} | {address}"` on both sides (Ditto-style serialization; empty address kept as `"{name} | "`).
Mechanism: covers the other half of the errors: aliases (13% of FNs have name token-set < 50 and address ≥ 80) and
empty-pool-address pairs (41% of FNs), which an address-only model cannot see (a) `error_patterns.json`. Same training
set, same output contract. Evidence: (b) Ditto, Li et al. 2020, large F1 gains over non-deep matchers on hard EM
benchmarks; (d) here.
While the GPU trains, B's CPU can run `plan/checks/hparam_search.py` case A (`plan/HPARAM_SEARCH_HANDOFF.md`).
Expected value is low: tuning moves train-like accuracy, and only ~10% of that reached the leaderboard (a). Also
report the band-stratum delta for the best trial.

**Me (M3, this repo): data + integration + evaluation**
1. Build and ship `pairs_{DEV_REST_180K,MATCH_TRAIN,MATCH_VALID,TEST}.parquet` (s1_idx, pool_idx, label (train sets
   only), raw name/address both sides, country). The friends then need no pipeline checkout. Test file = inputs only;
   log in `reports/test_usage_log.md`.
2. Add optional `p_ce_addr` / `p_ce_rec` columns to the matcher (plus within-list rank and gap to the list max),
   read from `work/ce/<name>/<SET>.parquet`, NaN when missing; unit tests.
3. Tier-0 diagnostics D1–D3 from `RCA_RESEARCH_PLAN.md`, including the densified test-like validation set.
4. Run the matcher experiments in §4 and build a submission from each adopted run.

## 4. Experiments that decide adoption (all on MATCH_TRAIN 180k, paired vs x1_strsim, same folds)

| ID | Matcher features | Adopt if |
|---|---|---|
| T1 | x1_strsim + `p_ce_addr` (+ rank, gap) | OOF Δ ≥ +0.0005, CI > 0; band-stratum Δ reported; guards per `FEATURE_EXPERIMENTS.md` §2.6 |
| T2 | x1_strsim + `p_ce_rec` (+ rank, gap) | same |
| T3 | both | same, vs the better of T1/T2 |
| T4 | best of T1–T3, LOCO US→India | report the change in the unseen-country drop (baseline −0.0195, `loco_x1_strsim.json`) |

Each run saves `work/match/runs/<T#>/models/` + `matcher.json`. The submittable version retrains on
MATCH_ALL (full_single recipe, round cap per `ROUND_CAP_EVIDENCE.md`). That needs `p_ce` on DEV_REST_REST too, so
the cross-encoders must score all train sets except their own training set. Rows from DEV_REST_180K itself
(the cross-encoder's training set) are excluded from that matcher retrain.

## 5. Output contract (both friends)

`work/ce/<ce_addr|ce_rec>/<SET>.parquet`, one row per candidate row of that set:
`s1_idx:int32, pool_idx:uint32, p_ce:float32 (sigmoid, NaN if not scorable)`, plus `meta.json`
(model id, checkpoint sha, train set, input format, max_length, epochs, lr, MATCH_VALID ROC-AUC/PR-AUC/logloss).
Sets to score: MATCH_TRAIN, MATCH_VALID, DEV_REST_REST, TEST.

## 6. Rules check before any of this ships
- Pretrained checkpoint use: team sign-off, open question in `plan/CROSS_ENCODER_NOTES.md` §1. Model runs
  offline; license Apache-2.0 (bge-reranker-v2-m3) / MIT (bge-m3); 568M ≤ 8B.
- No hosted inference, no external data; test is inputs only.

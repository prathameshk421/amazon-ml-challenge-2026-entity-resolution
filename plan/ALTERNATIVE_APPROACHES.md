# Alternative approaches after submission 2

Status: analysis, 2026-09-26. Nothing here is implemented. Evidence labels per `AGENTS.md`: (a) measured on
our data, (b) published result on other data, (c) should follow from the mechanism, (d) guess.

## 0. What the leaderboard told us

| | Internal F0.5 | Public leaderboard | Gap |
|---|---|---|---|
| Submission 1 (50 features, 180k S1) | 0.97449 MATCH_VALID | 0.966109 | -0.0084 |
| Submission 2 (68 features, 1.65M S1) | 0.98161 held-out 165k S1 | 0.966797 | -0.0148 |

(a) `reports/test_usage_log.md`, `artifacts/metrics/match_train_full_single.json`. Internal gain +0.0071,
leaderboard gain +0.0007. About 90% of the matcher improvement did not reach test. Every internal number is
measured on train S1 against the train pool; test differs in three measured ways:

1. Test pool is denser: 5.754 pool records per S1 vs 4.677 on train (+23%) (a, `plan/checks/split_design.json`).
2. France is 15.0% of test S1 (259,452) and absent from training (a, `match_predict_TEST.json`).
3. Test has its own pool text: 7.7% of test candidate rows have an empty pool address vs 12.4% on train
   (a, `match_features_TEST.json` vs `match_features_MATCH_TRAIN.json` nan shares).

Consequence: an approach is worth trying only if it plausibly acts on one of these three, or on the stage
before the matcher (candidate recall). Better pair classification on train-like pairs has been shown to
barely move the leaderboard.

## 1. Approaches, ranked by fit to the observed gap

### 1.1 Check the candidate cut on test first (diagnostic, ~15 min, no new libraries)
- **Mechanism:** the stage-2 cut keeps ~5.4 candidates per S1 (K_min 5). If test S1 have more true matches
  each (denser pool), the cut drops more of them before the matcher sees them.
- **Check, inputs only:** compare the re-ranker score at ranks 5-10 on test candidates vs train candidates
  (`work/candidates/TEST/query_stats.parquet` has p at ranks 1-5; train candidates carry `p_rerank`). If
  test S1 have high p at rank 5 far more often, the cut is truncating real matches on test.
- **Evidence:** (a) for the density difference, (c) for the effect.
- **If positive:** widen the cut or make it adaptive for all sets, regenerate TEST candidates (~2.5 h), retrain.
  The earlier "don't widen K_min" decision (`ADDRESS_EXPERIMENTS.md` S4) used train data only.

### 1.2 Dense bi-encoder retrieval as an extra candidate channel (GPU helps)
- **Mechanism:** a multilingual sentence encoder (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  Apache-2.0, 118M) embeds name+address; FAISS (MIT) nearest-neighbour search over the pool adds candidates
  that share no token key with the S1 (transliteration, abbreviation, reordering).
- **Targets:** candidate recall (blocking misses: 0.0064 of OOF loss on train, likely more on test).
- **Evidence:** (b) dense retrieval recovers pairs missed by token blocking in entity-resolution work;
  (d) for our data.
- **Limitation:** embedding ~20M records on CPU is many hours; on a GPU it is feasible. Adds a stage-1
  channel, so candidates for every set must be regenerated.

### 1.3 Cross-encoder transformer scorer, stacked as one matcher feature (GPU)
- **Mechanism:** fine-tune a small multilingual encoder on concatenated record pairs (Ditto pattern, Li et
  al. 2020); feed its probability into LightGBM as a column. See `plan/CROSS_ENCODER_NOTES.md`.
- **Targets:** France (multilingual pretraining is the only source of French knowledge we are allowed) and
  the semantic hard tail.
- **Evidence:** (b) Ditto reported large F1 gains over non-deep matchers on public benchmarks; (d) here.
- **Limitation:** rules question on pretrained checkpoints (see that file); CPU inference on 10.6M test
  pairs is slow, restrict to uncertain-band pairs.

### 1.4 Local open LLM (≤8B, MIT/Apache) as a judge on the hardest pairs
- **Mechanism:** a 7B instruction model (e.g. Qwen2.5-7B-Instruct or Mistral-7B, both Apache-2.0; Llama is
  NOT MIT/Apache and is excluded) reads both records and answers match / no-match; used only for the
  uncertain band, or fine-tuned (LoRA) on MATCH_TRAIN pairs.
- **Targets:** France and noisy-name cases.
- **Evidence:** (b) LLMs as entity matchers are competitive on public benchmarks, especially fine-tuned;
  (d) here. Cost: even 5% of test pairs (~530k) is a long GPU job.
- **Limitation:** same rules question as 1.3, heaviest compute of all options.

### 1.5 Country-invariance for France (no new model)
- **Mechanism:** audit features whose distribution depends on country (LOCO drop -0.0087 to -0.0195,
  `artifacts/metrics/loco_x1_strsim.json`); drop or normalise per-country-sensitive columns, check LOCO.
- **Evidence:** (a) LOCO drop exists; (c) that reducing it helps France. France itself stays unmeasurable.

### 1.6 GBDT swaps and ensembles: XGBoost, CatBoost, LightGBM seeds
- **Mechanism:** same 68 features, different boosting implementation; or average 2-3 models.
- **Evidence:** (b) on tabular benchmarks the three libraries land within a small margin of each other
  (e.g. Grinsztajn et al. 2022 treat them as one family); (c) an ensemble reduces variance slightly.
- **Expected effect here:** small internally, and by section 0 even smaller on the leaderboard.
- **Cost:** low (XGBoost Apache-2.0, CatBoost Apache-2.0; add to `reports/licenses.md`). Worth doing only as a
  cheap add-on, not as the next main bet.

### 1.7 Neural nets on the same tabular features (MLP, TabNet, FT-Transformer)
- **Evidence:** (b) Grinsztajn et al. 2022 and later benchmarks: tree ensembles usually beat deep models on
  medium-size tabular data. No reason to expect a win on 68 engineered features. Not recommended.

### 1.8 Things to avoid
- Pseudo-labelling or self-training on test inputs: uses test data for training; conflicts with
  `AGENTS.md` rule 5 in spirit. Needs an explicit organiser ruling first.
- Any hosted API or hosted LLM (`AGENTS.md` rule 2).
- Choosing between variants by leaderboard score (`VALIDATION_PLAN.md` §9).

## 2. Suggested order

1. 1.1 diagnostic (15 min). Decides whether candidate generation is where test loses.
2. If yes: widen/adaptive cut, regenerate candidates, retrain on MATCH_ALL (~4-5 h total).
3. On the friend's GPU, in parallel: 1.2 retrieval recall check, then 1.3 cross-encoder feature.
4. 1.6 only as a cheap ensemble once the pipeline is otherwise frozen.

Any model-family addition (1.2, 1.3, 1.4, 1.6) needs team sign-off (`AGENTS.md`) and a new-library license
entry before it ships.

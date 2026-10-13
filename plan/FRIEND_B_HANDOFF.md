# Handoff: cross-encoder feature for the matcher (RTX 4060 laptop, from scratch)

For your agent: read this whole file first, then `AGENTS.md` in the repo (rules; it overrides anything here
except where this file is stricter). Then say in one paragraph what you will do and what you will send back
before running anything heavy.

---

## 0. Why this experiment (what we already measured)

Pipeline: stage 1 inverted-index blocking → stage 2 LightGBM re-ranker + cut (~5.4 candidates per S1) →
stage 3 LightGBM pair matcher (68 hand-built features) + threshold. Metric: per-S1 F0.5, macro-averaged
(precision-weighted: false positives cost more than misses; a singleton scores 1 only if we predict nothing).

| | Internal score | Public leaderboard |
|---|---|---|
| Submission 1 | 0.97449 | 0.966109 |
| Submission 2 (current, 1.65M training S1) | 0.98161 | 0.966797 |

Internal gain +0.0071, leaderboard gain +0.0007. Diagnostics on the owner's machine (test inputs only, no
test labels exist) found:
- **Test candidate lists are more ambiguous.** Rows with calibrated matcher p in [0.1, 0.9) per S1:
  0.25 on train vs 0.47 test-US, 0.34 test-India, 0.91 test-France. On train, S1 with ≥1 such row score
  F0.5 0.950 vs 0.990 for the rest, and hold 56% of all loss. Test has 1.25–2.7× more of them.
- **Hand-built string features have hit diminishing returns** (+0.0021 → +0.0008 → +0.0005 → +0.0002 per
  feature group), and ~10% of internal gains reached the leaderboard.
- Remaining false negatives: 45% have near-identical names, 41% have an empty pool address, 13% are aliases
  (very different names, same address). Pair string metrics cannot separate these; a model that reads both
  records together might.
- France (15% of test) is absent from training; a multilingual pretrained model is the only allowed source
  of French knowledge.

**The bet:** fine-tune a multilingual cross-encoder on (S1 record, pool record) pairs and add its score as
new columns in the stage-3 LightGBM. It does not replace anything. A cross-encoder cannot replace stage 2:
that would mean ~460M test pairs, several days on a 4060.

---

## 1. Machine setup

Needs: NVIDIA RTX 4060 (8 GB), **≥ 16 GB system RAM**, **~60 GB free disk**, 8+ CPU cores.
On Windows use **WSL2 Ubuntu** and keep the repo on the Linux filesystem (`~/`, not `/mnt/c`, which is
several times slower for this I/O).

```bash
sudo apt install git git-lfs python3-venv python3-dev build-essential
git lfs install
git clone https://github.com/prathameshk421/amazon_ml_challenge.git
cd amazon_ml_challenge
git checkout submission-v2
git lfs pull
ls -la work/rerank/model_v1.txt work/match/models/ submission/output/
```

**Check:** `work/rerank/model_v1.txt` must be megabytes, not a ~130-byte text pointer. If it is a pointer,
the LFS objects were not pushed. Stop and tell the owner.

Contest data (not in git): put the original dataset at `student_resource/dataset/{train,test}/*.tsv`
(unchanged files: `train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `test_source{1,2,3}.tsv`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r submission/code/business_entity_resolution/requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA driver
.venv/bin/pip install transformers peft accelerate
.venv/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Add every new library with its license to `reports/licenses.md` (peft Apache-2.0, accelerate Apache-2.0;
bitsandbytes MIT if you use it).

Model (download once, then everything runs offline):
`BAAI/bge-reranker-v2-m3`: Apache-2.0, 567.8M params, XLM-RoBERTa-large, multilingual, outputs one logit.

---

## 2. Rebuild the pipeline intermediates (CPU, one job at a time)

All commands from the repo root:
```bash
export PYTHONPATH=submission/code/business_entity_resolution/src
PY=.venv/bin/python
```

| Step | Commands | Owner's time (8-core M3) |
|---|---|---|
| ingest + split | `$PY -m ber.cli ingest --split train`; `ingest --split test`; `splits` | ~10 min |
| train normalization | `mine-abbrev --split train`; `normalize --split train --no-dict`; `learn-dict`; `normalize --split train` | ~45 min |
| train index | `build-index --split train`; `index-sanity --split train` | ~13 min |
| test side | `mine-abbrev --split test`; `normalize --split test`; `build-index --split test`; `index-sanity --split test` | ~15 min |
| candidates | `candidates --set MATCH_TRAIN`; `candidates --set MATCH_VALID`; `candidates --set TEST` | 16 min, 3 min, ~2.4 h |
| DEV_REST_180K | run only the `generate_candidates` + `build_features` parts of `plan/checks/dev_rest_180k.py` (write a 10-line wrapper; do not run its training part) | ~30 min |
| matcher features | `match-features --set MATCH_TRAIN`; `--set MATCH_VALID`; `--set TEST` | ~10 min, 1 min, ~40 min |

Do **not** re-run `stage1-train`, `rerank-build/train/cut`: the stage-1 and stage-2 models come from git and
must stay identical (the candidate step refuses to run if they are stale).

**Gate G1: the rebuild must match the owner's pipeline exactly.**
1. Candidate row counts: MATCH_TRAIN **975,699**, MATCH_VALID **108,493**, DEV_REST_180K **976,902**;
   TEST mean list size **6.1257** over 1,732,544 S1 (`artifacts/metrics/candidates_TEST.json`).
2. Live model reproduction: `match-predict` with the git copy of `work/match/models/` (submission 2,
   `full_single`) must write **5,789,591** predicted pairs (`match_predict_TEST.json`). Back up
   `work/outputs/` first.
3. Baseline matcher: `match-train --run x1_repro` (68 features, 5-fold, MATCH_TRAIN). OOF F0.5 must be
   **0.97807 ± 0.0003** (`artifacts/metrics/match_train_x1_strsim.json`). Check every fold's
   `best_iteration` is below the cap (`plan/ROUND_CAP_EVIDENCE.md`).
   **Note:** `match-train` may overwrite `work/match/models/`. Copy the live models aside first
   (`cp -r work/match/models work/match/models_full_single_backup`).

If any check fails, stop and report the numbers. Do not continue on a different pipeline.

---

## 3. Build the pair files

Script `ce/prepare_pairs.py` → `work/ce/pairs_<SET>.parquet` for SET in DEV_REST_180K, MATCH_TRAIN,
MATCH_VALID, TEST.

- Rows: exactly the candidate rows of that set, via `match.read_candidates(SET, match.set_queries(SET))`
  (for DEV_REST_180K pass its S1 array from `pick_dev_rest_slice()`). Keep `s1_idx, pool_idx, rank, p_rerank`.
- Text: raw fields from `work/records/{train,test}_s1.parquet` (by `s1_idx`) and
  `work/records/{train,test}_pool.parquet` (by `pool_idx`): `name_raw`, `addr_raw`. Treat empty or
  `<NULL>`-only addresses as empty strings.
- Serialized input: `text_a = f"{s1_name} | {s1_addr}"`, `text_b = f"{pool_name} | {pool_addr}"`.
  Name + address together: the address-only variant cannot see aliases or empty-address pairs.
- Label (train sets only): `match.pair_labels(s1_idx, pool_idx)` (1 = true link). TEST gets no label column.
- Checks to print: rows per set, positive rate (~0.62 expected), S1 overlap between DEV_REST_180K,
  MATCH_TRAIN and MATCH_VALID = **0** (assert it).
- Log the TEST read in `reports/test_usage_log.md` (`io_utils.log_test_usage`).

**Why train on DEV_REST_180K and not MATCH_TRAIN:** the cross-encoder score becomes a LightGBM feature, and
LightGBM trains on MATCH_TRAIN. A cross-encoder trained on MATCH_TRAIN would memorize those labels, its
score would look perfect exactly where LightGBM learns to trust it, and it would fail on test (stacking
leakage). DEV_REST_180K is 180,000 other train S1 that the 180k-scale matcher never sees.

---

## 4. Train the cross-encoder (GPU)

Script `ce/train.py`. `AutoModelForSequenceClassification` + tokenizer from `BAAI/bge-reranker-v2-m3`,
`tokenizer(text_a, text_b, truncation=True, max_length=160)`, one logit, `BCEWithLogitsLoss`.

**8 GB constraint:** full fine-tuning with AdamW needs ~9.1 GB for weights + gradients + optimizer state
before activations. It does not fit. Default setup:
- **LoRA** (`peft`): r=16, alpha=32, dropout 0.1, target `query,key,value,dense`; classification head fully
  trainable. lr 1e-4, linear decay, 3% warmup.
- bf16 autocast (the 4060 supports it), gradient checkpointing on, batch 16 × grad-accum 2 (effective 32),
  dynamic padding, length-bucketed batches.
- Early-stop / checkpoint selection on a **5% S1-grouped slice of DEV_REST_180K**, not MATCH_VALID.
- 1 epoch over DEV_REST_180K (~976k pairs). A second epoch only if held-out logloss is still falling at the end.
- Fallback if LoRA underperforms: full fine-tune with bitsandbytes 8-bit AdamW, lr 2e-5.

Order:
1. **Throughput probe:** train 200 steps and time them; run inference on 10,000 pairs and time it. Write both
   numbers down and extrapolate full-train and full-scoring hours before the big run.
2. **Smoke run:** 10k pairs, 1 epoch. Loss must fall; held-out AUC must beat the untrained model.
3. **Full run.** Save LoRA adapter + head + tokenizer + config to `work/ce/ce_rec/model/`, plus `meta.json`
   (model id, adapter hash, train set, input format, max_length, epochs, lr, throughput numbers).

---

## 5. Evaluate standalone (MATCH_VALID, once)

Script `ce/evaluate.py` on `pairs_MATCH_VALID` (108,493 rows, 20,000 S1):
- ROC-AUC, PR-AUC, logloss for: untrained bge-reranker-v2-m3, fine-tuned `ce_rec`, and our existing columns
  `n_tfidf_cos`, `a_tfidf_cos`, `n_tset`, `a_tset`, `p_rerank` (from `work/match/features/MATCH_VALID`).
- The same metrics restricted to the hard subsets: pool address empty; name token-set ≥ 95; alias (name
  token-set < 50 and address ≥ 80).
- Score histograms for positives vs negatives.

A standalone win is necessary, not sufficient. LightGBM on all 68 features already has ROC-AUC ~0.9975.
Adoption is decided in §7 only.

---

## 6. Score every set

Script `ce/inference.py` → `work/ce/ce_rec/<SET>.parquet`, one row per candidate row:
`s1_idx:int32, pool_idx:uint32, p_ce:float32` (sigmoid of the logit), for MATCH_TRAIN (975,699),
MATCH_VALID (108,493), TEST (~10.6M). Sort by input length before batching (large speed-up), fp16/bf16,
batch as large as fits. Checkpoint every 1M rows, since test scoring will take hours. Log the TEST read.

Reload test: fresh Python process, load from `work/ce/ce_rec/model/`, score 4 hand-written pairs
(obvious match, obvious non-match, same chain name at a different address, an alias with the same address).

If full-test scoring is estimated at > 8 h: score only rows with `rank <= 8` and write NaN elsewhere. Apply
the **same rule to every set**, so train and test see the same NaN pattern.

---

## 7. Matcher experiment T1 (this decides adoption)

Follow `plan/FEATURE_EXPERIMENTS.md` §2 exactly (snapshot before editing, unit tests, smoke, full, compare,
restore if rejected).

- Add 3 columns to `FEATURES` in `src/ber/match.py`: `p_ce`; `p_ce_rank` (rank of the row's p_ce inside its
  S1 list, 1 = best); `p_ce_gap` (list max p_ce minus this row's p_ce). Read from `work/ce/ce_rec/<SET>.parquet`
  and join on (s1_idx, pool_idx). If the file is missing, raise; never fill silently. Missing `p_ce` → NaN.
- Rebuild `match-features` for MATCH_TRAIN, MATCH_VALID, TEST. `match-train --run t1_ce`.
- `match-compare --a x1_repro --b t1_ce --set OOF` and `--set MATCH_VALID`.
- **Extra required report: band-stratum delta.** Paired per-S1 delta restricted to S1 whose x1_repro OOF list
  has ≥ 1 row with calibrated p in [0.1, 0.9). That is the test-relevant stratum. Small script, CI by S1 bootstrap.
- Also: LOCO (`plan/checks/loco_matcher.py`) for t1_ce vs x1_repro. Baseline unseen-country drop:
  US→India −0.01945, India→US −0.00868 (`artifacts/metrics/loco_x1_strsim.json`). A smaller drop is the best
  available sign of a France gain.

**Adopt if:** OOF paired delta ≥ +0.0005 with 95% CI lower bound > 0; MATCH_VALID CI upper bound ≥ 0;
no guard segment (`country_US`, `country_India`, `singleton`) drops > 0.002 with CI upper bound < 0.
Report the band-stratum and LOCO numbers either way.

Every training run must end submittable (`AGENTS.md`). `match-train --run t1_ce` saves
`work/match/runs/t1_ce/models/` + `matcher.json`. Then `match-predict` + `write-outputs` produce
`matching_results.tsv` + `candidate_pairs.tsv`; validate both with the official validator
(README §6). Before training, state in one line: data, save path, hours to a submission file.

---

## 8. Optional, only after T1 is decided

- **T1-full (if T1 adopted):** retrain at submission-2 scale (all 1.65M MATCH_ALL S1, `plan/checks/train_final_single.py`
  recipe, round cap ≥ 12,000). That needs DEV_REST_REST candidates + features + `p_ce` (~6.5M more rows to
  score). **Exclude DEV_REST_180K rows** (the cross-encoder's own training set). Wall is many hours; ask the
  owner first.
- **Address-only ablation** (`text = addr` only): tells whether name+address was needed.
- **LightGBM hyperparameter search** on CPU while the GPU is busy: `plan/HPARAM_SEARCH_HANDOFF.md`. Expected
  value is low (train-like gains transferred ~10%); report the band-stratum delta for the best trial too.

---

## 9. Rules (from AGENTS.md; breaking these can disqualify us)

- No hosted APIs or hosted models at train or inference time. The checkpoint is downloaded once, then offline.
- No extra data, no scraping. Contest TSVs stay unchanged.
- Country is an open set: never one-hot, filter or hard-code US/India. Do not add country text to the
  cross-encoder input.
- Test labels do not exist. Never hand-label test rows, never tune or pick variants by test statistics or by
  leaderboard score. Log every test read in `reports/test_usage_log.md`.
- Validation S1 (`HOLDOUT_TEST`) labels are for scoring only. Do not train on them.
- Every number you report comes from a script that wrote `artifacts/metrics/*.json`. Failed or skipped runs
  are reported as such. Add a `RESEARCH_LOG.md` entry per run (template at the top of that file).
- Do not `git push` (the owner merges). Commit locally only if asked.

---

## 10. What to send back

1. `artifacts/metrics/*.json` written by your runs (G1 checks, CE evaluation, t1_ce train/compare, LOCO, band delta).
2. Your `RESEARCH_LOG.md` entries and `reports/test_usage_log.md` lines.
3. `work/ce/ce_rec/model/` (LoRA adapter + head + tokenizer + meta; tens of MB) and the 4 `ce/*.py` scripts +
   the `match.py` diff.
4. If T1 was adopted: `matching_results.tsv` + `candidate_pairs.tsv` from t1_ce, validator output, and
   `work/match/runs/t1_ce/models/`.
5. A 10-line summary: G1 results, CE standalone numbers vs existing features, T1 deltas with CIs (all S1,
   band stratum, segments), LOCO change, hours spent per step, anything that failed.

# Business entity resolution: reproduction guide

This folder rebuilds submission 18 (public leaderboard F0.5 0.984075) from the contest TSVs:

| File | Rows | Pairs | sha1 of the submitted file |
|---|---|---|---|
| `output/matching_results.tsv` | 1,732,544 S1 (101,788 with no match) | 5,812,655 | `7334635ce65ff93c9b1ca02395f48860e343de49` |
| `output/candidate_pairs.tsv` | 1,732,544 S1 | 17,546,805 (10.13 per S1) | `8c97a66b9d08ef722d6cf34b2760c963f9c1e3b9` |

The method is described in `Documentation_template.md` at the zip root. This file covers setup, the command to
run, what each step does, and how long it took.

## 1. Pipeline in one paragraph

Text normalization (Unicode NFC, romanisation of 9 Indic scripts, accent/case folding, abbreviation maps mined from
the pool, a transliteration dictionary learned from train pairs) → stage 1: inverted index over 10 hashed key
families + logistic scoring → stage 2: LightGBM re-ranker and cut (~6.1 candidates per S1) → cross-encoder `c0`
(LoRA on `BAAI/bge-reranker-v2-m3`) scores every candidate → stage 3: LightGBM pair matcher on 85 features,
isotonic calibration, threshold 0.73, one-owner → DRACE: nearest-neighbour search with `multilingual-e5-small`
(untrained and a copy distilled from `c0`), new pairs accepted when `c0` p ≥ 0.995, one-owner → output files.

## 2. Setup

Requirements:

- Python 3.12 or newer (built with CPython 3.14.5).
- CPU steps: 8 cores and 16 GB RAM were used (Apple M3). About 15 GB of free disk for `work/`.
- GPU steps: one CUDA GPU. The submission used an NVIDIA A100 80 GB and an H200 for scoring and retrieval, and one
  T4 16 GB for cross-encoder training. The exact nearest-neighbour search holds 1,024 queries × one country's pool
  in fp16 at a time (about 13 GB for the India pool); on a smaller GPU set `RETR_QCHUNK=256`.
- Internet on first use only: `transformers` downloads the two public base models (`BAAI/bge-reranker-v2-m3`,
  `intfloat/multilingual-e5-small`) from the Hugging Face Hub into its cache. No other network access. After the
  first download, `HF_HUB_OFFLINE=1` works.

Folder layout. Unzip, then put the contest data next to `code/`, unchanged:

```
<run folder>/                         run every command from here
  code/business_entity_resolution/    this package
  student_resource/dataset/train/     train_source{1,2,3}.tsv, train_ground_truth.tsv
  student_resource/dataset/test/      test_source{1,2,3}.tsv
  student_resource/utils/             validate_submission.py (optional; used by the last step if present)
```

Create the virtual environment and install the pinned packages.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

`torch` on macOS has no CUDA. The GPU steps fall back to Apple MPS or CPU, which is too slow for the ~22 million
cross-encoder pairs; run the GPU steps on a Linux CUDA machine (section 4) or use `--models shipped`.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r code/business_entity_resolution/requirements.txt
```

Pick the `--index-url` that matches the installed CUDA driver (`cu126`, `cu128`, …).

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r code\business_entity_resolution\requirements.txt
```

Not run on Windows by us. The code uses spawn-based multiprocessing and guards the one POSIX-only import
(`resource`), but no step has been executed there. WSL2 with Ubuntu and the Linux commands is the closer match to
what was run.

## 3. Run

One command runs every step in order. Output of each step is printed and appended to `logs/<step>.log`.

macOS / Linux:

```bash
python code/business_entity_resolution/src/reproduce.py
```

Windows (PowerShell):

```powershell
python code\business_entity_resolution\src\reproduce.py
```

Options (same on every OS):

| Option | Effect |
|---|---|
| `--list` | print the steps and exit |
| `--models shipped` | copy the trained models from `models/` into `work/` and skip every step marked `train` |
| `--from STEP` / `--to STEP` | run a contiguous range, e.g. resume after a failure |
| `--only STEP1,STEP2` | run only these steps |

Two ways to run:

- `--models train` (default): trains every model from the training data. This is how the submission was built.
- `--models shipped`: inference on test with the models the submission used. Runs the `data` steps only.
  Step 23 (drace_distilled) needs the distilled e5 from step 22 (distill). distill is a train step, so shipped
  mode skips it, and its inputs (DEV_REST_REST fold-1 pairs and c0 scores) also come only from train steps.
  e5_distil is not in the zip, so a shipped run stops at step 23. --models train is the complete pipeline.

Run the CPU steps on one machine and the GPU steps on another by copying `work/` between them, for example:

```bash
python code/business_entity_resolution/src/reproduce.py --to ce_pairs_test
```

then, on the GPU machine with the same `work/`:

```bash
python code/business_entity_resolution/src/reproduce.py --from ce_train
```

On Windows use the same arguments with `code\business_entity_resolution\src\reproduce.py`.

## 4. Steps

`kind`: `data` steps run in both modes; `train` steps are skipped with `--models shipped`. Times are measured on
the machines named in section 2 unless marked "est.". "A100" means one NVIDIA A100 80 GB.

| # | Step | Kind | Device | What it does | Time |
|---|---|---|---|---|---|
| 1 | `ingest` | data | CPU | read the 7 TSVs into parquet; S1 train/validation split (seed 42, strata country × singleton, 20% validation) and work slices | 1.2 min |
| 2 | `normalize` | data | CPU | mine abbreviation maps (train pool, test pool); normalize; learn the transliteration dictionary from train pairs | ~10 min (est.) |
| 3 | `index_train` | train | CPU | stage-1 inverted index of the train pool (S2 + S3) | 13.4 min |
| 4 | `index_test` | data | CPU | stage-1 inverted index of the test pool | 10.1 min |
| 5 | `stage1` | train | CPU | logistic model over per-key-family scores | 0.6 min |
| 6 | `stage2` | train | CPU | LightGBM re-ranker on 35 features; cut (rank ≤ 5, or p ≥ 0.05 up to rank 60) | 4 min |
| 7 | `candidates_train` | train | CPU | candidate lists and 68 pair features for MATCH_TRAIN, MATCH_VALID, DEV_REST_180K, DEV_REST_REST | ~2.5 h (est.) |
| 8 | `candidates_test` | data | CPU | candidate lists (10,613,056 pairs) and features for test | 2.35 h + 13 min |
| 9 | `ce_pairs_train` | train | CPU | "name \| address" text pairs for the cross-encoder; DEV_REST_REST fold-1 subset | ~10 min (est.) |
| 10 | `ce_pairs_test` | data | CPU | same for test | ~5 min (est.) |
| 11 | `ce_train` | train | GPU | `c0`: the `ce_rec` LoRA adapter continued 1 epoch on DEV_REST_REST fold 0 (3,285,477 pairs, lr 5e-5, batch 32, fp16) | 6.5 h on one T4 |
| 12 | `ce_score_train` | train | GPU | `c0` p for MATCH_VALID (stops if AUC < 0.99), MATCH_TRAIN, DEV_REST_REST fold 1 (3,459,853 pairs) | fold 1: 60 min on one L4; others not recorded |
| 13 | `ce_score_test` | data | GPU | `c0` p for the 10.6M test candidate pairs | 49 min on A100 |
| 14 | `matcher_s9` | train | CPU | LightGBM `full_c0` (68 features + `c0` p). Its test predictions are the seeds of copy expansion and the groups of step 18 | 4 min |
| 15 | `predict_s9` | data | CPU | `full_c0` on test → `work/outputs/full_c0/` | ~3 min |
| 16 | `rev_train` | train | GPU | reverse-name retrieval: each name-only pool candidate's 20 nearest S1 names (e5, same country) → 7 features | not recorded |
| 17 | `rev_test` | data | GPU | same for test | not recorded |
| 18 | `grp_train` / `grp_test` | train / data | CPU | 5-fold first-pass decisions on training rows; 6 group-profile features (candidate vs. records already accepted for the same S1) | 15.2 min / 3.2 min |
| 19 | `matcher_s12` | train | CPU | LightGBM `full_c0_rgrp`, 85 features, 3,864,095 fit rows, best iteration 1,190 of 12,000 | 4.8 min |
| 20 | `predict_s12` | data | CPU | test predictions at threshold 0.70 (base of the submission-13 file) and 0.73 (base of the final file) | ~7 min |
| 21 | `drace_untrained` | data | GPU | e5 embeddings of 10.0M pool records and 7.5M queries; S1 top 5 and top 5 around every `full_c0` acceptance; new pairs scored by `c0` | 35 min + 8 min on A100 |
| 22 | `distill` | train | GPU | e5-small fine-tuned on 435,454 triplets from DEV_REST_REST fold 1: in-batch InfoNCE + MarginMSE against `c0` | 7 min on A100 |
| 23 | `drace_distilled` | data | GPU | distilled-e5 S1 top 10; copy expansion ranks 3-5; `c0` scores for 5.2M new pairs | ~13 min retrieval + ~12 min scoring on H200 (est. from the per-job logs) |
| 24 | `merge` | data | CPU | matcher pairs + DRACE pairs with `c0` p ≥ 0.995, one-owner; writes `output/`; official validator; sha1 | ~5 min |

Step 23 builds its pair sets in the order the submissions added them (submission 12: distilled ranks 1-2; 13: ranks
3-5; 14: copy-expansion ranks 3-5, excluding pool records the submission-13 file assigns; 17: ranks 6-7; 18: ranks
8-10). It computes the top 10 once and takes the rank bands from that file; the original runs retrieved top 2, 5, 7
and 10 in four separate jobs with the same model and exact search.

Main intermediate files (under `work/`):

| Path | Content |
|---|---|
| `records/`, `norm/`, `index/` | parsed records, normalized text, stage-1 postings |
| `candidates/<SET>/`, `match/features/<SET>/` | candidate lists and pair features per set |
| `ce/pairs_<SET>.parquet`, `ce/ce_rec_c0/<SET>.parquet` | cross-encoder inputs and scores |
| `retrieval/` | e5 top-k files, reverse-name and group features, `e5_distil/` |
| `match/runs/<run>/models/` | LightGBM matchers (`matcher.json` = features, calibration, decision rule) |
| `outputs/<run>/matching_results.tsv` | matcher-only test predictions |

## 5. Shipped models (`models/`)

| Folder | Model | Produced by |
|---|---|---|
| `stage1/` | stage-1 logistic model (JSON) | step `stage1` |
| `rerank/` | stage-2 LightGBM re-ranker + cut | step `stage2` |
| `ce_rec_adapter/` | `ce_rec`: LoRA r 16, α 32, dropout 0.05 on query/key/value + classifier head of `BAAI/bge-reranker-v2-m3`; 1 epoch on 971,466 DEV_REST_180K pairs | `src/ce/train_ce_rec.py` (teammate's script; see below) |
| `ce_rec_c0_adapter/` | `c0`: `ce_rec` continued on DEV_REST_REST fold 0; MATCH_VALID AUC 0.99896, logloss 0.0402 | step `ce_train` |
| `matcher/full_c0/` | LightGBM, 69 features (submission 9) | step `matcher_s9` |
| `matcher/full_c0_rgrp/` | LightGBM, 85 features, threshold 0.70 (step 20 writes the 0.73 copy) | step `matcher_s12` |

`ce_rec` is the starting point of step 11 in both modes and is always taken from `models/`. It was trained
with an earlier revision of `src/ce/train_ce_rec.py` (the log's setup line predates the dtype print) on one NVIDIA
RTX 4060 Laptop GPU (CUDA capability 8.9, per the teammate; the log does not name the card). The weights were bf16
with no autocast or GradScaler (log: 1.15 GB allocated for 571M parameters). Log
(`cross_encoder_lora/ce_rec_ep1_train.log`): 976,897 pairs, 971,466 train / 5,431 held-out (last 1,000 S1 groups),
60,717 steps of batch 16, held-out AUC 0.9986. Learning rate, seed and gradient accumulation are the script defaults
(2e-4, 20260926, 4), not verified against the run. `ce_rec` is not wired into `reproduce.py`: the script reads
`name_s1`/`addr_s1`/`name_pool`/`addr_pool`, while `src/ce/prepare_pairs.py` writes `text_a`/`text_b` (976,902 rows
for DEV_REST_180K, 5 more than the teammate's file). The serialization `"{name} | {addr}"` is the same on both sides.

The distilled retriever `e5_distil` (118M parameters) is not in the zip because of its size; step `distill` trains
it again.

## 6. What is expected to match exactly

- CPU steps use fixed seeds; LightGBM runs with `deterministic=true, force_row_wise=true`, which gives the same
  model for the same data, parameters and thread count. The thread count is `min(14, cores − 2)`; the submission
  used 6 (8-core machine). A different thread count can change the LightGBM trees slightly.
- GPU steps are not bit-deterministic. bf16 and fp16 matrix products differ slightly between GPU models and library versions,
  so a rerun of `ce_train`, `distill` or the retrievals can move scores near the 0.995 admission threshold and change
  a small number of pairs. `ce_train` runs fp16 + GradScaler on GPUs below CUDA capability 8 (the T4 run). With `--models shipped` only scoring and retrieval run on the GPU.
- Not done for this package: a clean end-to-end rerun from an empty folder. The final `merge` step was rerun from the
  existing intermediate files and reproduced `matching_results.tsv` with the sha1 above.

## 7. Code map

```
src/reproduce.py            the step list above; runs each command, writes logs/<step>.log
src/ber/                    stages 1-3: python -m ber.cli <command>
  ingest.py, splits.py        TSV → parquet, S1 split and work slices
  textnorm.py, translit.py, translit_dict.py, abbrev_mine.py   normalization
  keys.py, index_build.py, index_store.py, hashing.py          stage-1 key families and postings
  retrieve.py, stage1_model.py, rerank.py, run_candidates.py   stage 1 query, stage 2 re-ranker and cut
  match.py, exp.py            68 pair features, extra feature groups (cex_p, rev, nfreq, grp_full), LightGBM, decision
  write_outputs.py, validate_stream.py, evaluate.py, monitor.py, io_utils.py, cli.py
src/ce/                     cross-encoder
  prepare_pairs.py            pair files in feature-row order
  train_continue.py, train.py, common.py   LoRA training (c0), shared scoring helpers
  hf_job/score_job.py         c0 scoring (local mode: CE_LOCAL=work)
  hf_job/retrieve_job.py      e5 embeddings + per-country exact top-k (local mode: RETR_LOCAL=work)
  hf_job/distill_job.py       e5 distillation (local mode: D_LOCAL=work)
  hf_job/ann_search.py, hf_job/collect.py   FAISS variant (not used in the submission), Hub download of scores
src/pipeline/               stage-3 feature groups, matcher training, DRACE
  build_dev_rest.py, dev_rest_180k.py, dev_rest_full.py, split_rest_folds.py   extra training slices
  rev_prep.py, firstpass_oof.py, grp_feats.py   ownership features
  train_final_comp.py         final LightGBM training (fit 85% / early stop 5% / calibration 10% of S1)
  drace_test.py, seed5_test_pairs.py, distill_e5.py   DRACE pairs, merge, distillation triplets
configs/                    blocking.json (stage 1-2 settings), match.json (LightGBM and decision settings), paths.json
tests/                      unit tests for stages 1-3
```

The `hf_job/*.py` scripts also run as Hugging Face Jobs (how the submission's GPU work was run); the local modes
execute the same model code without the Hub.

## 8. Tests

macOS / Linux:

```bash
cd code/business_entity_resolution
PYTHONPATH=src python -m unittest discover -s tests
```

Windows (PowerShell):

```powershell
cd code\business_entity_resolution
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## 9. Validation data use

- Train S1 is split 80/20 (seed 42). The 20% (441,363 S1) is read only by scoring commands.
- Inside the 80%, each slice has one role: stage-1 tuning (DEV_A, DEV_B), stage-1 model (S1R_TRAIN), re-ranker (RR_TRAIN,
  RR_CAL), `ce_rec` (DEV_REST_180K), `c0` (DEV_REST_REST fold 0), matchers and e5 distillation (MATCH_TRAIN,
  MATCH_VALID, DEV_REST_REST fold 1). A matcher never trains on rows whose `c0` score is in-sample.
- S2 and S3 are never split. Test has no labels; test files are read as inputs only.

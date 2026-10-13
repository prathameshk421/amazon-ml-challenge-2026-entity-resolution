# Business Entity Resolution: Candidate Generation (Bucketing) — Implementation Plan v1.1

Status: reviewed twice (see `plan/PLAN_PROGRESS.md`). Owner: human team. Executor: a coding agent
on the friend's laptop. Scope: raw TSV → `candidate_pairs.tsv` + the candidate parquet the matcher
will use. The pairwise matcher (model, threshold, decision rules) is a separate, later plan.

Read order for the executor: §0 → §1 → §3 → §4 → the module being implemented (§5) → the phase in §7.

Normative references (port verbatim into the new code; prose in this file explains them):
- `plan/reference/translit_ref.py` → `src/ber/translit.py` + `src/ber/textnorm.py` (§5.2, §5.5.1)
- `plan/reference/keys_ref.py` → `src/ber/keys.py` (§5.5)
- `plan/reference/goldens.json` → expected outputs for unit tests (§9)
If prose and reference code disagree, the reference code wins; log the disagreement in PROGRESS.md.

---

## 0. Rules for the executor (non-negotiable)

R1. Implement exactly what this document specifies. If something is impossible or contradictory,
    stop that step, write it to `PROGRESS.md` under `## BLOCKERS`, and continue with the next
    independent step. Never invent a replacement design silently.
R2. Parameters marked TUNE change only through the experiment named next to them (§8) and its
    decision rule. Parameters marked FIXED change only with a human decision logged under
    `## Deviations`.
R3. Data and package rules (official problem statement): no external data, APIs, registries,
    geocoders, gazetteers, transliteration libraries or models; nothing downloaded at runtime.
    Allowed packages: Python stdlib, numpy, pandas, pyarrow, scipy, scikit-learn, rapidfuzz,
    jellyfish, lightgbm, tqdm, matplotlib. Nothing else (no pytest, pyyaml, numba, polars).
    Tests use stdlib `unittest`; configs are JSON.
R4. Never fit, tune, or choose thresholds on VAL or on test. Test files are read only to build the
    test-pool index, run inference, and write outputs. Unsupervised statistics on test inputs
    (document frequencies, the edge-token set G) are allowed and must be logged in
    `reports/test_usage_log.md` (what, why, file).
R5. Every number in any report comes from a JSON written by a script into `artifacts/metrics/`.
R6. Update `PROGRESS.md` after every step (template §3.4). `git commit` after every phase.
R7. Determinism: all random choices use the seeds given here; every sort that can tie uses the
    tie-break in §3.3. Re-running a phase must produce byte-identical outputs.

---

## 1. Verified facts this plan relies on

Checked on the real data on 2026-09-25 (`plan/checks/data_facts.json`,
`plan/checks/romanize_quality.json`, `plan/reference/ref_quality.json`, `artifacts/metrics/diag_*.json`).
P1 re-asserts F1–F5 and F14 on the laptop's copy; any mismatch = stop (R1).

| # | Fact | Consequence |
|---|---|---|
| F1 | Every data row has exactly 4 tab-separated fields; no `\r`, no BOM; header `entity_id business_name business_address country` | Parse with `line.rstrip("\n").split("\t")`; assert 4 fields |
| F2 | `"` occurs inside fields | Never parse raw files with a quoting CSV reader (pandas default) |
| F3 | IDs match `^S[123]-\d+$`; no leading zeros; number ≤ 999,999,995 | Store (src:int8, num:uint32); `f"S{src}-{num}"` round-trips exactly (asserted) |
| F4 | S2 and S3 reuse numbers (26,801 train / 25,008 test) | A pool record is keyed by (src, num), never num alone |
| F5 | Rows: train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316 | Pools: train 10,320,219, test 9,969,589 |
| F6 | No empty names/countries. Empty pool addresses: train 344,883; test 265,506 | Name channels must carry these |
| F7 | Literal `<NULL>` inside pool addresses (35–44K per file); never in names | Removed in normalization |
| F8 | `<CITY_NAME>` (+1 `<CIT_NAME>`) placeholder in names (S1 and pool) | Becomes token `cityname` |
| F9 | Countries: train {US, India}; test {US, India, France}; same spelling across files | Open-set string; soft feature only |
| F10 | Non-ASCII letters: 9 Indic scripts, accented Latin, `º`; S1 is all Latin; no Arabic/CJK/Cyrillic | Romanize Indic; NFKD for the rest |
| F11 | 752,869 train-pool names contain Indic letters; the old normalizer erased them all | Normalization v2 is the biggest recall fix |
| F12 | Romanized Indic names are English words spelled phonetically; on 30K true pairs the reference romanizer gives: ≥1 identical token 79.7%, ≥1 identical skeleton 98.2% | Skeleton families SK/SP + learned dictionary |
| F13 | ZWNJ/ZWJ occur inside Indic words | Delete category `Cf` characters before tokenizing |
| F14 | GT: one row per train S1; 123,247 singletons; 7,638,365 links; max 11; no pool record has 2+ owners | Recall well defined; singletons excluded from recall |
| F15 | Official validator strips only `\n`, compares header lower-cased, splits IDs on `,` | Write with `newline="\n"`, exact headers, no spaces |
| F16 | Old blocker: 98.72% of true pairs reachable; top-150 of best single score 92.5%; quota union 0.962 at 497 (before the normalization fix) | Quota union + learned re-ranker |

---

## 2. Architecture

```
raw TSV ─► M1 ingest ─► records parquet (raw text, ids, GT)                  per split (train, test)
              │
              ▼
          M2 textnorm v2: NFC → placeholders → drop Cf → Indic romanization → NFKD → fold →
              │           casefold → & / apostrophes → punctuation → (addr) letter|digit split →
              │           abbreviation maps → (Indic fields) M3 learned dictionary
              ▼
          norm parquet + pool string store (mmap)
              │
              ▼
          M4 key families EN NT NG NP PH SK SP NC AT AB  (same code for pool and query)
              │
              ▼
          M5 index per split: family → sorted uint64 key hashes + CSR postings (.npy, mmap)
              │
S1 query ─────┼─► M6 stage-1: gather postings (df caps) → family scores → 6 composite channels →
              │      per-channel quota top-q → union  (≈500–1,000 candidates)
              ▼
          M7 stage-2: 33 cheap features → LightGBM p → adaptive cut (K_min / p_min / K_max)
              │
              ▼
          candidates parquet (s1_idx, pool_idx, rank, p_rerank) ─► M9 writer ─► candidate_pairs.tsv
              │
              └─► later plan: pairwise matcher → matching_results.tsv
```

### 2.1 Key decisions and trade-offs

| Decision | Chosen | Alternative | Why | Accepted cost |
|---|---|---|---|---|
| Bucket model | Many small buckets per record (inverted index over tokens, grams, pairs, phonetic codes, address tokens) | One composite key per record (the video's picture) | A single typo in a composite key loses the match; multi-key union reaches 98.7% | Unions of 25K–50K per query → ranking needed |
| Fusing buckets | Per-channel quota union | One additive score + top-K | Same recall with 2.6×–4× fewer candidates (`diag_quota.json`) | A second stage is needed to shrink |
| Shrinking | LightGBM re-ranker + adaptive cut | Fixed top-K / hand formula | Hand formula: +0.001 at K=150 (`diag_rerank.json`) | A trained model inside blocking (documented; LOCO-checked) |
| `candidate_pairs.tsv` | Stage-2 output | Stage-1 output | Spec: "the final candidate list just before the ML model scores them" | Matcher recall ceiling = stage-2 recall |
| Non-Latin text | Romanize inside normalization (stdlib `unicodedata` + fixed tables) + train-learned dictionary | Keep native script / external transliterator | Old approach erased 17% of Indian pool names; external tools banned | Approximate romanization (skeleton families compensate) |
| Key storage | 64-bit blake2b hash arrays, `.npy`, mmap | Python dicts | Shared by all spawn workers through the OS page cache; portable to Windows | ~1e-6 collision probability (accepted) |
| Frequent keys | Dropped above a df cap (fraction of pool size) | Keep all | Bounds per-query cost | Common-word names lose single-token keys → pair families rescue |
| Country | Stage-2 feature (same/different/unknown) | Partition index by country | France unseen; "country is an open set" | ~2× larger unions |
| Pool duplicates | Kept | Deduplicate | Only ~44K groups; both copies can be true matches | Minor ties |

### 2.2 Explicitly out of scope for v1
- BK-tree / Levenshtein automaton (cost grows with pool size; `blocking_report.md`).
- Postal-only bucket (postal tokens already are address tokens; 0 exclusive recoveries).
- Embedding / ANN family (allowlisted models are English-only; romanization first). Later decision.
- Reverse pool→S1 blocking: optional experiment X3 with a trigger (§8).

---

## 3. Environment, layout, conventions

### 3.1 Laptop setup (human, once)
1. Windows: `wsl --install -d Ubuntu-24.04`.
2. `C:\Users\<user>\.wslconfig`:
   ```
   [wsl2]
   memory=20GB
   processors=<logical CPUs − 2>
   swap=8GB
   ```
   then `wsl --shutdown`. (Default WSL memory = 50% of RAM = 12 GB: too small.)
3. Ubuntu: `sudo apt update && sudo apt install -y python3.12 python3.12-venv git unzip`.
4. Project lives INSIDE the Linux filesystem: `~/ber/` (never `/mnt/c/...`; 5–20× slower I/O).
5. Code transfer: on the Mac, a human runs `git bundle create ber.bundle --all` (or zips the repo
   without `work/`, `.venv/`, `student_resource/`); on the laptop `git clone ber.bundle ~/ber`.
   Copy `student_resource/` (the dataset) to `~/ber/student_resource/`.
6. `python3.12 -m venv ~/ber/.venv && source ~/ber/.venv/bin/activate`
7. `pip install numpy==2.5.3 pandas==3.0.6 pyarrow==25.0.1 rapidfuzz==3.14.6 jellyfish==1.2.1 scikit-learn==1.9.1 scipy==1.18.1 lightgbm==4.7.0 tqdm==4.70.1 matplotlib==3.11.2`
   If a pin has no wheel for Python 3.12, install the newest version that does; then
   `pip freeze > code/business_entity_resolution/requirements.txt`. This is the only allowed deviation.
8. AC power; Windows power mode "Best performance"; check `df -h ~` shows ≥ 40 GB free.

Mac use: writing code and running unit tests on tiny inputs only. Every artifact that feeds a result
is produced on the laptop (one Python/Unicode version for all goldens and indexes).

### 3.2 Repository layout (new code; old `src/` is reference only)
```
code/business_entity_resolution/
  README.md                 reproduce end to end (data → blocking → candidates)
  requirements.txt
  PROGRESS.md               executor progress (§3.4)
  configs/paths.json        {"data_root": "student_resource/dataset", "work_root": "work", "metrics_root": "artifacts/metrics"}
  configs/blocking.json     every FIXED/TUNE parameter (§3.5); the single source of truth
  src/ber/
    __init__.py  io_utils.py  ingest.py  translit.py  textnorm.py  translit_dict.py  splits.py
    hashing.py  keys.py  index_build.py  index_store.py  retrieve.py  features.py  rerank.py
    evaluate.py  run_candidates.py  write_outputs.py  validate_stream.py  cli.py
  tests/  test_textnorm.py test_translit.py test_keys.py test_topq.py test_metrics.py test_outputs.py
```
All paths in `paths.json` are relative to the repository root and resolved with `pathlib`.
`work/` and `artifacts/` are git-ignored except `artifacts/metrics/*.json` (committed).

### 3.3 Conventions
- Reading text: `open(p, encoding="utf-8", newline="")`. Writing text: `open(p, "w", encoding="utf-8", newline="\n")`.
- Types: s1_idx int32; pool_idx uint32 (row in the split's pool table); src int8 ∈ {2, 3}.
- Pool row order: S2 rows in file order, then S3 rows in file order. S1 order = file order.
- Tie-break ("top-q by score"): order by (score desc, pool_idx asc) = `np.lexsort((pool_idx, -score))`.
  Top-q of n > q items: `v = np.partition(score, n - q)[n - q]`; take all with score > v; then add
  items with score == v in ascending pool_idx until q are taken.
- Scores computed in float64; stored as float32.
- Parallelism: `multiprocessing.get_context("spawn").Pool(W, initializer=init_worker)`;
  `init_worker` opens indexes and stores with `np.load(..., mmap_mode="r")`, sets `OMP_NUM_THREADS=1`.
  Never rely on fork. `W = min(cfg.max_workers, os.cpu_count() − 2)`, never < 1.
  LightGBM inside workers: `num_threads=1`. LightGBM training (main process): `num_threads=W`.
- Long commands: log to `work/logs/<command>_<YYYYmmdd-HHMMSS>.log`; outputs as `part-XXXXX.parquet`
  plus `part-XXXXX._SUCCESS`; a part with `_SUCCESS` is skipped on re-run (resumable).
- Atomic writes: write `<file>.tmp`, then `os.replace`.
- Every metrics JSON contains: `n_queries`, `n_true_pairs`, `config` (full), `config_hash`
  (sha1 of the canonical JSON of `blocking.json`), `norm_version`, `git_commit`, `created_at`.

### 3.4 `PROGRESS.md` template
```
# PROGRESS
LAST_UPDATE: <date -Is>
CURRENT: <phase>.<step> <one line>
## Phases
P0 setup + tests: TODO|DOING|DONE
P1 ingest + splits:
P2 textnorm + dictionary:
P3 train index:
P4 stage-1 experiments (E1–E6):
P5 stage-2 re-ranker:
P6 full runs + outputs:
## Results (one line each, citing a metrics json)
## BLOCKERS
## Deviations (what, why, approved-by)
```

### 3.5 `configs/blocking.json` (initial content; TUNE values are defaults until tuned)
```json
{
  "norm_version": 2,
  "max_workers": 14,
  "families": ["EN", "NT", "NG", "NP", "PH", "SK", "SP", "NC", "AT", "AB"],
  "build_df_max_frac": 0.005,
  "cap_frac": {"EN": 0.002, "NT": 0.002, "NG": 0.0008, "NP": 0.002, "PH": 0.002,
               "SK": 0.002, "SP": 0.002, "NC": 0.002, "AT": 0.002, "AB": 0.002},
  "max_postings_per_query": 300000,
  "composites": {
    "BOTH": {"EN": 10, "NT": 1, "NG": 1, "PH": 0.2, "AT": 1},
    "NAME": {"EN": 10, "NT": 1, "NG": 1, "PH": 0.2},
    "ADDR": {"AT": 1, "AB": 1},
    "PAIR": {"NP": 1, "SP": 1},
    "PHON": {"SK": 1, "PH": 1},
    "CAT":  {"NC": 1}
  },
  "quota": {"BOTH": 300, "NAME": 100, "ADDR": 100, "PAIR": 100, "PHON": 50, "CAT": 20},
  "chain_rule": {"enabled": true, "chain_df": 50, "mult": 3, "channels": ["ADDR", "PAIR"]},
  "edge_tokens": {"min_frac": 0.0005, "min_edge_share": 0.80, "nc_min_len": 6, "nc_pool_token_min_len": 8},
  "rerank": {
    "sample": {"top_neg": 100, "rand_neg": 100},
    "lgb": {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "max_bin": 255, "seed": 42, "deterministic": true, "force_row_wise": true,
            "num_boost_round": 2000, "early_stopping_rounds": 50, "holdout_frac": 0.1},
    "cut": {"K_min": 10, "K_max": 100, "p_min": 0.005},
    "cut_grid": {"K_min": [5, 10, 20], "K_max": [60, 80, 100, 150],
                 "p_min": [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]},
    "mean_size_budget": 100
  },
  "tuned_by": {}
}
```
Composite rule: a composite skips terms of dropped families; a composite with no remaining family
gets quota 0. `tuned_by` records `"<param path>": "<experiment id>"` for every tuned value.

---

## 4. Data contracts (everything under `work/` unless stated)

Parquet via pyarrow, compression zstd.

| File | Columns (dtype) / content |
|---|---|
| `records/{split}_s1.parquet` | s1_idx int32, entity_id string, name_raw string, addr_raw string, country_raw string |
| `records/{split}_pool.parquet` | pool_idx uint32, src int8, num uint32, name_raw, addr_raw, country_raw |
| `records/train_gt.parquet` | s1_idx int32, pool_idx uint32 (one row per link; 7,638,365 rows) |
| `records/train_s1_nmatch.npy` | int16[n_s1]; 0 = singleton |
| `norm/{split}_s1.parquet`, `norm/{split}_pool.parquet` | idx, name_norm, addr_norm (string), name_indic, addr_indic (bool) |
| `norm/{split}_pool_store/` | `name.bin` + `name.off.npy` (int64, n+1), `addr.bin` + `addr.off.npy`, `country.npy` (int16), `countries.json` (list of strings, index = code), `src.npy` (int8), `num.npy` (uint32) |
| `translit/dict_v1.json` | `{"name": {rom: latin}, "addr": {rom: latin}, "meta": {...}}` |
| `splits/query_sets.json` | `{"fit", "val", "DEV_A", "DEV_B", "RR_TRAIN", "RR_CAL", "MATCH_POOL"}`: lists of train s1_idx; plus `"sizes"`, `"seeds"` |
| `index/{split}/{F}.keys.npy` / `.indptr.npy` / `.post.npy` | uint64 sorted unique / int64 CSR / uint32 pool_idx ascending per key |
| `index/{split}/meta.json` | n_pool, per-family n_keys + n_postings + df quantiles (50/90/99/max), G (sorted list), build caps, versions, wall time, peak RSS |
| `rerank/rr_train.parquet` | query_s1_idx, pool_idx, label int8, weight float32, 33 feature columns (§5.8) |
| `rerank/rr_cal.parquet` | same, unsampled, weight 1 |
| `rerank/model_v1.txt`, `rerank/cut_v1.json` | LightGBM model (best iteration); `{"K_min","K_max","p_min","stage1_config_hash"}` |
| `candidates/{SET}/part-XXXXX.parquet` | s1_idx int32, pool_idx uint32, rank int16 (1-based), p_rerank float32 |
| `outputs/candidate_pairs.tsv`, `outputs/matching_empty.tsv` | test outputs (§5.10) |

---

## 5. Module specifications

### 5.1 M1 ingest (`ingest.py`)
- Read each TSV line by line (F1): assert header, assert 4 fields per row.
- IDs: regex `^S([123])-(\d+)$`; assert prefix matches the file; num → uint32; assert
  `f"S{src}-{num}" == entity_id` on every row; assert uniqueness per file; assert row counts = F5.
- GT: map every `source1_entity_id` to s1_idx and every matched ID to pool_idx; all must exist;
  empty list = singleton; assert exactly one GT row per train S1 and total links = 7,638,365 (F14).
- Write the §4 `records/` files. Stream in 1M-row chunks; peak RSS < 4 GB.

### 5.2 M2 text normalization v2 (`translit.py`, `textnorm.py`) — FIXED
`NORM_VERSION = 2`. `normalize_field(text, kind, tdict) -> (norm, had_indic)`, kind ∈ {"name", "addr"}.
Normative code: `plan/reference/translit_ref.py` (`romanize`, `normalize_field`, `skeleton`).
Order of steps (as implemented there):
1. NFC. 2. `<NULL>` → space; any other `<[A-Z_]+>` → ` cityname `. 3. delete category `Cf`.
4. Romanize Indic (U+0900–U+0D7F) per §5.2.1. 5. NFKD, delete category `Mn`.
6. Fold `œ→oe æ→ae ß→ss ø→o ł→l đ→d ð→d þ→th ı→i ħ→h` (upper-case forms too). 7. casefold.
8. `&` → ` and `; apostrophes `' ’ ‘ ʼ \`` deleted. 9. non-`isalnum` → space.
10. addr only: space at letter↔digit boundaries (`enclave236` → `enclave 236`).
11. token maps: NAME_ABBREV / ADDR_ABBREV (listed in the reference; train-derived v1 lists);
    then, only if `had_indic`, the learned dictionary `tdict[kind]`.
12. join with single spaces ("" allowed).

#### 5.2.1 Romanizer rules (summary of the reference)
- Class from `unicodedata.name(c)`: independent vowel, consonant, dead consonant (CHILLU X, KHANDA TA),
  vowel sign, virama, anusvara, candrabindu/visarga, digit; other SIGNs and LENGTH MARKs ignored;
  anything else in the Indic range (danda, etc.) = word boundary.
- Consonant and vowel tables: in the reference (`CONS`, `VOW`). Unknown consonant name X → lowercase X
  minus a trailing "a". CHILLU X → `CONS[X + "A"]`.
- Inherent "a": owed after a consonant; written before a following consonant/independent vowel/
  anusvara/candrabindu/visarga; replaced by a vowel sign; cancelled by virama; dropped at a word
  boundary or end of text (schwa deletion).
- Anusvara → "m" before p/b/m, else "n".
- RRA clusters (Malayalam/Tamil): RRA+VIRAMA+RRA → "t" (the first "r" removed); NA/NNNA+VIRAMA+RRA → "n"+"t".
- Goldens: `plan/reference/goldens.json` (74 cases: 68 names over all 9 scripts + Latin/France edge cases,
  6 addresses), e.g. `राम मार्केटिंग प्राइवेट लिमिटेड` → `ram marketing praivet limited`,
  `ಸನ್‌ರೈಸ್ ಕನ್ಸಲ್ಟಿಂಗ್` → `sanrais kansalting`, `സിൽവർ` → `silvar`, `Œuvre d'Art` → `oeuvre dart`,
  `1945 214TH LN, BUCKEYE, AZ` → `1945 214 th lane buckeye az`.
- Measured quality (reference, 30K fit true pairs with Indic pool names): ≥1 identical token 79.7%,
  ≥1 identical skeleton 98.2%, S1 tokens with an identical romanized token 31.0%, with an identical
  skeleton 77.1% (`plan/reference/ref_quality.json`).

### 5.3 M3 learned transliteration dictionary (`translit_dict.py`) — FIXED rule
- Pairs: true (S1, pool) pairs with S1 ∈ fit \ (DEV_A ∪ DEV_B) and pool field `had_indic`; names and
  addresses handled separately.
- Texts: S1 = `normalize_field(..., tdict={})`; pool = `normalize_field(..., tdict={})`.
- For each pair and each pool token r (len ≥ 2) NOT present verbatim among the S1 tokens: c(r) += 1;
  l* = S1 token with maximal `rapidfuzz.distance.JaroWinkler.normalized_similarity(r, l)` (ties: first in
  S1 order); if similarity ≥ 0.60: c(r, l*) += 1.
- Keep r → l* when c(r, l*) ≥ 5 and c(r, l*)/c(r) ≥ 0.80.
- Write `translit/dict_v1.json` (`meta`: n pairs, n entries per kind, 50 most frequent entries).
- Metric `artifacts/metrics/translit_dict.json`: on DEV_A true pairs with Indic pool names, the share of
  pairs whose pool name shares ≥ 1 token with the S1 name, without vs with the dictionary.
  Expected from the positional-mismatch list (`ref_quality.json`): praivet→private (15,037),
  limitet→limited, piraivet→private, etc. Gate: the share must increase.
- Then write `norm/` for both splits WITH the dictionary (test uses the train-learned dictionary).

### 5.4 Query sets (`splits.py`) — FIXED
1. `ids = sorted(train S1 entity_id)`; strata = `country + "|" + ("single" if nmatch == 0 else "multi")`.
2. `fit, val = sklearn.model_selection.train_test_split(ids, test_size=0.2, stratify=strata, random_state=42)`.
3. `perm = numpy.random.default_rng(20260925).permutation(sorted(fit))`;
   DEV_A = perm[0:10000], DEV_B = perm[10000:20000], RR_TRAIN = perm[20000:70000],
   RR_CAL = perm[70000:90000], MATCH_POOL = perm[90000:].
4. Save s1_idx lists; assert pairwise disjoint and union = all train S1 (2,206,821).

Use: DEV_A tunes stage-1; DEV_B confirms once; RR_TRAIN trains stage-2; RR_CAL picks the cut;
MATCH_POOL = later matcher training entities; VAL = reported once at the end.
Set → split mapping for retrieval: DEV_A, DEV_B, RR_TRAIN, RR_CAL, MATCH_POOL, VAL → query = train S1,
index = train pool; TEST → query = test S1, index = test pool.

### 5.5 M4 key families (`keys.py`) — FIXED
Normative code: `plan/reference/keys_ref.py`. Tokens = `name_norm.split()`; tok2 = tokens with
len ≥ 2; first-8 = the first 8 distinct tok2.

| Family | Keys of one record | Query side vs pool side |
|---|---|---|
| EN exact name | `[name_norm]` (if non-empty) | same |
| NT name token | distinct tok2 | same |
| NG name 3-gram | distinct `s[i:i+3]`, `s = " " + name_norm + " "` | same |
| NP name token pair | unordered pairs of first-8 (`min|max`) | same |
| PH metaphone | distinct non-empty `jellyfish.metaphone(t)` for tok2 with len ≥ 3 | same |
| SK skeleton | distinct `skeleton(t)` for tok2 with skeleton length ≥ 2 | same |
| SP skeleton pair | unordered pairs of the distinct skeletons (length ≥ 2) of first-8 | same |
| NC concatenation | `concat_variants(tokens, G)` (below) | pool ALSO adds every single token of length ≥ 8 |
| AT address token | distinct tokens of addr_norm | same |
| AB address bigram | distinct adjacent pairs `a + " " + b` | same |

- Hash: `key64(F, k) = little-endian int of blake2b((F + "\x1f" + k).utf8, digest_size=8)`.
- `concat_variants(tokens, G)`: candidate token lists = [all]; + [all minus last] if last ∈ G, and
  + [all minus last two] if the last two are both ∈ G; + [all minus first] if first ∈ G, and
  + [all minus first and last] if first ∈ G and last ∈ G. Each is concatenated without spaces and kept
  if its length ≥ 6 (dedup, this order). Fixed variants on BOTH sides make matching symmetric
  (e.g. `happy nails llc` and `happynails com` share `happynails`; goldens in `keys_check.json`).
- Edge-token set G (per split, on pool names, at index build): c_all(t) = #names containing t;
  c_edge(t) = #names where t is the first token or one of the last two. G = {t : c_all(t) ≥
  0.0005 · n_pool and c_edge(t)/c_all(t) ≥ 0.80}. Train-pool value is recorded in
  `plan/reference/keys_check.json` for comparison.

#### 5.5.1 Skeleton (own phonetic class code)
ASCII token → (1) replace in order `x→ks, ph→f, ch→s, sh→s, q→k`; (2) map p b f v w → P; t d → T;
k g c → K; s z j → S; l → L; r → R; m → M; n → N; digits kept; a e i o u y h dropped;
(3) collapse runs. Non-ASCII token → "". Goldens: developers/tevalappars → TPLPRS; silver/chilvar/silvar →
SLPR; private/praivet/praibhet → PRPT; limited/limitet/limatid → LMT; consulting/kansalting → KNSLTNK.

### 5.6 M5 index build (`index_build.py`, `index_store.py`)
Per split s, per family F:
1. Workers take 250,000-row chunks of `norm/{s}_pool.parquet`, generate pool-side keys, hash, dedupe
   per record, and partition (key64, pool_idx) pairs into 64 shards by `key64 >> 58`, writing
   `work/tmp/{s}/{F}/shard-SS/part-CCCCC.npz`.
2. One shard at a time: load, `np.lexsort((pool_idx, key64))`, unique keys + df + indptr; drop keys
   with df > `build_df_max_frac · n_pool`.
3. Concatenate shards in shard order (keys remain globally sorted). Save `{F}.keys/indptr/post.npy`.
4. Verify: keys strictly increasing; indptr monotone with last = n_postings; every posting < n_pool;
   postings ascending within each key. Then delete `work/tmp/{s}/{F}/`.
5. G is computed before step 1 (needed by NC) and saved to `meta.json` with all stats.
Lookup: `IndexStore(split)` opens families with `mmap_mode="r"`; `lookup(F, keys_u64)` → list of
(postings view, df) via `np.searchsorted`.
Sanity (`index-sanity`): 1,000 random pool records (`default_rng(1)`); each must be in its own EN
postings (non-empty names) and in the postings of every NT key it generated whose df ≤ build limit.
Expected size ≤ 3 GB per split; time 10–20 min per split (estimate).

### 5.7 M6 stage-1 retrieval (`retrieve.py`)
Input: query (name_norm, addr_norm), IndexStore, config. Output for the union U (pool_idx ascending):
family scores S[F] (float64 arrays), query self-weights Q[F], composite scores, per-channel ranks,
`sel` bitmask of channels that selected the item, `d_EN`.
1. Query keys per family (query side). Look up. d_EN = df of the EN key regardless of cap (0 if absent).
   A key is used if df ≤ `floor(cap_frac[F] · n_pool)`. Weight: EN → 1.0; others → ln((n_pool+1)/(df+1)).
   Q[F] = sum of weights of used keys of F.
2. If the total number of postings of used keys > `max_postings_per_query`: drop used keys in
   descending df order (ties: family order in `families`, then key64 ascending) until ≤ limit; count
   dropped keys in the query log.
3. `u, inv = np.unique(concat_postings, return_inverse=True)`; S[F] = bincount over F's postings.
4. Composite scores per `composites` (dropped families skipped).
5. Quotas: effective quota per channel = `quota[C]`, multiplied by `chain_rule.mult` for channels in
   `chain_rule.channels` when `chain_rule.enabled` and d_EN ≥ `chain_df`. Only items with composite
   score > 0 are eligible. Selected = union of per-channel top-q (tie rule §3.3).
6. Output ordered by BOTH desc, pool_idx asc.
Evaluation mode (for E2–E5): for each union item compute its rank in every channel (1-based, among
eligible items; ineligible = ∞). An item is selected by a quota config iff ∃ channel c with
rank_c ≤ q_c. Evaluate many quota configs at once by vectorized comparison on the items whose
min rank ≤ max grid quota. Returns per config: selected count, true hits.
Performance target: median ≤ 8 ms/query single-thread; log p50/p99 of ms and |U|.

### 5.8 M7 stage-2 re-ranker (`features.py`, `rerank.py`)
Features (float32, in this order; the backticked names are the parquet column names):
1–10 `s_EN … s_AB`: S[F] for the 10 families (0 if dropped).
11–14 `cov_NT, cov_NG, cov_NP, cov_AT`: S[F]/Q[F] (0 if Q[F] = 0).
15–17 `rk_BOTH, rk_NAME, rk_ADDR`: rank in U by that composite (1-based, ties per §3.3, capped at 10,000;
      10,000 if ineligible).
18 `log_union` = ln(1+|U|). 19 `log_dEN` = ln(1+d_EN).
20–23 names (query name_norm vs pool name_norm), all ÷100 where rapidfuzz returns 0–100:
      `fz_tset` = fuzz.token_set_ratio, `fz_ratio` = fuzz.ratio, `fz_partial` = fuzz.partial_ratio,
      `jw_concat` = JaroWinkler.normalized_similarity on space-free strings (already 0–1).
24–25 addresses: `fa_tset` = fuzz.token_set_ratio/100, `fa_tsort` = fuzz.token_sort_ratio/100 (0 if either empty).
26 `addr_empty`: 0 both non-empty, 1 pool empty, 2 query empty, 3 both empty.
27 `postal`: postal token = all-digit token of length 4–6 in addr_norm; 2 = both have and share one;
   0 = both have, none shared; 1 = otherwise.
28 `country`: 1 same string, 0 different, 2 either empty.
29 `src` (2/3). 30 `pool_indic` (0/1). 31 `q_ntok`, 32 `p_ntok` (token counts). 33 `len_ratio` =
   len(pool name_norm)/max(1, len(query name_norm)).
All rapidfuzz via `process.cdist([query], candidates, scorer=..., workers=1)`.

Training data (`rerank-build`, requires the final stage-1 config; stores its `config_hash`):
- RR_TRAIN (50,000 queries): run stage-1; label = candidate ∈ truth. Among the SELECTED candidates:
  keep all positives; negatives: all negatives ranked in the top-100 by BOTH among selected (weight 1)
  + a uniform sample (rng seed = 1000 + s1_idx) of min(100, n_rest) of the remaining selected
  negatives, weight = n_rest / n_sampled. Features computed only for kept rows.
- RR_CAL (20,000 queries): all selected candidates, weight 1. `rerank-build` also writes the stage-1
  pair/entity recall of RR_CAL to `artifacts/metrics/rr_cal_stage1.json` (used by the cut rule).
Model (`rerank-train`): LightGBM with `rerank.lgb` params; holdout = 10% of RR_TRAIN queries via
`GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)` grouped by query; early stopping on
`binary_logloss`; save best iteration.
Cut (`rerank-cut`, on RR_CAL): rank candidates of a query by p desc (ties: BOTH desc, pool_idx asc);
keep rank r if `r ≤ K_min or (p ≥ p_min and r ≤ K_max)`. Evaluate all `cut_grid` combinations.
Decision: among configs with pair recall ≥ 0.99 × stage-1 pair recall (RR_CAL) and entity recall
≥ stage-1 entity recall − 0.002 and mean size ≤ `mean_size_budget`, choose the smallest mean size
(ties: higher pair recall). If none qualifies: highest pair recall with mean ≤ budget; log a deviation.
Report (`rerank-eval --set DEV_B`, later VAL): pair/entity recall, sizes, segments.
LOCO (`rerank-loco`): train on RR_TRAIN India-only → evaluate on DEV_B US-only (cut from RR_CAL
India-only), and US → India. Report the drop vs the all-country model on the same DEV_B slice.
Staleness guard: `run_candidates` refuses to run if `cut_v1.json.stage1_config_hash` ≠ current
stage-1 config hash (stage-1 params = everything in `blocking.json` except `rerank` and `tuned_by`).

### 5.9 M8 metrics (`evaluate.py`)
- Pair recall = Σ|cand ∩ truth| / Σ|truth|. Entity recall = share of non-singleton queries with ≥ 1
  hit. Full recall = share of non-singleton queries with all truth found.
- Reachability (stage-1 only) = pair recall of the union U before quotas.
- Exclusive pairs of family F = true pairs reachable only through keys of F.
- Sizes: mean, p50, p90, p99, max (singletons included). Reduction ratio = 1 − Σ|cand|/(n_queries·n_pool).
- Segments (pair recall and mean size): query chain size (count of the query's name_norm among the
  split's S1 name_norms) in {1, 2–9, ≥10}; query country; pool name class {indic, ascii, other}
  where indic = `name_indic`, ascii = raw name is ASCII, other = rest; pool address empty (raw).

### 5.10 M9 outputs (`write_outputs.py`, `validate_stream.py`)
- `candidate_pairs.tsv`: header `source1_entity_id\tcandidate_entity_ids`; one row per test S1 in
  `test_source1.tsv` order; IDs `f"S{src}-{num}"` in rank order joined by ","; empty list → nothing
  after the tab. `newline="\n"`, utf-8.
- `matching_empty.tsv`: header `source1_entity_id\tmatched_entity_ids`, one row per test S1, all empty
  (format placeholder for validation only; never submitted).
- `validate_stream.py` (mandatory): streams the file and checks every rule of the official validator
  (header, one row per required S1, no extra S1, no duplicate IDs in a row, only S2-/S3- prefixes,
  every ID exists in the test pool via a sorted uint64 array of (src<<32|num), no `\r`).
- Official validator (best effort, memory-heavy with the candidate file): from `student_resource/`:
  `python3 utils/validate_submission.py --matching ../work/outputs/matching_empty.tsv --candidate ../work/outputs/candidate_pairs.tsv --test-dir dataset/test`
  Run it with no other job running. If it is killed for memory, record that and rely on `validate_stream.py`.

---

## 6. Evaluation protocol
- Stage-1 tuned on DEV_A only (E1–E5). Each experiment writes `artifacts/metrics/s1_<exp>.json`.
- Stage-2 trained on RR_TRAIN; cut chosen on RR_CAL.
- DEV_B is used twice, each check once: (a) E6 in P4 checks the final stage-1 config: if DEV_B
  stage-1 pair recall < DEV_A − 0.005, switch to the next point of the E4 Pareto list (next larger mean
  size) and re-check once. (b) `rerank-eval --set DEV_B` in P5 checks stage-2: if DEV_B final pair
  recall < RR_CAL final pair recall − 0.005, use the next smaller p_min in the grid and re-check once.
  Any stage-1 change after P5 started invalidates stage-2 (staleness guard): rerun all of P5.
- VAL: full pipeline once at the end; report only.
- Reference points: old system pair recall 0.917 (top-150, exact +100) / 0.925 (exact +10), entity 0.986–0.989.

---

## 7. Runbook (repository root, venv active: `python -m ber.cli <command>`)

| Command | Arguments | Does |
|---|---|---|
| `ingest` | `--split {train,test}` | M1 |
| `splits` | none | §5.4 |
| `normalize` | `--split {train,test} [--no-dict]` | M2 (+ dictionary unless `--no-dict`) |
| `learn-dict` | none | M3 (needs `normalize --split train --no-dict` and `splits`) |
| `build-index` | `--split {train,test}` | M5 |
| `index-sanity` | `--split` | §5.6 sanity |
| `s1-eval` | `--set DEV_A --exp <id> [--families ...] [--cap F=x ...] [--quota-grid]` | M6 evaluation mode; writes `s1_<id>.json` |
| `rerank-build` | none | RR_TRAIN / RR_CAL parquet |
| `rerank-train`, `rerank-cut` | none | model, cut |
| `rerank-eval` | `--set {DEV_B,VAL}` | runs stage-1+2 for the set if `candidates/{SET}/` is missing, then writes final-candidate metrics |
| `rerank-loco` | none | LOCO |
| `candidates` | `--set {VAL,MATCH_POOL,TEST} [--limit N]` | M6 + M7 → `candidates/{SET}/` |
| `write-outputs` | none | `candidate_pairs.tsv`, `matching_empty.tsv`, streaming validation |

| Phase | Steps | Gate | Est. time (laptop, W=14) |
|---|---|---|---|
| P0 | §3.1 setup; implement + pass unit tests of §9 (goldens from `plan/reference/`) | all tests pass | 1–2 h human + coding |
| P1 | `ingest --split train`, `ingest --split test`, `splits` | §5.1/§5.4 asserts | ~10 min |
| P2 | `normalize --split train --no-dict`, `learn-dict`, `normalize --split train`, `normalize --split test` | goldens pass; dictionary gate (§5.3) | ~30 min |
| P3 | `build-index --split train`, `index-sanity --split train` | sanity 100% | ~20 min |
| P4 | E1–E6 (§8) | E1 gate | 2–4 h |
| P5 | `rerank-build`, `rerank-train`, `rerank-cut`, `rerank-eval --set DEV_B`, `rerank-loco` | §6 DEV_B rule | 1–2 h |
| P6 | `build-index --split test`, `index-sanity --split test`, `candidates --set VAL`, `rerank-eval --set VAL`, `candidates --set MATCH_POOL`, `candidates --set TEST`, `write-outputs` | streaming validator PASS | 1.5–3 h |

Times are estimates (Mac measurements scaled); the executor logs real ones.
Memory (20 GB WSL): shared mmap per split ≤ 3 GB index + ≈ 1 GB string store; per worker ≤ 400 MB
(log RSS); W = 14 → ≈ 10 GB. If the summed RSS exceeds 16 GB, reduce W (minimum 4) and log it.
Disk: `work/` ≈ 15–20 GB.

---

## 8. Experiments (P4, DEV_A) with decision rules

E1 Baseline replication: families {EN, NT, NG, PH, AT}, BOTH only, top-K for K ∈ {50, 150, 500, 2000},
   plus reachability. Gate: pair recall@150 ≥ 0.92. Lower → bug hunt; do not continue.
E2 Family value: from the E1 set, add one at a time NP, SK, SP, NC, AB (default caps and quotas).
   Keep F if (exclusive pairs of F ≥ 0.05% of DEV_A true pairs) OR (pair recall with default quotas
   improves ≥ 0.002). Then test removing PH with the same rule (keep PH if its removal loses either
   amount). Record kept families in `blocking.json.families`.
E3 Caps (kept families; order NT, NG, AT, NP, SP, AB, SK): try cap × {0.5, 1, 2.5} (≤ build limit).
   Take the larger cap only if pair recall (default quotas) gains ≥ 0.002 and median ms/query rises ≤ 30%;
   take the smaller cap if recall loses < 0.001 and median ms falls ≥ 15%. One pass per family, in order.
E4 Quota grid (one retrieval pass, evaluation mode): BOTH {150, 300, 500} × NAME {50, 100, 200} ×
   ADDR {50, 100, 200} × PAIR {50, 100, 200} × PHON {0, 50, 100} × CAT {0, 20} = 486 configs.
   Save the Pareto front (pair recall vs mean size). Decision: cheapest config with pair recall ≥ 0.970
   and mean ≤ 1,000; if none, highest pair recall with mean ≤ 1,000.
E5 Chain rule: chain_df ∈ {20, 50, 100} × mult ∈ {1, 2, 3} (mult 1 = off). Choose the best setting only
   if the chain ≥ 10 segment gains ≥ 0.01 pair recall and overall mean size grows ≤ 15%; else off.
E6 DEV_B confirmation (§6).
Optional after P6 works (only with the trigger):
X2 name × address cross keys (name token × last 3 address tokens) — trigger: chain ≥ 10 recall < 0.93 after E5.
X3 reverse blocking (pool record → top-3 S1, added to those S1 lists) — trigger: chain ≥ 10 recall < 0.93 after X2.
Write every chosen value into `blocking.json` with `tuned_by`.

---

## 9. Tests (stdlib `unittest`)
- test_translit / test_textnorm: every case in `plan/reference/goldens.json` (exact `out`, `had_indic`,
  `skeletons`); outputs ASCII for Indic inputs; idempotence `normalize(normalize(x)) == normalize(x)`
  on 1,000 generated ASCII strings (seed 3); `<NULL>`, `<CITY_NAME>`, ZWNJ, digit-split cases.
- test_keys: query and pool keys identical for every family except NC; NP count = n(n−1)/2 for n
  distinct first-8 tokens; `key64("EN", "ram marketing private limited")` equals the value in
  `plan/reference/keys_check.json`; hash equality across two spawned processes.
- test_topq: the §3.3 tie rule on hand-made arrays (ties at the boundary).
- test_metrics: pair/entity/full recall on a toy example with a singleton.
- test_outputs: writer on a 5-row fake test directory passes both `validate_stream.py` and the official
  validator (subprocess); the file contains no `\r`.
- Integration: M1 asserts; `index-sanity`.

---

## 10. Risks and mitigations
| Risk | Detection | Mitigation |
|---|---|---|
| Romanization misses (Indic) | `translit_dict.json`; segment "indic" recall | dictionary + SK/SP + NG; accept the rest |
| Common-word names silent | E2 exclusive counts | NP/SP; cap sweep |
| Very large unions (p99 latency) | p99 ms, dropped-key counts | caps; `max_postings_per_query` |
| Re-ranker fails to transfer (France) | LOCO drop; DEV_B | language-agnostic features, no country identity, caps as fractions |
| Worker memory under spawn | per-worker RSS log | mmap everything; reduce W |
| `\r\n` corrupting IDs | tests; streaming validator | `newline="\n"` everywhere |
| Unicode version differences | goldens | all artifacts on the laptop's Python 3.12 |
| Candidate file too large to submit | size logged | mean ≤ 100; open question Q1 |
| Hash collision | none practical | 64-bit; accepted |

---

## 11. Open questions for the human
- Q1 Upload size limit of the final zip? (`candidate_pairs.tsv` ≈ 13 bytes × mean size × 1.73M; mean 80 ≈ 1.8 GB raw.)
- Q2 Laptop CPU model, and is WSL2 acceptable? (Code is portable either way; W auto-sizes.)

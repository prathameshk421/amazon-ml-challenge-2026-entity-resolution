# Business Entity Resolution: Candidate Generation (Bucketing) — Implementation Plan v2

Status: v2 = v1.1 + labelled HOLDOUT_TEST split + fixes from the hostile review
(`plan/critique/CRITIQUE.md`), each re-verified on real data before adoption (§11 lists every
accepted/rejected point). Previous version: `plan/BUCKETING_PLAN_v1.1.md`.
Owner: human team. Executor: a coding agent on the friend's laptop.
Scope: raw TSV → `candidate_pairs.tsv` + the candidate parquet the matcher will use. The pairwise
matcher (model, threshold, decision rules) is a separate, later plan.

Read order for the executor: §0 → §1 → §3 → §4 → the module being implemented (§5) → the phase in §7.

Normative references (port verbatim; prose explains them; if they disagree the code wins and the
disagreement is logged in PROGRESS.md):
- `plan/reference/translit_ref.py` → `src/ber/translit.py` + `src/ber/textnorm.py` (§5.2, §5.5.1)
- `plan/reference/keys_ref.py` → `src/ber/keys.py` (§5.5)
- `plan/reference/goldens.json` → unit-test expectations (§9); regenerated on the laptop in P0 (§3.1)

---

## 0. Rules for the executor (non-negotiable)

R1. Implement exactly what this document specifies. If something is impossible or contradictory,
    stop that step, write it under `## BLOCKERS` in `PROGRESS.md`, continue with the next independent
    step. Never invent a replacement design silently.
R2. TUNE parameters change only through the experiment named next to them (§8) and its decision rule.
    FIXED parameters change only with a human decision logged under `## Deviations`.
R3. Data/package rules (official problem statement): no external data, APIs, registries, geocoders,
    gazetteers, transliteration libraries or models; nothing downloaded at runtime. Allowed packages:
    Python stdlib, numpy, pandas, pyarrow, scipy, scikit-learn, rapidfuzz, jellyfish, lightgbm, tqdm,
    matplotlib. Nothing else. Tests use stdlib `unittest`; configs are JSON.
R4. val-S1 (`train_source1_validation.tsv`, also stored as HOLDOUT_TEST in query_sets.json) is never
    used to fit, tune, choose thresholds or pick between designs. It is scored once per frozen
    pipeline version. S2 and S3 stay the full train pool. The official test files are
    read only to build the test-pool index, run inference, write outputs, and compute unsupervised
    statistics (document frequencies, edge-token set G, mined abbreviation map, S1 chain sizes); every
    such use is logged in `reports/test_usage_log.md` (what, why, file).
R5. Every number in any report comes from a JSON written by a script into `artifacts/metrics/`.
R6. Update `PROGRESS.md` after every step (template §3.4). `git commit` after every phase.
R7. Determinism: all random choices use the seeds given here; every sort that can tie uses §3.3.
    Re-running a phase must produce byte-identical outputs.

---

## 1. Verified facts this plan relies on

All measured on the real files on 2026-09-25. Evidence files: `plan/checks/*.json`,
`plan/reference/*.json`, `plan/critique/out/*.json`, `artifacts/metrics/diag_*.json`. P1 re-asserts
F1–F5 and F14 on the laptop's copy; any mismatch = stop (R1).

| # | Fact (evidence) | Consequence |
|---|---|---|
| F1 | Every data row has exactly 4 tab-separated fields; no `\r`, no BOM; header `entity_id business_name business_address country` (data_facts) | Parse with `line.rstrip("\n").split("\t")` |
| F2 | `"` occurs inside fields (data_facts) | Never use a quoting CSV reader on raw files |
| F3 | IDs `^S[123]-\d+$`, no leading zeros, number ≤ 999,999,995 (data_facts) | (src int8, num uint32); `f"S{src}-{num}"` round-trips (asserted) |
| F4 | S2 and S3 reuse numbers: 26,801 train / 25,008 test (data_facts) | A pool record is (src, num) |
| F5 | Rows: train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316 | Pools: train 10,320,219; test 9,969,589 |
| F6 | No empty names/countries; empty pool addresses: train 344,883, test 265,506; no address consists only of `<NULL>`/punctuation (data_facts, critique b_gt_facts) | Name channels must carry empty-address records |
| F7 | `<NULL>` inside pool addresses 35–44K per file, never in names | Removed in normalization |
| F8 | `<CITY_NAME>` (+1 `<CIT_NAME>`) in names, S1 and pool | Token `cityname` |
| F9 | Countries: train {US, India}; test {US, India, France}; France = 259,452 test S1 (14.98%) | Open set; France has no labels anywhere |
| F10 | Non-ASCII: 9 Indic scripts, accented Latin, `º`; S1 Latin only; no Arabic/CJK/Cyrillic | Romanize Indic; NFKD for Latin |
| F11 | 752,869 train-pool names contain Indic letters; 551,240 true links (7.2%) have an Indic pool name | Romanization is required |
| F12 | On all 551,240 Indic true links (reference romanizer): ≥1 identical token 79.72%, ≥1 identical skeleton 98.38%, but counting only keys with df ≤ 0.002·n_pool: token **26.15%**, skeleton **5.54%** (verify_critique C3). Skeleton `LMT` df 2,621,103 | Single-skeleton keys are useless for retrieval; the learned dictionary is the Indic lever (usable-token overlap 0.2598→0.6063 on held-out pairs: critique d_evidence, NOT re-run by me; re-measured by the §5.3 gate) |
| F13 | ZWNJ/ZWJ inside Indic words; Gurmukhi TIPPI/ADDAK/BINDI and DEVANAGARI OM were mishandled by v1 (verified: `ਅਨੰਦ` → `an d`) | Fixed in v2.1 (§5.2) |
| F14 | GT: 2,206,821 rows; 123,247 singletons (US 0.0558, India 0.0559); 7,638,365 links; max 11; no pool record has 2+ owners; 2,681,854 orphan pool records (26.0%) | Recall well defined |
| F15 | Validator strips only `\n`, header compared lower-case, IDs split on `,`; with the candidate file it holds all IDs in Python sets | `newline="\n"`; streaming validator |
| F16 | 100% of 7,638,365 true links have the same country on both sides; 0 true pool matches have an empty name; 337,018 true links (4.41%) have an empty pool address (split_design, critique b_gt_facts) | Country agreement is a strong feature (still soft) |
| F17 | Exact normalized-name equality: 28.55% of all links; 24.68% of links of chain ≥ 10 S1 (chain = # train S1 with the same name); chain ≥ 10 = 18.17% of links; their exact-name pool bucket d_EN median 31, p90 138 (verify_critique C2, C4) | Exact-name bonus must not dominate; chain rule keyed on S1 chain size |
| F18 | Name overlap: 35.26% of test S1 names occur among train S1 names (US 35.91%, India 46.02%, France 0.01%); a random 20% of train S1 vs the other 80%: 37.8% (US 33.48%, India 44.28%) (split_design) | Random entity split is realistic for US/India; France needs its own proxy |
| F19 | France pool vs France S1 per-record token shares: `r` 0.2486 vs 0.0002, `rue` 0.3886 vs 0.6582, `av` 0.0384 vs 0.0085, `street` (from `st`→street) 0.0170 vs 0.0005, name tokens `s` 0.0447 vs 0.0, `a` 0.0431 vs 0.0002 (from `S.A.S.`) (verify_critique C1) | v2.1 fixes: letter-run collapse, st/ste classes, mined abbreviations |
| F20 | Unsupervised abbreviation miner (§5.2.2) at cosine ≥ 0.95 on the France part of the test pool finds r→rue, av→avenue, all→allee, imp→impasse, rte→route, ch→chemin, pl→place, q→quai, bd→blvd→boulevard, n→no, b→bis, t→ter and one error l→labbe (df(labbe)/N = 0.0006); on the train pool it finds US/India state codes (tx→texas, mh→maharashtr), dr→drive, ave→avenue, apt, hwy, blvd (abbrev_miner) | Mined map with a target-frequency floor (§5.2.2), gated by E0 |
| F21 | Normalization v2.1 vs v1 on 519,542 true links: exact name equality 0.2859→0.2960; ≥1 shared token 0.9089→0.9093; address Jaccard 0.6980→0.6996; Gurmukhi shared token 0.2253→0.2590; no script decreased (norm_v21_check) | Adopt v2.1 |
| F22 | Learned linear stage-1 score vs hand composite (old indexes, same 10K queries): top-150 pair recall 0.9529 vs 0.9308; top-300 0.9607 vs 0.9405; in the quota union B300+N100+A100+P100+R50, replacing the hand B channel with the learned score: pair recall 0.9623→0.9682, entity 0.9950→0.9963, F0.5 ceiling 0.9854→0.9882 at mean size 497→487 (learned_stage1) | Stage-1 main channel = learned logistic score |
| F23 | Old top-150 candidates: pair recall 0.9251, entity 0.9866, full recall 0.8129, perfect-matcher F0.5 ceiling 0.9686 (critique b_gt_facts) | Primary blocking objective = F0.5 ceiling (§6) |
| F24 | Mac `.venv`: Python 3.14.5, Unicode 16.0.0, numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, rapidfuzz 3.14.6, lightgbm 4.7.0, scikit-learn 1.9.1, scipy 1.18.1 (critique M7; re-checked) | Laptop uses the same Python and pins (§3.1) |
| F25 | Stage-2 hot path on the Mac: 6 rapidfuzz scorers 2.69 ms/query at 700 candidates; LightGBM (33 features, 63 leaves, 500 trees, synthetic model) 22.6 µs/row (critique f_throughput, NOT re-run by me) | Tree cap 400; compute budget §7 (estimate) |

---

## 2. Architecture

```
raw TSV ─► M1 ingest ─► records (raw text, ids, GT)                               per split: train, test
              ▼
          M2 textnorm v2.1: NFC → placeholders → drop Cf → Indic romanization → NFKD/fold/casefold →
              │  & / apostrophes → punctuation → (addr) letter|digit split → fixed maps (+ st/ste classes)
              │  → (addr) mined per-country abbreviation map (E0-gated) → (names) single-letter collapse
              │  → (Indic fields) M3 learned dictionary
              ▼
          norm parquet + pool string store (mmap)
              ▼
          M4 key families EN NT NG NP PH SP NC AT AB XK (identical code for pool and query)
              ▼
          M5 index per split: family → sorted uint64 key hashes + CSR postings (.npy, mmap)
              │
S1 query ─────┼─► M6 stage-1: postings (df caps) → family scores → LEARN (logistic, §5.7) + NAME, ADDR,
              │      PAIR, XK, CAT channels → per-channel quota (chain-aware) → union (~500–1,000)
              ▼
          M7 stage-2: 36 cheap features → LightGBM p → adaptive cut (K_min / p_min / K_max ≤ 100 mean)
              ▼
          candidates parquet (s1_idx, pool_idx, rank, p_rerank) ─► M9 writer ─► candidate_pairs.tsv
              └─► later plan: pairwise matcher → matching_results.tsv
```

### 2.1 Decisions and trade-offs
| Decision | Chosen | Alternative | Why (evidence) | Accepted cost |
|---|---|---|---|---|
| Bucket model | Inverted index over many key families | One composite key per record | One typo kills a composite key; union reaches 98.7–99.35% (diag_blocking, diag_blocking2) | Large unions → ranking needed |
| Stage-1 ranking | Learned logistic score over family scores as the main quota channel, plus 5 hand channels for diversity | Hand weights (v1.1) | F22: +2.2 pp pair recall at K=150; quota union +0.6 pp at equal size | A small model trained on DEV labels (documented) |
| Shrinking | LightGBM re-ranker + adaptive cut | Fixed top-K | Hand re-rank gained +0.001 at K=150 (diag_rerank) | Stage-2 is a learned filter; documented as blocking |
| `candidate_pairs.tsv` | Stage-2 output | Stage-1 output | Spec p.3–4: "the last one: whatever your model actually runs inference over" | Matcher ceiling = stage-2 recall |
| Indic text | Romanize (stdlib `unicodedata` + fixed tables) + train-learned dictionary | External transliterator / keep script | Banned / erased 17% of Indian pool names | Tamil (0.23) and Malayalam (0.34) token match stay low → dictionary + stage-2 skeleton features |
| Chains | S1-chain-size-triggered quota boost + XK cross keys (name skeleton × usable address token) | EN bonus (v1.1) | F17: only 24.7% of chain ≥10 links share the exact name; XK reaches 86.5% of chain ≥10 true pairs (critique d_evidence) | XK index ≈ largest family (E2 measures) |
| France | Language-agnostic fixes + unsupervised per-country abbreviation mining on the pool + "name unseen" proxy segment + per-country monitoring on test | Hand French rules | Hand rules = hard-coding a country; mining is data-driven for any country label | France quality stays unmeasurable (no labels) |
| Split | Random 20% HOLDOUT_TEST stratified by country × match count | Name-grouped split | F18: random split reproduces test's name overlap for US/India | France not represented (proxy only) |
| Objective | Mean per-entity F0.5 ceiling (singletons = 1) with an entity-recall floor | Micro pair recall | Metric is macro per-entity F0.5 (F23) | Pair recall reported as secondary |
| Key storage | 64-bit blake2b hashes, `.npy`, mmap | Python dicts | Spawn-safe, shared page cache, portable | ~1e-6 collision probability |
| Country | Stage-2 feature (same/different/unknown) | Partition by country | Open-set rule; France unseen | Larger unions |

### 2.2 Out of scope for v2
BK-tree/automaton; postal-only keys (inside AT; 0 exclusive recoveries); embeddings/ANN (allowlisted
models are English-only); reverse pool→S1 blocking (optional X3, §8); NC word-order flips inside
domains (NG covers partially; unmeasured).

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
   then `wsl --shutdown`. (Default WSL memory is 50% of RAM = 12 GB: too small.)
3. Ubuntu: `sudo apt update && sudo apt install -y git unzip curl`, then install `uv` (human step,
   network allowed during setup only) and `uv python install 3.14.5`. Python 3.14.5 matches the Mac
   (Unicode 16.0.0, F24) so goldens and romanization are identical.
4. Project lives inside the Linux filesystem: `~/ber/` (never `/mnt/c/...`).
5. Code transfer: on the Mac, `git bundle create ber.bundle --all`; on the laptop `git clone ber.bundle ~/ber`.
   Copy `student_resource/` to `~/ber/student_resource/`.
6. `cd ~/ber && uv venv --python 3.14.5 .venv && source .venv/bin/activate`
7. `uv pip install numpy==2.5.3 pandas==3.0.6 pyarrow==25.0.1 rapidfuzz==3.14.6 jellyfish==1.2.1 scikit-learn==1.9.1 scipy==1.18.1 lightgbm==4.7.0 tqdm==4.70.1 matplotlib==3.11.2`
   (identical to the Mac pins). Any install failure = BLOCKER (do not substitute versions silently).
   Then `pip freeze > code/business_entity_resolution/requirements.txt`.
8. P0 check: `python -c "import unicodedata; print(unicodedata.unidata_version)"` must print `16.0.0`;
   run `plan/reference/make_goldens.py` on the laptop to a temp file and diff against the committed
   `goldens.json`; any difference = BLOCKER.
9. AC power; Windows power mode "Best performance"; `df -h ~` ≥ 40 GB free.
Mac: code writing + unit tests on tiny inputs only.

### 3.2 Repository layout (new code; old `src/` is reference only)
```
code/business_entity_resolution/
  README.md  requirements.txt  PROGRESS.md
  configs/paths.json        {"data_root": "student_resource/dataset", "work_root": "work", "metrics_root": "artifacts/metrics"}
  configs/blocking.json     all FIXED/TUNE parameters (§3.5)
  src/ber/
    __init__.py io_utils.py ingest.py translit.py textnorm.py abbrev_mine.py translit_dict.py
    splits.py hashing.py keys.py index_build.py index_store.py retrieve.py stage1_model.py
    features.py rerank.py evaluate.py run_candidates.py write_outputs.py validate_stream.py
    monitor.py cli.py
  tests/  test_textnorm.py test_translit.py test_keys.py test_topq.py test_metrics.py test_outputs.py test_abbrev.py
```
`work/` and `artifacts/` are git-ignored except `artifacts/metrics/*.json`.

### 3.3 Conventions
- Read text: `open(p, encoding="utf-8", newline="")`. Write text: `open(p, "w", encoding="utf-8", newline="\n")`.
- Types: s1_idx int32; pool_idx uint32; src int8 ∈ {2, 3}. Pool order: S2 rows (file order), then S3.
- Tie-break for "top-q by score": (score desc, pool_idx asc). Top-q of n > q: `v = np.partition(score, n-q)[n-q]`;
  take all with score > v; add score == v in ascending pool_idx until q are taken.
- Scores float64 in computation, float32 in storage.
- Parallelism: `multiprocessing.get_context("spawn").Pool(W, initializer=init_worker)`; workers open all
  big arrays with `np.load(..., mmap_mode="r")`; `OMP_NUM_THREADS=1`; never rely on fork.
  `W = min(max_workers, os.cpu_count() − 2)`, ≥ 1. LightGBM in workers: `num_threads=1`; training: `num_threads=W`.
- Logs: `work/logs/<command>_<YYYYmmdd-HHMMSS>.log`. Outputs `part-XXXXX.parquet` + `part-XXXXX._SUCCESS`;
  finished parts are skipped on re-run. Atomic writes (`.tmp` + `os.replace`).
- Every metrics JSON has `n_queries`, `n_true_pairs`, `config`, `config_hash` (sha1 of canonical JSON of
  `blocking.json`), `norm_version`, `git_commit`, `created_at`.

### 3.4 `PROGRESS.md` template
```
# PROGRESS
LAST_UPDATE: <date -Is>
CURRENT: <phase>.<step> <one line>
## Phases
P0 setup + goldens + tests:
P1 ingest + splits:
P2 textnorm (E0) + dictionary:
P3 train index:
P4 stage-1 (E1–E6):
P5 stage-2:
P6 HOLDOUT_TEST (once) + test index + outputs:
## Results (one line each, citing a metrics json)
## BLOCKERS
## Deviations (what, why, approved-by)
```

### 3.5 `configs/blocking.json` (initial; TUNE values are defaults until tuned)
```json
{
  "norm_version": "2.1",
  "max_workers": 14,
  "abbrev_mining": {"enabled": true, "min_frac": 0.0005, "short_max_len": 4, "cos_min": 0.95,
                    "target_min_frac": 0.001, "sample": 1500000, "min_country_records": 50000, "seed": 0},
  "families": ["EN", "NT", "NG", "NP", "PH", "SP", "NC", "AT", "AB", "XK"],
  "build_df_max_frac": 0.005,
  "cap_frac": {"EN": 0.002, "NT": 0.002, "NG": 0.0008, "NP": 0.002, "PH": 0.002, "SK": 0.002,
               "SP": 0.002, "NC": 0.002, "AT": 0.002, "AB": 0.002, "XK": 0.002},
  "max_postings_per_query": 300000,
  "hand_both": {"EN": 10, "NT": 1, "NG": 1, "PH": 0.2, "AT": 1},
  "channels": {
    "LEARN": "stage1_model",
    "NAME": {"EN": 1, "NT": 1, "NG": 1, "PH": 0.2},
    "ADDR": {"AT": 1, "AB": 1},
    "PAIR": {"NP": 1, "SP": 1},
    "XK":   {"XK": 1},
    "CAT":  {"NC": 1}
  },
  "quota": {"LEARN": 300, "NAME": 100, "ADDR": 100, "PAIR": 100, "XK": 100, "CAT": 20},
  "chain_rule": {"enabled": true, "min_chain": 10, "mult": 2, "channels": ["ADDR", "PAIR", "XK"]},
  "edge_tokens": {"min_frac": 0.0005, "min_edge_share": 0.80, "nc_min_len": 6, "nc_pool_token_min_len": 8},
  "stage1_model": {"neg_top": 300, "neg_rand": 300, "C": 1.0, "max_iter": 2000},
  "rerank": {
    "sample": {"top_neg": 100, "rand_neg": 100},
    "lgb": {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "max_bin": 255, "seed": 42, "deterministic": true, "force_row_wise": true,
            "num_boost_round": 400, "early_stopping_rounds": 50, "holdout_frac": 0.1},
    "cut": {"K_min": 10, "K_max": 100, "p_min": 0.005},
    "cut_grid": {"K_min": [5, 10, 20], "K_max": [60, 80, 100, 150],
                 "p_min": [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]},
    "mean_size_budget": 100
  },
  "tuned_by": {}
}
```
Channel rule: a channel's hand formula skips dropped families; a channel with no remaining family gets quota 0.
`NAME` uses EN weight 1 (not 10): F17 shows the exact name is present for only ~25–29% of links.

---

## 4. Data contracts (under `work/` unless stated; parquet zstd)

| File | Content |
|---|---|
| `records/{split}_s1.parquet` | s1_idx int32, entity_id, name_raw, addr_raw, country_raw |
| `records/{split}_pool.parquet` | pool_idx uint32, src int8, num uint32, name_raw, addr_raw, country_raw |
| `records/train_gt.parquet` | s1_idx int32, pool_idx uint32 (7,638,365 rows) |
| `records/train_s1_nmatch.npy` | int16[n_s1] |
| `norm/abbrev_{split}.json` | mined map per country label: `{country: {short: long}}` + stats (§5.2.2) |
| `norm/{split}_s1.parquet`, `norm/{split}_pool.parquet` | idx, name_norm, addr_norm, name_indic, addr_indic |
| `norm/{split}_s1_chain.npy` | int32[n_s1]: # S1 of the same split with the same name_norm |
| `norm/{split}_pool_store/` | name.bin + name.off.npy, addr.bin + addr.off.npy, country.npy (int16), countries.json, src.npy, num.npy |
| `translit/dict_v1.json` | {"name": {...}, "addr": {...}, "meta": {...}} |
| `splits/query_sets.json` | lists of train s1_idx per set (§5.4) + sizes + seeds |
| `index/{split}/{F}.keys.npy/.indptr.npy/.post.npy` + `meta.json` | per family |
| `stage1/model_v1.json` | logistic coefficients, feature means/stds, feature order, config_hash |
| `rerank/rr_train.parquet`, `rerank/rr_cal.parquet`, `rerank/model_v1.txt`, `rerank/cut_v1.json` | stage-2 |
| `candidates/{SET}/part-XXXXX.parquet` | s1_idx int32, pool_idx uint32, rank int16, p_rerank float32 |
| `outputs/candidate_pairs.tsv`, `outputs/matching_empty.tsv` | test outputs |

---

## 5. Module specifications

### 5.1 M1 ingest — FIXED
Line-by-line parsing (F1); assert header, 4 fields, ID regex/prefix, round-trip, uniqueness, row counts
(F5). GT: every S1 has one row, all IDs exist, 7,638,365 links (F14). Stream in 1M-row chunks.

### 5.2 M2 normalization v2.1 — FIXED (except E0-gated mining)
`normalize_field(text, kind, tdict, abbrev_map) -> (norm, had_indic)`; normative code
`plan/reference/translit_ref.py` (mined map is applied by the new code at the point marked below).
Order:
1. NFC. 2. `<NULL>` → space; other `<[A-Z_]+>` → ` cityname `. 3. delete category `Cf`.
4. Romanize Indic (§5.2.1). 5. NFKD; delete category `Mn`. 6. fold œ æ ß ø ł đ ð þ ı ħ (and capitals).
7. casefold. 8. `&` → ` and `; delete apostrophes `' ’ ‘ ʼ \``. 9. non-`isalnum` → space.
10. addr: space at letter↔digit boundaries.
11. token maps: names `NAME_ABBREV`; addresses `ADDR_ABBREV` where `st`, `street`, `saint` → `st` and
    `ste`, `suite`, `sainte` → `ste` (canonical classes, not expansions; F19).
11b. addr only, if E0 adopts it: the record's country map from `norm/abbrev_{split}.json`
    (only for tokens not already mapped in step 11).
12. names only: collapse runs of ≥ 2 single-letter alphabetic tokens (`s a s` → `sas`, `j s w` → `jsw`).
13. if `had_indic`: learned dictionary `tdict[kind]` per token.
14. join with single spaces.

#### 5.2.1 Romanizer (summary of the reference)
Class from `unicodedata.name(c)` for U+0900–U+0D7F: independent vowel; consonant (table `CONS`);
dead consonant (CHILLU X → `CONS[X+"A"]`, KHANDA TA); vowel sign; virama; ANUSVARA, Gurmukhi TIPPI and
SIGN BINDI → "m" before p/b/m, else "n"; CANDRABINDU → "n"; VISARGA → "h"; Gurmukhi ADDAK and other
SIGNs/LENGTH MARKs ignored; DEVANAGARI OM → "om"; digits → ASCII; anything else → word boundary.
Inherent "a" written before a following consonant/vowel/nasal sign, replaced by a vowel sign, cancelled by
virama, dropped at a word boundary. RRA clusters: RRA+VIRAMA+RRA → "t"; NA/NNNA+VIRAMA+RRA → "n"+"t".
Goldens: `goldens.json` (83 cases: 74 names over 9 scripts + Latin/France cases; 9 addresses), e.g.
`ਅਨੰਦ` → `anand`, `BORDEAUX CLUB S.A.S.` → `bordeaux club sas`, `57 R. JEAN MOULIN STE MARIE` →
`57 r jean moulin ste marie`, `12 Main St, St Louis` → `12 main st st louis`.

#### 5.2.2 Unsupervised per-country address-abbreviation mining (`abbrev_mine.py`) — gated by E0
Runs per split on the POOL, separately for each country label with ≥ `min_country_records` pool records
(any label, no hard-coded countries). `mine-abbrev` reads `records/{split}_pool.parquet` and applies
steps 1–11 itself (no 11b), so it does not depend on `normalize` having run.
Algorithm (exactly as `plan/checks/abbrev_miner.py`, plus the target floor):
1. Sample `sample` records (seeded shuffle). df over records; N = sample size.
2. Short S: alphabetic, 1 ≤ len ≤ `short_max_len`, df ≥ `min_frac`·N. Long L: alphabetic, len ≥ 2, df ≥ `min_frac`·N.
3. (s, l) eligible iff l[0] == s[0], len(l) > len(s), s is a subsequence of l.
4. Context vector = [left-neighbour counts | right-neighbour counts] over the vocabulary of tokens with
   df ≥ `min_frac`·N plus `<num>`, `<s>`, `</s>` (digits → `<num>`); L2-normalized.
5. s → l* = argmax cosine; keep if cosine ≥ `cos_min`, df(l*) ≥ 0.1·df(s) and df(l*) ≥ `target_min_frac`·N.
6. Resolve chains (bd → blvd → boulevard) up to depth 3; drop cycles; drop s already mapped in step 11.
Output `norm/abbrev_{split}.json`. On test this is an unsupervised statistic of test inputs → log (R4).
Applied to S1 AND pool records of that country (same split, same map).

### 5.3 M3 learned transliteration dictionary — FIXED rule, new gate
Pairs: true pairs with S1 ∈ DICT_SRC (§5.4) whose pool field `had_indic`; names and addresses separately.
Texts from `normalize_field(..., tdict={})`. For each pool token r (len ≥ 2) not present verbatim among the
S1 tokens: c(r) += 1; l* = S1 token with max `JaroWinkler.normalized_similarity` (ties: first); if ≥ 0.60,
c(r, l*) += 1. Keep r → l* if c(r, l*) ≥ 5 and c(r, l*)/c(r) ≥ 0.80.
Gate (metric `translit_dict.json`, on DEV_A true pairs with Indic pool names): share of pairs sharing ≥ 1
token whose train-pool df ≤ cap_frac["NT"]·n_pool ("usable token"), before vs after, overall and per
script. Must increase overall and must not decrease for any script (F12: expected ≈ 0.26 → ≈ 0.61).

### 5.4 Splits and query sets — FIXED
Canonical TSVs (not in git): `data/student_resource/dataset/train/train_source1_train.tsv` and
`train_source1_validation.tsv` plus matching `train_ground_truth_{train,validation}.tsv`. See
`data/student_resource/dataset/split.md`. S2 and S3 are not split.
1. Walk train S1 in file order. Stratum = country × {singleton if nmatch==0 else matched}.
2. Per stratum, `random.Random(42).shuffle`; validation = `round(0.2 * n)` (441,363 rows);
   train-S1 = the rest (1,765,458). Same assignment as `data/write_s1_split.py`.
3. `perm = numpy.random.default_rng(20260925).permutation(sorted(train-S1 entity_id))`:
   DEV_A = perm[0:10000] (stage-1 tuning), DEV_B = perm[10000:20000] (confirmation),
   S1R_TRAIN = perm[20000:40000] (stage-1 logistic model), RR_TRAIN = perm[40000:90000] (stage-2 model),
   RR_CAL = perm[90000:110000] (stage-2 cut), MATCH_TRAIN = perm[110000:290000],
   MATCH_VALID = perm[290000:310000], DEV_REST = perm[310000:].
   DICT_SRC = train-S1 \ (DEV_A ∪ DEV_B). HOLDOUT_TEST in `query_sets.json` is an alias of val-S1.
4. Assert pairwise disjointness of the work slices plus val-S1, covering all 2,206,821 train S1.
Why random, not grouped by name: F18 (random 20% of S1 reproduces the official test's name overlap for
US/India; grouping would give 0% overlap). France has no labelled proxy; the "name unseen in train-S1"
segment (§5.9) and country-holdout eval (§5.8) are the closest train-side stand-ins.
Rules: every model/threshold/labelled statistic uses train-S1 only (mining and df on the
pool are unsupervised and include validation-owned pool records, exactly as the test pool will include
test-owned records). Evaluation of any set is always against the FULL train S2+S3. Any later global
"one pool record → one owner" step must run over all train S1 queries, not only the evaluated set.
Set → split: all train sets use train S1 + train index; TEST uses test S1 + test index.

### 5.5 M4 key families — FIXED definitions (families kept by E2)
Normative: `plan/reference/keys_ref.py`. tokens = `name_norm.split()`; tok2 = len ≥ 2; first-8 = first
8 distinct tok2.

| Family | Keys | Query vs pool |
|---|---|---|
| EN | `[name_norm]` | same |
| NT | distinct tok2 | same |
| NG | distinct 3-grams of `" " + name_norm + " "` | same |
| NP | unordered pairs of first-8 | same |
| PH | distinct non-empty metaphone of tok2 with len ≥ 3 | same |
| SP | unordered pairs of the distinct skeletons (len ≥ 2) of first-8 | same |
| NC | `concat_variants(tokens, G)` | pool also adds single tokens with len ≥ 8 |
| AT | distinct address tokens | same |
| AB | distinct adjacent address bigrams | same |
| XK | `xk_keys`: skeleton(first-8 token, len ≥ 2) × address token with AT df ≤ floor(cap_frac["AT"]·n_pool), key `skeleton + "#" + token` | same (AT df of the split's pool) |
| SK (not default; E2 candidate; always a stage-2 feature) | distinct skeletons (len ≥ 2) of tok2 | same |

Hash `key64(F, k)` = little-endian int of blake2b((F + "\x1f" + k).utf8, digest_size=8).
`concat_variants`, G (edge tokens) and `skeleton` exactly as in the reference; G on the train pool =
268 tokens (`plan/reference/keys_check.json`).

### 5.6 M5 index build — FIXED
Per split: compute G and AT df first (needed by NC and XK), then for each family: workers over
250,000-row chunks → (key64, pool_idx) → 64 shards by `key64 >> 58` → per shard lexsort, unique, df,
drop df > build_df_max_frac·n_pool → concatenate shards → `.npy`. Verify keys increasing, indptr
monotone, postings < n_pool and ascending per key; delete temp shards. `meta.json` with counts, df
quantiles, G, AT-df cap, sizes, wall time, peak RSS. `index-sanity`: 1,000 random pool records
(`default_rng(1)`) are found in their own EN postings and their NT postings (df ≤ build limit).

### 5.7 M6 stage-1 retrieval + stage-1 model
Per query (name_norm, addr_norm, chain_S1 = `norm/{split}_s1_chain.npy[s1_idx]`):
1. Query keys per family; key used if df ≤ floor(cap_frac[F]·n_pool). Weight: EN 1.0; others
   ln((n_pool+1)/(df+1)). Q[F] = Σ weights of used keys. d_EN = df of the EN key (0 if absent).
2. If total postings > `max_postings_per_query`: drop used keys by df desc (ties: family order, key64 asc)
   until ≤ limit; log count.
3. `u, inv = np.unique(postings, return_inverse=True)`; S[F] per family via bincount.
4. Stage-1 feature vector per union item (order fixed): S[F] for the kept families (10 default),
   cov = S[F]/Q[F] for NT, NG, NP, AT, XK (0 if Q = 0), log1p(S[F]) for the kept families.
   LEARN score = standardized features · coefficients (`stage1/model_v1.json`); hand channels per
   `channels`.
5. Quotas: `quota[C]`; if `chain_rule.enabled` and chain_S1 ≥ `min_chain`, multiply the listed channels
   by `mult`. Eligible = channel score > 0 (LEARN: all union items). Selected = union of per-channel top-q.
6. Output ordered by LEARN desc, pool_idx asc; keep the feature vector for stage 2.
Stage-1 model (`stage1-train`): on S1R_TRAIN queries, union with the E2/E3 families and caps; label =
truth; keep all positives, all items in the top `neg_top` by `hand_both`, and `neg_rand` uniform random
(rng seed = 7000 + s1_idx) from the rest with weight n_rest/n_sampled. Standardize (mean/std over the
training rows), `LogisticRegression(C, max_iter, random_state=0)`. Save coefficients + scaler. Evidence F22.
Evaluation mode (E1–E5): per union item compute each channel's rank; an item is selected by a quota
config iff ∃ channel c with rank_c ≤ q_c; evaluate many configs at once (vectorized on items whose min
rank ≤ max grid quota). Returns per config: size, hits, per-entity recall arrays (for §6 metrics).
Performance target: median ≤ 8 ms/query single-thread (log p50/p99 ms and |U|).

### 5.8 M7 stage-2 re-ranker
Features (float32, this order; backticked = column names):
1–10 `s_<F>` for the 10 default families (0 if dropped). 11–15 `cov_NT, cov_NG, cov_NP, cov_AT, cov_XK`.
16 `learn` (stage-1 LEARN score). 17–19 `rk_LEARN, rk_NAME, rk_ADDR` (rank in U, capped 10,000; 10,000 if
ineligible). 20 `log_union`. 21 `log_dEN`. 22 `log_chain` = ln(1+chain_S1).
23–26 names: `fz_tset`, `fz_ratio`, `fz_partial` (÷100), `jw_concat`.
27 `sk_jacc`: Jaccard of the name skeleton sets (len ≥ 2). 28–29 addresses: `fa_tset`, `fa_tsort` (÷100; 0 if either empty).
30 `addr_empty` (0 both, 1 pool empty, 2 query empty, 3 both). 31 `postal` (2 shared 4–6-digit token,
0 both have none shared, 1 otherwise). 32 `country` (1 same, 0 different, 2 either empty).
33 `src`. 34 `pool_indic`. 35 `q_ntok`, 36 `p_ntok`. (`len_ratio` from v1.1 removed: redundant with 35/36.)
Training (`rerank-build`): RR_TRAIN queries through stage-1 (final config); among selected candidates keep all
positives, negatives in the top `top_neg` by LEARN (weight 1), and `rand_neg` uniform (seed 1000 + s1_idx)
from the remaining selected negatives with weight n_rest/n_sampled; features only for kept rows. RR_CAL:
all selected, weight 1; write `rr_cal_stage1.json` (stage-1 metrics of RR_CAL).
Model: `rerank.lgb`; holdout 10% of RR_TRAIN by query (`GroupShuffleSplit(1, test_size=0.1, random_state=42)`);
early stopping on binary_logloss; ≤ 400 rounds.
Cut (`rerank-cut`, RR_CAL): rank by p desc (ties LEARN desc, pool_idx asc); keep rank r if
`r ≤ K_min or (p ≥ p_min and r ≤ K_max)`; all `cut_grid` combos. Decision: among configs with mean size ≤
`mean_size_budget`, entity recall ≥ stage-1 entity recall − 0.002 and F0.5 ceiling ≥ stage-1 F0.5 ceiling
− 0.002, choose the smallest mean size (ties: higher F0.5 ceiling). If none: highest F0.5 ceiling with mean
≤ budget; log a deviation.
LOCO (`rerank-loco`): stage-2 trained on RR_TRAIN India-only, cut from RR_CAL India-only, evaluated on
DEV_B US-only; and the reverse. Report the drop vs the all-country model on the same slices.
Staleness guard: stage-2 artifacts store the stage-1 config hash (+ stage-1 model hash); mismatch → refuse.

### 5.9 M8 metrics — primary objective = F0.5 ceiling
Per query: r = |cand ∩ truth|/|truth| (non-singletons); ceiling = 1.25·r/(0.25+r) (0 if r = 0); singleton
ceiling = 1.0 (a perfect matcher predicts empty). Reported per set:
- **F0.5 ceiling** = mean over ALL queries (primary). Entity recall, full recall, pair recall (secondary).
- Reachability (stage-1 union before quotas); exclusive pairs per family.
- Sizes: mean, p50, p90, p99, max; reduction ratio.
- Segments (all metrics): chain_S1 {1, 2–9, ≥10}; country; pool name class {indic, ascii, other};
  pool address empty; **name unseen** = no DEV S1 entity other than the query itself has the same
  name_norm (France proxy; F18); match count {1, 2–3, 4–5, 6+}.

### 5.10 M9 outputs and monitoring
- `candidate_pairs.tsv`: header `source1_entity_id\tcandidate_entity_ids`; one row per test S1 in file
  order; IDs `f"S{src}-{num}"` in rank order joined by ","; empty allowed; `newline="\n"`.
- `matching_empty.tsv` (format placeholder, never submitted).
- `validate_stream.py` (mandatory): every official rule + ID existence via sorted uint64 (src<<32|num) + no `\r`.
- Official validator: best effort (memory-heavy with the candidate file), run alone; record the outcome.
- `monitor.py` (test run, unsupervised; logged): per country label: candidate-list size distribution,
  distribution of max p_rerank and of p at rank 1–5, share of empty lists, share of queries hitting
  `max_postings_per_query`. Alarm (written to PROGRESS.md) if a country's mean max-p is < 0.5 × the
  mean of the train-covered countries or its empty-list share is > 3× theirs. An alarm does not change
  the pipeline automatically; it goes to the humans.

---

## 6. Evaluation protocol
- Primary objective everywhere: F0.5 ceiling (§5.9), with entity recall as a floor. Pair recall secondary.
- Stage-1: tuned on DEV_A (E0–E5); stage-1 model trained on S1R_TRAIN; confirmed once on DEV_B (E6).
- Stage-2: trained on RR_TRAIN; cut on RR_CAL; confirmed once on DEV_B.
- DEV_B rules: (a) E6: if DEV_B stage-1 F0.5 ceiling < DEV_A − 0.003, switch to the next point of the E4
  Pareto list (larger mean size) and re-check once. (b) Stage-2 on DEV_B: if F0.5 ceiling < RR_CAL − 0.003,
  use the next smaller p_min and re-check once. A stage-1 change after P5 started → rerun P5.
- HOLDOUT_TEST: once, after P5, full pipeline frozen; report all metrics + segments. No decision may be
  changed because of it except a human "go/no-go".
- Reference points (old system, old split): top-150 pair recall 0.9251, entity 0.9866, full 0.8129,
  F0.5 ceiling 0.9686 (F23).

---

## 7. Runbook (repository root, venv active: `python -m ber.cli <command>`)

| Command | Arguments | Does |
|---|---|---|
| `ingest` | `--split {train,test}` | M1 |
| `splits` | none | §5.4 |
| `mine-abbrev` | `--split {train,test}` | §5.2.2 (test = logged unsupervised use) |
| `normalize` | `--split {train,test} [--no-dict] [--no-abbrev]` | M2 |
| `learn-dict` | none | M3 |
| `build-index` / `index-sanity` | `--split` | M5 |
| `s1-eval` | `--set DEV_A --exp <id> [--families ...] [--cap F=x ...] [--quota-grid] [--scorer hand|learn]` | M6 evaluation mode |
| `stage1-train` | none | stage-1 logistic model |
| `rerank-build`, `rerank-train`, `rerank-cut`, `rerank-loco` | none | M7 |
| `rerank-eval` | `--set {DEV_B,HOLDOUT_TEST}` | runs stage-1+2 for the set if needed, writes metrics |
| `candidates` | `--set {MATCH_TRAIN,MATCH_VALID,HOLDOUT_TEST,TEST} [--limit N]` | M6+M7 → `candidates/{SET}/` |
| `write-outputs` | none | TSVs + streaming validation |
| `monitor` | none | §5.10 |

| Phase | Steps | Gate | Est. time (laptop, W=14) |
|---|---|---|---|
| P0 | §3.1; implement + pass unit tests (§9) | goldens diff empty; tests pass | human + coding |
| P1 | `ingest` ×2, `splits` | asserts | ~10 min |
| P2 | `mine-abbrev --split train`; E0; `normalize --split train --no-dict`; `learn-dict`; `normalize --split train` (writes norm parquet, pool store and `train_s1_chain.npy`) | E0 decision; dictionary gate | ~45 min |
| P3 | `build-index --split train`, `index-sanity` | sanity 100% | ~25 min |
| P4 | E1–E6 (§8), incl. `stage1-train` inside E3b | E1 gate | 2–4 h |
| P5 | `rerank-build`, `rerank-train`, `rerank-cut`, `rerank-eval --set DEV_B`, `rerank-loco` | §6 DEV_B rule | 1–2 h |
| P6 | `rerank-eval --set HOLDOUT_TEST` (once); `mine-abbrev --split test` (only if E0 adopted it); `normalize --split test`; `build-index --split test`; `index-sanity --split test`; `candidates --set TEST`; `write-outputs`; `monitor`; `candidates --set MATCH_TRAIN`; `candidates --set MATCH_VALID` | streaming validator PASS | 1.5–3 h |

Compute estimate (ESTIMATE from Mac measurements F25 and the 8 ms stage-1 target; the executor logs
real numbers): per query ≈ 8 ms stage-1 + 2.7 ms features + ~700 × 18 µs (400 trees) ≈ 23 ms. Queries:
TEST 1,732,544 + HOLDOUT_TEST ~441,365 + MATCH_TRAIN 180,000 + MATCH_VALID 20,000 + DEV/S1R/RR ~110,000 ≈
2.48M → ≈ 16 CPU-hours → ≈ 1.2 h wall at W = 14 (plus I/O). If real stage-1 latency or union size is
2× the target, expect ≈ 2.5 h.
Memory (20 GB WSL): per split mmap index ≤ 4 GB (XK included; E2 logs the real size) + ≈ 1 GB string
store, shared; per worker ≤ 400 MB (log RSS). If summed RSS > 16 GB, reduce W (min 4). Disk ≈ 20–25 GB.

---

## 8. Experiments (DEV_A unless stated) with decision rules
All report the §5.9 metric set; "better" = higher F0.5 ceiling unless stated.
E0 Abbreviation mining (train; runs in P2, before any index exists). On DEV_A, compare normalization with vs
   without step 11b: (a) mean address-token Jaccard of true pairs; (b) the same for random negatives = for
   each DEV_A query, the true pool records of 5 other DEV_A queries of the same country (rng seed 11).
   Adopt 11b if (a) rises ≥ 0.005 AND (a) rises more than (b). Record the decision in
   `blocking.json.abbrev_mining.enabled`.
E1 Baseline (bug gate): families {EN, NT, NG, PH, AT}, scorer `hand_both`, top-K K ∈ {50, 150, 500, 2000},
   reachability. Gate: pair recall@150 ≥ 0.91 (old system 0.9251 on a different split with v1 normalization;
   v2.1 should not be worse). Lower → bug hunt; stop.
E2 Family value (scorer `hand_both` for ranking inside channels; default quotas with LEARN replaced by
   `hand_both` until E3b): add one at a time NP, SP, NC, AB, XK, SK to the E1 set. Keep F if exclusive pairs
   ≥ 0.05% of DEV_A true pairs OR F0.5 ceiling improves ≥ 0.001. Test removing PH by the same rule. Log index
   size and median ms per family.
E3 Caps (kept families; order NT, NG, AT, NP, SP, AB, XK): cap × {0.5, 1, 2.5} (≤ build limit). Take the
   larger cap if F0.5 ceiling gains ≥ 0.001 and median ms rises ≤ 30%; the smaller if it loses < 0.0005 and
   median ms falls ≥ 15%.
E3b Stage-1 model: `stage1-train` with the E2/E3 configuration; compare on DEV_A: learned top-K vs
   `hand_both` top-K (K = 150, 300). Keep LEARN as the main channel if its F0.5 ceiling at K = 300 is ≥ the
   hand one (F22 predicts +). Otherwise use `hand_both` as the main channel and log it.
E4 Quota grid (evaluation mode, one pass): LEARN {150, 300, 500} × NAME {0, 50, 100} × ADDR {50, 100, 200} ×
   PAIR {0, 50, 100} × XK {0, 50, 100} × CAT {0, 20} = 486 configs. Save the Pareto front (F0.5 ceiling vs mean
   size). Decision: let best = max F0.5 ceiling among configs with mean ≤ 1,000; choose the smallest-mean
   config whose F0.5 ceiling ≥ best − 0.001 and entity recall ≥ best config's entity recall − 0.001.
E5 Chain rule: min_chain ∈ {5, 10, 20} × mult ∈ {1 (off), 2, 3}. Choose the best setting if the chain ≥ 10
   segment F0.5 ceiling gains ≥ 0.005 and overall mean size grows ≤ 15%; else off.
E6 DEV_B confirmation (§6).
Optional (after P6 works, only with the trigger): X2 name × address token (non-skeleton) cross keys —
trigger: chain ≥ 10 F0.5 ceiling < 0.95 after E5. X3 reverse blocking — trigger: X2 not kept and the same
condition holds.
Every chosen value goes into `blocking.json` with `tuned_by`.

---

## 9. Tests (stdlib `unittest`)
- textnorm/translit: all 83 `goldens.json` cases (exact `out`, `had_indic`, `skeletons`); Indic outputs ASCII;
  idempotence on 1,000 generated ASCII strings (seed 3); `<NULL>`, `<CITY_NAME>`, ZWNJ, TIPPI, S.A.S.,
  st/ste classes, digit split.
- abbrev: miner on a synthetic corpus recovers planted pairs (e.g. `12 rd x` / `12 road x`) and rejects a
  planted low-frequency target; chain resolution; no cycles.
- keys: query/pool identity for all families except NC; NP count n(n−1)/2; XK uses the AT-df cap; `key64`
  golden from `keys_check.json`; hash equality across two spawned processes.
- topq tie rule; metrics (F0.5 ceiling incl. singleton = 1, r = 0 → 0); outputs (5-row fake test dir passes
  `validate_stream.py` and the official validator; no `\r`).
- Integration: M1 asserts; `index-sanity`.

---

## 10. Risks and mitigations
| Risk | Detection | Mitigation |
|---|---|---|
| France collapse (15% of score) | `monitor.py` alarms; name-unseen segment | language-agnostic keys; per-country mined abbreviations; humans decide |
| Tamil/Malayalam/Gurmukhi romanization weak | per-script dictionary gate; indic segment | dictionary; `sk_jacc` feature; XK |
| Chains | chain ≥ 10 segment | S1-chain quota boost; XK; learned stage-1 |
| Mined abbreviation errors | E0 gate; `abbrev_*.json` reviewed by a human before P6 | cosine + target-frequency floors |
| Learned stage-1/2 overfit | DEV_B rules; LOCO; HOLDOUT_TEST once | disjoint query sets |
| Unions too large | p99 ms, dropped keys | caps; postings limit |
| Compute | logged ms | tree cap 400; MATCH sets 200K |
| Memory under spawn | RSS log | mmap; reduce W |
| `\r\n` | tests, streaming validator | `newline="\n"` |
| Candidate file too large | size logged | mean ≤ 100 (Q1) |

---

## 11. Review log (hostile critique → decision, each re-verified)
| Critique item | Verified? (evidence) | Decision |
|---|---|---|
| B1 France: no proxy, `r`/`rue`, `st`→street, `S.A.S.` | Yes (verify_critique C1; reference outputs) | Accepted: letter collapse, st/ste classes, per-country mined abbreviations (F20), name-unseen segment, monitoring |
| M1 skeleton usable only 5.5%; dictionary gate vacuous | Yes (C3: 0.0554 / 0.2615) | Accepted: SK removed from default retrieval (feature only); gate on usable tokens per script |
| M2 pair recall ≠ macro F0.5 | Yes (arithmetic; F23) | Accepted: F0.5 ceiling primary everywhere |
| M3 hand weights, learned scorer | Yes, measured myself (F22) | Accepted: LEARN channel |
| M4 chains: EN decoys, chain rule on d_EN | Yes (C2, C4: 24.68% exact, d_EN median 31) | Accepted: chain rule on S1 chain size; XK family; EN weight 1 in NAME |
| M5 split realism, leakage traps | Yes (split_design; same numbers) | Accepted: random stratified HOLDOUT_TEST; global-assignment rule; "pool density" stress test NOT adopted (cannot add realistic distractors; noted as limitation) |
| M6 compute | Partly (critique timing on Mac; not re-run) | Accepted: 400-tree cap; MATCH sets 200K; estimate labelled |
| M7 Python/Unicode mismatch | Yes (F24) | Accepted: laptop uses Python 3.14.5 via uv; goldens diff in P0 |
| m1 Gurmukhi TIPPI/ADDAK/OM | Yes (reference outputs) | Accepted: fixed; F21 |
| m2 re-ranker compliance | Spec text re-read (p.3–4) | Accepted: document stage-2 as a learned blocking filter; keep stage-1 reproducible |
| m3 contradictions | Yes | Fixed: budget 100 everywhere; E1 gate re-based; quota numbers labelled old-era |
| m4 NC word-order flips | Not measured | Not adopted (NG covers partially); listed out of scope |

---

## 12. Open questions for the human
- Q1 Upload size limit of the final zip? (`candidate_pairs.tsv` ≈ 13 bytes × mean size × 1.73M; mean 100 ≈ 2.3 GB raw.)
- Q2 Laptop CPU model; is WSL2 + uv (Python 3.14.5) acceptable?
- Q3 Human review of the mined abbreviation maps (`norm/abbrev_*.json`) before P6 — who signs off?

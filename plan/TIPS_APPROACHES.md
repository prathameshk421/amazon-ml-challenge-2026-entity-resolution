# Contest tips → concrete approaches for stages 1, 2, 3

Status: analysis and proposals, 2026-09-26. **Nothing in this file is implemented unless marked
"exists" or "in working tree".** Priority of rules: `AGENTS.md` > `split.md` > `VALIDATION_PLAN.md` >
this file.

The contest gave five tips. For each tip, this document says what the pipeline already does, where it
still loses score (measured), and which approaches could close the gap: how each one works, what it
could gain, what it changes in our architecture, and what it costs. Section 7 lists fuzzy-matching,
heuristic and country-pattern (regex) features that can go into the matcher, and section 8 gives the
test that decides whether a feature is kept, so that no feature makes the model worse.

Every number cites a metrics file in `artifacts/metrics/`. The error-pattern counts come from
`plan/checks/error_patterns.py` → `artifacts/metrics/error_patterns.json`. That script reads only the
MATCH_TRAIN out-of-fold error exports: train S1, never validation S1 or test.

---

## 0. Short version

Current result: **OOF F0.5 0.9741** on MATCH_TRAIN (180,000 train S1) and **0.9745** on MATCH_VALID
(20,000 S1). The best possible score with the current candidates is 0.9930 / 0.9928
(`match_train.json`, `match_eval_MATCH_VALID.json`). Remaining loss: 0.0259.

| Rank | Approach | Stage | Loss it goes after (OOF) | Cost | Evidence for gain |
|---|---|---|---|---|---|
| 1 | **N1** Entity-level "has any match?" model | 3 | ~0.0055 (matched S1 given an empty answer) + 0.0014 (singleton FP) | ~0.5 day, no reruns upstream | (c) |
| 2 | **F1** Sibling features (compare pool candidates with each other) | 3 | part of the 0.0122 missed-match loss; 26.5% of FN rows are exact-name + empty pool address | ~0.5 day | (c) |
| 3 | **F2** TF-IDF char n-gram cosine for name and address | 3 | missed matches + FPs | ~0.5 day | (a) best name signal in EDA, hard-neg AUC 0.7497 |
| 4 | **B0** Split blocking misses into "stage 1 never found it" vs "stage-2 cut dropped it" | 1–2 | measures the 0.0063 bucket | 1–2 h, diagnostic only | — |
| 5 | **R1** Wider stage-2 cut (K_min 20), **in working tree** | 2 | part of the 0.0063 blocking loss; the cut costs 0.0019 of ceiling | TEST candidates rerun ~2.5 h | (a) ceiling 0.9946 → 0.9926 at the cut |
| 6 | **B1** Domain-name segmentation key | 1 | 9.2% of blocking-miss rows | ~1 day + stage-1/2 rerun | (c) |
| 7 | **B2** Name-only channel limited to empty-address pool records | 1 | 38.5% of blocking-miss rows have an empty pool address | ~0.5 day + rerun | (c) |
| 8 | **C1–C4** Country-pattern features and a France guard | 3 | India trails US by 0.0072 F0.5; France (15% of test) is unmeasured | ~1 day | (a) for the gap, (c) for the fix |
| — | Stop investing in: one-owner variants, per-segment thresholds, unit/PO-box regex alone | | measured ≈ 0 or too rare | | (a) |

What the evidence labels mean (from `AGENTS.md`): **(a)** measured on our data, with a metrics file;
**(b)** someone else's published number on their data; **(c)** should work from how the algorithm
works, not measured; **(d)** a guess.

---

## 1. Background for readers new to this pipeline

### 1.1 The task in one paragraph

Each S1 record (business name, address, country) must be linked to every record in S2 and S3 that is
the same real business. S2 + S3 together are the **pool**: 10.3M records for train and 10.0M for test.
Some S1 records have no match at all; these are **singletons**, 5.6% of train S1 (`data_summary.json`).
The score is **F0.5 computed per S1 record, then averaged over all S1 records**.

### 1.2 Why F0.5 per entity behaves the way it does

F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP). A false positive (FP) costs 4× as much as a missed match
(FN) inside the formula, so F0.5 is precision-weighted. Per entity, for an S1 with 4 true matches:

| What we predict | F0.5 for that entity |
|---|---|
| all 4, nothing else | 1.000 |
| 3 of 4 (one missed) | 0.938 |
| all 4 + one wrong | 0.833 |
| **nothing (empty)** | **0.000** |
| (singleton) nothing | 1.000 |
| (singleton) anything | 0.000 |

Two consequences drive this document:
1. Missing one of several matches is cheap, but **predicting nothing for a matched S1 costs the whole
   entity**.
2. Any prediction for a singleton costs the whole entity.

### 1.3 The pipeline today

```
raw TSVs
  └─ textnorm v2.1: Unicode fold, Indic romanization, abbreviation expansion (mined per country)
STAGE 1  blocking / candidate generation
  └─ 9 key families (name tokens, name n-grams, name pairs, address tokens, cross name×address key…)
     inverted index over the pool → ~264–291 candidates per S1, quota per channel
STAGE 2  re-ranker
  └─ LightGBM on 35 cheap features → p_rerank → cut (K_min, K_max, p_min) → ~5.4 candidates per S1
STAGE 3  matcher
  └─ LightGBM on 50 pair + list features → isotonic calibration → threshold 0.70 + one-owner
     → matching_results.tsv
```

Terms used below:
- **Blocking / candidate generation**: cheap retrieval that picks a short list of pool records per S1,
  so the expensive model never scores all 10M.
- **Recall ceiling**: the best F0.5 a perfect matcher could reach given the candidates. A true match
  that blocking drops can never be recovered later.
- **Bi-encoder**: encodes each record separately into a vector; matches are nearest vectors. Fast
  enough for retrieval.
- **Cross-encoder**: reads both records together and outputs one score. More accurate per pair, but
  far too slow for retrieval; only usable on a short candidate list.
- **OOF (out-of-fold)**: every MATCH_TRAIN row is scored by a model that did not train on its S1.
- **Calibration**: making "p = 0.8" mean "80% of such pairs are true". The isotonic map is fit on OOF
  predictions.

### 1.4 Where the 0.0259 of remaining loss sits (OOF, MATCH_TRAIN, `match_train.json`)

| Bucket | Entities | Loss | Share |
|---|---|---|---|
| Matched S1, some true matches missed, no FP | 16,966 | 0.0122 | 47% |
| Blocking miss only (true match never reached stage 3) | 11,296 | 0.0063 | 24% |
| Matched S1, FP only | 3,895 | 0.0051 | 20% |
| Singleton with any FP | 248 | 0.0014 | 5% |
| Matched S1, FP and FN | 505 | 0.0009 | 4% |

**A number the plan did not state.** From `match_train.json`: 6.08% of S1 get an empty prediction,
which is 10,937 entities. Singletons correctly left empty: 0.9757 × 10,199 = 9,951. So **986 matched S1
received an empty prediction**. Each scores 0, which adds up to 986 / 180,000 = **0.0055 of loss, 21%
of the total**. Some of these have no true match among their candidates (blocking). Val-S1 entity recall
is 0.9984 (`rerank_eval_HOLDOUT_TEST.json`), which puts that at roughly 300 entities. The other ~680
(~0.0038) had a true match in hand and still got an empty answer. This is derived arithmetic from
metrics files, not a direct count. Saving OOF predictions would turn it into one (see N1).

Segments (OOF): US 0.9770, India 0.9698; chain size 1: 0.9803, chain ≥ 10: 0.9636; **S1 with exactly 1
true match: 0.9132** (9,692 entities), the weakest segment.

Model importance (`match_train.json` `feature_gain`): `p_rerank` 62%, list-context features about 20%,
all name/address similarity features together about 18%. The matcher mostly re-uses the stage-2
score. That is why new pair features have room to help.

### 1.5 What the errors look like (`error_patterns.json`)

| Pattern (share of rows) | False negatives (19,275) | False positives (4,946) | Blocking misses (15,117) |
|---|---|---|---|
| Pool address empty | **40.7%** | 19.4% | **38.5%** |
| Name near-exact (token_set ≥ 95) **and** pool address empty | **26.5%** | 12.0% | 15.4% |
| "Alias" name: name token_set < 50 but address token_set ≥ 80 | 13.1% | 18.7% | 12.7% |
| Domain-style pool name (`foo.com`) | 3.8% | 2.9% | **9.2%** |
| Pool name in non-Latin script | 3.6% | 4.2% | **11.3%** |
| Leading-zero number in pool address (`01092`) | 10.8% | 5.0% | 6.5% |
| Ordinal word in pool address (`Eleventh`) | 1.5% | 1.3% | 1.4% |
| Compound house-id conflict (`8-1-293/B/4/4` vs `…/13`) | 2.8% | 4.8% | — |
| Unit/suite number conflict | 0.3% | 0.7% | — |
| S1 country India | 38.9% | 42.6% | **59.5%** (India is 40% of S1) |

For comparison, 3.3% of all pool records have an empty address (`eda_report.md`). The share among
candidate rows is not measured yet; the `a_empty` feature can measure it.

How to read the table: a pattern helps if it separates FN from FP. **Empty address + exact name** is
2.2× more common in FNs than in FPs, so the model under-trusts these pairs. **Alias names** appear in
both, so they are ambiguous, not under-trusted. **House-id conflict** is more common in FPs (4.8%) than
FNs (2.8%), but it fires in fewer than 5% of either, so its ceiling is small.

---

## 2. Tip 1: "Invest in a strong blocking/candidate generation strategy"

### What exists

- Stage 1 (`s1_E4.json`, DEV_A): pair recall 0.9854 and entity recall 0.9993 at a mean of 291
  candidates.
- After the stage-2 cut (`rerank_eval_HOLDOUT_TEST.json`, val-S1, scored once): pair recall 0.9755,
  **full recall** (every true match of the S1 found) 0.9217, ceiling 0.9928.
- The cut costs 0.0019 of ceiling: 0.9946 before, 0.9926 after (`rerank_cut.json`, RR_CAL).
- Weak segments at val-S1: India pair recall 0.9635 (US 0.9836); pool name in Indic script 0.9388;
  chain ≥ 10 0.9371.
- Blocking misses cost 0.0063 of OOF F0.5, which stage 3 cannot recover.

### B0. Diagnostic first: where was each blocking miss lost?

- **What:** for each MATCH_TRAIN blocking miss, record whether stage 1 retrieved the pair (it was in
  the ~290) and the stage-2 cut dropped it, or whether stage 1 never retrieved it. Also record which
  key family retrieved it, if any.
- **Why first:** the two causes need different fixes: a wider cut or better re-ranker features in one
  case, new keys in the other. Today `train_blocking_misses.csv` mixes both.
- **Outcome:** a table of blocking-miss share by cause and by pattern (section 1.5). No score change.
- **Cons:** needs stage-1 output for MATCH_TRAIN, which is recomputed and not stored. That is ~15 min
  (`candidates_MATCH_TRAIN.json` 938 s).
- **Cost:** 1–2 h.

### B1. Domain-name segmentation key

- **Problem:** 9.2% of blocking-miss rows have pool names like `fortuneone.com`, `metalbestcustom.com`,
  `5py.com`. Normalization produces one token, `fortuneone com`, which shares no token or bigram with
  `One Fortune Trading Private Limited`. Word order is often flipped inside the domain.
- **Mechanism:** strip the TLD (`.com`, `.in`, `.fr`, …). Split the remaining string into words by
  **Viterbi word segmentation**: dynamic programming over split points that maximizes the sum of
  log-unigram frequencies. The frequencies come from pool name tokens (unsupervised, per split). Then
  emit the segmented tokens as sorted-token keys, so `fortune one` and `one fortune` produce the same
  key.
- **Expected outcome:** recovers part of the 9.2% (≈ 1,400 MATCH_TRAIN pairs), (c). Upper bound on F0.5:
  well under the whole 0.0063 bucket, because a domain name usually covers one of several matches for
  that S1.
- **Architecture change:** a new key family `DM` in `keys.py`, a new quota channel, plus the same
  segmented tokens as a stage-3 feature (F7).
- **Cons:** segmentation is ambiguous for short strings (`5py` → `5 py`?). Common words (`best`,
  `metal`) create large postings lists, so they need the existing `cap_frac` cap. Stage 1 → 2 → TEST
  rerun (~2.5 h).

### B2. Name-only channel limited to pool records with an empty address

- **Problem:** 38.5% of blocking-miss rows have an empty pool address. These records can only be found
  by name. The E4 grid chose a general NAME quota of 0 (`s1_E4.json`): name-only candidates across
  all pool records did not raise the overall ceiling enough for their list-size cost.
- **Mechanism:** a separate quota (e.g. 10–20) filled only from pool records whose address is empty,
  ranked by name-key score. The `ADDR` and `XK` channels cannot find these records by construction, so
  this channel adds candidates no other channel can.
- **Expected outcome:** (c). Common names (`Primary Care Group`, 253 S1 entities) will flood the
  channel. The existing `cap_frac` limits that, and stage 2 needs a feature `pool_addr_empty` to rank
  these candidates fairly.
- **Architecture change:** config quota plus a filter in `retrieve.py`; E4-style re-tune of the quota.
- **Cons:** raises mean candidate size, and the matcher then faces the hardest pairs (see F1). Needs a
  full rerun.

### B3. Address-only channel for alias names

- **Problem:** 12.7% of blocking misses have an unrelated pool name (`Ariakelo`, `Rizaflux`,
  `DELTAVANTAGEARC`) but a near-identical address. The `ADDR` channel (quota 50) exists. B0 will show
  whether stage 1 finds these and the cut drops them (then the fix is **R3**) or whether stage 1 misses
  them.
- **Mechanism (if stage 1 misses them):** a strict address key = canonical house number (leading zeros
  stripped) + rarest street token + last admin token (city). It retrieves only records at the same
  building.
- **Cons:** office buildings and malls host many businesses, so this key must be capped by postings
  size. Alias pairs are also 18.7% of FPs: finding more of them raises recall but gives the matcher
  more ambiguous pairs.

### B4. Better transliteration for Indic-script pool names

- **Problem:** pool names in Indic script have pair recall 0.9388 and ceiling 0.980 at val-S1, against
  0.9755 / 0.9928 overall. 11.3% of blocking-miss rows have a non-Latin pool name. Example: `శివ్
  సాఫ్ట్‌వేర్ ప్రైవేట్ లిమిటెడ్` vs `Shiv Software Private Limited`.
- **Mechanism options:**
  - (i) Extend the learned token dictionary (`translit_dict.json`, 445 name entries) by mining more
    aligned token pairs from train positives.
  - (ii) Emit the consonant-skeleton key (`SK` family, exists but off, E2) only for Indic-origin
    records. Vowel spelling is the main transliteration noise; the skeleton drops vowels.
- **Evidence:** the dictionary raised the usable-token share on DEV_A from 0.2896 to 0.4847
  (`translit_dict.json`, (a)). The effect of SK restricted to Indic records is not measured.
- **Cons:** SK as a general family showed no gain in E2. Limiting it to Indic records makes the key
  depend on a script detector, not on country, so it stays legal under `AGENTS.md` rule 3.

### B5. MinHash-LSH on character shingles

- **Mechanism:** represent each name as a set of character 3-grams. MinHash approximates Jaccard
  similarity with k hash minima. Banding (b bands of r rows) makes two records collide in some band
  with probability 1 − (1 − J^r)^b, which is a steep S-curve around a chosen Jaccard threshold. This
  catches typo-heavy names (`Golden Miolity Miolity Works`, `Pnncidle`) that share few whole tokens.
- **Evidence:** (b) standard near-duplicate detection (Broder 1997; Leskovec et al., *Mining of Massive
  Datasets*, ch. 3). On our data: none. The `NG` (name n-gram) family may already cover most of it, and
  B0 will show whether typo-only misses are frequent.
- **Cons:** new library (`datasketch`, MIT) or ~150 lines of numpy. Memory for 10M × 64 hashes ≈ 5 GB at
  uint64, so it needs uint32 plus chunking. Low priority until B0 shows typo misses matter.

### B6. Dense bi-encoder retrieval (needs team approval: changes the model family)

- **Mechanism:** a small multilingual sentence encoder (e.g. `paraphrase-multilingual-MiniLM-L12-v2`,
  Apache-2.0, 118M params) embeds each record. FAISS (MIT) finds the nearest pool vectors. Near vectors
  can catch transliteration and synonyms that shared-token keys miss.
- **Evidence:** (d) for this data. The bucketing plan notes that the allowlisted embedding models are
  English-only.
- **Cons:** embedding ~20M records on an 8-core CPU without a GPU would take many hours (d). The model
  file must ship. Not recommended before B0–B4.

---

## 3. Stage 2: re-ranker and cut (connects tip 1 to tip 4)

### R1. Wider cut, in working tree (`configs/blocking.json`: K_min 5 → 20)

- **Mechanism:** keep at least 20 candidates per S1 instead of 5. The config note records that
  candidate-list size is not scored by the contest. The ceiling then approaches stage 1's (0.9946 vs
  0.9926, `rerank_cut.json`).
- **Outcome:** up to ~+0.002 of ceiling (a). The realized F0.5 gain depends on the matcher: the extra
  candidates are mostly negatives, so the positive rate in lists falls from 0.62 toward ~0.2.
- **Cons:** stage-3 rows ×~3.7, TEST candidates rerun (~2.5 h), and the matcher must be retrained on the
  new list distribution. List features (`l_*`) change meaning: `l_n` is ~20 for everyone.

### R2. Train the re-ranker longer

- **Fact:** stage 2 hit its 400-round cap without early stopping (`rerank_train.json`, (a)).
- **Outcome:** (d) small logloss gain, which feeds the 62%-importance `p_rerank` feature.
- **Cons:** changes `p_rerank` everywhere, so stage 3 must be retrained and candidates regenerated.
  Batch it with R1 so the ~2.5 h rerun happens once.

### R3. Reserved slots past the cut

- **Mechanism:** after the cut, always keep the best address-only candidate and the best empty-address
  name-only candidate, even if their `p_rerank` is below the cut. The matcher decides.
- **Outcome:** targets the alias and empty-address blocking misses if B0 shows the cut drops them. (c)
- **Cons:** with K_min = 20 (R1) this may already happen. Measure after R1.

### R4. Pass stage-2 features to stage 3 (MATCHER_PLAN M3)

- **Mechanism:** dump the 35 stage-2 features (per-family key scores, coverage) with the candidates, so
  the matcher sees why a pair was retrieved, not only the final `p_rerank`.
- **Outcome:** (c). Keep only if OOF F0.5 improves by ≥ 0.001 (MATCHER_PLAN rule).
- **Cons:** needs a `--dump-features` flag and a candidates rerun.

---

## 4. Tip 2: "Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine)"

### What exists (`match.py`, 50 features in the shipped model)

- **Name:** rapidfuzz `token_set_ratio`, `token_sort_ratio`, `ratio`, `partial_ratio`; Jaro-Winkler on
  sorted tokens; token Jaccard; IDF-weighted Jaccard and coverage; highest-IDF token missing on each
  side; exact / core (edge tokens removed) / no-space equality; acronym; consonant-skeleton Jaccard.
- **Address:** token set/sort ratio, IDF Jaccard, number Jaccard, house number, ≥ 3-digit number
  shared/conflict, empty state.
- **In working tree, not yet measured:** `a_partial`, `a_char3_jacc`, `a_num_minlev`, `a_numstreet`,
  `n_shared_maxidf`, `a_shared_maxidf`, pool counts of the same name/address (`q_/p_name_cnt`,
  `q_/p_addr_cnt`), `l_n_addr_close`.
- Numbers are already compared with leading zeros stripped (`_canon_num`). The text-level token
  features still see `01092` and `1092` as different tokens.

### EDA ranking of single signals (`eda_report.md`, hard negatives = name-similar non-matches)

Address signals separate hard negatives best: token Jaccard AUC 0.972, TF-IDF 0.964. The best **name**
signal is **TF-IDF char n-gram cosine, AUC 0.750**. Next come Soundex set-Jaccard 0.711, Metaphone
0.706, token Jaccard 0.701, Levenshtein 0.657. **TF-IDF cosine is not in the matcher.**

### F2. TF-IDF char n-gram cosine (name and address)

- **Mechanism:** each string becomes a sparse vector of character 2–4-gram counts, weighted by
  IDF = ln(N / df) with df counted over the **pool of the same split** (unsupervised). Cosine of the two
  vectors: rare shared fragments count a lot, common fragments (`ltd`, `road`) count little, and typos
  change only a few n-grams.
- **Outcome:** (a) highest single-name-signal AUC in our EDA; gain on top of the current 50 features
  is (c).
- **Architecture change:** fit `HashingVectorizer(analyzer="char_wb", ngram_range=(2,4))` +
  `TfidfTransformer` on the pool once per split. Compute row-wise cosine only for candidate pairs, so
  no dense S1 × pool matrix.
- **Cons:** the df table is ~2^20 floats (small). Per-pair sparse dot products in Python are slow, so
  vectorize per chunk. Train and test pools get different IDF, which is correct (same definition) but
  should be checked with the monitor.

### F3. Monge-Elkan with Jaro-Winkler inner similarity

- **Mechanism:** for every token in name A, take its best Jaro-Winkler match among tokens in name B,
  then average; symmetrize by averaging both directions. Each token may have its own typo, and token
  order does not matter. Covers `Aveneu`/`Avenue`, `Tcaoma`/`Tacoma`, `Brown Gulley Robinson PLLC
  CENTER`.
- **Evidence:** (b) Cohen, Ravikumar & Fienberg (2003) found Monge-Elkan and "soft TF-IDF" (the same
  idea with IDF weights) among the best name-matching measures on their benchmarks. (c) for our data.
- **Cost:** `rapidfuzz.process.cdist` on ≤ 10 × 10 token matrices, cheap.
- **Cons:** overlaps with `token_set_ratio` + JW. Expect a small gain; the paired test (section 8)
  decides.

### F4. Phonetic codes (Soundex, Double Metaphone)

- **Mechanism:** map each token to a code of how it sounds (`Smith`/`Smyth` → `S530`), then take
  Jaccard of the code sets.
- **Evidence (a):** hard-negative AUC Metaphone 0.8065 on US but 0.6146 on India; Editex 0.5220 on
  India (`eda_report.md`).
- **Cons:** English-centric. Worth adding for US; expect little for India and nothing reliable for
  France. **Do not add Editex.** jellyfish (MIT) is already a dependency.

### F5. Name-number and name-numeral conflict (heuristic)

- **Problem (FP examples):** `Human Rights Council of Sylva II LLC` vs `… IC LLC`; `Piedmont Charities
  IV` vs `IIV`; `DO Strategic Clearthink` vs `Fischer Strategic Clearthink`.
- **Mechanism:** extract digits and Roman numerals (regex `\b[ivxlc]+\b` checked against a valid
  Roman-numeral pattern) from both names. Output 3 states: equal / conflict / absent, NaN if both are
  absent. Related existing feature: `n_q_miss_maxidf` (rarest unmatched token).
- **Outcome:** (c). These are chain cases (same name, different branch), where name features are ~1.0
  for both true and false pairs.

### F6. Leading-zero and ordinal canonicalization at the text level

- **Problem:** 10.8% of FN rows have a zero-padded number in the pool address (`Flat No. 00502`,
  `014700 Second St`); 1.5% have ordinal words (`205 Eleventh Street` vs `224 11th Street`).
- **Mechanism:** in the feature path only (not in textnorm, so stage 1 is untouched), strip leading
  zeros from digit tokens and map ordinal words to numbers (English `first…ninety-ninth` → `1…99`;
  French `premier/première, deuxième…` for France). Then recompute the address token features.
- **Outcome:** (c). The numeric features already strip zeros, so the gain is limited to the text-token
  features (`a_tset`, `a_idf_jacc`).
- **Cons:** `1st` vs `first` is language-dependent. The rule set is small and generic, not a gazetteer.

### F7. Domain-name features

- **Mechanism:** reuse B1's segmentation. Features: `p_is_domain`, and the token-set ratio between the
  S1 name tokens and the segmented domain tokens.
- **Outcome:** (a) 3.8% of FN rows are domain-style; the gain is (c).

---

## 5. Tip 3: "Pay attention to country-specific address patterns"

### The constraint

`AGENTS.md` rule 3: country is an open set; no one-hot, no hard-coding US/India, no filtering. Test
has **France: 259,452 of 1,732,544 test S1 = 15.0% of the test score**, with no training labels
(`data_summary.json`). India trails US by 0.0072 OOF F0.5.

The way to use country patterns without breaking the rule: **write each feature as a pattern
detector that is defined the same way for every row.** A regex that finds `Plot No. 815/A` fires
wherever that pattern appears, whatever the country label says. If the pattern is absent, the feature
is NaN, and LightGBM routes NaN down its own learned branch. For France, most India/US patterns will be
NaN, so the model falls back to the generic features. That is the intended behavior, and it is
checkable (C4).

### C1. Pattern features (all return NaN when the pattern is absent on either side)

| Feature | Regex idea | Pattern seen in | Measured headroom (`error_patterns.json`) |
|---|---|---|---|
| Compound house id eq / conflict | `[a-z]{0,2}\d+[a-z]?([-/][0-9a-z]+)+` → `8-1-293/b/4/4`, `815/a`, `h2-87` | India | conflict in 4.8% of FP rows vs 2.8% of FN rows; separates, but rarely fires |
| Unit / suite / floor eq / conflict | `(apt|suite|ste|unit|fl|floor|bldg|rm|#)\s*([0-9a-z]+)` | US | conflict in 0.7% of FP rows. Too rare alone |
| PO Box eq / conflict / one-sided | `p\.?o\.?\s*box\s*(\d+)` | US | one-sided in 1.5% of FN rows (`800 Arnold St, PO BOX 1709`) |
| House-number fraction | `\d+\s+1/2` (`111 1/2 Robinbrooke`) | US | rare (d) |
| Landmark clause removed | cut the address at `near|opp|opposite|behind|beside|adjacent` up to the next comma | India | 2,552 of 44K EDA positives contain landmarks (`eda_report.md`) |
| Sector / block / phase / ward number | `(sector|sec|block|blk|phase|ward|khasra|gut|survey)\s*(no\.?)?\s*([0-9a-z]+)` → key–value compare | India | (d) |
| Postal code eq / conflict | 5 digits at the end or before a city (US ZIP, France CP), 6 digits (India PIN); plus `cedex` | all 3 | EDA 3-state postal AUC 0.664 (weak); S1 rarely carries ZIP/PIN |
| `bis` / `ter` number suffix | `\d+\s*(bis|ter)\b` | France | none (no labels) |
| Admin area agreement | last 1–2 address tokens after abbreviation mapping, compared by skeleton | all | see C2 |

How to use it: add all C1 features as one experiment, not one per run. Each alone is too rare to pass
the adoption bar (section 8).

### C2. Fix the mined state-abbreviation maps

The mined maps (`work/norm/abbrev_{train,test}.json`) already map `il → illinois`, `ka → karnatak`,
France `bd → boulevard`, and so on. Inspection shows errors that hurt admin-area agreement:
- `fl → floor` in US: Florida is lost.
- `dl → dilli`, while S1 writes `delhi`.
- `ka → karnatak` / `mh → maharashtr`, while S1 writes `karnataka` / `maharashtra`. The map target is
  the romanized pool spelling, not the S1 spelling.
- test US `b → blue`.

Two fixes:
- Compare admin tokens with the consonant skeleton or JW ≥ 0.9, not by equality.
- Require the mined target to also appear in S1 text. This is still data-mined, so it is not a
  gazetteer.

Gain: (c). Cons: these maps feed stage 1, so changing them there means a full rerun. Doing it only in
the stage-3 feature path is cheap.

### C3. Legal-suffix handling for France

This exists: edge tokens are mined from data (`edge_tokens` config), so `sarl`, `sas`, `sasu`, `eurl`,
`sci` get the same treatment as `llc`/`pvt` without a hand list. Verify on the test pool that these
tokens are in the mined edge set. This is a structural input check, allowed by the test inspection log
rules. Log it in `reports/test_inspection_log.md`.

### C4. France guard: we cannot measure France, so bound the risk

- **LOCO as a proxy.** Train the matcher on US-only MATCH_TRAIN and score India, then the reverse. A
  feature that helps in-country but hurts cross-country is acting as a stand-in for country. Reject it
  (VALIDATION_PLAN §6 adoption rule, LOCO guard). Precedent: stage-2 LOCO drop was ≈ 0.004 US → India
  (`rerank_loco.json`).
- **Monotone constraints (LightGBM `monotone_constraints`).** Force p to be non-decreasing in
  similarity features (name/address ratios, IDF overlap). On an unseen distribution such as France, the
  model then cannot learn quirks like "p falls as address similarity rises in this bin". (c). Cons: the
  true relation is not monotone everywhere (exact name in a chain ≥ 10 group is weak evidence), so
  constrain only the similarity features, not counts or list features.
- **Monitor** (`monitor.py`, exists): compare per-country predicted-list size, share of empty
  predictions, and mean max p between France and train countries on test. Check this, but it is not
  tuning: `AGENTS.md` forbids tuning on test.

---

## 6. Tip 4: "Consider the precision-recall trade-off carefully"

### What exists

- Threshold 0.70 on calibrated p, chosen by entity F0.5 on OOF. Pairwise precision 0.9916, recall
  inside candidates 0.968 (`match_train.json`).
- Expected-F0.5 top-k rule: exact computation from the Poisson-binomial distribution of the number of
  true matches. OOF 0.97433 vs threshold 0.97412 (both with one-owner). **+0.0002**, below the
  adoption bar.
- One-owner filter (a pool record goes only to the S1 with the highest p): paired Δ +0.000007 on
  MATCH_VALID (`validation_noise.json`). **No measurable effect; stop investing in variants.**
- Calibration: reliability bins within ~0.02 of observed rates (`match_eval_MATCH_VALID.json`).

### The key measured fact

FN median p = 0.33 (`EXPERIMENT_ROUTINE.md` §2, from the OOF export). Most missed matches sit far below
any threshold that keeps precision, so **moving the threshold cannot recover them**. That takes better
features (F1–F7) or a better empty-vs-non-empty decision (N1).

### D1. Keep the loss asymmetry in the decision rule, not in training

Do not use `is_unbalance` or class weights. Positives are already 62% of candidate rows, and weighting
breaks the calibration the expected-F rule needs (`EXPERIMENT_ROUTINE.md` §2). Measured earlier:
hard-negative ×3 weighting scored 0.9050 vs 0.9063 unweighted (`ml_baseline_report.md`, (a)).

### D2. Joint decision for near-duplicate pool records

- **Mechanism:** 85.2% of matched S1 match in both S2 and S3 (`eda_report.md`). When two candidates in
  one list (one from S2, one from S3) are near-identical to each other (name + address), decide them
  together: both in or both out, using the higher calibrated p.
- **Outcome:** (c). Fixes cases where one copy is accepted and its twin missed.
- **Cons:** errors come in pairs too. A wrong twin decision now costs 2 FPs. Overlaps with F1.

### D3. Per-segment thresholds: not recommended

Expected-F already adapts the cut per list. Per-country thresholds would violate rule 3. Per-chain
thresholds add parameters that are tuned on the same OOF, which risks overfitting it.

---

## 7. Tip 5: "Do not neglect singletons", and the other half of the same problem

### What exists

Singletons are 5.67% of MATCH_TRAIN S1. Singleton accuracy is 0.9757 (248 singletons given an FP:
loss 0.0014).

### N1. Entity-level "does this S1 have any match?" model (MATCHER_PLAN M6, not implemented)

- **Why it is bigger than the plan says:** MATCHER_PLAN caps M6 at the singleton-FP loss. By section
  1.4, **986 matched S1 were given an empty prediction (~0.0055 loss)**. The same decision, "empty or
  not", governs both. The total headroom is ≈ 0.0014 + up to ~0.004, so it is the largest single
  bucket a stage-3-only change can attack.
- **Mechanism:** one row per S1 with list features: max p, top-2 p, sum p, count of name-close /
  address-close candidates, `log_chain`, whether any candidate has a non-empty matching address, and
  the S1's own name frequency in the pool. Target = "S1 has ≥ 1 true match". LightGBM binary, grouped
  OOF like the pair model.
- **How it connects:** the expected-F0.5 rule currently estimates P(no match) as ∏(1 − p_i), which
  assumes independent candidates. Candidates in one list are not independent (S2/S3 twins). Replace
  that k = 0 term with the entity model's calibrated P(no match). The rule then compares "predict
  nothing" against "predict top-k" with a better estimate.
- **Outcome:** (c). Cap ≈ 0.005 of F0.5.
- **Cons:** a second model to calibrate. Entities with all true matches outside the candidates (~300)
  stay wrong either way.
- **Prerequisite:** save OOF pair predictions to a parquet, so the 986 split into "true match was in
  candidates" vs "was not" becomes a count instead of an estimate.

### F1. Sibling (pool-to-pool) features, the main fix for "exact name, empty address"

- **Problem:** 26.5% of FN rows are a near-exact name with an empty pool address (`Harbor Sóciety Inc`,
  `Heritage Printing Networks`). The model under-trusts them (12.0% of FP rows vs 26.5% of FN rows).
  An empty address alone cannot tell the right `Jones Industries` from another.
- **Mechanism (transitivity):** for candidate c, compare c with the other candidates in the same list,
  not only with S1. If c's name equals the name of another candidate d, d comes from the other source,
  and d is confirmed by address (high `a_tset`, house number equal), then c is very likely the twin
  copy of the same business. Features:
  - `sib_best_p`: highest pair-model p among candidates whose name matches c
  - `sib_name_sim_to_best`: c's name similarity to the list's top candidate
  - `sib_other_src_confirmed`: does the other source have an address-confirmed candidate with c's name
  - `n_same_name_in_list`
- **Outcome:** (c). This targets the largest FN pattern. `a_empty` has zero gain in the model today, so
  the model has no signal that resolves these pairs.
- **Architecture change:** a second pass of stage 3. Pass 1 computes p. Pass 2 adds sibling features
  from pass-1 **OOF** p (never in-sample, or the feature leaks the label) and retrains. List size ~5–20,
  so pairwise pool–pool comparison is cheap.
- **Cons:** two-pass training doubles stage-3 time (~17 → ~35 min). If the sibling is itself an FP, the
  error spreads. Chain ≥ 10 lists may contain several same-name records from different branches, where
  transitivity is wrong; `n_same_name_in_list` lets the model learn that.

### N2. Name specificity for empty-address pairs, in working tree

`q_name_cnt`/`p_name_cnt` (number of pool records with the same core name) are implemented but not
measured. They cover the other half of the empty-address problem: a name that appears once in 10M
records is strong evidence alone, a name that appears 400 times is not.

---

## 8. Fuzzy, heuristic and regex features: how to add them without making the model worse

"Without degradation" is a measurable claim. The rule already fixed in `VALIDATION_PLAN.md` §6:

1. **Same folds, paired comparison.** Both arms are scored on the same S1 with the same 5-fold split
   (`fold_seed` 42). Compute the per-entity F0.5 difference and bootstrap its mean. Paired SE is far
   below the unpaired SE (0.0007 at 20K, `validation_noise.json`).
2. **Adopt** only if paired Δ ≥ +0.0005 with CI lower bound > 0.
3. **Guard segments:** no drop worse than 0.002 (CI upper bound < 0) in `country_US`, `country_India`,
   singletons, or LOCO in either direction. The LOCO guard is our only proxy for France.
4. **Never** tune on val-S1 or test. Val-S1 is scored once per frozen pipeline.

Feature-definition rules that prevent silent damage:

| Rule | Why |
|---|---|
| Pattern absent → NaN, never 0 | 0 means "conflict" or "no similarity"; NaN means "no information". An S1 without a unit number must not look like a unit mismatch |
| Any statistic (IDF, df, n-gram weights, pool counts) is fit on the **pool** of the same split, never on labels | unsupervised, identical definition for train and test (`split.md` rule 5) |
| No feature reads the country label; regexes fire on text patterns | `AGENTS.md` rule 3 |
| Features computed from other predictions (F1 sibling, N1) use **OOF** predictions only | in-sample p leaks the label and inflates OOF |
| Add related features as one group (C1 as one run) | single rare features cannot pass +0.0005, and one test per feature multiplies false adoptions |
| Keep a feature only if it passes the paired test; otherwise delete its code path | LightGBM usually ignores a useless feature, but not always, and it costs featurize time on 10.6M test rows |

What LightGBM already tolerates: redundant features (correlated ratios) rarely hurt a tree model, so
the risk from adding a group is mostly a small overfit plus compute time. The real degradation risks
are **leakage** (in-sample predictions, label-derived statistics) and **country proxies** (features
that are always NaN for one country). Rules 3 and the LOCO guard catch these.

### Candidate feature list (all for stage 3; ✓ = exists, WT = in working tree, new = proposed)

| Group | Feature | Status | Evidence |
|---|---|---|---|
| Name fuzzy | token_set / sort / ratio / partial, JW sorted, Jaccard, IDF Jaccard | ✓ | (a) in model |
| Name fuzzy | TF-IDF char 2–4-gram cosine (F2) | new | (a) EDA AUC 0.750 |
| Name fuzzy | Monge-Elkan JW (F3) | new | (b) Cohen et al. 2003 |
| Name phonetic | Double Metaphone / Soundex set Jaccard (F4) | new | (a) US AUC 0.81, India 0.61 |
| Name heuristic | numeral / Roman numeral conflict (F5) | new | (c) |
| Name heuristic | domain segmentation similarity (F7) | new | (a) 3.8% of FN |
| Name heuristic | pool name frequency (N2) | WT | (c) |
| Address fuzzy | TF-IDF char cosine (F2) | new | (a) EDA AUC 0.964 |
| Address fuzzy | char 3-gram Jaccard, partial ratio | WT | (c) |
| Address canonical | leading zeros + ordinal words in token features (F6) | new | (a) 10.8% / 1.5% of FN |
| Address regex | house-id, unit, PO box, landmark cut, sector/block, postal, bis/ter (C1) | new | (a) headroom < 5% of rows each |
| Address | admin area agreement with fixed mined maps (C2) | new | (c) |
| List / graph | sibling features (F1) | new | (c), targets 26.5% of FN |
| Entity | P(no match) model (N1) | new | (c), cap ≈ 0.005 |

---

## 9. Suggested order of work

Stage-3-only changes run on frozen candidates in ~20 min per experiment (`EXPERIMENT_ROUTINE.md` §1).
Changes upstream cost ~2.5 h of reruns each, so batch them.

1. Save OOF pair predictions (`work/match/oof.parquet`). This turns the 986-entity estimate into a
   count.
2. Measure the working-tree features (N2, `a_char3_jacc`, …) against the 0.9741 baseline with the
   paired test.
3. **N1** entity model, then **F1** sibling features, then **F2** TF-IDF cosine. Each is one experiment.
4. **C1 + C2** as one "country-pattern" group, checked with the LOCO guard; add monotone constraints
   (C4) and check LOCO again.
5. **B0** diagnostic, then decide between R3 (cut problem) and B1/B2/B3 (retrieval problem). Batch R1,
   R2 and any stage-1 change into **one** upstream rerun.
6. Freeze, score val-S1 once (T3), write the headline from `match_eval_HOLDOUT_TEST.json`.

## 10. Limits of this analysis (flat)

- All error-pattern shares come from regexes and raw-text rapidfuzz on the OOF exports at the shipped
  rule (threshold 0.70 + one-owner). They describe the errors of **that** model. The regexes are
  heuristic, and each count includes some rows that are not truly the named pattern.
- The 986 "matched but empty" entities are derived from two rates in `match_train.json`. The ~300 / ~680
  split uses val-S1 entity recall as a stand-in for MATCH_TRAIN's.
- France quality is unmeasured. Every France claim here is (c) or (d).
- No approach here has a measured F0.5 gain on our data yet. The "loss it goes after" column is an
  upper bound, not an expected gain.

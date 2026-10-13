# Prathamesh's stage-3 matcher experiment log

Purpose: hand this to the training machine so every ADOPTED change can be re-implemented from scratch on
a clean checkout, and every REJECTED change is not repeated. Numbers are copied from
`artifacts/metrics/*.json` in this repo (never estimated). Full detail, protocol and per-run ledger:
`plan/FEATURE_EXPERIMENTS.md`. Code lives in
`submission/code/business_entity_resolution/src/ber/match.py`.

Baseline this log starts from: **submission 1**, 50 features, OOF F0.5 (MATCH_TRAIN, 180,000 S1)
**0.97412**, MATCH_VALID (20,000 S1) **0.97449**.

Current best after all experiments below: **x1_strsim**, 68 features, OOF **0.97807**, MATCH_VALID
**0.97848**. Net gain over submission 1: **+0.00399 MATCH_VALID**.

Adoption rule used for every experiment (do not skip any clause when re-testing on the training machine):
1. Paired OOF delta (same 5-fold GroupKFold, same 180,000 MATCH_TRAIN S1, 95% bootstrap CI over per-entity
   score differences) **>= +0.0005** and CI lower bound **> 0**.
2. MATCH_VALID paired delta CI upper bound **>= 0** (confirms; MATCH_VALID never selects).
3. No guard segment (`country_US`, `country_India`, `singleton`) drops by more than 0.002 with its CI
   upper bound < 0.
4. Any change that reuses a first-pass model's own predictions as a feature (stacking) needs an
   additional condition: the MATCH_VALID CI lower bound itself must be **> 0** (not just >= 0), because
   pass-1 p already carries most of the signal and small OOF gains from stacking do not reliably repeat.

---

## PART A — ADOPTED (implement all of this)

### A1. Leading-zero digit canonicalization (bug fix)

**Problem measured**: pool addresses zero-pad house/door numbers (`0412 Brunswick Dr`, `Door No. 0381`)
where the S1 record writes `412` / `381`. Every digit-token comparison treated these as different tokens
— 2.7% of MATCH_VALID true pairs with numbers on both sides shared a number only after stripping leading
zeros.

**Exact change** — one function, used everywhere a digit token is compared:

```python
def _canon_num(t):
    """Digit token without leading zeros: pool addresses write 0412 / door no 0381 for 412 / 381."""
    return t.lstrip("0") or "0"
```

Apply it inside `_numbers()` (builds the digit-token set, the >=3-digit "longnum" set, and the first
digit token) and inside `_num_street()` (number-then-word pairs used by `a_numstreet`). Every existing
digit feature (`a_num_jacc`, `a_num_q_only`, `a_num_p_only`, `a_longnum`, `a_house`) is computed from
these canonicalized sets, so the fix propagates automatically — no other feature code changes.

**Evidence**: (a) measured, 1,214 false negatives vs 19 false positives attributable to the leading-zero
gap alone (teammate's `ADDRESS_EXPERIMENTS.md` §4 S1, cross-checked against our own error export).

### A2. Four-state digit agreement code (`a_digits_state`) — new column

Fellegi-Sunter style: agreement and disagreement are different evidence, not opposite ends of one
number. Sits alongside `a_longnum` (which only covers digit tokens of length >= 3); this one covers
every digit token.

```python
def _digits_state(q_dig, p_dig):
    """4-state code on stripped digit token sets: 3 match (equal sets), 2 overlap (both nonempty, share
    >=1 token, sets differ), 1 conflict (both nonempty, disjoint), 0 one-side (exactly one side has
    digits); NaN when neither side has any digit token (no information, not 0)."""
    if not q_dig and not p_dig:
        return float("nan")
    if not q_dig or not p_dig:
        return 0.0
    if q_dig == p_dig:
        return 3.0
    return 2.0 if q_dig & p_dig else 1.0
```

Called as `_digits_state(q_dig, p_dig)` where `q_dig, p_dig` are the canonicalized digit-token sets
already computed by `_numbers()` in A1 — no extra parsing needed if A1 is in place.

**Combined result of A1+A2** (run name `k5_base`, everything else = submission 1 plus the K_min=5 cut
unchanged): OOF F0.5 **0.97517** (delta **+0.00105** vs submission 1's 0.97412), MATCH_VALID **0.97594**
(delta **+0.00145**). `artifacts/metrics/match_train_k5_base.json`,
`artifacts/metrics/match_eval_MATCH_VALID_k5_base.json`.

### A3. Pool-uniqueness counts (name and address) — 4 new columns

**Mechanism**: a name or address that appears once in the 10.3M-record pool is much stronger evidence
than one that appears 400 times. Two count tables, built once per split, unsupervised (pool text only):

```python
COUNT_KINDS = ("name", "addr")

def _count_key(kind, text, G):
    """Key whose pool frequency is counted: core name tokens (edge tokens G removed) or the full address
    token set; both sorted, so word-order transposition / component reordering share a key."""
    toks = text.split()
    if kind == "name":
        return " ".join(sorted(set(_core(toks, G))))   # _core() already exists: strips G, keeps order if empty
    return " ".join(sorted(set(toks)))
```

Build (once per split, cached with a size+mtime stamp on the normalized pool file, same pattern as the
index build's staleness guard): hash every pool record's key (`pandas.util.hash_array`), `np.unique` with
`return_inverse=True, return_counts=True`, save `{kind}_uniq.npy` (sorted hash array) + `{kind}_cnt.npy`
(count per unique hash) + `pool_{kind}_cnt.npy` (count broadcast back to every pool row, for O(1) lookup
by `pool_idx`). Query-side lookup (`query_counts`) hashes the query's own key and binary-searches
`{kind}_uniq.npy`.

Four columns, values = `log1p(count)`, **NaN if the text is empty** (not 0 — "no address" is not "count
zero"):
```
q_name_cnt, p_name_cnt, q_addr_cnt, p_addr_cnt
```
`q_*` = S1-side count (same value for every candidate of that S1 — a genuinely constant-per-list feature,
which is fine for LightGBM). `p_*` = pool-side count (varies per candidate).

### A4. Address-shape features — 6 new columns

All computed inside `pair_features()`, using the canonicalized digit/token sets from A1:

```python
def _num_street(toks):
    """(number, following word) pairs: '1612 grandview avenue' -> {('1612', 'grandview')}."""
    return {(_canon_num(a), b) for a, b in zip(toks, toks[1:])
            if a.isdigit() and len(b) >= 2 and b.isalpha()}

def _char3(s):
    s = s.replace(" ", "")
    return {s[i:i+3] for i in range(len(s) - 2)}

def _shared_maxidf(q_toks, p_toks, idf):
    """Largest idf among tokens both sides share; NaN if either side has no tokens."""
    qs, ps = set(q_toks), set(p_toks)
    if not qs or not ps:
        return float("nan")
    return max((idf.get(t, 0.0) for t in qs & ps), default=0.0)
```

Columns and their exact formulas (q_long/p_long = the >=3-digit canonicalized sets from `_numbers()`):
- `n_shared_maxidf` = `_shared_maxidf(name_tokens_q, name_tokens_p, name_idf)`
- `a_shared_maxidf` = `_shared_maxidf(addr_tokens_q, addr_tokens_p, addr_idf)`
- `a_partial` = `fuzz.partial_ratio(addr_q, addr_p) / 100.0` if both addresses non-empty else NaN
- `a_char3_jacc` = Jaccard of `_char3(addr_q)` / `_char3(addr_p)` if both non-empty else NaN
- `a_num_minlev` = `min(Levenshtein.distance(a, b) for a in q_long for b in p_long)` if both sides have a
  >=3-digit number, else NaN (near-equal numbers like "19010" vs "9010")
- `a_numstreet` = 1.0 if `_num_street(addr_q) & _num_street(addr_p)` is non-empty, 0.0 if both sides have
  at least one number-street pair but none shared, NaN if either side has none

**Combined result of A3+A4** (run `k5_feat`, on top of k5_base): OOF **0.97729** (delta vs k5_base
**+0.00211**, 95% CI **[0.00190, 0.00235]**), MATCH_VALID **0.97786** (delta **+0.00192**, CI
**[0.00129, 0.00257]**). `artifacts/metrics/match_train_k5_feat.json`,
`compare_k5_base_vs_k5_feat_{OOF,MATCH_VALID}.json`.

### A5. List feature `l_n_addr_close` — 1 new column

Count of other candidates in the same S1's list whose `a_tset` (address token-set ratio to the S1) is
>= 0.9 (mirrors the existing `l_n_name_close` at the same 0.9 threshold, for name). Computed in
`list_features()` after `a_tset` is filled for the whole list:
```python
X[:, c["l_n_addr_close"]] = float((a >= 0.9).sum())   # a = X[:, c["a_tset"]] with NaN -> -1.0
```

### A6. String-similarity group (X1) — 6 new columns, biggest single win

**Mechanism, three signals, all on name tokens except the two TF-IDF columns:**

**F2 — TF-IDF character n-gram cosine** (name and address). Best single name signal in the team's EDA
(hard-negative AUC 0.750, ahead of plain token Jaccard 0.701). Character n-grams tolerate typos and
transliteration noise that whole-token Jaccard cannot.
```python
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2**20,
                       alternate_sign=False, norm=None)
```
IDF fit once per split, unsupervised, over the FULL pool (10,320,219 train records): binarize each
record's hashed n-gram vector (presence, not count) and sum to get document frequency `df` per hashed
dimension; `idf = ln((1 + N) / (1 + df)) + 1`. Save as a flat `float32[2**20]` array, cached with a
size+mtime stamp on the normalized pool file.

Per pair: multiply each side's hashed count vector by the idf vector (element-wise), L2-normalize, cosine
= dot product. Vectorize per feature-build chunk (deduplicate the chunk's distinct query/pool strings,
transform each distinct string once, then index back out per row) — never build a dense S1×pool matrix.
`n_tfidf_cos` on name text. `a_tfidf_cos` on address text, with the empty-address rule: **NaN** (not 0.0)
whenever either side's address is empty.

**F3 — Monge-Elkan (best-token match, both directions)**:
```python
def _monge_elkan(qt, pt):
    """For each token of one side, take its best JaroWinkler match on the other side, then average
    (not symmetric: A->B != B->A); returns (mean of both directions, min of both directions)."""
    if not qt or not pt:
        return float("nan"), float("nan")
    m = process.cdist(qt, pt, scorer=JaroWinkler.normalized_similarity)
    q_to_p = float(m.max(axis=1).mean())
    p_to_q = float(m.max(axis=0).mean())
    return (q_to_p + p_to_q) / 2.0, min(q_to_p, p_to_q)
```
Columns: `n_me_mean`, `n_me_min`. Covers per-token typos (`Aveneu`/`Avenue`) regardless of word order,
without needing the whole strings to match.

**F4 — Phonetic set Jaccard** (Metaphone and Soundex, via `jellyfish`, MIT-licensed, already a
dependency):
```python
def _phonetic_tokens(tokens):
    return [t for t in tokens if len(t) >= 2 and t.isascii() and t.isalpha()]

def _phonetic_jacc(qt, pt, code_fn):
    q_elig, p_elig = _phonetic_tokens(qt), _phonetic_tokens(pt)
    if not q_elig or not p_elig:
        return float("nan")
    qs, ps = {code_fn(t) for t in q_elig}, {code_fn(t) for t in p_elig}
    return len(qs & ps) / len(qs | ps)
```
Columns: `n_metaphone_jacc` (`code_fn=jellyfish.metaphone`), `n_soundex_jacc` (`code_fn=jellyfish.soundex`).

**Result**: OOF **0.97807** (delta vs k5_feat **+0.00078**, CI **[0.00060, 0.00097]**), MATCH_VALID
**0.97848** (delta **+0.00062**, CI **[0.00021, 0.00104]**). Every OOF segment improved; largest gains:
`match_1` (S1 with exactly 1 true match, the hardest segment) **+0.00322** [0.00121, 0.00529], India
**+0.00109**, US **+0.00058**; singleton **+0.00039** (CI includes 0, no guard issue).
Feature-gain rank of 68 total: `n_tfidf_cos` 12th, `n_me_min` 23rd, `n_me_mean` 28th, `a_tfidf_cos` 29th,
`n_metaphone_jacc` 41st, `n_soundex_jacc` 55th — all contribute, none dominate.
`artifacts/metrics/match_train_x1_strsim.json`, `compare_k5_feat_vs_x1_strsim_{OOF,MATCH_VALID}.json`.

### Final adopted feature list (68 columns) and decision rule

`PAIR_FEATURES` (54) + `LIST_FEATURES` (14) = 68. The 50 original submission-1 columns are unchanged in
definition (only A1's zero-strip changes their *values*, not their code shape) plus the 18 new columns
from A2–A6 above. Decision rule chosen on MATCH_TRAIN OOF: **global threshold 0.67 on calibrated p, plus
the one-owner filter** (a pool record kept only for the S1 giving it the highest calibrated p). Isotonic
calibration fit on out-of-fold predictions, 5-fold GroupKFold by S1, `fold_seed=42`.

Candidate generation (stage 1+2) is UNCHANGED from submission 1: `K_min=5, K_max=60, p_min=0.05`. A
separate widen-to-K_min=20 test was run and reverted — see Part B.

---

## PART B — TESTED AND REJECTED (do NOT implement; kept here so nobody re-tries them without new evidence)

### B1. Widen the stage-2 candidate cut from K_min=5 to K_min=20/K_min=10

**What**: force at least 20 (or 10) candidates per S1 instead of 5, since candidate-list size is not
scored by the contest (only `matching_results.tsv` is) and the stage-1 recall ceiling improves from
0.9926 to 0.9945–0.9946 at wider cuts (`artifacts/metrics/rerank_cut.json`).

**Why rejected**: not tested to a conclusion in this repo — a teammate's independent analysis
(`ADDRESS_EXPERIMENTS.md` §4 S4) measured the same ceiling numbers and concluded the gain (+0.002
ceiling) is not worth the cost (~3.7x more stage-3 rows, list features change meaning, TEST candidates
must be regenerated, ~2.5h). Decision: **do not widen K_min**; if candidate width is revisited later,
lower `p_min` first (cheaper, keeps the same K_max), per that analysis. `blocking.json` `rerank.cut` is
back at `{K_min: 5, K_max: 60, p_min: 0.05}`.

### B2. Heuristic/pattern group (X2) — 11 columns

**What was built** (all still described here in case a lower adoption bar is later approved — see
"Open items" below): `n_num_state` (name digit/Roman-numeral 3-state agreement), `a_tset_canon` /
`a_idf_jacc_canon` (address token-set/IDF-Jaccard after ordinal-word-to-number conversion + zero-strip),
`p_is_domain` / `q_is_domain` / `n_domain_tset` (domain-style pool/S1 name detection + Viterbi
word-segmentation of the domain body compared to the other side's name tokens), `a_hid_state` /
`a_unit_state` / `a_pobox_state` / `a_sector_state` (RAW-address regex patterns: compound house id,
unit/suite/floor, PO box, sector/block/phase — each a 4-state NaN/one-side/conflict/match code),
`a_nolandmark_tset` (token-set ratio after cutting the RAW address at "near/opposite/behind/beside" up to
the next comma).

**Numbers**: OOF F0.5 0.97854, delta vs x1_strsim **+0.00047**, CI **[0.00031, 0.00065]** — CI excludes
zero, but the point estimate is **below the +0.0005 minimum gain** (rule bullet 1). MATCH_VALID delta
**+0.00069** [0.00025, 0.00115] does confirm, and no guard segment breached. **Rejected purely on the
minimum-gain threshold**, not on noise or a regression. Gain ranks (of 79 features):
`a_idf_jacc_canon` 24th, `a_nolandmark_tset` 30th, `n_domain_tset` 40th, `a_hid_state` 53rd,
`a_tset_canon` 56th, `n_num_state` 64th, `a_pobox_state` 66th, `a_unit_state` 68th, `p_is_domain` 71st,
`a_sector_state` 73rd, `q_is_domain` 79th (zero gain — S1 text is almost never itself a raw domain
string). `artifacts/metrics/match_train_x2_pattern.json`,
`compare_x1_strsim_vs_x2_pattern_{OOF,MATCH_VALID}.json`.

**Open item**: this is the closest miss in the whole log. If the team lowers the adoption bar from
+0.0005 to +0.0003 (allowed under `VALIDATION_PLAN.md` §6, "change if the team wants to chase smaller
gains"), X2 would pass outright with both CIs already computed. Full code is not preserved on disk (it
was reverted per protocol) — re-implement from the exact specs above, re-run, and it should reproduce
these numbers.

### B3. Sibling / stacked-list features (X3) — REJECTED on a guard-rail, not on the headline number

**What**: two-pass stacking on the frozen x1_strsim pair model. Pass-1 calibrated p (out-of-fold for
MATCH_TRAIN, fold-averaged for MATCH_VALID) is joined onto each candidate row. Pool-to-pool sibling
columns computed only from other candidates in the same list (unsupervised, no S1 text): best name
similarity to another candidate, whether that best-match sibling is from the other source (S2 vs S3),
count of same-name siblings, address similarity between the two. Then pass-2 stacked columns from pass-1
p: own p, own rank-by-p in the list, the best sibling's pass-1 p, the max pass-1 p among same-name
siblings, whether the best sibling is address-confirmed.

**Numbers**: OOF **+0.00074** [0.00053, 0.00095] and MATCH_VALID **+0.00175** [0.00109, 0.00239] —
**both primary gain rules pass, including the extra stacking condition** (MATCH_VALID CI lower bound >
0). **Rejected anyway** because the `singleton` guard segment broke: OOF singleton F0.5 delta
**-0.01088** [-0.01343, -0.00892], MATCH_VALID **-0.00868** [-0.01476, -0.00347] — both far past the
-0.002 guard with CI entirely negative.

**Mechanism of the failure** (important — any future sibling/stacking idea will hit this unless it's
fixed): a true-singleton S1's candidate list can still contain an address-confirmed pair of WRONG pool
records (two unrelated businesses that happen to share an address, e.g. a mall). Sibling features measure
candidate-to-candidate similarity, not candidate-to-S1 correctness, so they can push p up for a candidate
that borrows a confident neighbor's score even when neither candidate is this S1's actual match. Under
F0.5's rule, any non-empty prediction on a true singleton scores 0 regardless of how "confident" it
looked, so this single failure mode outweighs the genuine recall gain. Measured: `singleton_with_fp`
entities rose 190 -> 301; `matched_missed_only` loss (the thing X3 targeted) did fall 0.01005 -> 0.00943
as intended — the trade-off is real, just not worth it under F0.5.

**If revisited**: gate the sibling boost so it cannot fire on a candidate whose OWN pass-1 p is very low
(e.g. only apply `sb_p`/`sb_confirmed` when the row's own p1 already clears some floor), or add a
singleton-specific penalty term. Not attempted here — flagging as the natural next idea, not a promise it
works.

### B4. Entity-level "has any match?" gate (X4) — REJECTED

**What**: a second LightGBM model, one row per S1 (23 features: candidate count, top-1/2/3 pass-1 p and
their gaps, sum p, count p>=0.5, entropy of the p distribution, max/mean of several pair-similarity
columns, pool-uniqueness counts, max `p_rerank`), target = "does this S1 have >=1 true match among its
candidates?" (positive rate 0.9420, OOF AUC 0.99947 — the model itself is very accurate). Its calibrated
`P_any` gates the base threshold decision: force empty if `P_any < tau_empty`, force the top-1 candidate
in if the base rule left the S1 empty and `P_any >= tau_force`. Best grid point: base threshold 0.70,
`tau_empty=0.40`, `tau_force=0.62`.

**Numbers**: OOF **+0.00016** [0.00004, 0.00029] — below the +0.0005 minimum gain (same failure mode as
X2). MATCH_VALID **+0.00033** [-0.00001, 0.00072] — CI lower bound is negative, failing the stacking
condition on its own. Guard: singleton OOF **-0.00206** [-0.00333, -0.00069] — a real (if small) breach,
same mechanism direction as X3 (fixing 599->551 "empty but should have answered" S1 at the cost of
190->211 new singleton false positives). **Triple rejection** — none of the three rules pass.

### B5. Monotone constraints (X5a) — REJECTED, made things worse everywhere

**What**: forced LightGBM's `monotone_constraints` (+1, "advanced" method) on 19 pair-similarity columns
(every rapidfuzz/Jaccard/IDF/TF-IDF-style column, e.g. `n_tset`, `a_tset`, `n_tfidf_cos`), reasoning that
higher similarity should never be allowed to lower match probability — meant as a safety measure for
France (unseen at training time).

**Numbers**: OOF **-0.00102** [-0.00122, -0.00084], MATCH_VALID **-0.00097** [-0.00150, -0.00046] — a
clean loss, not noise. Every guard segment moved negative (country_India -0.00141, country_US -0.00076,
singleton -0.00333, match_1 -0.00363).

**Mechanism**: the unconstrained model was using real interactions between similarity columns and other
signals — e.g. down-weighting a high `n_tset` when `a_digits_state` shows a number conflict elsewhere in
the same pair. Forcing monotonicity on the similarity columns in isolation removes those interactions
entirely, and the interactions carried genuine separating signal on this data. **Do not reuse this idea
as-is**; if country-robustness is revisited, a narrower constraint (fewer columns, or interaction
constraints instead of monotone ones) would need its own test.

### B6. LambdaRank objective (X5b) — REJECTED, severely

**What**: replaced the binary logloss objective with LightGBM's `lambdarank` (group = S1's candidate
list, `metric=ndcg`, `eval_at=[1,3,5]`), keeping the same isotonic calibration and threshold+one-owner
decision rule downstream.

**Numbers**: OOF **-0.01983** [-0.02037, -0.01935], MATCH_VALID **-0.01892** [-0.02044, -0.01743] —
singleton alone fell **-0.14129** [-0.14816, -0.13530], catastrophic.

**Mechanism**: a ranking objective optimizes within-list ORDER, not the absolute probability value. It
early-stopped in 90–184 rounds (ndcg@1 already >0.999) versus x1's 1000+ rounds on logloss, because
ranking the top candidate first is easy and saturates fast — but the isotonic-calibrated ranking score is
a poor stand-in for a real probability, and the threshold+one-owner rule needs a real probability to
decide empty-vs-non-empty per S1, not just a within-list order. **Do not combine a ranking objective with
a probability-threshold decision rule**; if ranking is revisited, it would need its own decision rule
(e.g. always keep top-k for a learned k), not this one.

### B7. Leave-one-country-out (LOCO) diagnostic (X5c) — report only, no code to adopt or reject

Measures how much the x1_strsim model relies on having seen a country's own data (a proxy for the
completely-unseen France, 15% of TEST):
- Train on US-only MATCH_TRAIN S1 (107,859), score India MATCH_TRAIN S1 (72,141) with those fold models +
  their own OOF isotonic calibration: F0.5 **0.95387** vs. x1_strsim's own OOF on the same India S1
  **0.97332** — paired delta **-0.01945** [-0.02018, -0.01875].
- Reverse (train India, score US): F0.5 **0.97256** vs. **0.98125** — paired delta **-0.00868**
  [-0.00914, -0.00830].

Both drops are large and don't overlap zero. Losing a country from training costs noticeably more than
any single feature group in this log gained — and this doesn't even isolate the "unseen country" effect
from the "less training data" effect (each one-country model only saw 40–60% of the rows). Treat as a
LOWER BOUND on France risk, not a France-specific number. No monotone-LOCO cross-check was run since X5a
(B5) was rejected.

---

## What to actually run on the training machine

1. Apply A1 (leading-zero fix) inside `_numbers()`/`_num_street()`. Re-run `pair_features()`'s address
   block unchanged — it already reads from the canonicalized sets.
2. Add A2 (`a_digits_state`), A3 (4 pool-count columns + their once-per-split count-table build), A4 (6
   address-shape columns), A5 (`l_n_addr_close`), A6 (6 string-similarity columns + the once-per-split
   TF-IDF IDF build). Append all 18 new column names to `PAIR_FEATURES`/`LIST_FEATURES` in the exact
   order they were added (order matters only for readability, not correctness — LightGBM is
   order-invariant — but keep it for reproducibility of `feature_gain` rank comparisons).
3. Keep `K_min=5, K_max=60, p_min=0.05` (B1 — do not widen).
4. Train with 5-fold GroupKFold (`fold_seed=42`) grouped by `s1_idx`, isotonic calibration on OOF, decide
   between threshold and one-owner exactly as `run_match_train` already does (paired-CI adoption, not a
   fixed +0.001 gate).
5. Do NOT implement B2–B6 as written. B2 is the one worth a second look if the team lowers the adoption
   bar (see "Open item" under B2) — everyone else has a real reason not to.
6. Expect final numbers close to: OOF F0.5 ≈ 0.978, MATCH_VALID F0.5 ≈ 0.978, i.e. **+0.004 over
   submission 1** on MATCH_VALID.

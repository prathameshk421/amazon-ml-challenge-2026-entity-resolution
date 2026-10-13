# Hostile review of `plan/BUCKETING_PLAN.md` v1.1 (candidate generation)

Reviewer stance: adversarial. Every number below comes from a script under `plan/critique/scripts/`
that wrote a JSON under `plan/critique/out/`, or from a named project file. Anything I could not
measure is in the last section, labelled SPECULATION.

How to reproduce (repo root, one job at a time):
```
nice -n 10 .venv/bin/python plan/critique/scripts/a_normalize_all.py   # reference normalizer over train S1, train pool, test S1, test-pool France rows -> out/norm_*.parquet (~8 min, peak RSS ~1.4 GB)
nice -n 10 .venv/bin/python plan/critique/scripts/b_gt_facts.py        # ALL 7,638,365 links, chains, name overlap, F0.5 ceilings -> out/b_gt_facts.json (peak RSS 3.67 GB)
nice -n 10 .venv/bin/python plan/critique/scripts/c_france.py          # France token statistics (test, structural only) -> out/c_france.json
nice -n 10 .venv/bin/python plan/critique/scripts/d_evidence.py        # df-capped key evidence, per-script romanization, dictionary re-implementation, cross keys -> out/d_evidence.json, out/d_examples.json (peak RSS 3.65 GB)
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python plan/critique/scripts/f_throughput.py   # stage-2 hot-path timing -> out/f_throughput.json
nice -n 10 .venv/bin/python plan/critique/scripts/g_chars.py           # Indic characters that fall back to word breaks -> out/g_chars.json
nice -n 10 .venv/bin/python plan/critique/scripts/h_france_df.py       # df of France chain-name tokens vs the cap -> out/h_france_df.json
.venv/bin/python plan/critique/scripts/i_derived.py                    # arithmetic on the verified counts -> out/i_derived.json
```
Test files were used only for structural and token statistics, plus a count of how many test S1
names also appear among train S1 names. No test S1 record was matched against the test pool, and
nothing was tuned on test.

---

## Summary verdict
1. Retrieval is not the bottleneck; ranking is. Even with df caps, 99.3% of true pairs already share at least one usable name or address key (see M3). Five of the ten families mostly add more retrieval, while the part that decides recall (composite weights, EN = 10) is FIXED by hand and never tuned.
2. The headline romanization number is misleading. "≥1 identical skeleton 98.2%" drops to **5.5%** once the planned df cap is applied, so SK and SP do almost nothing for Indic names. The learned dictionary is what helps (26% → 61% usable-token overlap), but its planned gate would pass trivially.
3. France is 15% of test S1 and effectively none of it can be evaluated. Only 0.013% of France S1 names occur in train, 47% of the top-200 France chain names consist only of tokens above the cap, and the France pool abbreviates addresses differently ("r" for "rue" in 24.9% of pool records vs 0.02% of S1). Nothing in the plan measures or mitigates this.
4. The plan optimises micro pair recall, but the score is a macro per-entity F0.5. On the old K=150 candidates, pair recall is 0.925 while the perfect-matcher F0.5 ceiling is 0.969, and the two objectives weight entities differently.
5. Compliance risk is low if documented. Feasibility is underestimated: about 20 CPU-hours for stage-2 alone at 700 candidates and 500 trees, against the plan's 1.5–3 h for all of P6. Executability risk: the goldens and version pins come from Python 3.14 (Unicode 16.0), but the laptop plan uses Python 3.12.

---

## Issues (sorted by severity)

### B1 [BLOCKER]: France (15% of test S1) has no evaluation, no proxy, and normalization that is wrong for it
- **Attacked:** §2.1 "Country | Stage-2 feature … France unseen"; §10 risk row "Re-ranker fails to transfer (France) | LOCO drop; DEV_B | language-agnostic features…"; §5.2 step 11 "token maps: NAME_ABBREV / ADDR_ABBREV (… train-derived v1 lists)".
- **Evidence:**
  - France is 259,452 / 1,732,544 = **14.98%** of test S1 (`out/i_derived.json`), so about 15% of the final score.
  - Name novelty: **0.013%** of France test S1 names occur among train S1 names. For US it is 35.9%, for India 46.0%, and for a random 20% of train held out against the other 80% it is 37.9% (`out/b_gt_facts.json` → `test_s1.by_country`, `heldout20_share_name_seen_in_fit80`). LOCO US↔India therefore does not approximate France: both of those countries share about 40% of their names with train.
  - France chains are made of generic tokens. The largest are `bordeaux club sarl` ×205, `nantes club sarl` ×157, `lille club sarl` ×147 … (`out/h_france_df.json`). The df of these tokens in the France part of the test pool alone already exceeds the cap (0.002 × 9,969,589 = 19,939): club 106,695, sarl 298,035, sas 200,604, amicale 69,239, ecole 72,260, bordeaux 26,232, nantes 22,761, lille 21,512. In the address field, bordeaux 233,153 and nantes 200,247. **47%** of the top-200 France chain names have *every* token capped. For those names, NT, SK, PH and city-level AT are all silent, so they depend on EN, NP, NC (quota 20!) and street tokens.
  - Address abbreviations the maps do not cover. Per-record share, France S1 vs France pool (`out/c_france.json`): `r` 0.0002 vs **0.2486**; `rue` 0.658 vs 0.389; `av` 0.0085 vs 0.038; `all` (allée) 0.0025 vs 0.018; `imp` 0.0008 vs 0.0084; `rte` 0.0015 vs 0.0081; `no` 0.0 vs 0.066 (from `Nº`). The pool also carries department names that S1 lacks: `nord` 151,677 vs 200, `gironde` 150,701 vs 62, `atlantique` 127,095 vs 49 (`top_pool_addr_tokens_rare_in_s1`).
  - Collisions in the fixed maps. `ADDR_ABBREV["st"]="street"` rewrites French *Saint*: raw `ST` appears in 1.75% of France pool addresses (example `NO. 46 RUE DES SAPHIIS, ST-HERBLAIN`), and after normalization `street` appears in 1.70% of France pool records vs 0.05% of S1. `ste` → `suite` rewrites *Sainte* (`57 R. JEAN MOULIN STE MARIE`). The fixed maps are symmetric, so this does not break exact keys, but it corrupts AB bigrams and the `fa_tset` features.
  - Legal-form punctuation. `S.A.S.` normalizes to `s a s`, and tokens of length 1 are dropped by `tok2`. The France pool has 64,195 records with token `s` and 61,856 with `a`, vs 8 and 52 in France S1 (`top_pool_name_tokens_rare_in_s1`). So `bordeaux club s a s` ≠ `bordeaux club sas`, which kills the EN key, and EN is the main name key left for these chains.
- **Why it matters:** if France recall or ranking collapses, about 15% of entities score near 0 (or near 1 only if the entity is a singleton and the matcher predicts empty). No DEV set can show this.
- **Alternative:** (a) Add a "query name unseen in the fitting part" segment to every metric. That is the only train-side proxy for France: with a random split it is about 62% of held-out entities. (b) Collapse runs of single-letter tokens (`s a s` → `sas`, `s s` → `ss`) before key generation. (c) Mine address abbreviations *unsupervised* from the pool, which R4 allows if logged: pairs (short, long) where short is a prefix or consonant skeleton of long, with a high co-occurrence ratio in the same slot (e.g. before a street number). This recovers `r/rue`, `av/avenue`, `imp/impasse`, `rte/route`, `all/allee`. (d) Make `st`/`ste` expansion country-agnostic: map both `st` and `saint` to one canonical token instead of `street`. (e) Report per-country candidate-size and score distributions on the test run (unsupervised, allowed) to detect a France collapse.
- **Cheap test:** rerun `c_france.py` after changes. The `r`/`rue` and `s a s` asymmetries should disappear. Recompute `h_france_df.py` on the new tokens. On train, measure recall for DEV_A entities whose name is unseen in the rest of fit vs seen (one extra segment in `evaluate.py`).

### M1 [MAJOR]: The romanization justification does not survive the df cap; SK/SP are nearly useless for Indic, and the dictionary gate is vacuous
- **Attacked:** F12 / §5.2.1 "≥1 identical token 79.7%, ≥1 identical skeleton 98.2% | Skeleton families SK/SP + learned dictionary"; §5.3 gate "the share of pairs whose pool name shares ≥ 1 token with the S1 name … Gate: the share must increase."
- **Evidence** (all 551,240 true pairs whose pool name contains Indic letters, not a 30K sample; `out/d_evidence.json` → `romanization_all`):
  - Identical token 79.72%, identical skeleton 98.38%. This reproduces the plan's numbers.
  - The same measures counting only keys with df ≤ 0.002 × n_pool (20,640), i.e. keys the index can actually use: **token 26.15%, skeleton 5.54%**. Only 277 skeletons exceed the cap, but they dominate: `LMT` df 2,621,103, `PRPT` 2,094,803, `LK` 1,159,618, `NKRPRT` 895,557 (`top_SK_df`). Skeletons collapse 1,432,434 distinct tokens into 439,495 codes.
  - Any usable name evidence (EN or usable token or usable skeleton) exists for only 29.8% of Indic pairs. Usable address-token overlap exists for 91.0%. Both are silent for 6.3%.
  - My re-implementation of §5.3, fitted on a random 80% of S1 entities and evaluated on the other 20% (110,065 pairs, `dictionary_eval_hold20`): "≥1 shared token" goes 0.7976 → **0.9987**. That jump comes almost entirely from generic suffixes: 297 of the 446 dictionary entries map to a capped target token such as private or limited. The **usable**-token share goes 0.2598 → **0.6063**. That is real, and it is the actual Indic recall lever.
- **Why it matters:** the planned gate passes whether or not the dictionary helps blocking. SK/SP/PHON (quota 50) and SP pairs may be kept by E2 on noise. Meanwhile the dictionary, which needs the most care, gets a trivial gate.
- **Alternative:** gate §5.3 on "shares ≥1 token with df ≤ cap" (and report per script). Drop SK as a retrieval family unless E2 shows exclusive value on the *indic* segment; keep the skeleton as a stage-2 feature. Consider applying the dictionary to S1 as well, reversed (S1 token → canonical form), so rare proper nouns can meet halfway.
- **Cheap test:** `d_evidence.py` already computes both versions. Port its 20-line dictionary block into the P2 metric.

### M2 [MAJOR]: The optimisation target is micro pair recall; the leaderboard is macro per-entity F0.5
- **Attacked:** §5.9 metrics, E1 gate "pair recall@150 ≥ 0.92", E4 "cheapest config with pair recall ≥ 0.970", cut rule "pair recall ≥ 0.99 × stage-1 pair recall".
- **Evidence:**
  - On the old 60,000-entity `fit_sample_K150` candidates (`out/b_gt_facts.json` → `old_K150_fit_sample`): pair recall 0.9251, entity recall 0.9866, full recall 0.8129, **F0.5 ceiling with a perfect matcher, singletons = 1.0: 0.9686**.
  - The ceiling is concave in per-entity recall: missing one of k matches costs 0.167 (k=2), 0.091 (k=3), 0.0625 (k=4), 0.048 (k=5), but missing all costs 1.0 (`F05_ceiling_if_miss_one_of_k`).
  - Match counts go up to 11 (histogram in `match_count_hist`). Pair recall weights an entity with 8 matches 8× more than an entity with 1 match (119,157 entities have exactly one match). The metric weights them equally.
- **Why it matters:** E4 and the cut rule can pick configurations that trade whole single-match entities for extra pairs on large entities. That raises pair recall and lowers the achievable score.
- **Alternative:** make the primary blocking objective the mean per-entity ceiling `1.25·r/(0.25+r)` over *all* queries (singletons = 1), with entity recall as a hard floor. Report pair recall as secondary.
- **Cheap test:** in E4 evaluation mode, compute both objectives for all 486 configs and check whether the chosen config changes. This costs nothing extra: per-entity hits are already in the arrays.

### M3 [MAJOR]: Over-engineered retrieval, under-powered ranking; composite weights are arbitrary and never tuned
- **Attacked:** §3.5 `composites` (BOTH = 10·EN + NT + NG + 0.2·PH + AT, all FIXED); §8, where only families, caps, quotas and the chain rule are tuned.
- **Evidence:**
  - On a 20K S1 sample, 69,260 true pairs (`out/d_evidence.json` → `evidence_20k_sample.all`), before NG/PH/NC/AB are even counted: 86.3% share a usable name key (EN, NT, SK or NP), 92.7% a usable address token, and only **0.70%** share nothing. The old diagnostics agree: 98.72% reachable with the old families, 99.35% after PAIR+ROM (`artifacts/metrics/diag_blocking2.json`). The loss is ranking: recall@150 is 0.925 against 0.987 reachability (`diag_blocking.json`).
  - Exact name equality holds for only 28.55% of all 7.6M links (`true_pairs_exact_name_equal`), yet EN weighs 10× a token idf.
  - `diag_blocking2.json`: adding families into one sum *hurt* small-K recall (F8 @150 0.9062 vs F1 0.9252). The plan's answer is quotas, but the per-channel scores themselves remain hand sums with untested weights (NP+SP = 1:1, SK+PH = 1:1).
- **Why it matters:** BOTH gets 300 of about 700 slots. Its hand weights decide what enters stage-2, and nothing in P4 can correct them.
- **Alternative:** learn the stage-1 scorer. Stage-1 already computes S[F] per family for the union, so fit a *linear* (logistic) model on DEV_A-disjoint queries over the 10 S[F] and cov features. It is cheap and vectorisable in the same numpy pass, and it replaces BOTH/NAME. Or add a weight grid for BOTH to E4. Simpler baseline to beat: TF-IDF char-3gram sparse top-N (scipy CSR dot per batch) on "name + ' | ' + top address tokens". That is SPECULATION as to its recall; I did not run it on the full pool.
- **Cheap test:** on 2,000 DEV_A queries, dump the union with S[F] and labels (the diag scripts already do this: `artifacts/diag_pairs*.parquet`), fit logistic regression, and compare recall@300 with BOTH.

### M4 [MAJOR]: Chains: the EN bonus promotes same-name decoys, and the chain rule threshold misses most chain queries
- **Attacked:** `"BOTH": {"EN": 10, …}`, `"chain_rule": {"chain_df": 50, "mult": 3, "channels": ["ADDR","PAIR"]}`.
- **Evidence** (all links, `out/b_gt_facts.json` → `segments_by_chain`):
  - Chain ≥10 queries hold 18.17% of all links. Only **24.68%** of their true pairs have an exactly equal name.
  - Their EN bucket (pool records with the query's exact name) is small: d_EN median **31**, p90 138. It exceeds 300 for only 1.0% of these pairs and never exceeds the cap (max pool EN df is 1,071 for `physical therapy`).
  - So for a typical chain query, about 31 same-name records, mostly other entities' matches, get +10. Meanwhile the 75% of true matches with a non-exact name are name-silent: usable NT only 46.2% and usable SK 19.9% in this segment (`evidence_20k_sample."chain=>=10"`). About 25% of these pairs have *only* address evidence.
  - With `chain_df = 50` above the median d_EN of 31, the chain rule does not fire for most chain ≥10 queries.
  - Old result: chain ≥10 pair recall was 0.7958 @150 and 0.8999 @2000 (`diag_blocking.json`).
- **Why it matters:** chains are 40% of S1 entities (`s1_entities_in_chain_ge2` 0.399) and the main source of false merges, so they dominate F0.5 precision too.
- **Alternative:** (a) Trigger the chain rule on the chain size of the query name among S1 names, which is known at query time, not on d_EN. (b) Add a cross-key channel: name skeleton × usable address token, i.e. conjunctive keys that turn capped name tokens into selective keys. Measured: recall 85.0% overall and **86.5% on chain ≥10**, with per-query sum-of-df p50 1,175, p90 19,446 (chain ≥10: p50 3,333) (`cross_key_*`). This is an upper bound on channel size, not a union. (c) Within an EN bucket, rank by address similarity instead of adding a constant.
- **Cheap test:** E5 grid with "S1 chain size" as trigger; X2 promoted from optional to E2 candidate. `d_evidence.py` §4 already generates the keys.

### M5 [MAJOR]: Evaluation design for the requested "held-out TEST from train" split
- **Attacked:** §5.4 split, §6.
- **Evidence:**
  - A random entity split is already *realistic* for name overlap on US/India: held-out names seen in the rest of train are 37.9%, vs 35.9% (US) and 46.0% (India) for test S1 against train S1 (`out/b_gt_facts.json`). Grouping chain siblings by name would make the held-out set pessimistic (0% seen) for US/India, and it still would not model France (0.013%).
  - The pool is denser in test: 5.75 pool records per S1 vs 4.68 in train (`out/i_derived.json`). ESTIMATE, assuming the train link rate of 3.46 per S1: test orphans would be about 39.8% of the pool vs 26.0% in train, so more distractors per query.
  - One-owner holds on all 7.6M links (7,638,365 unique pool records, 2,681,854 orphans).
- **Why it matters / leakage traps:**
  1. Any later "one pool record → at most one S1" assignment is only realistic if inference runs over **all** S1 entities that compete for the pool. Scoring only the held-out 20% against the full pool removes 80% of the competitors, and the result is optimistic.
  2. Never remove fit-owned records from the pool when evaluating held-out entities.
  3. The dictionary, df and G are learned on train data that includes the held-out entities' pool records. df and G are unsupervised, so this is fine. The dictionary is learned only from fit pairs, and in my re-implementation **0** of 446 entries rest on a single S1 entity (`dictionary_fit80`), so memorisation leakage is negligible.
- **Alternative:** keep a random entity split for the held-out TEST part, and add two segments: (i) query name unseen in fit, as a France proxy; (ii) a "pool-density" stress test that adds held-out-only distractors, e.g. S2/S3 records from orphans. Run the held-out set with all train S1 as competing queries whenever a global assignment is used.
- **Cheap test:** segment (i) is one boolean per query in `evaluate.py`.

### M6 [MAJOR]: Compute is underestimated; MATCH_POOL runs stage-2 on 1.68M queries it doesn't need
- **Attacked:** §7 "P6 … 1.5–3 h"; §5.4 MATCH_POOL = perm[90000:].
- **Evidence** (`out/f_throughput.json`, Mac M3, single thread):
  - rapidfuzz cdist for the 6 planned scorers: 2.69 ms per query at 700 candidates.
  - LightGBM predict (33 features, 63 leaves, 500 trees): **22.6 µs per row**, 15.6 ms per 700-row call.
  - Planned volumes: TEST 1,732,544 + MATCH_POOL 1,675,457 + VAL 441,364 + about 100K. Total **≈ 20.1 CPU-hours for stage-2 alone**, without stage-1 (target 8 ms/query ≈ 8.6 CPU-h more), feature glue or I/O. At W=12 on a laptop that is several hours of wall time, and more if early stopping keeps more than 500 of the 2,000 rounds.
- **Alternative:** subsample MATCH_POOL (e.g. 200K entities is plenty for a pairwise matcher). Cap LightGBM at about 300 trees, or distil stage-2 into a linear model for the first cut. Run stage-2 only on the top about 300 by the learned stage-1 score (M3).
- **Cheap test:** rerun `f_throughput.py` with the real model file after P5.

### M7 [MAJOR, executability]: The version pins and goldens come from a different Python/Unicode than the laptop
- **Attacked:** §3.1 step 7 pins; §3.1 "one Python/Unicode version for all goldens and indexes"; R7 determinism.
- **Evidence:** the Mac `.venv` is Python 3.14.5 with `unicodedata.unidata_version` 16.0.0, and numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, rapidfuzz 3.14.6, lightgbm 4.7.0, scikit-learn 1.9.1, scipy 1.18.1. These are exactly the §3.1 pins. Command: `.venv/bin/python -c "import unicodedata,numpy,…; print(unicodedata.unidata_version, numpy.__version__, …)"`. The laptop plan installs Python 3.12, which ships Unicode 15.0. `goldens.json` was generated by the reference running on the Mac (`plan/reference/make_goldens.py`), which contradicts the plan's own rule.
- **Why it matters:** the romanizer classifies characters by `unicodedata.name`, so golden tests can fail on a different Unicode DB. Pins that lack cp312 wheels trigger the "newest that does" fallback, so versions end up unpinned in practice.
- **Alternative:** either install Python 3.14 on the laptop (deadsnakes or uv) or regenerate the goldens on the laptop in P0 and diff them against the Mac copy, logging the differences. Do not treat Mac goldens as normative.
- **Cheap test:** on the laptop, run `plan/reference/check_ref.py` / the goldens under 3.12 and diff.

### m1 [MINOR→MAJOR for Punjabi/Tamil names]: Script-specific romanizer defects
- **Evidence:**
  - `out/g_chars.json`: `GURMUKHI TIPPI` (nasal sign, 3,652 train / 4,409 test occurrences) and `GURMUKHI ADDAK` (466 / 500) become **word breaks**, because their Unicode names lack "SIGN". Examples from `out/d_examples.json`: `ਇੰਡਸਟ੍ਰੀਜ਼` → `i dasatrij`, `ਅਨੰਦ` → `an d`. `DEVANAGARI OM` (2,338) also becomes a break, and `GURMUKHI SIGN BINDI` (2,325) is silently dropped.
  - Identical-token share by script (`romanization_by_script`): Tamil **0.236**, Gurmukhi **0.225**, Malayalam **0.364**, vs 0.85–0.93 for Devanagari, Telugu, Kannada, Bengali, Gujarati and Oriya. Examples: Tamil `பாபா கேர்` → `papa ker` (Baba Care; the script has no voicing), and Malayalam `limitad`/`praivat`. These three scripts cover 69,106 of 551,240 Indic pairs.
- **Alternative:** classify TIPPI and ADDAK as nasal/gemination (TIPPI → "n"; ADDAK doubles the next consonant). For Tamil, rely on the dictionary plus voicing-insensitive skeletons, which already map p/b and t/d to the same code. Add goldens for these cases.
- **Cheap test:** rerun `d_evidence.py` §2. Gurmukhi `tok_eq` should rise.

### m2 [MINOR]: Compliance of a learned re-ranker as the producer of `candidate_pairs.tsv`
- **Attacked:** §2.1 "`candidate_pairs.tsv` | Stage-2 output".
- **Evidence:** the PDF (p.3–4) says: "If your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last one: whatever your model actually runs inference over." A LightGBM filter is a filtering stage, so emitting its output is literally compliant. However, the PDF also says organisers "audit your blocking" and use the file for "recall ceiling, reduction ratio".
- **Risk:** a reviewer could read stage-2 as a matching model that "narrowed it down". LightGBM (MIT, verified from dist-info metadata) is license-compliant. The romanizer uses only stdlib `unicodedata` plus hand-typed consonant/vowel tables. That is linguistic knowledge, not an external lookup, and I find no clause against it. The dictionary is learned only from provided training data. SPECULATION: the risk of disqualification is low.
- **Alternative:** in `Documentation_template.md`, describe stage-2 explicitly as a learned blocking filter trained only on train labels, and report stage-1 and stage-2 recall and reduction ratio side by side. Keep the stage-1 file reproducible via a CLI flag.

### m3 [MINOR]: Internal contradictions and ambiguities an autonomous executor will trip on
1. `mean_size_budget: 100` (§3.5) vs "Plan targets mean ≤ 80" (`PLAN_PROGRESS.md` Q1). At 13 bytes per ID the file is 1.83 GB at mean 80 and 2.28 GB at mean 100 (`out/f_throughput.json` → `candidate_pairs_tsv_bytes_at_mean`), and the zip limit is unknown (Q1).
2. F12 cites 79.7% token equality (reference romanizer), while `PLAN_PROGRESS.md` cites 77.3% (older romanizer, `romanize_quality.json`). Both are "verified facts".
3. F16 "quota union 0.962 at 497" was measured with channels B/N/A/P/**ROM** from the old aux index (`diag_quota.json`), not the plan's BOTH/NAME/ADDR/PAIR/PHON/CAT on normalization v2. The E4 target (≥ 0.970 at mean ≤ 1,000) has no measured support: the best old point was 0.9681 at 888 and 0.9717 at 1,755.
4. The E1 gate (0.92 @150) comes from a different split and normalizer. If normalization v2 changes df (it will: romanized tokens now exist), a failure is ambiguous between a bug and a real change.
5. The chain rule uses d_EN, while the segment definition in §5.9 uses the S1 chain size. These are different quantities (see M4), and the plan uses both under the word "chain".
6. `NAME_ABBREV` maps `co` → `company` in every country. That is harmless for France (raw name `co` appears in 0.03% of France S1 records), but the "train-derived" claim in §5.2 is not reproducible: no script derives these lists.

### m4 [MINOR]: The NC concat channel misses word-order flips inside domains
- **Evidence:** `reports/eda_report.md` noise pattern 8 shows `Continental Foundation LLC` vs `foundationcontinental.com`. `concat_variants` produces `continentalfoundation` on the query side and `foundationcontinental` on the pool side, so they never meet. The report counts 1,757 domain-name positives in its 44K sample, but I did not measure how many are flipped (SPECULATION).
- **Alternative:** on the pool side, index long tokens (≥ 8 characters) also by their sorted character 4-gram set, or rely on NG. Measure on train before adding anything.

---

## Data facts re-verified (all links, not samples)
All from `out/b_gt_facts.json` unless noted.
- GT: 2,206,821 rows; 123,247 singletons; 7,638,365 links; max 11; **one-owner holds** (7,638,365 unique pool records); 2,681,854 orphan pool records (26.0%).
- **Cross-country true links: 0** of 7,638,365 (US→US 4,578,522; India→India 3,059,843).
- True pool matches with an empty name: **0**. Pool names empty after normalization: 0.
- True pairs with an empty pool address: **337,018 (4.41%)**. Pool addresses consisting only of `<NULL>`/punctuation: **0** in the whole pool. Empty pool addresses: 344,883 (F6 confirmed). 6,225 non-singleton entities have *all* their matches with an empty address. 798 true pairs have both an Indic name and an empty address.
- `<NULL>` rows (grep): train S2 43,799, S3 43,572; test S2 35,024, S3 35,153; never in names (F7 confirmed).
- True pairs whose pool name has Indic letters: 551,240 (7.2%). 40,636 entities have only Indic-named matches.

## Things the plan got right (evidence-based)
- Parsing, ID and row-count facts F1–F7 and F14 hold on the full data (above).
- Treating country as a soft feature is safe on train (0 cross-country links), and correctly not hard-coded for France.
- Romanizing instead of erasing Indic names is necessary: 551,240 true pairs have Indic pool names, and the old normalizer erased 699,389 pool names (`diag_scripts.json`).
- The learned dictionary is the right tool and does not memorise single entities: 0 of 446 entries rest on one S1 entity, and usable-token overlap goes 26% → 61% on held-out pairs (`d_evidence.json`).
- The quota union is better than one additive score at equal size (0.9623 vs 0.9469 at about 500; `diag_quota.json`).
- df caps as fractions of the pool, a streaming validator, `\n`-only output, spawn-safe mmap indexes, and staleness hashing are all sound engineering.
- EN buckets never exceed the cap (max pool EN df 1,071, cap 20,640), so d_EN is always available.

## Unverified / speculation
- Whether wheels for the pinned versions exist for Python 3.12, and whether Unicode 15.0 vs 16.0 changes any golden. Not testable here: no Python 3.12 in the project.
- Real stage-1 latency with 10 families on normalization v2, and the real tree count of the stage-2 model. My timing uses a synthetic 500-tree model.
- France recall or ranking quality. It cannot be measured without labels. The structural evidence (B1) only shows risk.
- Test orphan share of about 39.8% is an ESTIMATE that assumes the train link rate.
- Recall of a TF-IDF char-gram top-N or sorted-neighbourhood baseline on the full pool: not run. Cross-key channel sizes are sum-of-df upper bounds, not exact unions.
- How organisers will judge a learned stage-2 filter (m2). Upload size limit (Q1).
- Share of flipped-order domain names (m4).

# Deep-research brief: pushing a business entity-resolution pipeline past F0.5 0.981

Paste this whole file into a deep-research session. It is self-contained. All numbers come from our own runs
(metrics files in `artifacts/metrics/`, narrative in `RESEARCH_LOG.md`). "Internal" = held-out train S1; "LB" = the
public leaderboard.

---

## 1. The task

- **Amazon ML Challenge 2026, business entity resolution.** Each Source-1 (S1) business record must be linked to every
  record of the same real business in a pool made of Source 2 + Source 3 (S2, S3).
- **Fields:** business name, free-text address, country. No IDs, no phones, no categories.
- **Scale:**
  - Train: S1 2,206,821; pool 10,320,219 (S2 5,034,616 + S3 5,285,603); 7,638,365 true links.
  - Test: S1 1,732,544; pool 9,969,589.
- **Structure:** an S1 has 0 to 11 true pool records (mean ~3.45; 5.6% have none). No pool record belongs to two S1. True links
  never cross countries. Train countries: US and India. **Test adds France, never seen in train.**
- **Metric:** F0.5 computed per S1 and macro-averaged over all S1. A singleton S1 (no true match) scores 1.0 only if we
  predict nothing, else 0. F0.5 weights precision 2x: one wrong merge costs about as much as 3.6 correct matches gain.
- **Scoring / ranking:** public leaderboard on test, plus a final review of code and `candidate_pairs.tsv`. **Organisers
  said a smaller candidate set per S1 ranks higher** at equal quality, and blocking must scale to Amazon size (billions).

## 2. Hard rules (any suggestion must obey these)

- Only the provided data. **No external business databases, web enrichment, geocoders, gazetteers, libpostal, hosted
  LLM/APIs, or external labels.** No scraping, no invented records.
- Pretrained models are allowed if open-licensed (MIT / Apache-2.0) and ≤ 8B parameters, run by our own code.
- No test labels exist; nothing may be tuned on test. No pseudo-labelling on test.
- Country is an open set: no US/India/France hardcoding.

## 3. Resources left

- About 7 hours until the deadline, one leaderboard submission left.
- An 8-core Apple M3 laptop (CPU + MPS), about $4 of Hugging Face GPU credit (A100 ~$2.50/h), free Kaggle 2xT4 (~21 GPU-h
  left this week).
- Measured throughputs:
  - bge-reranker-v2-m3 scoring: 3,600 pairs/s on A100, ~990 on L4, 76 on the M3.
  - LoRA training of the same model: 161 pairs/s on one T4.
  - multilingual-e5-small embedding: ~14-18K texts/s on A100.
  - LightGBM on 180K S1 (975K pairs): 2-5 min per 5-fold run on the laptop.

## 4. Current system (S10, public LB 0.98135)

1. **Normalisation.** Unicode NFC, romanisation of 9 Indic scripts, accent folding, legal-suffix and punctuation cleanup,
   abbreviations mined from the unlabelled pool, and a transliteration dictionary learned from train pairs.
2. **Stage 1: lexical blocking.** An inverted index over 10 key families:
   - whole name, name tokens, name character 3-grams, first-8 token pairs, Metaphone codes, consonant-skeleton pairs,
     joined/split name variants;
   - address tokens, address bigrams, name-skeleton × rare-address-token.
   A logistic ranker over per-family scores plus per-channel quotas keeps ~290 candidates per S1.
   Ceiling F0.5 0.9957 on a tuning slice.
3. **Stage 2: shortlist.** A LightGBM re-ranker on 35 cheap features keeps rank ≤ 5, or p ≥ 0.05 up to rank 60:
   ~6.1 candidates per test S1, ceiling F0.5 0.9928 on validation S1.
4. **Stage 3: pair features.** 68 handcrafted features (name: token-set/sort, Jaro-Winkler, IDF Jaccard, Monge-Elkan,
   phonetic, char TF-IDF cosine; address: token-set, IDF, house-number and digit agreement, empty flag; context:
   country match, source, Indic flag, chain size, stage-2 score/rank; 14 list-position features).
5. **Cross-encoder c0.** `BAAI/bge-reranker-v2-m3` (568M) with LoRA r=16, α=32 on query/key/value + classifier head.
   - Epoch 1 on 971K DEV_REST_180K pairs (`ce_rec`); epoch 2 on 3.29M new DEV_REST_REST fold-0 pairs (`c0`).
   - Input "S1 name | address" vs "pool name | address", 128 tokens; only 0.06% of pairs exceed that.
   - MATCH_VALID AUC 0.99896, logloss 0.0402. On empty-address pairs, AUC only 0.946.
   - Used as one extra LightGBM feature (p_ce).
6. **Matcher.** One LightGBM (binary logloss) on ~837K train S1, excluding the S1 the cross-encoder trained on (their
   p_ce would be in-sample). Isotonic calibration, threshold 0.72, then one-owner: each pool record goes only to the S1
   where it scores highest.
7. **DRACE (dense retrieval + copy expansion), added after the matcher:**
   - multilingual-e5-small (118M, no fine-tuning) embeds "name | address".
   - For each S1: its 5 nearest pool records in the same country (exact cosine, GPU matmul).
   - Copy expansion: for each pool record the matcher already accepted, its 2 nearest pool records (other copies of the
     same business).
   - Drop pairs already in the list or already owned. c0 scores the rest (~1.0 per S1); add a pair only if
     p_ce ≥ 0.995; one-owner again.
   - Result on test: 49,859 matches added; final candidate set 7.13 per S1.

## 5. Results history

### Leaderboard

| Submission | What changed | Public LB |
|---|---|---|
| S1 | 3-stage baseline, 50 features | 0.966109 |
| S2 | 68 features (string-sim group) | 0.966797 |
| S3 | + pool-side contention features (list overlaps) | ~0.966 (**below S2**) |
| S6 | + cross-encoder `ce_rec` as a feature | 0.976128 (**+0.0093**) |
| S7 | + name frequency (`nfreq`) | 0.976205 |
| **S10** | c0 (better CE) + DRACE, LightGBM on 837K S1 | **0.98135** (+0.0051) |

The internal→LB gap shrank from −0.0148 (S2) to −0.0093 (S7) to about −0.0071 (S10).

### Internal (180K MATCH_TRAIN S1, 5-fold GroupKFold by S1, paired bootstrap CI vs parent)

| Step | F0.5 | Δ vs parent |
|---|---|---|
| Baseline matcher (M000) | 0.97412 | — |
| 68-feature set (x1_strsim) | 0.97807 | cumulative +0.0040 |
| + p_ce from `ce_rec` (M018) | 0.98438 | +0.00631 |
| p_ce only, no rank/gap columns (M020) | 0.98446 | +0.00008 vs M018 |
| **c0 instead of ce_rec (M027)** | **0.98552** | **+0.00106 [0.00091, 0.00121]** |
| **+ DRACE, c0 ≥ 0.99 (Gate 2, cross-fitted by S1 half)** | **0.98802** | **+0.00251 [0.00239, 0.00264]**; 97.9% precision of added pairs |
| + reverse-name ownership features (M031) | 0.98579 | +0.00028 [0.00016, 0.00039] vs M027 |
| + nfreq (M034) | 0.98572 | +0.00020 [0.00009, 0.00031] vs M027 |
| + reverse-name + nfreq (M035, = S11 recipe) | 0.98589 | +0.00037 vs M027 |

S11 = S10 + reverse-name + nfreq is being built now; expected LB about 0.9817-0.9824 (a guess).

### Tried and did NOT help (do not re-suggest without a new reason)

| Idea | Result |
|---|---|
| CE-derived features: logit, entropy, rank, margins, list z-score, softmax share, CE-vs-stage-2 disagreement, CE × evidence products (7 families) | all within ±0.00008 of p_ce alone |
| Second CE score from the same lineage (ce_rec + c0) | −0.00007 |
| XGBoost blend with LightGBM | +0.0000; raw-p correlation 0.99975 |
| MLP, then MLP with PLR numeric embeddings (tuned), alone and blended | −0.00111 / −0.00023 alone; blend +0.00008 (CI crosses 0) |
| LambdaRank (listwise) objective | −0.0198 |
| Monotone constraints | −0.00102 |
| Target-side competition from candidate-list overlaps (owner rank, margin, claimant counts) | +0.003 internal, **negative on LB**: test pool is denser (5.75 vs 4.68 records per S1) |
| Copy-cluster support / S2-S3 triangulation (text), e5 copy paths to rejected candidates | dropped; copy-path precision ≤ 55% |
| TransClean-style transitive-consistency cleaning with c0 on pool-pool pairs | +0.00000; c0 pool-pool AUC only 0.831; only 0.25% FPs inside multi-match sets |
| Name-twin rescue (accept rejected candidates with the same normalised name as an accepted one) | precision 8%, negative |
| Separate threshold for empty-address pairs | −0.00010 |
| DRACE-specific LightGBM scorer (provenance, cosines, ranks) vs plain c0 threshold | −0.00001 |
| Macro-aware S1 weighting; hard-negative upweighting in LightGBM | dropped |
| Fellegi-Sunter m/u log-likelihood feature | +0.00007, CI crosses 0 |
| Wider stage-2 cut (K_min 5→20) | ceiling +0.002, mostly negatives; not worth the rebuild |
| Qwen3-Reranker-0.6B | untrained AUC 0.824; LoRA run OOM'd on a T4; never compared |

Parked (small positive, not built at full scale): name/address/context specialist models stacked +0.00021; sibling
near-duplicate features +0.00012.

## 6. Where the remaining loss is (S10-equivalent on 180K train S1, F0.5 0.98801)

| Bucket | Pairs | Oracle gain if fixed | Notes |
|---|---|---|---|
| True pair in list, matcher rejects | 9,878 | **+0.00546** | 77% pool address empty; 97% of these S1 already have an accepted true match |
| True pair found by nothing (keywords, e5 top-50) | 4,900 | +0.00271 | 73% empty address, 57% India |
| Wrong pair accepted by LightGBM | 1,395 | +0.00192 | chain branches, different house numbers, near-identical names |
| Found by e5 at rank 6-50, outside DRACE's top 5 | 3,554 | +0.00162 | |
| DRACE retrieved, c0 < 0.995 | 1,216 | +0.00049 | |
| Wrong pair added by DRACE | 87 | +0.00011 | |

**Core diagnosis.** The biggest problem is **name-only pool records** (empty address). Typical case: an S1 "Legacy
Restaurants | 1109 Houserville Rd" already has 4 accepted copies with addresses; the list also holds three name-only
records "Legacy Restaurants", "Légacy-Restaurants", "Legacy Restaurants Inc" — only one is ours, the others belong to
other S1 with the same name. The data generator alters names on these copies (legal forms, accents, splits, websites
like "rammarketing.com"), so exact-name tricks fail. Ownership among same-name S1 is the open question. Checked and
ruled out: IDs and row order carry no signal (Spearman ~0).

## 7. What we want from the research

Find methods, **published or used in real systems**, that could add new information, not more of the same. For each
suggestion give: exact method name, how it works in 1-2 sentences, published evidence (paper + reported numbers + their
dataset), why it should transfer to our data, the specific bucket in section 6 it attacks, cost in our resources
(section 3), and the main risk. Label evidence as (a) measured on data like ours, (b) published on other data,
(c) should work from the mechanism, (d) guess.

Priority questions:

1. **Name-only ownership.** How do multi-source ER systems decide which of several same-name entities owns a record with
   a missing field? Collective / relational ER, entity-level (cluster) matching, probabilistic assignment across S1
   (e.g. optimal transport or assignment under the one-owner constraint), set-to-set matching of an S1's whole group of
   copies. What can separate "Legacy Restaurants Inc" from "Legacy Restaurants" when the address is gone?
2. **Retrieval beyond off-the-shelf e5.** Fine-tuned bi-encoders for blocking (SC-Block, UniBlocker, supervised
   contrastive), multilingual / transliteration-robust retrieval (India 57% of never-found), hybrid sparse+dense fusion,
   learned sparse (SPLADE) for names, multi-vector. How much recall do they add at a *small* candidate budget
   (≤ ~2 extra per S1)?
3. **Cross-encoder training.** Hard-negative curricula that help most for ER (which negatives, how many, iterative
   mining), listwise/selective CE objectives (e.g. ComEM-style select, contrastive margin), field-aware input formats,
   training with the one-owner constraint. Reported gains on ER benchmarks.
4. **Precision on chain branches.** Methods that separate branches of one brand (same name, different house number or
   street), where 34% of our false merges have an empty pool address.
5. **Robustness to an unseen country (France) and denser test pools.** Techniques that avoid our S3 failure (features
   whose distribution shifts with pool density).
6. **Scaling to billions** with small candidate sets: ANN indexes for dense blocking, CE cost reduction (distillation,
   early exit, scoring only uncertain pairs).
7. **Public 2026 challenge solutions**, if any are visible: what they did for blocking, competition features, and
   collective matching, with their reported scores.

Please rank the final list by expected gain on the buckets in section 6 per hour of work, given 7 hours and ~$4 of GPU.

## 8. References we already used

- Ditto: Li et al., "Deep Entity Matching with Pre-Trained Language Models", PVLDB 14(1), 2021 (pretrained cross-encoders for EM).
- BGE M3 / bge-reranker-v2-m3: Chen et al., "BGE M3-Embedding", arXiv 2402.03216, 2024.
- Multilingual E5: Wang et al., "Multilingual E5 Text Embeddings: A Technical Report", arXiv 2402.05672, 2024.
- LoRA: Hu et al., arXiv 2106.09685, 2021.
- LightGBM: Ke et al., NeurIPS 2017.
- SC-Block: Brinkmann, Shraga, Bizer, "Supervised Contrastive Blocking within Entity Resolution Pipelines", ESWC 2024, arXiv 2303.03132.
- UniBlocker: Wang et al., "Towards Universal Dense Blocking for Entity Resolution", arXiv 2404.14831 (dense and sparse blocking are complementary).
- ComEM: "Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching", COLING 2025.
- TransClean: arXiv 2506.04006, 2025 (tested; no gain here).
- "Entity Resolution in Practice: Lessons from a Self-Serve Pipeline", arXiv 2607.26298 (precision needs vetoes, recall needs diverse retrieval).
- Grinsztajn, Oyallon, Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS 2022.
- Gorishniy, Rubachev, Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022.
- Fellegi and Sunter, "A Theory for Record Linkage", JASA 1969.

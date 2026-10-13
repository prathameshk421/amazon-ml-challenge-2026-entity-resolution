# Entity Resolution Challenge — Ideas

## Problem recap
- Match S1 (clean reference) records to S2/S3 records — 0, 1, or many matches per S1 entity.
- No external lookups (no APIs, no geocoding, no registries). Final model must be MIT/Apache, ≤8B params.
- Test set includes France (unseen in training) — nothing hardcoded to US/India only.
- Scoring: F0.5 per S1 entity, macro-averaged. Precision >> recall. Singletons (correctly predicting no match) score 1.0 — worth getting right.
- Two outputs: matching_results.tsv (scored) and candidate_pairs.tsv (checked for blocking recall ceiling/reduction ratio — not scored directly but graded for quality).

## Pipeline shape
1. Normalize (rules/FST derived from actual EDA, not guessed)
2. Block/generate candidates (algo/automata stage — this sets the recall ceiling)
3. Score + classify candidate pairs (ML/DL/NLP stage — this is the "final model" they require)
4. Post-process (consolidate clusters, resolve conflicts)
5. Validate output format before every submission

---

## Algo / Automata
- **BK-tree**: edit distance is a metric (triangle inequality) — build a BK-tree over S2/S3 names, query "everything within distance k" in sub-linear time instead of brute-force pairwise. Cheap to implement (~50 lines), high value. Priority pick for blocking.
- **Levenshtein automaton**: automaton accepting all strings within edit distance k of a query; build once per S1 name, run all candidates through in linear time. More powerful than BK-tree but nontrivial to implement correctly (NFA→DFA) under time pressure — treat as stretch goal.
- **Q-gram/n-gram indexing**: classic pre-ML record linkage blocking — index character n-grams like a database, retrieve records sharing enough n-grams with query. Rarely used in hackathons, real production technique.
- **FST (finite state transducer) for normalization**: deterministic, composable, provably-terminating alternative to a pile of regex (Rd→Road, Pvt→Private, etc). Cleaner than regex hacks for ordering issues.
- **Suffix automaton / generalized suffix tree**: for substring matching and component-reordering in addresses (shuffled parts) — longest common substring between tokenized address strings.
- **Banded edit distance DP**: if exact edit distance value (not just accept/reject) is needed, bound to O(n·k) via diagonal band instead of full O(n·m).
- **Union-Find (disjoint set)**: after pairwise match decisions, consolidate into consistent clusters instead of raw pairs — fixes inconsistency (A~B, B~C but A~C untested). Cheap, easy, worth doing regardless of model choice.
- **Bipartite/Hungarian matching**: for resolving conflicting high-confidence matches via optimal assignment instead of greedy first-match-wins.
- **Fellegi-Sunter probabilistic record linkage**: classic formal ER theory (pre-ML) — computes match/non-match likelihood ratios per field, combines with data-derived weights. Strong methodology-doc angle, ties directly into "principled, not buzzword" framing.

### Heuristics (within Algo)
- Fuzzy matching/regex must be generalized from EDA-observed patterns, not hardcoded guesses — needs to generalize to unseen country (France) and unseen noise types.
- Phonetic matching (Soundex/Metaphone/Double Metaphone): catches transliteration variants, cheap extra blocking key or feature.
- Word-order invariant name matching: problem explicitly calls out transpositions — token-set overlap needed alongside sequence-based edit distance.
- Address parsing into components (street/locality/city/state/pincode): compare component-wise since noise types differ per component (landmarks only affect street-level, PIN sometimes just missing).
- Country-specific threshold calibration: check if single global threshold underperforms vs per-country adaptation, without hardcoding logic that breaks on France.
- Chain/franchise trap: same name legitimately at multiple real branches — check ground truth for how this is actually handled before assuming it's always a false match.
- Run utils/validate_submission.py continuously during iteration, not just before final submission.

---

## NLP
- Semantic similarity on business_name (and cautiously on address) — one feature among several, not a standalone decision-maker (pure semantic sim overmatches similar-sounding different businesses).
- Character-level / subword embeddings: more robust to typos and transliteration than word-level embeddings — matches this problem's exact noise patterns.
- SLM/vector models: best use is as an embedding-similarity feature, not as the actual pairwise decision-maker (too slow/unpredictable at scale for a precision-heavy metric).

---

## ML
- EDA-first: check per-column match rate needed (name vs address vs country) to see which columns carry signal before combining into a model.
- Feature-based classifier (GBT/XGBoost/LightGBM): main engine — engineered similarity features (edit distance, Jaccard, TF-IDF cosine, phonetic, address-component-wise) fed into a small interpretable model, easy to tune toward precision.
- Learning-to-rank framing: rank candidates per S1 entity by match likelihood rather than independent binary classify per pair — matches the actual one-to-many structure of the problem.
- Calibrated probabilities (isotonic regression / Platt scaling): F0.5 is precision-sensitive — calibrated output lets threshold be tuned precisely instead of guessing off raw scores.
- Hard negative mining: same-name-different-address pairs (the trap case) are the most valuable training examples — deliberately construct/oversample these rather than relying on random negatives.
- Ensembling multiple signal types (string-edit, phonetic, semantic, address-component) with weighted combination — more robust to any single method's blind spot.

---

## DL / AI
- Fine-tune a small bi-encoder on the actual training pairs (contrastive/triplet loss): strongest DL lever — don't use generic pretrained embeddings as-is, fine-tune on this dataset's own ground-truth positive/negative pairs so it learns this dataset's specific noise style. Fits ≤8B constraint easily. This is what real production matching systems (Ditto-style) do.
- Specialized DL entity-resolution architectures (Ditto/DeepMatcher-style — BERT fine-tuned as sentence-pair classification): legit but heavier lift to get right by deadline — stretch goal, not core plan.
- RL: no clear fit — this is a static classification/ranking problem, not sequential decision-making. Drop unless a specific angle comes up.

---

## Agent strategies (research/exploration tool, not the matcher itself)
- Automated EDA sweep: agent loops through feature combos + quick model fits against train_ground_truth, reports which features actually separate true/false matches — faster than manual eyeballing.
- Ablation search on normalization rules: brute-force test each rule's marginal effect on validation F0.5 to find which actually matter vs which are noise.
- Error bucket analysis: after a baseline model exists, agent categorizes false positives/negatives (transliteration fails, landmark-address fails, chain-store traps, etc) to point at where to spend remaining time.
- Caution: use narrowly as a hypothesis-tester on our own data to compress "what to focus on" — not as a replacement for actually building/tuning the pipeline. Open-ended agent exploration can burn hackathon time without landing anywhere.

---

## Honest fit-check vs problem statement
- Automata/algo ideas (BK-tree, Levenshtein automaton, q-gram indexing) are strongest for **blocking/candidate generation** — this directly matches Amazon's own tip ("invest in a strong blocking strategy") and is exactly what candidate_pairs.tsv is graded on.
- The problem statement explicitly requires a **final trained model** (MIT/Apache, ≤8B) making the match decision — pure heuristics/automata alone as the final decision layer likely won't satisfy this or tune as well for precision. Need: automata/algo for blocking → ML/DL feature classifier for final decision.
- Priority for tomorrow: EDA on columns → normalization rules from EDA → BK-tree blocking (Levenshtein automaton as stretch) → feature-based classifier (GBT) with string/phonetic/semantic features → calibrate threshold for precision → union-find cleanup → validate often.

## Open questions / to check with real data
- Feature set exact composition for the classifier
- How address noise actually breaks down by type/country in the real data
- Whether chain/franchise cases exist in ground truth and how they're labeled
- Best threshold calibration approach once validation split is built
